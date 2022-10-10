from dataclasses import dataclass, field
import dataclasses
from enum import Enum
import pathlib
import time
from typing import Dict, List, Union
from warnings import warn
from metrics import EpisodeMetricsTracker, EpisodeMetricsEntry, OptimizationTracker
from game import Game, GameConfig, State, StateTensorBatch

from memory import DequeReplayMemory, Transition
import torch
import random
import math
from agents import Action, DefenderAction, DefenderActionTensorBatch, WolpertingerDefenderAgent, Agent

import json 

import numpy as np
from utils import get_device, get_prefix
import utils
from tqdm.notebook import tqdm

from vehicles import VehicleProvider, Vulnerability

criterion = torch.nn.MSELoss()
# PolicyUpdateType = Union["soft","hard"]
@dataclass
class WolpertingerDefenderAgentTrainerConfig:
    batch_size:int
    game_config: GameConfig
    vehicle_provider: VehicleProvider
    train_steps: int
    warmup_steps: int
    max_steps_per_episode: int
    defender_agent: WolpertingerDefenderAgent
    attacker_agent: Agent
    update_policy_interval: int
    train_interval: int
    checkpoint_interval: Union[int, None]
    policy_update_type: str
    metrics_callback: lambda metrics: None = lambda metrics: ()

    def as_dict(self) -> Dict: 
        return {
            "game_config": self.game_config.as_dict(),
            "vehicle_provider": self.vehicle_provider.__class__.__name__,
            "train_steps": self.train_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps_per_episode": self.max_steps_per_episode,
            "defender_agent": str(self.defender_agent),
            "defender_config": self.defender_agent.as_dict(),
            "attacker_agent": str(self.attacker_agent),
            "update_policy_interval": self.update_policy_interval,
            "policy_update_type": self.policy_update_type,
            "checkpoint_interval": self.checkpoint_interval,
            # "metrics_callback": None,
        }

@dataclass
class OptimizationResult:
    loss: float
    diff_min: float
    diff_max: float
    diff_mean: float
    policy_loss: float
    policy_updated: bool

    @staticmethod
    def default():
        return OptimizationResult(0,0,0,0,0,False)

class WolpertingerDefenderAgentTrainer:
    gamma:float = 0.99
    tau:float = 0.001
    eps_start:float = 0.9
    eps_end:float = 0.05
    # eps_decay:float = 200
    eps_decay:float = 10000
    target_update_interval:int = 10

    game: Game
    prev_state: State
    metrics_history: List[EpisodeMetricsTracker]
    optimization_metrics: OptimizationTracker
    step: int
    optim_step: int
    episode: int
    episode_step: int

    def __init__(self, config: WolpertingerDefenderAgentTrainerConfig) -> None:
        self.config = config

    def get_epsilon_threshold(self) -> float:
        # https://www.desmos.com/calculator/ylgxqq5rvd
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.optim_step / self.eps_decay)
 
    def get_action(self, agent: WolpertingerDefenderAgent, state: State) -> DefenderAction:
        if random.random() > self.get_epsilon_threshold():
            return agent.get_action(state)
        else:
            return agent.get_random_action(state)

    def track_run_start(
        self,
        config: WolpertingerDefenderAgentTrainerConfig,
        save_dir: str,
    ):
        path = pathlib.Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{utils.get_prefix()} config.json"
        with open(path, "w") as f:
            json.dump(config.as_dict(), f, indent=4)

    def prepare_next_episode(self) -> None:
        self.current_episode_metrics = EpisodeMetricsTracker()
        self.game = Game(config=self.config.game_config, vehicle_provider=self.config.vehicle_provider)
        self.prev_state = self.game.state
        self.episode_step = 0
        self.current_episode_metrics.track_stats(trainer=self)
        self.episode += 1

    def explore_step(self) -> None:
        _, defender_action = self.game.take_step(
            attacker_agent=self.config.attacker_agent,
            defender_agent=self.config.defender_agent
        )
        done = self.episode_step == self.config.max_steps_per_episode - 1
        reward = 0 if done else self.config.defender_agent.get_utility(self.game.state)
        next_state = None if done else self.game.state
        self.memory.push(Transition(
            state=self.prev_state,
            next_state=next_state,
            action=defender_action,
            reward=reward,
            terminal=done
        ))
        if done:
            self.prepare_next_episode()
        else:
            self.prev_state = next_state
            self.episode_step += 1


    def take_step(self) -> None:
        print(f"{get_prefix()} episode {self.episode} step {self.episode_step} ", end="")

        self.explore_step()

        print("optimizing ", end="")
        optim = self.optimize_policy()
        print(f"loss={optim.loss:.4f} diff={{max={optim.diff_max:.4f}, min={optim.diff_min:.4f}, mean={optim.diff_mean:.4f}}} policy_loss={optim.policy_loss:.4f} ", end="")
        if optim.policy_updated:
            print("policy updated! ", end="")
        self.current_episode_metrics.track_stats(trainer=self)
        self.optimization_metrics.track_stats(optim)

        should_checkpoint = self.config.checkpoint_interval is not None and self.optim_step % self.config.checkpoint_interval == 0 and self.optim_step > 0
        if should_checkpoint:
            self.config.defender_agent.save(dir="checkpoints")
            print("checkpointed! ", end="")
    
        print() # write newline
        self.metrics_history.append(self.current_episode_metrics)
        # allow for tracking metrics while being able to cancel training
        self.config.metrics_callback(self.current_episode_metrics)
        self.step += 1

    def train(
        self,
    ) -> List[List[EpisodeMetricsEntry]]:
        
        self.metrics_history = []
        self.episode = 0
        self.optim_step = 0
        self.step = 0
        self.optimization_metrics = OptimizationTracker()
        self.track_run_start(self.config, save_dir="checkpoints")
        self.memory = DequeReplayMemory(10000)
        self.prepare_next_episode()

        print("Warming up...")        
        for _ in tqdm(range(max(self.config.warmup_steps, self.config.batch_size))):
            self.explore_step()
            self.step += 1
        print("Warmup complete~!")

        while self.optim_step < self.config.train_steps:
            self.take_step()

        return self.metrics_history

    def optimize_policy(self) -> OptimizationResult:
        batch = self.memory.sample(self.config.batch_size)
        shape_data = self.config.defender_agent.state_shape_data
            
        state_batch = [v.state.as_tensors(shape_data) for v in batch]
        state_batch = StateTensorBatch(
            vulnerabilities=torch.cat([v.vulnerabilities for v in state_batch]).to(get_device()),
            vehicles=torch.cat([v.vehicles for v in state_batch]).to(get_device()),
        )
        
        assert state_batch.vehicles.shape == (self.config.batch_size, shape_data.num_vehicles, shape_data.num_vehicle_features)
        assert state_batch.vulnerabilities.shape == (self.config.batch_size, shape_data.num_vehicles, shape_data.num_vulns, shape_data.num_vuln_features)
        assert state_batch.vehicles.shape[0] == state_batch.vulnerabilities.shape[0]

        action_batch = [v.action.as_tensor(shape_data) for v in batch]
        action_batch = DefenderActionTensorBatch(
            members=torch.cat([v.members for v in action_batch]).to(get_device()),
            monitor=torch.cat([v.monitor for v in action_batch]).to(get_device()),
        )

        assert action_batch.members.shape == (self.config.batch_size, 1, shape_data.num_vehicles)
        assert action_batch.monitor.shape == (self.config.batch_size, 1, shape_data.num_vehicles)

        # get the batch of next states for calculating q
        def coalesce_next_state(transition: Transition) -> StateTensorBatch:
            if transition.next_state is not None:
                return transition.next_state.as_tensors(shape_data)
            else:
                shape = State.get_shape(shape_data, batch_size=1)
                return StateTensorBatch(
                    vulnerabilities=torch.zeros(shape.vulnerabilities),
                    vehicles=torch.zeros(shape.vehicles),
                )
        next_state_batch = [coalesce_next_state(entry) for entry in batch]
        next_state_batch = StateTensorBatch(
            vulnerabilities=torch.cat([v.vulnerabilities for v in next_state_batch]).to(get_device()),
            vehicles=torch.cat([v.vehicles for v in next_state_batch]).to(get_device()),
        )

        assert next_state_batch.vulnerabilities.shape == state_batch.vulnerabilities.shape
        assert next_state_batch.vehicles.shape == state_batch.vehicles.shape

        with torch.no_grad():
            # get proto actions for the next states
            # should return [batch_size, 1, num_vehicles] tensors for members and monitor
            proto_actions:DefenderActionTensorBatch = self.config.defender_agent.actor_target(next_state_batch)
            assert proto_actions.members.shape == proto_actions.monitor.shape == (self.config.batch_size, 1, shape_data.num_vehicles)

            # get the evaluation of the proto actions in their state
            next_q_values = self.config.defender_agent.critic_target(
                next_state_batch,
                proto_actions,
            )
            # for each batch (which has 1 proto action), there should be 1 q value
            assert next_q_values.shape == (self.config.batch_size, 1)

            # boolean vector indicating states that aren't terminal
            non_terminal_states = [not v.terminal for v in batch]
            
            # target q is initialized from the actual observed reward
            target_q_batch = torch.as_tensor([v.reward for v in batch], dtype=torch.float32).to(get_device())
            # target q is increased by discounted predicted future reward
            target_q_batch[non_terminal_states] += self.gamma * next_q_values[non_terminal_states].flatten()
            assert target_q_batch.shape == (self.config.batch_size,)

        #region critic update
        # reset the gradients
        self.config.defender_agent.critic.zero_grad()
        # predict/grade the proposed actions
        q_batch = self.config.defender_agent.critic(state_batch, action_batch)
        assert q_batch.shape == (self.config.batch_size, 1)
        # get the loss for the predicted grades
        value_loss: torch.Tensor = criterion(q_batch.flatten(), target_q_batch)
        diff = (q_batch.flatten() - target_q_batch).abs()

        # track the loss to model weights
        value_loss.backward()
        # apply model weight update
        self.config.defender_agent.critic_optimizer.step()
        #endregion critic update

        #region actor update
        # reset the gradients
        self.config.defender_agent.actor.zero_grad()
        # get proposed actions from actor, then get the critic to grade them
        # loss goes down as the critic makes better assessments of the proposed actions
        policy_loss: torch.Tensor = -1 * self.config.defender_agent.critic(state_batch, self.config.defender_agent.actor(state_batch))
        # ensure the actor proposes mostly good (according to the critic) actions
        policy_loss = policy_loss.mean()
        # back propagate the loss to the model weights
        policy_loss.backward()
        # apply model weight update
        self.config.defender_agent.actor_optimizer.step()
        #endregion actor update

        # Soft update wasn't training fast, trying hard update
        should_update_policy = self.optim_step % self.config.update_policy_interval == 0
        if should_update_policy:
            if self.config.policy_update_type == "soft":
                # from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py#L26
                def soft_update(target, source, tau):
                    for target_param, param in zip(target.parameters(), source.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - tau) + param.data * tau
                        )
                soft_update(self.config.defender_agent.actor_target, self.config.defender_agent.actor, self.tau)
                soft_update(self.config.defender_agent.critic_target, self.config.defender_agent.critic, self.tau)
            elif self.config.policy_update_type == "hard":
                self.config.defender_agent.actor_target.load_state_dict(self.config.defender_agent.actor.state_dict())
                self.config.defender_agent.critic_target.load_state_dict(self.config.defender_agent.critic.state_dict())
            else:
                raise ValueError(f"unknown policy update type: \"{self.config.policy_update_type}\"")

        return OptimizationResult(
            loss=float(value_loss.detach().cpu().numpy()),
            diff_max = float(diff.max().detach().cpu().numpy()),
            diff_min = float(diff.min().detach().cpu().numpy()),
            diff_mean = float(diff.mean().detach().cpu().numpy()),
            policy_loss=float(policy_loss.detach().cpu().numpy()),
            policy_updated=should_update_policy,
        )
        
