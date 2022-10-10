from dataclasses import dataclass, field
import pathlib
import time
from typing import List, Union
from warnings import warn
from metrics import EpisodeMetricsTracker, EpisodeMetricsEntry
from game import Game, GameConfig, State, StateTensorBatch

from memory import DequeReplayMemory, Transition
import torch
import random
import math
from agents import Action, DefenderAction, DefenderActionTensorBatch, WolpertingerDefenderAgent, Agent

import numpy as np
from utils import get_device

from vehicles import VehicleProvider, Vulnerability

criterion = torch.nn.MSELoss()

@dataclass
class WolpertingerDefenderAgentTrainer:
    batch_size:int = 128
    gamma:float = 0.99
    tau:float = 0.001
    eps_start:float = 0.9
    eps_end:float = 0.05
    # eps_decay:float = 200
    eps_decay:float = 10000
    target_update_interval:int = 10
    steps_done:int = 0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.steps_done = 0
        self.memory = DequeReplayMemory(10000)
    
    def get_epsilon_threshold(self) -> float:
        # https://www.desmos.com/calculator/ylgxqq5rvd
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
 
    def get_action(self, agent: WolpertingerDefenderAgent, state: State) -> DefenderAction:
        if random.random() > self.get_epsilon_threshold():
            return agent.get_action(state)
        else:
            return agent.get_random_action(state)

    def train(
        self,
        game_config: GameConfig,
        vehicle_provider: VehicleProvider,
        episodes: int,
        max_steps_per_episode: int,
        defender_agent: WolpertingerDefenderAgent,
        attacker_agent: Agent,
        warmup: int,
        update_policy_interval: int,
        checkpoint_interval: int,
        metrics_callback: lambda metrics: None = lambda metrics: (),
    ) -> List[List[EpisodeMetricsEntry]]:
        if (warmup < self.batch_size):
            raise ValueError("warmup must be greater than batch size")
        metrics_history: List[EpisodeMetricsTracker] = []
        i = 0
        for episode in range(episodes):
            metrics = EpisodeMetricsTracker()
            game = Game(
                config=game_config,
                vehicle_provider=vehicle_provider
            )
            metrics.track_stats(
                game=game,
                loss=0,
                epsilon_threshold=self.get_epsilon_threshold(),
            ) # log starting positions
            from_state = game.state

            for episode_step in range(max_steps_per_episode):
                i += 1
                now = time.strftime("%Y-%m-%d %H%M-%S")
                print(f"{now} episode {episode} step {episode_step} ", end="")
                
                #region manually invoke game loop
                game.logger.debug("stepping")

                game.logger.debug(f"attacker turn begin")
                action = attacker_agent.get_action(game.state)
                game.state = attacker_agent.take_action(game.state, action)
                game.logger.debug(f"attacker turn end")
                
                game.logger.debug(f"defender turn begin")
                action = self.get_action(defender_agent, game.state)
                game.state = defender_agent.take_action(game.state, action)
                game.logger.debug(f"defender turn end")

                game.cycle()
                game.step_count += 1
                #endregion manually invoke game loop
                
                done = episode_step == max_steps_per_episode - 1

                # calculate reward
                reward = 0 if done else defender_agent.get_utility(game.state)

                # add to memory
                to_state = None if done else game.state
                self.memory.push(Transition(
                    state=from_state,
                    next_state=to_state,
                    action=action,
                    reward=reward,
                    terminal=done
                ))

                # update state
                from_state = to_state

                # optimize
                loss = 0
                # todo: only perform training every {j} steps to allow for more newer memories in buffer
                should_train = self.steps_done > warmup
                if should_train:
                    print("optimizing ", end="")
                    loss = self.optimize_policy(defender_agent)
                    

                    #region target update
                    #from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py#L26
                    # def soft_update(target, source, tau):
                    #     for target_param, param in zip(target.parameters(), source.parameters()):
                    #         target_param.data.copy_(
                    #             target_param.data * (1.0 - tau) + param.data * tau
                    #         )
                    # soft_update(defender_agent.actor_target, defender_agent.actor, self.tau)
                    # soft_update(defender_agent.critic_target, defender_agent.critic, self.tau)

                    # Soft update wasn't training fast, trying hard update
                    if self.steps_done % update_policy_interval == 0:
                        defender_agent.actor_target.load_state_dict(defender_agent.actor.state_dict())
                        defender_agent.critic_target.load_state_dict(defender_agent.critic.state_dict())
                        print("policy update! ", end="")
                        # target_net.load_state_dict(policy_net.state_dict())

                # track stats
                metrics.track_stats(
                    game=game,
                    loss=loss,
                    epsilon_threshold=self.get_epsilon_threshold(),
                )

                if i % checkpoint_interval == 0 and should_train:
                    defender_agent.save(save_dir="checkpoints")
                    print("checkpointed! ", end="")



                self.steps_done += 1
                print()

            # log stats before wiping
            metrics_history.append(metrics)
            # allow for tracking metrics while being able to cancel training
            metrics_callback(metrics)

        return metrics_history

    def optimize_policy(
        self,
        defender_agent: WolpertingerDefenderAgent,
    ) -> None:
        batch = self.memory.sample(self.batch_size)
        shape_data = defender_agent.state_shape_data
            
        state_batch = [v.state.as_tensors(shape_data) for v in batch]
        state_batch = StateTensorBatch(
            vulnerabilities=torch.cat([v.vulnerabilities for v in state_batch]).to(get_device()),
            vehicles=torch.cat([v.vehicles for v in state_batch]).to(get_device()),
        )
        
        assert state_batch.vehicles.shape == (self.batch_size, shape_data.num_vehicles, shape_data.num_vehicle_features)
        assert state_batch.vulnerabilities.shape == (self.batch_size, shape_data.num_vehicles, shape_data.num_vulns, shape_data.num_vuln_features)
        assert state_batch.vehicles.shape[0] == state_batch.vulnerabilities.shape[0]

        action_batch = [v.action.as_tensor(shape_data) for v in batch]
        action_batch = DefenderActionTensorBatch(
            members=torch.cat([v.members for v in action_batch]).to(get_device()),
            monitor=torch.cat([v.monitor for v in action_batch]).to(get_device()),
        )

        assert action_batch.members.shape == (self.batch_size, 1, shape_data.num_vehicles)
        assert action_batch.monitor.shape == (self.batch_size, 1, shape_data.num_vehicles)

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
            proto_actions:DefenderActionTensorBatch = defender_agent.actor_target(next_state_batch)
            assert proto_actions.members.shape == proto_actions.monitor.shape == (self.batch_size, 1, shape_data.num_vehicles)

            # get the evaluation of the proto actions in their state
            next_q_values = defender_agent.critic_target(
                next_state_batch,
                proto_actions,
            )
            # for each batch (which has 1 proto action), there should be 1 q value
            assert next_q_values.shape == (self.batch_size, 1)

            # boolean vector indicating states that aren't terminal
            non_terminal_states = [not v.terminal for v in batch]
            
            # target q is initialized from the actual observed reward
            target_q_batch = torch.as_tensor([v.reward for v in batch], dtype=torch.float32).to(get_device())
            # target q is increased by discounted predicted future reward
            target_q_batch[non_terminal_states] += self.gamma * next_q_values[non_terminal_states].flatten()
            assert target_q_batch.shape == (self.batch_size,)

        #region critic update
        # reset the gradients
        defender_agent.critic.zero_grad()
        # predict/grade the proposed actions
        q_batch = defender_agent.critic(state_batch, action_batch)
        assert q_batch.shape == (self.batch_size, 1)
        # get the loss for the predicted grades
        value_loss: torch.Tensor = criterion(q_batch.flatten(), target_q_batch).mean()
        # print(f"loss={value_loss} predicted={q_batch.flatten()} target={target_q_batch}", end="")
        print(f"loss={value_loss:.4f} ({value_loss / self.batch_size:.4f} , {(q_batch.flatten() - target_q_batch).abs().max():.4f}) ", end="")
        
        # track the loss to model weights
        value_loss.backward()
        # apply model weight update
        defender_agent.critic_optimizer.step()
        #endregion critic update

        #region actor update
        # reset the gradients
        defender_agent.actor.zero_grad()
        # get proposed actions from actor, then get the critic to grade them
        # loss goes down as the critic makes better assessments of the proposed actions
        policy_loss: torch.Tensor = -1 * defender_agent.critic(state_batch, defender_agent.actor(state_batch))
        # ensure the actor proposes mostly good (according to the critic) actions
        policy_loss = policy_loss.mean()
        # back propagate the loss to the model weights
        policy_loss.backward()
        # apply model weight update
        defender_agent.actor_optimizer.step()
        #endregion actor update

        return float(value_loss.detach().cpu().numpy())
