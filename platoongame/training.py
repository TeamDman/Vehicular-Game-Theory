from dataclasses import dataclass
import pathlib
from torch import Tensor
from typing import Dict, Union
from metrics import TrainingMetricsTracker
from game import Game, GameConfig, StateTensorBatch

from memory import ReplayMemory, Transition, TransitionTensorBatch
import torch
import math
from agents import AttackerAgent, DefenderActionTensorBatch, RandomDefenderAgent, WolpertingerDefenderAgent

import json 

from utils import get_device
import utils
from tqdm.notebook import tqdm


from vehicles import VehicleProvider

criterion = torch.nn.MSELoss()
# PolicyUpdateType = Union["soft","hard"]
@dataclass
class WolpertingerDefenderAgentTrainerConfig:
    batch_size:int
    game_config: GameConfig
    vehicle_provider: VehicleProvider
    train_steps: int
    warmup_replay: int
    max_steps_per_episode: int
    defender_agent: WolpertingerDefenderAgent
    attacker_agent: AttackerAgent
    update_policy_interval: int
    exploration_per_step: int
    checkpoint_interval: Union[int, None]
    attacker_headstart: int
    policy_update_type: str

    metrics_tracker: TrainingMetricsTracker
    memory: ReplayMemory[TransitionTensorBatch]

    reward_gamma: float
    soft_update_tau: float

    def as_dict(self) -> Dict: 
        return {
            "game_config": self.game_config.as_dict(),
            "vehicle_provider": self.vehicle_provider.__class__.__name__,
            "train_steps": self.train_steps,
            "warmup_steps": self.warmup_replay,
            "max_steps_per_episode": self.max_steps_per_episode,
            "defender_agent": str(self.defender_agent),
            "attacker_headstart": self.attacker_headstart,
            "defender_config": self.defender_agent.as_dict(),
            "attacker_agent": str(self.attacker_agent),
            "update_policy_interval": self.update_policy_interval,
            "policy_update_type": self.policy_update_type,
            "checkpoint_interval": self.checkpoint_interval,
            "reward_gamma": self.reward_gamma,
            "soft_update_tau": self.soft_update_tau,
            "memory": {
                "maxlen": self.memory.get_max_len(),
            },
        }

    def dump(self, dir: str, prefix: Union[None, str]) -> None:
        path = pathlib.Path(dir)
        path.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = utils.get_prefix()
        path = path / f"{prefix} config.json"
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=4)

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
    game: Game
    step: int
    optim_step: int
    episode: int
    episode_step: int
    random_defender_agent: RandomDefenderAgent

    def __init__(self, config: WolpertingerDefenderAgentTrainerConfig) -> None:
        assert config.memory.get_max_len() >= config.batch_size
        self.game = Game(config=config.game_config, vehicle_provider=config.vehicle_provider)
        self.step = 0
        self.optim_step = 0
        self.episode = 0
        self.episode_step = 0
        self.config = config
        self.random_defender_agent = RandomDefenderAgent()

    def prepare_next_episode(self) -> None:
        self.game = Game(config=self.config.game_config, vehicle_provider=self.config.vehicle_provider)
        # have the attacker get a free starting action to ensure the defender can generalize
        # my theory is that having a starting state with no compromises is not good for the defender's ability to learn
        for _ in range(self.config.attacker_headstart):
            self.game.state = self.config.attacker_agent.take_action(self.game.state, self.config.attacker_agent.get_action(self.game.state))

        self.episode_step = 0
        self.episode += 1

    def take_explore_step(self, random: bool) -> float: # returns reward
        prev_state = self.game.state
        _, defender_action = self.game.take_step(
            attacker_agent=self.config.attacker_agent,
            defender_agent=self.config.defender_agent if not random else self.random_defender_agent
        )
        reward = self.config.defender_agent.get_utility(self.game.state)
        next_state = self.game.state
        transition = Transition(
            state=prev_state,
            next_state=next_state,
            action=defender_action,
            reward=reward,
            terminal=False
        )
        transition = transition.as_tensor_batch(self.config.defender_agent.state_shape_data)
        self.config.memory.push(transition)
        if self.episode_step == self.config.max_steps_per_episode - 1:
            self.prepare_next_episode()
        else:
            self.episode_step += 1
        
        return reward


    def take_optim_step(self) -> None:
        print("optimizing ", end="")
        optim = self.optimize_policy()
        print(f"loss={optim.loss:07.4f} diff={{max={optim.diff_max:07.4f}, min={optim.diff_min:07.4f}, mean={optim.diff_mean:07.4f}}} policy_loss={optim.policy_loss:07.4f} ", end="")
        if optim.policy_updated:
            print("policy updated! ", end="")
        self.config.metrics_tracker.track_stats(
            optimization_results=optim,
        )

        should_checkpoint = self.config.checkpoint_interval is not None \
            and self.config.checkpoint_interval != -1 \
            and self.optim_step % self.config.checkpoint_interval == 0 \
            and self.optim_step > 0
        if should_checkpoint:
            self.config.defender_agent.save(dir="checkpoints")
            print("checkpointed! ", end="")
    
        print() # write newline
        # allow for tracking metrics while being able to cancel training
        self.step += 1
        self.optim_step += 1

    def warmup(self) -> None:
        self.episode = 0
        self.optim_step = 0
        self.step = 0
        self.prepare_next_episode()
        print("Warming up...")
        # desired minimum replay size after warmup
        warmup_steps = max(self.config.warmup_replay, self.config.batch_size)
        # find remaining steps needed
        warmup_steps = warmup_steps - len(self.config.memory)
        warmup_steps = max(0, warmup_steps)
        assert warmup_steps <= self.config.memory.get_max_len()
        for _ in tqdm(range(warmup_steps)):
            self.take_explore_step(random=True)
            self.step += 1
        print("Warmup complete~!")

    def train(self) -> None:
        self.warmup()
        # ensure the agent knows to use epsilon decay
        self.config.defender_agent.training = True
        digits = math.ceil(math.log10(self.config.train_steps))
        for i in tqdm(range(self.config.train_steps)):
            print(f"train step {i:0{digits}d} ", end="")

            # exploration between optimization steps
            rewards = [
                self.take_explore_step(random=False)
                for _ in range(self.config.exploration_per_step)
            ]
            print(f"reward={{max={max(rewards):07.4f}, min={min(rewards):07.4f}, mean={sum(rewards)/len(rewards):07.4f}}} ", end="")

            self.take_optim_step()

    def optimize_policy(self) -> OptimizationResult:
        batch = self.config.memory.sample(self.config.batch_size)
        batch = TransitionTensorBatch.cat(batch).to(get_device())
        shape_data = self.config.defender_agent.state_shape_data
            
        assert batch.state.vulnerabilities.shape == (self.config.batch_size, shape_data.num_vehicles, shape_data.num_vulns, shape_data.num_vuln_features)
        assert batch.next_state.vulnerabilities.shape == batch.state.vulnerabilities.shape
        assert batch.action.members.shape == (self.config.batch_size, shape_data.num_vehicles)
        assert batch.reward.shape == (self.config.batch_size,)
        assert batch.terminal.shape == (self.config.batch_size,)

        # zero out state for terminal transitions
        terminal_indices = batch.terminal == True
        non_terminal_indices = ~terminal_indices
        zero_state = StateTensorBatch.zeros(shape_data=shape_data, batch_size=int(terminal_indices.sum())).to(get_device())
        batch.next_state.vulnerabilities[terminal_indices] = zero_state.vulnerabilities

        # # calculate q value for next state
        # proto_actions:DefenderActionTensorBatch = self.config.defender_agent.actor_target(batch.next_state)
        # assert proto_actions.members.shape == (self.config.batch_size, shape_data.num_vehicles)
        # with torch.no_grad():
        #     next_q_values: Tensor = self.config.defender_agent.critic_target(batch.next_state, proto_actions)
        #     assert next_q_values.shape == (self.config.batch_size,)
        # next_q_values.requires_grad_()

        # # target q is discounted future reward calculated using q values from critic assessment of actor's action
        # target_q_batch = batch.reward.clone()
        # target_q_batch[non_terminal_indices] += self.config.reward_gamma * next_q_values[non_terminal_indices]
        # assert target_q_batch.shape == (self.config.batch_size,)

        # self.config.defender_agent.critic.zero_grad()
        # q_batch = self.config.defender_agent.critic(batch.state, batch.action)
        # assert q_batch.shape == (self.config.batch_size,)
        # # critic should be predicting reward considering the discounted future reward
        # value_loss: torch.Tensor = criterion(q_batch, target_q_batch)
        # value_loss.backward()
        # self.config.defender_agent.critic_optimizer.step()

        # critic loss goes down as it makes better predictions of immediate reward
        # future reward discounts aren't needed since the game isn't stateful
        self.config.defender_agent.critic.zero_grad()
        q_batch = self.config.defender_agent.critic_target(batch.state, batch.action)
        assert q_batch.shape == (self.config.batch_size,)
        assert batch.reward.shape == (self.config.batch_size,)
        value_loss: Tensor = criterion(q_batch, batch.reward)
        value_loss.backward()
        self.config.defender_agent.critic_optimizer.step()
        

        # actor loss goes down as the critic makes better assessments of the proposed actions
        self.config.defender_agent.actor.zero_grad()
        policy_loss: torch.Tensor = -1 * self.config.defender_agent.critic(batch.state, self.config.defender_agent.actor_target(batch.state)).mean()
        policy_loss.backward()
        self.config.defender_agent.actor_optimizer.step()

        should_update_policy = self.optim_step % self.config.update_policy_interval == 0
        if should_update_policy:
            if self.config.policy_update_type == "soft":
                # from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py#L26
                def soft_update(target, source, tau):
                    for target_param, param in zip(target.parameters(), source.parameters()):
                        ## shouldn't be necessary since we use target networks to calculate loss
                        # if isinstance(target_param, torch.nn.parameter.UninitializedParameter):
                        #     # target model uninitialize, hard update
                        #     target_param.data.copy_(param.data)
                        # else:
                        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                soft_update(self.config.defender_agent.actor_target, self.config.defender_agent.actor, self.config.soft_update_tau)
                soft_update(self.config.defender_agent.critic_target, self.config.defender_agent.critic, self.config.soft_update_tau)
            elif self.config.policy_update_type == "hard":
                self.config.defender_agent.actor_target.load_state_dict(self.config.defender_agent.actor.state_dict())
                self.config.defender_agent.critic_target.load_state_dict(self.config.defender_agent.critic.state_dict())
            else:
                raise ValueError(f"unknown policy update type: \"{self.config.policy_update_type}\"")


        diff = (q_batch - batch.reward).abs()
        rtn = OptimizationResult(
            loss=float(value_loss.detach().cpu().numpy()),
            diff_max = float(diff.max().detach().cpu().numpy()),
            diff_min = float(diff.min().detach().cpu().numpy()),
            diff_mean = float(diff.mean().detach().cpu().numpy()),
            policy_loss=float(policy_loss.detach().cpu().numpy()),
            policy_updated=should_update_policy,
        )
        
        # del target_q_batch
        del q_batch
        del zero_state
        del terminal_indices
        del non_terminal_indices
        del diff

        return rtn
