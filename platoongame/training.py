# from dataclasses import dataclass, field
# from typing import List, Tuple, Union
# import gym
# import math
# import random
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple
# from itertools import count
# from PIL import Image
# from evaluation import Evaluator, Metrics
# import vehicles

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

# from agents import Action, Agent, DefenderAction, AttackerAction, BasicAttackerAgent, BasicDefenderAgent
# from logging import Logger
# from game import Game, State,GameConfig
# from utils import get_logger, get_device
# from memory import ReplayMemory, Transition

from dataclasses import dataclass, field
from typing import List, Union
from evaluation import Evaluator, Metrics
from game import State, StateTensors

from memory import DequeReplayMemory, Transition
import torch
import random
import math
from agents import Action, DefenderAction, DefenderActionTensors, WolpertingerDefenderAgent, Agent

import numpy as np

from vehicles import Vulnerability

criterion = torch.nn.MSELoss()

@dataclass
class WolpertingerDefenderAgentTrainer:
    batch_size:int = 128
    gamma:float = 0.99
    tau:float = 0.001
    eps_start:float = 0.9
    eps_end:float = 0.05
    eps_decay:float = 200
    target_update_interval:int = 10
    steps_done:int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.steps_done = 0
        self.memory = DequeReplayMemory(10000)
    
    def get_action(self, agent: Agent, state: State) -> DefenderAction:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if random.random() > eps_threshold:
            return agent.get_action(state)
        else:
            return agent.get_random_action(state)

    def train(
        self,
        episodes: int,
        max_steps_per_episode: int,
        defender_agent: WolpertingerDefenderAgent,
        attacker_agent: Agent,
        evaluator: Evaluator,
        warmup: int,
    ) -> List[List[Metrics]]:
        stats_history: List[List[Metrics]] = []
        for episode in range(episodes):
            evaluator.reset()
            evaluator.track_stats() # log starting positions
            from_state = evaluator.game.state

            for step in range(max_steps_per_episode):
                #region manually invoke game loop
                evaluator.game.logger.debug("stepping")

                evaluator.game.logger.debug(f"attacker turn begin")
                action = attacker_agent.get_action(evaluator.game.state)
                evaluator.game.state = attacker_agent.take_action(evaluator.game.state, action)
                evaluator.game.logger.debug(f"attacker turn end")
                
                evaluator.game.logger.debug(f"defender turn begin")
                action = self.get_action(defender_agent, defender_agent, evaluator.game.state)
                evaluator.game.state = defender_agent.take_action(evaluator.game.state, action)
                evaluator.game.logger.debug(f"defender turn end")

                evaluator.game.cycle()
                evaluator.game.step_count += 1
                #endregion manually invoke game loop
                
                done = step == max_steps_per_episode - 1

                # calculate reward
                reward = 0 if done else defender_agent.get_utility(evaluator.game.state)
                # reward = torch.tensor([reward], device=self.device)

                # add to memory
                to_state = None if done else evaluator.game.state
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
                if self.steps_done > warmup:
                    self.optimize_policy(defender_agent)

                # track stats
                evaluator.track_stats()
                self.steps_done += 1
            # log stats before wiping
            stats_history.append(evaluator.stats)

            # # Update the target network, copying all weights and biases in DQN
            # # hard update
            # # todo: soft update
            # if episode % self.target_update_interval == 0:
            #     defender_agent.target_net.load_state_dict(defender_agent.policy_net.state_dict())

        return stats_history

    def optimize_policy(
        self,
        defender_agent: WolpertingerDefenderAgent,
    ) -> None:
        batch = self.memory.sample(self.batch_size)
        def coalesce_next_state(transition: Transition) -> StateTensors:
            if transition.next_state is None:
                return transition.next_state.as_tensors(defender_agent.state_shape_data)
            else:
                return StateTensors(
                    vulnerabilities=torch.zeros(State.get_shape(defender_agent.state_shape_data)[0]),
                    vehicles=torch.zeros(State.get_shape(defender_agent.state_shape_data)[1]),
                )
            
        state_batch = [v.state.as_tensors(defender_agent.state_shape_data) for v in batch]
        state_batch = StateTensors(
            vulnerabilities=torch.stack([v.vulnerabilities for v in state_batch]),
            vehicles=torch.stack([v.vehicles for v in state_batch]),
        )
        action_batch = [v.action.as_tensor() for v in batch]
        action_batch = DefenderActionTensors(
            members=torch.stack([v.members for v in action_batch]),
            monitor=torch.stack([v.monitor for v in action_batch]),
        )

        # get the batch of next states for calculating q
        next_state_batch = [coalesce_next_state(entry) for entry in batch]
        # convert List[Tuple[Tensor,Tensor]] to Tuple[List[Tensor],List[Tensor]]
        next_state_batch = zip(*[(v.vulnerabilities, v.vehicles) for v in next_state_batch])
        # convert Tuple[List[Tensor],List[Tensor]] to Tuple[Tensor,Tensor]
        next_state_batch = StateTensors(
            vulnerabilities=torch.stack(next_state_batch[0]),
            vehicles=torch.stack(next_state_batch[1]),
        )

        with torch.no_grad():
            next_q_values = defender_agent.critic_target(
                next_state_batch,
                defender_agent.actor_target(next_state_batch)
            )
            non_terminal_states = [not v.terminal for v in batch]
            # rewards = torch.as_tensor([v.reward for v in batch])
            target_q_batch = torch.as_tensor([v.reward for v in batch])
            target_q_batch[non_terminal_states] += self.gamma * next_q_values[non_terminal_states]

            #region critic update
            # reset the gradients
            defender_agent.critic.zero_grad()
            # predict/grade the proposed actions
            q_batch = defender_agent.critic(state_batch, action_batch)
            # get the loss for the predicted grades
            value_loss: torch.Tensor = criterion(q_batch, target_q_batch)
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

            #region target update
            #from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py#L26
            def soft_update(target, source, tau):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - tau) + param.data * tau
                    )
            soft_update(defender_agent.actor_target, defender_agent.actor, self.tau)
            soft_update(defender_agent.critic_target, defender_agent.critic, self.tau)
