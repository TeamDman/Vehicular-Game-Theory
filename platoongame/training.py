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

#region old
# @dataclass
# class DefenderAgentTrainer:
#     batch_size:int = 128
#     gamma:float = 0.999
#     eps_start:float = 0.9
#     eps_end:float = 0.05
#     eps_decay:float = 200
#     target_update:int = 10
#     steps_done:int = 0
#     device: torch.device = field(default_factory=lambda: torch.device("cpu"))

#     def __post_init__(self):
#         self.reset()

#     def reset(self):
#         self.steps_done = 0
#         self.memory = ReplayMemory(10000)

#     # epsilon-greedy?
#     def get_action(self, agent: RLAgent, state: State) -> Action:
#         eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
#         if random.random() > eps_threshold:
#             return agent.get_action(state)
#         else:
#             return agent.get_random_action(state)
        

#     def train(self, episodes: int, steps: int, evaluator: Evaluator) -> List[List[Metrics]]:
#         stats_history: List[List[Metrics]] = []
#         agent = evaluator.defender
#         for episode in range(episodes):
#             evaluator.reset()
#             evaluator.track_stats() # log starting positions
#             from_state = evaluator.game.state

#             for step in range(steps):
#                 ### manually invoke game loop
#                 evaluator.game.logger.debug("stepping")

#                 evaluator.game.logger.debug(f"attacker turn begin")
#                 action = evaluator.attacker.get_action(evaluator.game.state)
#                 evaluator.game.state = evaluator.attacker.take_action(evaluator.game.state, action)
#                 evaluator.game.logger.debug(f"attacker turn end")
                
#                 evaluator.game.logger.debug(f"defender turn begin")
#                 action = self.get_action(agent, evaluator.game.state)
#                 evaluator.game.state = agent.take_action(evaluator.game.state, action)
#                 evaluator.game.logger.debug(f"defender turn end")

#                 evaluator.game.cycle()
#                 evaluator.game.step_count += 1
#                 ###
                
#                 # calculate reward
#                 reward = agent.get_utility(evaluator.game.state)
#                 reward = torch.tensor([reward], device=self.device)

#                 # add to memory
#                 to_state = evaluator.game.state
#                 self.memory.push(from_state, action, to_state, reward)

#                 # update state
#                 from_state = to_state

#                 # optimize
#                 # self.optimize(agent, self.memory)

#                 # track stats
#                 evaluator.track_stats()
#                 self.steps_done += 1

#             # log stats before wiping
#             stats_history.append(evaluator.stats)

#             # Update the target network, copying all weights and biases in DQN
#             if episode % self.target_update == 0:
#                 agent.target_net.load_state_dict(agent.policy_net.state_dict())

#         return stats_history

#     def optimize(self, agent: RLAgent, memory: ReplayMemory) -> None:
#         if len(memory) < self.batch_size: return
#         optimizer = optim.RMSprop(agent.policy_net.parameters())

#         # get batch
#         transitions = memory.sample(self.batch_size)

#         # extract states from batch
#         states = [x.state for x in transitions]

#         # convert states to tensors
#         x_vulns, x_vehicles = agent.policy_net.quantify_state_batch(states)

#         # feed tensors to model
#         optimizer = optim.RMSprop(agent.policy_net.parameters())
#         y_members, y_monitor = agent.policy_net(x_vulns, x_vehicles)

#         # convert model output tensors to actions
#         actions = agent.policy_net.dequantify_state_batch(states, y_members, y_monitor)

#         # compute expected actions using older target_net
#         y_members, y_monitor = agent.target_net(x_vulns, x_vehicles)
#         expected_actions = agent.target_net.dequantify_state_batch(states, y_members, y_monitor)

#         # evaluate the action to find the resulting state
#         next_states = [agent.take_action(s, a) for s,a in zip(states, actions)]
#         expected_next_states = [agent.take_action(s, a) for s,a in zip(states, actions)]

#         # calculate rewards
#         next_reward = torch.tensor([agent.get_utility(s) for s in next_states], dtype=int)
#         expected_next_reward = torch.tensor([agent.get_utility(s) for s in expected_next_states], dtype=int)

#         # calculate loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(next_reward, expected_next_reward)
#         print(loss, "loss")

#         # optimize
#         optimizer.zero_grad()
#         loss.backward()
#         for param in agent.policy_net.parameters():
#             param.grad.data.clamp_(-1, 1)
#         optimizer.step()





#     def optimize_old(self, agent: RLAgent, memory: ReplayMemory) -> None:
#         if len(memory) < self.batch_size: return
#         optimizer = optim.RMSprop(agent.policy_net.parameters())
        
#         transitions = memory.sample(self.batch_size)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))

#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=self.device, dtype=torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state
#                                                     if s is not None])
#         state_batch = torch.cat(batch.state)
#         action_batch = torch.cat(batch.action)
#         reward_batch = torch.cat(batch.reward)

#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to policy_net
#         state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

#         # Compute V(s_{t+1}) for all next states.
#         # Expected values of actions for non_final_next_states are computed based
#         # on the "older" target_net; selecting their best reward with max(1)[0].
#         # This is merged based on the mask, such that we'll have either the expected
#         # state value or 0 in case the state was final.
#         next_state_values = torch.zeros(self.batch_size, device=self.device)
#         next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
#         # Compute the expected Q values
#         expected_state_action_values = (next_state_values * self.gamma) + reward_batch

#         # Compute Huber loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#         # Optimize the model
#         optimizer.zero_grad()
#         loss.backward()
#         for param in agent.policy_net.parameters():
#             param.grad.data.clamp_(-1, 1)
#         optimizer.step()
#endregion old