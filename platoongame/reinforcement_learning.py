from dataclasses import dataclass, field
from typing import List, Tuple, Union
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from evaluation import Evaluator, Metrics
import vehicles

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from agents import Action, Agent, DefenderAction, AttackerAction, BasicAttackerAgent, BasicDefenderAgent
from logging import Logger
from game import Game, State,GameConfig
from utils import get_logger, get_device

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

@dataclass(frozen=True)
class ShapeData:
    num_vehicles: int
    num_vehicle_features: int
    num_vulns: int
    num_vuln_features: int

class AttackerDQN(nn.Module):
    def __init__(self, game: Game):
        super(AttackerDQN, self).__init__()
        """
            2d conv
            vehicles x vulns
            features:
            - prob
            - severity
            - is_compromised
            - is_compromise_known
        """
        self.vuln_width = vehicles.Vulnerability(0,0).as_tensor().shape[0]
        self.max_vulns = game.vehicle_provider.max_vulns
        self.max_vehicles = game.config.max_vehicles
        self.vuln_conv = nn.Conv2d(
            in_channels=self.vuln_width,
            out_channels=8,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.BatchNorm2d(self.vuln_conv.out_channels)
        
        """
        1d conv
        vehicles data
        features: 
        - risk
        - in_platoon
        """
        self.vehicle_conv = nn.Conv1d(
            in_channels = 2,
            out_channels = 4,
            kernel_size=2,
            stride=1
        )
        self.vehicle_norm = nn.BatchNorm1d(self.vehicle_conv.out_channels)

        self.head = nn.Linear(
            in_features = 144+108,
            out_features = self.max_vehicles
        )
    
    def forward(
        self,
        x_vulns: torch.Tensor, # (BatchSize, Vehicle, Vuln, VulnFeature)
        x_vehicle: torch.Tensor, # (BatchSize, Vehicle, VehicleFeature)
    ):
        x_a = F.relu(self.vuln_conv(x_vulns.permute((0,3,1,2))))
        x_a = F.relu(self.vuln_norm(x_a))

        x_b = F.relu(self.vehicle_conv(x_vehicle.permute(0,2,1)))
        x_b = F.relu(self.vehicle_norm(x_b))

        x = torch.cat((x_a.flatten(), x_b.flatten()))
        x = F.sigmoid(self.head(x))
        return x


class RLAttackerAgent(BasicAttackerAgent):
    def __init__(self, game: Game) -> None:
        super().__init__(get_logger("RLAttackerAgent"))
        device = get_device()
        self.game = game
        self.policy_net = AttackerDQN(100,100,100).to(device)
        self.target_net = AttackerDQN(100,100,100).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def take_action(self, state: State, action: DefenderAction) -> State:
        return BasicAttackerAgent.take_action(self, state, action)
    
    def get_action(self, state: State) -> DefenderAction:
        state_quant = self.quantify(state)
        with torch.no_grad():
            action_quant = self.policy_net(state_quant).max(1)[1].view(1,1)
        return self.dequantify_action(action_quant)

    def get_random_action(self, state: State) -> AttackerAction:
        pass

    def quantify_state(self, state: State) -> torch.Tensor:
        pass

    def dequantify_action(self, quant: torch.Tensor) ->DefenderAction:
        pass
class DefenderDQN(nn.Module):
    def __init__(self, shape_data: ShapeData):
        super(DefenderDQN, self).__init__()
        """
            2d conv
            vehicles x vulns
            features:
            - prob
            - severity
            - is_compromised
            - is_compromise_known
        """
        self.shape_data = shape_data
        self.vuln_conv = nn.LazyConv2d(
            out_channels=8,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        """
        1d conv
        vehicles data
        features: 
        - risk
        - in_platoon
        """
        self.vehicle_conv = nn.LazyConv1d(
            out_channels = 4,
            kernel_size=2,
            stride=1
        )
        self.vehicle_norm = nn.LazyBatchNorm1d()

        """
        one-hot-ish vectors out
        determine which vehicles should be in the platoon
        """
        self.member_head = nn.LazyLinear(out_features = shape_data.num_vehicles)
        
        """
        one-hot-ish vectors out
        determine which vehicles should be monitored
        """
        self.monitor_head = nn.LazyLinear(out_features = shape_data.num_vehicles)
    
    def forward(
        self,
        x_vulns: torch.Tensor,  # (BatchSize, Vehicle, Vuln, VulnFeature)
        x_vehicles: torch.Tensor,# (BatchSize, Vehicle, VehicleFeature)
    ) -> Tuple[torch.Tensor, ...]:
        # print(x_vulns.shape, "x_vulns")
        x_a = F.relu(self.vuln_conv(x_vulns.permute((0,3,1,2))))
        # print(x_a.shape, "x_a after conv")
        x_a = F.relu(self.vuln_norm(x_a))
        # print(x_a.shape, "x_a after norm")

        # print(x_vehicles.shape, "x_vehicle")
        x_b = F.relu(self.vehicle_conv(x_vehicles.permute(0,2,1)))
        # print(x_b.shape, "x_b after conv")
        x_b = F.relu(self.vehicle_norm(x_b))
        # print(x_b.shape, "x_b after norm")

        # print(x_a.shape, x_b.shape, "x_a, x_b")

        x = torch.cat((x_a.flatten(start_dim=1), x_b.flatten(start_dim=1)), dim=1)
        # print(x.shape, "flat")

        members = torch.arctan(self.member_head(x))
        # print(members.shape, "member")
        monitor = torch.arctan(self.monitor_head(x))
        # print(monitor.shape, "monitor")

        return members, monitor

    def quantify_state_batch(self, states: List[State]) -> Tuple[torch.Tensor,...]:
        # list of state tensor tuples
        x: List[Tuple[torch.Tensor, torch.Tensor]] = [s.as_tensors(self.shape_data) for s in states]

        # list of tuples => tuple of lists
        # https://stackoverflow.com/a/51991395/11141271
        x: Tuple[List[torch.Tensor], List[torch.Tensor]] = tuple(map(list,zip(*x)))

        # stack in new dim
        x_vulns = torch.stack(x[0])
        x_vehicles = torch.stack(x[1])
        return x_vulns, x_vehicles

    def dequantify_state_batch(self, states: List[State], members: torch.Tensor, monitor: torch.Tensor) -> List[DefenderAction]:
        return [
            self.dequantify(state, mem, mon)
            for state, mem, mon in zip(states, members, monitor)
        ]

    def dequantify(self, state: State, members: torch.Tensor, monitor: torch.Tensor) -> DefenderAction:
        members = members.heaviside(torch.tensor(1.)) # threshold
        members = (members == 1).nonzero().squeeze() # identify indices
        members = frozenset(members.numpy()) # convert to set

        monitor = monitor.heaviside(torch.tensor(1.))
        monitor = (monitor == 1).nonzero().squeeze()
        monitor = frozenset(monitor.numpy())

        existing_members = [i for i,v in enumerate(state.vehicles) if v.in_platoon]

        return DefenderAction(
            monitor = monitor,
            kick = frozenset([x for x in existing_members if x not in members]),
            join = frozenset([x for x in members if x not in existing_members])
        )

    def get_actions(self, states: List[State]) -> List[DefenderAction]:
        with torch.no_grad():
            members, monitor = self(*self.quantify_state_batch(states))
        return self.dequantify_state_batch(states, members, monitor)

class RLDefenderAgent(BasicDefenderAgent):
    def __init__(self, monitor_limit: int, shape_data: ShapeData) -> None:
        super().__init__(
            monitor_limit=monitor_limit
        )
        self.logger = get_logger("RLDefenderAgent")
        device = get_device()
        self.shape_data = shape_data
        self.policy_net = DefenderDQN(shape_data).to(device)
        self.target_net = DefenderDQN(shape_data).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
    def get_action(self, state: State) -> DefenderAction:
        return self.policy_net.get_actions([state])[0]

    def get_random_action(self, state: State) -> DefenderAction:
        members = [i for i,v in enumerate(state.vehicles) if v.in_platoon]
        non_members = [i for i,v in enumerate(state.vehicles) if not v.in_platoon]
        return DefenderAction(
            monitor=frozenset(random.sample(members, min(self.monitor_limit, len(members)))),
            join=frozenset(random.sample(non_members, random.randint(0,len(non_members)))),
            kick=frozenset(random.sample(members, len(members)))
        )


RLAgent = Union[RLAttackerAgent, RLDefenderAgent]

@dataclass
class DefenderAgentTrainer:
    batch_size:int = 128
    gamma:float = 0.999
    eps_start:float = 0.9
    eps_end:float = 0.05
    eps_decay:float = 200
    target_update:int = 10
    steps_done:int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.steps_done = 0
        self.memory = ReplayMemory(10000)

    # epsilon-greedy?
    def get_action(self, agent: RLAgent, state: State) -> Action:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if random.random() > eps_threshold:
            return agent.get_action(state)
        else:
            return agent.get_random_action(state)
        

    def train(self, episodes: int, steps: int, evaluator: Evaluator) -> List[List[Metrics]]:
        stats_history: List[List[Metrics]] = []
        agent = evaluator.defender
        for episode in range(episodes):
            evaluator.reset()
            evaluator.track_stats() # log starting positions
            from_state = evaluator.game.state

            for step in range(steps):
                ### manually invoke game loop
                evaluator.game.logger.debug("stepping")

                evaluator.game.logger.debug(f"attacker turn begin")
                action = evaluator.attacker.get_action(evaluator.game.state)
                evaluator.game.state = evaluator.attacker.take_action(evaluator.game.state, action)
                evaluator.game.logger.debug(f"attacker turn end")
                
                evaluator.game.logger.debug(f"defender turn begin")
                action = self.get_action(agent, evaluator.game.state)
                evaluator.game.state = agent.take_action(evaluator.game.state, action)
                evaluator.game.logger.debug(f"defender turn end")

                evaluator.game.cycle()
                evaluator.game.step_count += 1
                ###
                
                # calculate reward
                reward = agent.get_utility(evaluator.game.state)
                reward = torch.tensor([reward], device=self.device)

                # add to memory
                to_state = evaluator.game.state
                self.memory.push(from_state, action, to_state, reward)

                # update state
                from_state = to_state

                # optimize
                self.optimize(agent, self.memory)

                # track stats
                evaluator.track_stats()
                self.steps_done += 1

            # log stats before wiping
            stats_history.append(evaluator.stats)

            # Update the target network, copying all weights and biases in DQN
            if episode % self.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        return stats_history

    def optimize(self, agent: RLAgent, memory: ReplayMemory) -> None:
        if len(memory) < self.batch_size: return
        optimizer = optim.RMSprop(agent.policy_net.parameters())

        # get batch
        transitions = memory.sample(self.batch_size)

        # extract states from batch
        states = [x.state for x in transitions]

        # convert states to tensors
        x_vulns, x_vehicles = agent.policy_net.quantify_state_batch(states)

        # feed tensors to model
        optimizer = optim.RMSprop(agent.policy_net.parameters())
        y_members, y_monitor = agent.policy_net(x_vulns, x_vehicles)

        # convert model output tensors to actions
        actions = agent.policy_net.dequantify_state_batch(states, y_members, y_monitor)

        # compute expected actions using older target_net
        y_members, y_monitor = agent.target_net(x_vulns, x_vehicles)
        expected_actions = agent.target_net.dequantify_state_batch(states, y_members, y_monitor)

        # evaluate the action to find the resulting state
        next_states = [agent.take_action(s, a) for s,a in zip(states, actions)]
        expected_next_states = [agent.take_action(s, a) for s,a in zip(states, actions)]

        # calculate rewards
        next_reward = torch.tensor([agent.get_utility(s) for s in next_states], dtype=int)
        expected_next_reward = torch.tensor([agent.get_utility(s) for s in expected_next_states], dtype=int)

        # calculate loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(next_reward, expected_next_reward)
        print(loss, "loss")

        # optimize
        optimizer.zero_grad()
        loss.backward()
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()





    def optimize_old(self, agent: RLAgent, memory: ReplayMemory) -> None:
        if len(memory) < self.batch_size: return
        optimizer = optim.RMSprop(agent.policy_net.parameters())
        
        transitions = memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()