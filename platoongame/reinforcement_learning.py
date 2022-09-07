from dataclasses import dataclass, field
from typing import Tuple, Union
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from evaluation import Evaluator
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
    def __init__(self, game: Game):
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

        """
        one-hot-ish vectors out
        determine which vehicles should be in the platoon
        """
        self.member_head = nn.Linear(
            in_features = 144+108,
            out_features = self.max_vehicles
        )
        
        """
        one-hot-ish vectors out
        determine which vehicles should be monitored
        """
        self.monitor_head = nn.Linear(
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

        members = torch.sigmoid(self.member_head(x))
        monitor = torch.sigmoid(self.monitor_head(x))
        return members, monitor


class RLDefenderAgent(BasicDefenderAgent):
    def __init__(self, game: Game) -> None:
        super().__init__(get_logger("RLDefenderAgent"))
        device = get_device()
        self.game = game
        self.policy_net = DefenderDQN(game).to(device)
        self.target_net = DefenderDQN(game).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def take_action(self, state: State, action: DefenderAction) -> State:
        return BasicDefenderAgent.take_action(self, state, action)
    
    def get_action(self, state: State) -> DefenderAction:
        state_quant = self.quantify(state)
        with torch.no_grad():
            action_quant = self.policy_net(state_quant).max(1)[1].view(1,1)
        return self.dequantify_action(action_quant)

    def get_random_action(self, state: State) -> DefenderAction:
        members = [i for i,v in enumerate(state.vehicles) if v.in_platoon]
        non_members = [i for i,v in enumerate(state.vehicles) if not v.in_platoon]
        return DefenderAction(
            monitor=random.sample(members, min(self.monitor_limit, len(members))),
            join=random.sample(non_members, random.randint(len(non_members))),
            kick=random.sample(members, len(members))
        )

    def quantify_state(self, state: State) -> Tuple[torch.Tensor,...]:
        return state.as_tensors()

    def dequantify_action(self, quant: Tuple[torch.Tensor,...]) ->DefenderAction:
        pass

RLAgent = Union[RLAttackerAgent, RLDefenderAgent]

@dataclass
class DefenderAgentTrainer(Game):
    batch_size:int = 128
    gamma:float = 0.999
    eps_start:float = 0.9
    eps_end:float = 0.05
    eps_decay:float = 200
    target_update:int = 10
    steps_done:int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def reset(self):
        self.steps_done = 0

    # epsilon-greedy?
    def get_action(self, agent: RLAgent, state: State) -> Action:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if random.random() > eps_threshold:
            return agent.get_action(state)
        else:
            return agent.get_random_action()
        

    def train(self, steps: int, evaluator: Evaluator, agent: RLAgent):
        memory = ReplayMemory(10000)
        evaluator.defender = agent
        evaluator.reset()
        evaluator.track_stats()
        for i in range(steps):
            # action = self.select_action(agent, )
            state = evaluator.game.state

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
            

            next_state = evaluator.game.state

            reward = agent.get_utility(evaluator.game.state)
            reward = torch.tensor([reward], device=self.device)
            memory.push(state, action, next_state, reward)

            self.optimize(agent, memory)

            evaluator.track_stats()
            self.steps_done += 1

            # Update the target network, copying all weights and biases in DQN
            if i % self.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    def optimize(self, agent: RLAgent, memory: ReplayMemory) -> None:
        optimizer = optim.RMSprop(agent.policy_net.parameters())
        if len(memory) < self.batch_size: return
        
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