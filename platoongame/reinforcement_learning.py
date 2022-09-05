import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import vehicles

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from agents import Agent, DefenderAction, AttackerAction, BasicAttackerAgent, BasicDefenderAgent
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


AttackerStateQuant = namedtuple("vulns", "in_platoon")

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

    def quantify_state(self, state: State) -> torch.Tensor:
        pass

    def dequantify_action(self, quant: torch.Tensor) ->DefenderAction:
        pass

RLAgent = RLAttackerAgent
# RLAgent = Union[RLAttackerAgent, RLDefenderAgent]


class Trainer:
    def __init__(self) -> None:
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10

    def optimize(agent: RLAgent):
        optimizer = optim.RMSprop(agent.policy_net.parameters())
        memory = ReplayMemory(10000)
        steps_done = 0
        