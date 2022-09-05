import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

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
            - attackerCost
            - defenderCost
            - attackerProb
            - defenderPRob
            - severity
            - is_compromised
            - is_compromise_known
        """
        self.vuln_conv = nn.Conv2d(
            in_channels = 7,
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
        self.vec_conv = nn.Conv1d(
            in_channels = 2,
            out_channels = 4,
            kernel_size=2,
            stride=1
        )
        self.vec_norm = nn.BatchNorm1d(self.vec_conv.out_channels)

        # self.head = nn.Linear()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        pass
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))

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
        