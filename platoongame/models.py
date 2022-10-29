from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

LazyLayer = Union[nn.LazyLinear, nn.LazyConv1d, nn.LazyConv2d, nn.LazyBatchNorm2d, nn.LazyBatchNorm1d]

if TYPE_CHECKING:
    from game import StateShapeData, StateTensorBatch, DefenderActionTensorBatch, AttackerActionTensorBatch

# Generates actions
class DefenderActor(nn.Module):
    def __init__(
        self,
        state_shape_data: StateShapeData,
    ) -> None:
        super(DefenderActor, self).__init__()
        
        self.vuln_conv = nn.LazyConv2d(
            out_channels=128,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        self.hidden1 = nn.LazyLinear(out_features = 10000)
        self.hidden2 = nn.LazyLinear(out_features = 1000)

        # probability vectors, each elem {i} represents probability of vehicle {i} being chosen
        self.member_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who should be in platoon

    ## not sure how this behaves with lazy modules so going to avoid for now
    ## https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L208-L213
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="gelu")
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        state: StateTensorBatch
    ) -> DefenderActionTensorBatch:
        x = F.gelu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        x = F.gelu(self.vuln_norm(x))

        x = torch.cat((
            x.flatten(start_dim=1),
            state.vulnerabilities.flatten(start_dim=1),
        ), dim=1)
        x = self.hidden1(x)
        x = F.gelu(x)
        x = self.hidden2(x)
        x = F.gelu(x)
        x = self.member_head(x)
        # x /= 10
        x = F.sigmoid(x)

        return DefenderActionTensorBatch(
            members=x,
        )

class DefenderCritic(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(DefenderCritic, self).__init__()

        self.vuln_conv = nn.LazyConv2d(
            out_channels=128,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        # self.vehicle_conv = nn.LazyConv1d(
        #     out_channels = 128,
        #     kernel_size=2,
        #     stride=1
        # )
        # self.vehicle_norm = nn.LazyBatchNorm1d()

        self.hidden1 = nn.LazyLinear(out_features = 10000)
        self.hidden2 = nn.LazyLinear(out_features = 1000)
        self.score = nn.LazyLinear(out_features = 1)

    def forward(
        self,
        state: StateTensorBatch, # the state as context for the action
        actions: DefenderActionTensorBatch, # the action that is being graded
    ) -> torch.Tensor: # returns Q value (rating of the action)
        assert len(state.vulnerabilities.shape) == 4 # [batch, vehicle, vuln, vuln_features]
        assert len(actions.members.shape) == 2 # [batch, binary_member_vectors]

        x_states = F.gelu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        x_states = F.gelu(self.vuln_norm(x_states))

        x = torch.cat((
            x_states.flatten(start_dim=1),
            state.vulnerabilities.flatten(start_dim=1),
            actions.members,
        ), dim=1)

        x = self.hidden1(x)
        x = F.gelu(x)
        x = self.hidden2(x)
        x = F.gelu(x)
        x = self.score(x)
        return x.flatten()
