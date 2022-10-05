from dataclasses import dataclass, field
from typing import List, Tuple, Union
from agents import DefenderActionTensors

from game import Game, State, GameConfig, StateTensors
from vehicles import Vulnerability

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

@dataclass(frozen=True)
class StateShapeData:
    num_vehicles: int           # max number of vehicles in the game
    num_vehicle_features: int   # in platoon, risk
    num_vulns: int              # max number of vulns in any vehicle
    num_vuln_features: int      # prob, severity, is_compromised, is_compromise_known

# Generates actions
class DefenderActor(nn.Module):
    def __init__(
        self,
        state_shape_data: StateShapeData,
        propose: int, # how many actions the actor should propose
    ) -> None:
        self.vuln_conv = nn.LazyConv2d(
            out_channels=8,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        self.vehicle_conv = nn.LazyConv1d(
            out_channels = 4,
            kernel_size=2,
            stride=1
        )
        self.vehicle_norm = nn.LazyBatchNorm1d()

        self.hidden1 = nn.LazyLinear(out_features = 400)
        self.hidden2 = nn.LazyLinear(out_features = 300)

        # probability vectors, each elem {i} represents probability of vehicle {i} being chosen
        self.member_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who should be in platoon
        self.monitor_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who to monitor

    ## not sure how this behaves with lazy modules so going to avoid for now
    ## https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L208-L213
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        state: StateTensors
    ) -> DefenderActionTensors:
        # print(x_vulns.shape, "x_vulns")
        x_a = F.relu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        # print(x_a.shape, "x_a after conv")
        x_a = F.relu(self.vuln_norm(x_a))
        # print(x_a.shape, "x_a after norm")

        # print(x_vehicles.shape, "x_vehicle")
        x_b = F.relu(self.vehicle_conv(state.vehicles.permute(0,2,1)))
        # print(x_b.shape, "x_b after conv")
        x_b = F.relu(self.vehicle_norm(x_b))
        # print(x_b.shape, "x_b after norm")

        # print(x_a.shape, x_b.shape, "x_a, x_b")

        x = torch.cat((x_a.flatten(start_dim=1), x_b.flatten(start_dim=1)), dim=1)
        # print(x.shape, "flat")

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)

        members_proto = torch.arctan(self.member_head(x))
        # print(members.shape, "member")
        monitor_proto = torch.arctan(self.monitor_head(x))
        # print(monitor.shape, "monitor")

        return DefenderActionTensors(
            members=members_proto,
            monitor=monitor_proto,
        )

class DefenderCritic(nn.Module):
    def __init__(
        self,
    ) -> None:
        self.vuln_conv = nn.LazyConv2d(
            out_channels=8,
            kernel_size=5,
            stride=2
        )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        self.vehicle_conv = nn.LazyConv1d(
            out_channels = 4,
            kernel_size=2,
            stride=1
        )
        self.vehicle_norm = nn.LazyBatchNorm1d()


        self.hidden1 = nn.LazyLinear(out_features = 400)
        self.hidden2 = nn.LazyLinear(out_features = 300)
        self.score = nn.LazyLinear(out_features = 1)

        # # probability vectors, each elem {i} represents probability of vehicle {i} being chosen
        # self.member_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who should be in platoon
        # self.monitor_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who to monitor
    def forward(
        self,
        state: StateTensors, # the state as context for the action
        action: DefenderActionTensors, # the action that is being graded
    ) -> torch.tensor: # returns Q value (rating of the action)
        # print(x_vulns.shape, "x_vulns")
        x_a = F.relu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        # print(x_a.shape, "x_a after conv")
        x_a = F.relu(self.vuln_norm(x_a))
        # print(x_a.shape, "x_a after norm")

        # print(x_vehicles.shape, "x_vehicle")
        x_b = F.relu(self.vehicle_conv(state.vehicles.permute(0,2,1)))
        # print(x_b.shape, "x_b after conv")
        x_b = F.relu(self.vehicle_norm(x_b))
        # print(x_b.shape, "x_b after norm")

        # print(x_a.shape, x_b.shape, "x_a, x_b")

        # todo: make sure the flatten doesn't run the batch dimension
        x = torch.cat((x_a.flatten(start_dim=1), x_b.flatten(start_dim=1), action.members, action.monitor), dim=1)
        # print(x.shape, "flat")

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.score(x)
        x = F.tanh(x)
        return x


#region old
# class AttackerDQN(nn.Module):
#     def __init__(self, game: Game):
#         super(AttackerDQN, self).__init__()
#         """
#             2d conv
#             vehicles x vulns
#             features:
#             - prob
#             - severity
#             - is_compromised
#             - is_compromise_known
#         """
#         self.vuln_width = Vulnerability(0,0).as_tensor().shape[0]
#         self.max_vulns = game.vehicle_provider.max_vulns
#         self.max_vehicles = game.config.max_vehicles
#         self.vuln_conv = nn.Conv2d(
#             in_channels=self.vuln_width,
#             out_channels=8,
#             kernel_size=5,
#             stride=2
#         )
#         self.vuln_norm = nn.BatchNorm2d(self.vuln_conv.out_channels)
        
#         """
#         1d conv
#         vehicles data
#         features: 
#         - risk
#         - in_platoon
#         """
#         self.vehicle_conv = nn.Conv1d(
#             in_channels = 2,
#             out_channels = 4,
#             kernel_size=2,
#             stride=1
#         )
#         self.vehicle_norm = nn.BatchNorm1d(self.vehicle_conv.out_channels)

#         self.head = nn.Linear(
#             in_features = 144+108,
#             out_features = self.max_vehicles
#         )
    
#     def forward(
#         self,
#         x_vulns: torch.Tensor, # (BatchSize, Vehicle, Vuln, VulnFeature)
#         x_vehicle: torch.Tensor, # (BatchSize, Vehicle, VehicleFeature)
#     ):
#         x_a = F.relu(self.vuln_conv(x_vulns.permute((0,3,1,2))))
#         x_a = F.relu(self.vuln_norm(x_a))

#         x_b = F.relu(self.vehicle_conv(x_vehicle.permute(0,2,1)))
#         x_b = F.relu(self.vehicle_norm(x_b))

#         x = torch.cat((x_a.flatten(), x_b.flatten()))
#         x = F.sigmoid(self.head(x))
#         return x


# class DefenderDQN(nn.Module):
#     def __init__(self, state_shape_data: ShapeData):
#         super(DefenderDQN, self).__init__()
#         """
#             2d conv
#             vehicles x vulns
#             features:
#             - prob
#             - severity
#             - is_compromised
#             - is_compromise_known
#         """
#         self.shape_data = shape_data
#         self.vuln_conv = nn.LazyConv2d(
#             out_channels=8,
#             kernel_size=5,
#             stride=2
#         )
#         self.vuln_norm = nn.LazyBatchNorm2d()
        
#         """
#         1d conv
#         vehicles data
#         features: 
#         - risk
#         - in_platoon
#         """
#         self.vehicle_conv = nn.LazyConv1d(
#             out_channels = 4,
#             kernel_size=2,
#             stride=1
#         )
#         self.vehicle_norm = nn.LazyBatchNorm1d()

#         """
#         one-hot-ish vectors out
#         determine which vehicles should be in the platoon
#         """
#         self.member_head = nn.LazyLinear(out_features = shape_data.num_vehicles)
        
#         """
#         one-hot-ish vectors out
#         determine which vehicles should be monitored
#         """
#         self.monitor_head = nn.LazyLinear(out_features = shape_data.num_vehicles)
    
#     def forward(
#         self,
#         x_vulns: torch.Tensor,  # (BatchSize, Vehicle, Vuln, VulnFeature)
#         x_vehicles: torch.Tensor,# (BatchSize, Vehicle, VehicleFeature)
#     ) -> Tuple[torch.Tensor, ...]:
#         # print(x_vulns.shape, "x_vulns")
#         x_a = F.relu(self.vuln_conv(x_vulns.permute((0,3,1,2))))
#         # print(x_a.shape, "x_a after conv")
#         x_a = F.relu(self.vuln_norm(x_a))
#         # print(x_a.shape, "x_a after norm")

#         # print(x_vehicles.shape, "x_vehicle")
#         x_b = F.relu(self.vehicle_conv(x_vehicles.permute(0,2,1)))
#         # print(x_b.shape, "x_b after conv")
#         x_b = F.relu(self.vehicle_norm(x_b))
#         # print(x_b.shape, "x_b after norm")

#         # print(x_a.shape, x_b.shape, "x_a, x_b")

#         x = torch.cat((x_a.flatten(start_dim=1), x_b.flatten(start_dim=1)), dim=1)
#         # print(x.shape, "flat")

#         members = torch.arctan(self.member_head(x))
#         # print(members.shape, "member")
#         monitor = torch.arctan(self.monitor_head(x))
#         # print(monitor.shape, "monitor")

#         return members, monitor

#     def quantify_state_batch(self, states: List[State]) -> Tuple[torch.Tensor,...]:
#         # list of state tensor tuples
#         x: List[Tuple[torch.Tensor, torch.Tensor]] = [s.as_tensors(self.shape_data) for s in states]

#         # list of tuples => tuple of lists
#         # https://stackoverflow.com/a/51991395/11141271
#         x: Tuple[List[torch.Tensor], List[torch.Tensor]] = tuple(map(list,zip(*x)))

#         # stack in new dim
#         x_vulns = torch.stack(x[0])
#         x_vehicles = torch.stack(x[1])
#         return x_vulns, x_vehicles

#     def dequantify_state_batch(self, states: List[State], members: torch.Tensor, monitor: torch.Tensor) -> List[DefenderAction]:
#         return [
#             self.dequantify(state, mem, mon)
#             for state, mem, mon in zip(states, members, monitor)
#         ]

#     def dequantify(self, state: State, members: torch.Tensor, monitor: torch.Tensor) -> DefenderAction:
#         members = members.heaviside(torch.tensor(1.)) # threshold
#         members = (members == 1).nonzero().squeeze() # identify indices
#         members = frozenset(members.numpy()) # convert to set

#         monitor = monitor.heaviside(torch.tensor(1.))
#         monitor = (monitor == 1).nonzero().squeeze()
#         monitor = frozenset(monitor.numpy())

#         existing_members = [i for i,v in enumerate(state.vehicles) if v.in_platoon]

#         return DefenderAction(
#             monitor = monitor,
#             kick = frozenset([x for x in existing_members if x not in members]),
#             join = frozenset([x for x in members if x not in existing_members])
#         )

#     def get_actions(self, states: List[State]) -> List[DefenderAction]:
#         with torch.no_grad():
#             members, monitor = self(*self.quantify_state_batch(states))
#         return self.dequantify_state_batch(states, members, monitor)
#endregion old