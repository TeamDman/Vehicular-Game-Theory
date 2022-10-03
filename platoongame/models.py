from dataclasses import dataclass, field
from typing import List, Tuple, Union

from game import Game, State, GameConfig
from vehicles import Vulnerability

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

@dataclass(frozen=True)
class ShapeData:
    num_vehicles: int
    num_vehicle_features: int
    num_vulns: int
    num_vuln_features: int




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
#     def __init__(self, shape_data: ShapeData):
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