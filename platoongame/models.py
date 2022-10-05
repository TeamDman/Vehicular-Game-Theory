from dataclasses import dataclass, field

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

@dataclass(frozen=True)
class StateTensors:
    vulnerabilities: torch.Tensor# (BatchSize, Vehicle, Vuln, VulnFeature)
    vehicles: torch.Tensor# (BatchSize, Vehicle, VehicleFeature)


@dataclass(frozen=True)
class DefenderActionTensors:
    members: torch.Tensor # batch, 'binary' vector len=|vehicles|
    monitor: torch.Tensor # batch, 'binary' vector len=|vehicles|

@dataclass(frozen=True)
class AttackerActionTensors:
    attack: torch.Tensor # batch, 'binary' vector len=|vehicles|


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
