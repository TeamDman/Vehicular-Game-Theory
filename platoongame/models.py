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
class StateTensorBatch:
    vulnerabilities: torch.Tensor# (BatchSize, Vehicle, Vuln, VulnFeature)
    vehicles: torch.Tensor# (BatchSize, Vehicle, VehicleFeature)


@dataclass(frozen=True)
class DefenderActionTensorBatch:
    members: torch.Tensor # batch, 'binary' vector len=|vehicles|
    monitor: torch.Tensor # batch, 'binary' vector len=|vehicles|

@dataclass(frozen=True)
class AttackerActionTensorBatch:
    attack: torch.Tensor # batch, 'binary' vector len=|vehicles|


# Generates actions
class DefenderActor(nn.Module):
    def __init__(
        self,
        state_shape_data: StateShapeData,
    ) -> None:
        super(DefenderActor, self).__init__()
        
        # self.vuln_conv = nn.LazyConv2d(
        #     out_channels=8,
        #     kernel_size=5,
        #     stride=2
        # )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        # self.vehicle_conv = nn.LazyConv1d(
        #     out_channels = 4,
        #     kernel_size=2,
        #     stride=1
        # )
        self.vehicle_norm = nn.LazyBatchNorm1d()

        # self.hidden1 = nn.LazyLinear(out_features = 400)
        self.hidden1 = nn.LazyLinear(out_features = 3000)
        # self.hidden2 = nn.LazyLinear(out_features = 300)
        self.hidden2 = nn.LazyLinear(out_features = 100)

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
        state: StateTensorBatch
    ) -> DefenderActionTensorBatch:
        x_a = self.vuln_norm(state.vulnerabilities)
        x_a = F.relu(x_a)
        # x_a = F.relu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        # x_a = F.relu(self.vuln_norm(x_a))

        # x_b = F.relu(self.vehicle_conv(state.vehicles.permute(0,2,1)))
        x_b = self.vehicle_norm(state.vehicles)
        x_b = F.relu(x_b)

        x = torch.cat((
            x_a.flatten(start_dim=1),
            x_b.flatten(start_dim=1)
        ), dim=1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)

        # members_proto = torch.relu(self.member_head(x))   # todo: investigate
        # monitor_proto = torch.relu(self.monitor_head(x))  # todo: investigate
        members_proto = torch.sigmoid(self.member_head(x))
        monitor_proto = torch.sigmoid(self.monitor_head(x))

        # convert from [state,action] to [state,0,action] (action sub-batches of size 1 for each state)
        members_proto = members_proto.unsqueeze(dim=1)
        monitor_proto = monitor_proto.unsqueeze(dim=1)

        return DefenderActionTensorBatch(
            members=members_proto,
            monitor=monitor_proto,
        )

class DefenderCritic(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(DefenderCritic, self).__init__()

        # self.vuln_conv = nn.LazyConv2d(
        #     out_channels=8,
        #     kernel_size=5,
        #     stride=2
        # )
        self.vuln_norm = nn.LazyBatchNorm2d()
        
        # self.vehicle_conv = nn.LazyConv1d(
        #     out_channels = 4,
        #     kernel_size=2,
        #     stride=1
        # )
        self.vehicle_norm = nn.LazyBatchNorm1d()


        self.hidden1 = nn.LazyLinear(out_features = 3000)
        self.hidden2 = nn.LazyLinear(out_features = 1000)
        self.score = nn.LazyLinear(out_features = 1)

        # # probability vectors, each elem {i} represents probability of vehicle {i} being chosen
        # self.member_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who should be in platoon
        # self.monitor_head = nn.LazyLinear(out_features = state_shape_data.num_vehicles) # who to monitor
    def forward(
        self,
        state: StateTensorBatch, # the state as context for the action
        actions: DefenderActionTensorBatch, # the action that is being graded
    ) -> torch.tensor: # returns Q value (rating of the action)
        assert len(state.vehicles.shape) == 3
        assert len(state.vulnerabilities.shape) == 4
        assert len(actions.members.shape) == 3
        assert len(actions.monitor.shape) == 3

        # vehicles and vulnerability batch sizes should match
        assert state.vehicles.shape[0] == state.vulnerabilities.shape[0]
        # members and monitor should match
        assert actions.members.shape == actions.monitor.shape
        # state and action batch sizes should match
        assert state.vehicles.shape[0] == actions.members.shape[0]

        actions_per_batch = actions.members.shape[1]

        x_a = state.vulnerabilities
        # x_a = F.relu(self.vuln_conv(state.vulnerabilities.permute((0,3,1,2))))
        x_a = F.relu(self.vuln_norm(x_a))
        x_a = x_a.flatten(start_dim=1).repeat(actions_per_batch,1)

        x_b = state.vehicles
        # x_b = F.relu(self.vehicle_conv(state.vehicles.permute(0,2,1)))
        x_b = F.relu(self.vehicle_norm(x_b))
        x_b = x_b.flatten(start_dim=1).repeat(actions_per_batch,1)

        x = torch.hstack((
            x_a,
            x_b,
            actions.members.flatten(0,1),
            actions.monitor.flatten(0,1),
        ))

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.score(x)
        return x.reshape(-1, actions_per_batch)
