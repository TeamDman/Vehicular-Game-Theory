from __future__ import annotations
import logging
import random
from typing import Deque, List, Set, Union, FrozenSet, TYPE_CHECKING
from models import StateShapeData
from utils import get_logger
from vehicles import CompromiseState, Vehicle
from pprint import pprint
from collections import deque
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from game import Game, State


@dataclass(frozen=True)
class DefenderAction:
    members: FrozenSet[int] # binary vector, indices of corresponding vehicle
    monitor: FrozenSet[int] # binary vector, indices of corresponding vehicle

    def as_tensor(self):
        return DefenderActionTensors(
            members=torch.as_tensor(self.members),
            monitor=torch.as_tensor(self.monitor),
        )
        

@dataclass(frozen=True)
class DefenderActionTensors:
    members: torch.Tensor # 'binary' vector len=|vehicles|
    monitor: torch.Tensor # 'binary' vector len=|vehicles|

@dataclass(frozen=True)
class AttackerAction:
    attack: FrozenSet[int] # 'binary' vector len=|vehicles|

    def as_tensor(self):
        return AttackerActionTensors(
            attack=torch.as_tensor(self.attack),
        )

@dataclass(frozen=True)
class AttackerActionTensors:
    attack: torch.Tensor


Action = Union[DefenderAction, AttackerAction]

###############################
#region Base stuff
###############################
class Agent(ABC):
    logger: logging.Logger

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass

    @abstractmethod
    def get_random_action(self, state: State) -> Action:
        pass

    @abstractmethod
    def take_action(self, state: State, action: Action) -> State:
        pass

    @abstractmethod
    def get_utility(self, state: State) -> int:
        pass

#agent that does nothing, can act as defender or attacker
class PassiveAgent(Agent):
    def __init__(self) -> None:
        super().__init__(get_logger("PassiveAgent"))

    def get_action(self, state: State) -> Action:
        return None

    def get_random_action(self, state: State) -> Action:
        return None

    def take_action(self, state: State, action: Action) -> State:
        return state

    def get_utility(self, state: State) -> int:
        return 0

class DefenderAgent(Agent):
    def get_utility(self, state: State) -> int:
        members = [vehicle for vehicle in state.vehicles if vehicle.in_platoon]
        compromises = sum([vuln.severity for vehicle in members for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED])
        # return len(members) * 10 - compromises ** 2
        # return int(len(members) * 10 - compromises ** 1.5)
        return int(len(members) * 1 - compromises)
        # return int(len(members) * 1 - compromises)

    def take_action(self, state: State, action: DefenderAction) -> State:
        # create mutable copy
        vehicles = list(state.vehicles)

        # monitor vehicles
        for i in action.monitor:
            vehicle = vehicles[i]
            self.logger.info(f"monitoring vehicle {i}")
            # compromised vulns become known
            vehicle = replace(vehicle, vulnerabilities=tuple([vuln if vuln.state != CompromiseState.COMPROMISED_UNKNOWN else replace(vuln, state = CompromiseState.COMPROMISED_KNOWN) for vuln in vehicle.vulnerabilities]))
            vehicles[i] = vehicle

        for i in vehicles:
            vehicles[i] = replace(vehicles[i], in_platoon = action.members[i] == 1)
            # self.logger.info(f"kicking vehicle {i} out of platoon")
            # self.logger.info(f"adding vehicle {i} to platoon")
            
        # update vehicles from mutated copy
        state = replace(state, vehicles=tuple(vehicles))

        # calculate util change
        change = self.get_utility(state)
        utility = state.defender_utility
        utility += change
        self.logger.debug(f"utility {utility} ({change:+d})")
        state = replace(state, defender_utility=utility)

        # return new state
        return state

    def get_random_action(self, state: State) -> DefenderAction:
        members = [i for i,v in enumerate(state.vehicles) if v.in_platoon]
        non_members = [i for i,v in enumerate(state.vehicles) if not v.in_platoon]
        return DefenderAction(
            monitor=frozenset(random.sample(members, min(self.monitor_limit, len(members)))),
            join=frozenset(random.sample(non_members, random.randint(0,len(non_members)))),
            kick=frozenset(random.sample(members, len(members)))
        )

class AttackerAgent(Agent):
    def get_utility(self, state: State) -> int:
        util = 0
        for vehicle in state.vehicles:
            for vuln in vehicle.vulnerabilities:
                if vehicle.in_platoon and vuln.state != CompromiseState.NOT_COMPROMISED:
                    util += vuln.severity
                elif not vehicle.in_platoon and vuln.state == CompromiseState.COMPROMISED_UNKNOWN:
                    util += vuln.severity / 4
        return int(util)

    def take_action(self, state: State, action: AttackerAction) -> State:
        vehicles = list(state.vehicles)
        for i in action.attack:
            self.logger.debug(f"attacking vehicle {i}")
            vehicle = vehicles[i]
            for j, vuln in enumerate(vehicle.vulnerabilities):
                if vuln.state != CompromiseState.NOT_COMPROMISED: continue
                if random.random() > vuln.prob: continue
                self.logger.info(f"successfully compromised vehicle {i} vuln {j} sev {vuln.severity}")
                new_vulns = vehicle.vulnerabilities[:j] + (replace(vuln, state=CompromiseState.COMPROMISED_UNKNOWN),) + vehicle.vulnerabilities[j+1:]
                vehicle = replace(vehicle, vulnerabilities=new_vulns)
            vehicles[i] = vehicle

        # calculate util change
        state = replace(state, vehicles=tuple(vehicles))
        change = self.get_utility(state)
        utility = state.attacker_utility
        utility += change
        self.logger.debug(f"utility {utility} ({change:+d})")
        state = replace(state, attacker_utility=utility)

        # return new state
        return state

    
    def get_random_action(self, state: State) -> AttackerAction:
        raise NotImplementedError() #todo

#endregion Base stuff

###############################
#region human design agents
###############################
class BasicDefenderAgent(DefenderAgent):
    num_vehicles_monitoring_constraint: int
    recently_monitored: Deque[Vehicle]
    tolerance_threshold: int
    # max_size: int

    def __init__(
        self,
        monitor_limit: int = 1,
        tolerance_threshold: int = 3
    ) -> None:
        super().__init__(get_logger("BasicDefenderAgent"))
        self.recently_monitored = deque([], maxlen=3)
        self.tolerance_threshold = tolerance_threshold
        # self.max_size = 10
        self.monitor_limit = monitor_limit

    def get_action(self, state: State) -> DefenderAction:
        # pick next vehicle to monitor
        choices = [
            i for i,v in enumerate(state.vehicles)
            # if v not in self.recently_monitored
            # todo: investigate ways of preventing picking same vehicle without using vehicle IDs
        ]
        random.shuffle(choices)
        monitor = set()
        if len(choices) == 0:
            self.logger.warn("no candidates found to monitor?")
            pprint(state.vehicles)
            pprint(self.recently_monitored)
        else:
            while len(monitor) < self.monitor_limit and len(choices) > 0:
                x = choices.pop()
                monitor.add(x)
                self.recently_monitored.append(x)

        # kick risky vehicles from platoon
        kick = list()
        for i,vehicle in enumerate(state.vehicles):
            if not vehicle.in_platoon: continue
            total = sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN])
            if total > self.tolerance_threshold:
                kick.append(i)

        # join low risk vehicles into the platoon
        join = list()
        # member_count = len([1 for v in state.vehicles if v.in_platoon])
        candidates = [
            i for i,vehicle in enumerate(state.vehicles)
            if not vehicle.in_platoon
            and vehicle.risk <= 10
            and sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN]) <= self.tolerance_threshold
            # and member_count < self.max_size
        ]
        random.shuffle(candidates)
        while len(candidates) > 0:
        # while member_count + len(join) < self.max_size and len(candidates) > 0:
            # take while there's room in the platoon
            join.append(candidates.pop())

        members = [1 if v.in_platoon else 0 for v in state.vehicles]
        members[kick] = 0
        members[join] = 1

        return DefenderAction(
            jmembers=frozenset(members),
            monitor=frozenset(monitor),
        )

class BasicAttackerAgent(AttackerAgent):
    def __init__(self, attack_limit:int = 1) -> None:
        super().__init__(get_logger("BasicAttackerAgent"))
        self.attack_limit = attack_limit

    def get_action(self, state: State) -> None:
        # Pick vehicle to attack
        candidates = list([(i,v) for i,v in enumerate(state.vehicles) if v.in_platoon])
        attack = set()
        if len(candidates) == 0:
            # attack anything if platoon is empty
            candidates = list(enumerate(state.vehicles))

        if len(candidates) == 0:
            self.logger.warn("sanity check failed, no vehicles to attack")
        else:
            def eval_risk(v: Vehicle) -> float:
                return [x.severity ** 2 * x.prob for x in v.vulnerabilities if x.state == CompromiseState.NOT_COMPROMISED]
            candidates = sorted(candidates, key=lambda x: eval_risk(x[1]))
            for _ in range(self.attack_limit):
                if len(candidates) == 0:
                    break
                attack.add(candidates.pop()[0])

        return AttackerAction(attack=frozenset(attack))

#endregion human design agents

###############################
#region ml agents
###############################

from utils import get_logger, get_device
from models import StateShapeData, DefenderActor, DefenderCritic
import torch
import torch.optim

class WolpertingerDefenderAgent(DefenderAgent):
    def __init__(self, state_shape_data: StateShapeData) -> None:
        super().__init__(get_logger("WolpertingerDefenderAgent"))

        self.state_shape_data = state_shape_data

        self.actor = DefenderActor(state_shape_data, propose=5)
        self.actor_target = DefenderActor(state_shape_data, propose=5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = DefenderCritic(state_shape_data)
        self.critic_target = DefenderCritic(state_shape_data)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0001)

        # hard update
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(
        self,
        state_obj: State
    ) -> DefenderAction:
        # convert from state object to tensors to be fed to the actor model
        state_tensor = state_obj.as_tensors(self.state_shape_data)
        
        # get action suggestions from the actor
        proto_actions = self.actor(*state_tensor)

        # convert suggestions to actual actions
        
        
#region old
# class RLDefenderAgent(BasicDefenderAgent):
#     def __init__(self, monitor_limit: int, shape_data: ShapeData) -> None:
#         super().__init__(
#             monitor_limit=monitor_limit
#         )
#         self.logger = get_logger("RLDefenderAgent")
#         device = get_device()
#         self.shape_data = shape_data
#         self.policy_net = DefenderDQN(shape_data).to(device)
#         self.target_net = DefenderDQN(shape_data).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
    
#     def get_action(self, state: State) -> DefenderAction:
#         return self.policy_net.get_actions([state])[0]



# class RLAttackerAgent(BasicAttackerAgent):
#     def __init__(self, game: Game) -> None:
#         super().__init__(get_logger("RLAttackerAgent"))
#         device = get_device()
#         self.game = game
#         self.policy_net = AttackerDQN(100,100,100).to(device)
#         self.target_net = AttackerDQN(100,100,100).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()


#     def take_action(self, state: State, action: DefenderAction) -> State:
#         return BasicAttackerAgent.take_action(self, state, action)
    
#     def get_action(self, state: State) -> DefenderAction:
#         state_quant = self.quantify(state)
#         with torch.no_grad():
#             action_quant = self.policy_net(state_quant).max(1)[1].view(1,1)
#         return self.dequantify_action(action_quant)

#     def quantify_state(self, state: State) -> torch.Tensor:
#         pass

#     def dequantify_action(self, quant: torch.Tensor) ->DefenderAction:
#         pass
    
# RLAgent = Union[RLAttackerAgent, RLDefenderAgent]
#endregion old
#endregion ml agents
