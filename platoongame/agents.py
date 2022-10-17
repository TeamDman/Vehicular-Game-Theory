from __future__ import annotations
import json
import logging
import random
from re import M
import time
from typing import Deque, Dict, List, Optional, Set, Union, FrozenSet, TYPE_CHECKING
from models import StateShapeData, DefenderActionTensorBatch, AttackerActionTensorBatch, StateTensorBatch
from utils import get_logger, get_prefix
from vehicles import CompromiseState, Vehicle
from pprint import pprint
from collections import deque
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
import pathlib

if TYPE_CHECKING:
    from game import Game, State


@dataclass(frozen=True)
class DefenderAction:
    members: FrozenSet[int] # binary vector, indices of corresponding vehicle
    monitor: FrozenSet[int] # binary vector, indices of corresponding vehicle

    def as_tensor_batch(self, state_shape: StateShapeData):
        members = torch.zeros(state_shape.num_vehicles, dtype=torch.float32)
        members[list(self.members)] = 1
        monitor = torch.zeros(state_shape.num_vehicles, dtype=torch.float32)
        monitor[list(self.monitor)] = 1

        return DefenderActionTensorBatch(
            members=members.unsqueeze(dim=0).unsqueeze(dim=1),
            monitor=monitor.unsqueeze(dim=0).unsqueeze(dim=1),
        )



@dataclass(frozen=True)
class AttackerAction:
    attack: FrozenSet[int] # binary vector len=|vehicles|

    def as_tensor(self, state_shape: StateShapeData):
        attack = torch.zeros(state_shape.num_vehicles, dtype=torch.float32)
        attack[list(self.attack)] = 1
        return AttackerActionTensorBatch(
            attack=attack.unsqueeze(dim=0).unsqueeze(dim=1),
        )


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

    def __str__(self) -> str:
        return self.__class__.__name__

class DefenderAgent(Agent):
    def get_utility(self, state: State) -> int:
        members = [vehicle for vehicle in state.vehicles if vehicle.in_platoon]
        compromises = sum([vuln.severity for vehicle in members for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED])
        # return len(members) * 10 - compromises ** 2
        # return int(len(members) * 10 - compromises ** 1.5)
        return int(len(members) * 2.5 - compromises)
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

        for i, v in enumerate(vehicles):
            vehicles[i] = replace(v, in_platoon = i in action.members)
            # self.logger.info(f"kicking vehicle {i} out of platoon")
            # self.logger.info(f"adding vehicle {i} to platoon")
        
        self.logger.debug(f"platoon={action.members}")

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
        max_rand_monitor = 1
        return DefenderAction(
            members=frozenset(random.sample(range(len(state.vehicles)), random.randint(0,len(state.vehicles)))),
            monitor=frozenset([] if len(members) < max_rand_monitor else random.sample(members, max_rand_monitor)),
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
        raise NotImplementedError()

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
        choices = [ i for i,v in enumerate(state.vehicles) ]
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

        members = torch.tensor([1 if v.in_platoon else 0 for v in state.vehicles])
        members[kick] = 0
        members[join] = 1
        members = members.nonzero().squeeze()
        members = members.numpy()

        return DefenderAction(
            members=frozenset(members),
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
    def __init__(self,
        state_shape_data: StateShapeData,
        learning_rate: float,
        num_proposals: int,
        utility_func: Optional[lambda state: int],
    ) -> None:
        super().__init__(get_logger("WolpertingerDefenderAgent"))
        self.learning_rate = learning_rate
        self.state_shape_data = state_shape_data
        self.num_proposals = num_proposals
        # allow to monkey patch the utility function
        # helps when adjusting game balance
        if utility_func is not None:
            self.get_utility = utility_func.__get__(self, utility_func)

        self.actor = DefenderActor(state_shape_data).to(get_device())
        self.actor_target = DefenderActor(state_shape_data).to(get_device())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = DefenderCritic().to(get_device())
        self.critic_target = DefenderCritic().to(get_device())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # hard update
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def get_action(
        self,
        state_obj: State
    ) -> DefenderAction:
        # convert from state object to tensors to be fed to the actor model
        state = state_obj.as_tensor_batch(self.state_shape_data)
        state = StateTensorBatch(
            vulnerabilities=state.vulnerabilities.to(get_device()),
            vehicles=state.vehicles.to(get_device()),
        )
        
        # get action suggestions from the actor
        proto_actions: DefenderActionTensorBatch = self.actor(state)
        # todo: introduce epsilon-decaying noise to this while training
        # todo: clip from -1 to 1

        # convert proto actions to actual actions: -lt 0 => 0, -gt 0 => 1
        actions = self.collapse_proto_actions(proto_actions)

        # should be a batch of 1
        assert actions.members.shape[0] == 1
        assert actions.monitor.shape[0] == 1

        # grade the acctions using the critic
        action_q_values: torch.Tensor = self.critic(state, actions)

        # find the best action
        best_action_index = action_q_values.argmax().cpu()
        assert best_action_index.numel() == 1

        # convert binary vectors to vector of indices
        return DefenderAction(
            members=frozenset(actions.members[0][best_action_index].cpu().nonzero().flatten().numpy()),
            monitor=frozenset(actions.monitor[0][best_action_index].cpu().nonzero().flatten().numpy()),
        )
    
    def as_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "num_proposals": self.num_proposals,
            "actor_hidden1_out_features": self.actor.hidden1.out_features,
            "actor_hidden2_out_features": self.actor.hidden2.out_features,
            "critic_hidden1_out_features": self.critic.hidden1.out_features,
            "critic_hidden2_out_features": self.critic.hidden2.out_features,
        }

    # Converts latent action into multiple potential actions
    def collapse_proto_actions(self, proto_actions: DefenderActionTensorBatch) -> DefenderActionTensorBatch:
        num_proposals = 5
        def propose(t: torch.Tensor) -> torch.Tensor:
            noise = torch.linspace(-0.5, +0.5, num_proposals).unsqueeze(dim=1).to(get_device())
            rtn = t.repeat(1,num_proposals,1)
            rtn += noise
            zerovalue = torch.tensor(1.).to(get_device())
            rtn = rtn.heaviside(zerovalue)
            return rtn
        return DefenderActionTensorBatch(
            members=propose(proto_actions.members),
            monitor=propose(proto_actions.monitor),
        )

    def save(self, dir: str, prefix: str = None):
        dir: pathlib.Path = pathlib.Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        models = ["actor", "actor_target", "critic", "critic_target"]
        if prefix is None:
            prefix = get_prefix()
        for model in models:
            save_path = dir / f"{prefix} {model}.pt"
            torch.save(getattr(self,model).state_dict(), save_path)

    def load(self, dir: str, prefix: str):
        dir = pathlib.Path(dir)
        models = ["actor", "actor_target", "critic", "critic_target"]
        for model in models:
            path = dir / f"{prefix} {model}.pt"
            getattr(self,model).load_state_dict(torch.load(path, map_location=get_device()))
    
    def as_dict(self) -> str:
        return {
            "name": self.__class__.__name__,
            "learning_rate": self.learning_rate,
        }