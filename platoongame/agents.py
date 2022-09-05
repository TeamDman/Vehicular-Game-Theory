from __future__ import annotations
import logging
import random
from typing import List, Set, Union, FrozenSet, TYPE_CHECKING
from utils import get_logger
from vehicles import CompromiseState, Vehicle
from pprint import pprint

from dataclasses import dataclass, replace

if TYPE_CHECKING:
    from game import Game, State


@dataclass(frozen=True)
class DefenderAction:
    monitor: FrozenSet[int] # indices of corresponding vehicle
    join: FrozenSet[int]
    kick: FrozenSet[int]


@dataclass(frozen=True)
class AttackerAction:
    attack: FrozenSet[int]


Action = Union[DefenderAction, AttackerAction]


class Agent:
    logger: logging.Logger

    def __init__(self, logger: logging.Logger) -> None:
        self.utility = 0
        self.logger = logger

    def get_action(self, state: State) -> Action:
        pass

    def take_action(self, state: State, action: Action) -> State:
        pass

    def get_utility(self, state: State) -> int:
        pass

# Very simple agent for defender


class BasicDefenderAgent(Agent):
    num_vehicles_monitoring_constraint: int
    recently_monitored: List[Vehicle]
    tolerance_threshold: int
    max_history: int
    max_size: int

    def __init__(self) -> None:
        super().__init__(get_logger("BasicDefenderAgent"))
        self.recently_monitored = []
        self.tolerance_threshold = 3
        self.max_history = 3
        self.max_size = 10

    def get_utility(self, state: State) -> None:
        members = len([vehicle for vehicle in state.vehicles if vehicle.in_platoon])
        compromises = - \
            sum([vuln.severity for vehicle in state.vehicles for vuln in vehicle.vulnerabilities if vehicle.in_platoon])
        return members * 10 + compromises

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

        # kick vehicles
        for i in action.kick:
            vehicle = vehicles[i]
            self.logger.info(f"kicking vehicle {i} out of platoon")
            vehicle = replace(vehicle, in_platoon = False)
            vehicles[i] = vehicle

        # join vehicles
        if i in action.join:
            vehicle = vehicles[i]
            self.logger.info(f"adding vehicle {i} to platoon")
            vehicle = replace(vehicle, in_platoon=True)
            vehicles[i] = vehicle

        # sanity check
        if len(action.kick.intersection(action.join)) > 0:
            self.logger.warn(f"sanity check failed, some vehicles were kicked and joined at the same time {action}")

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

    def get_action(self, state: State) -> DefenderAction:
        # pick next vehicle to monitor
        choices = [
            i for i,v in enumerate(state.vehicles)
            if v not in self.recently_monitored
        ]
        monitor = set()
        if len(choices) == 0:
            self.logger.warn("no candidates found to monitor?")
            pprint(state.vehicles)
            pprint(self.recently_monitored)
        else:
            monitor.add(random.choice(choices))
            # avoid monitoring the last 3 vehicles again
            if len(self.recently_monitored) >= self.max_history:
                self.recently_monitored.pop(0)
            for vehicle in monitor:
                self.recently_monitored.append(vehicle)

        # kick risky vehicles from platoon
        kick = set()
        for i,vehicle in enumerate(state.vehicles):
            if not vehicle.in_platoon: continue
            total = sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED])
            if total > self.tolerance_threshold:
                kick.add(i)

        # join low risk vehicles into the platoon
        join = set()
        member_count = len([1 for v in state.vehicles if v.in_platoon])
        candidates = [
            i for i,vehicle in enumerate(state.vehicles)
            if vehicle.risk <= 10
            and sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]) <= self.tolerance_threshold
            and member_count < self.max_size
        ]
        random.shuffle(candidates)
        while member_count + len(join) < self.max_size and len(candidates) > 0:
            # take while there's room in the platoon
            join.add(candidates.pop())

        return DefenderAction(
            monitor=frozenset(monitor),
            join=frozenset(join),
            kick=frozenset(kick)
        )

# Very simple agent for attacker


class BasicAttackerAgent(Agent):
    def __init__(self) -> None:
        super().__init__(get_logger("BasicAttackerAgent"))
        self.attack_limit = 2

    def get_utility(self, state: State) -> None:
        util = 0
        for vehicle in state.vehicles:
            for vuln in vehicle.vulnerabilities:
                if vuln.state == CompromiseState.NOT_COMPROMISED: continue
                if vehicle.in_platoon:
                    util += vuln.severity * 2
                else:
                    util += vuln.severity
        return util

    def take_action(self, state: State, action: AttackerAction) -> State:
        vehicles = list(state.vehicles)
        for i in action.attack:
            self.logger.debug(f"attacking vehicle {i}")
            vehicle = vehicles[i]
            for j, vuln in enumerate(vehicle.vulnerabilities):
                if vuln.state != CompromiseState.NOT_COMPROMISED: continue
                if random.random() > vuln.prob: continue
                self.logger.info(f"successfully compromised vehicle {i} vuln {j} sev {vuln.severity}")
                vehicle = replace(vehicle, vulnerabilities=vehicle.vulnerabilities[:j] + (
                    replace(vuln, state=CompromiseState.COMPROMISED_UNKNOWN),) + vehicle.vulnerabilities[j+1:])
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
