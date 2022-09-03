from __future__ import annotations 
import logging
import random
from typing import List, Set, Union, FrozenSet, TYPE_CHECKING
from utils import get_logger
from vehicles import Vehicle
from pprint import pprint

from dataclasses import dataclass, replace

if TYPE_CHECKING:
    from game import Game, State


@dataclass(frozen=True)
class DefenderAction:
    monitor: FrozenSet[Vehicle]
    join: FrozenSet[Vehicle]
    kick: FrozenSet[Vehicle]


@dataclass(frozen=True)
class AttackerAction:
    attack: FrozenSet[Vehicle]


Action = Union[DefenderAction, AttackerAction]


class Agent:
    utility: int
    logger: logging.Logger

    def __init__(self, logger: logging.Logger) -> None:
        self.utility = 0
        self.logger = logger

    def get_action(self, state: State) -> Action:
        pass

    def take_action(self, state: State, action: Action) -> State:
        pass

    def get_utility(self, state: State) -> int:
        return None

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
        members = len([v for v in state.vehicles if v.in_platoon])
        compromises = - \
            sum([s.severity for v in state.vehicles for s in v.compromises if v.in_platoon])
        return members * 10 + compromises

    def take_action(self, state: State, action: DefenderAction) -> State:
        vehicles = set()
        for v in state.vehicles:
            if v in action.monitor:
                self.logger.debug(f"monitoring vehicle id {v.id}")
                # logging the discovered compromises
                for c in v.compromises:
                    if c in v.known_compromises:
                        continue
                    self.logger.info(
                        f"discovered compromise on vehicle {v.id} vuln {c.id} sev {c.severity}")
                # update vehicle
                v = replace(v, known_compromises=v.compromises)

            if v in action.kick:
                sev = sum([s.severity for s in v.known_compromises])
                self.logger.info(
                    f"kicking vehicle {v.id} out of platoon, severity {sev} exceeded threshold {self.tolerance_threshold}")
                v = replace(v, in_platoon=False)

            if v in action.join:
                sev = sum([s.severity for s in v.known_compromises])
                self.logger.info(
                    f"adding vehicle {v.id} to platoon, severity {sev}")
                v = replace(v, in_platoon=True)

            if v in action.kick and v in action.join:
                self.logger.warn(
                    f"sanity check failed, vehicle {v.id} was kicked and joined at the same time")

            vehicles.add(v)

        # calculate util change
        state = replace(state, vehicles=frozenset(vehicles))
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
            v for v in state.vehicles
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
            for v in monitor:
                self.recently_monitored.append(v)

        # kick risky vehicles from platoon
        kick = set()
        for v in [v for v in state.vehicles if v.in_platoon]:
            total = sum([s.severity for s in v.known_compromises])
            if total > self.tolerance_threshold:
                kick.add(v)

        # join low risk vehicles into the platoon
        join = set()
        member_count = len([v for v in state.vehicles if v.in_platoon])
        candidates = [
            v for v in state.vehicles
            if v.risk <= 10
            and sum([s.severity for s in v.known_compromises]) <= self.tolerance_threshold
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
        for v in state.vehicles:
            for c in v.compromises:
                if v.in_platoon:
                    util += c.severity * 2
                else:
                    util += c.severity
        return util

    def take_action(self, state: State, action: AttackerAction) -> State:
        vehicles = set()
        for v in state.vehicles:
            if v in action.attack:
                self.logger.debug(f"attacking vehicle id {v.id}")
                for vuln in v.attacker_choices:
                    success = random.randint(
                        1, 1000) / 1000 > vuln.attackerProb
                    if success and vuln in v.defender_choices:
                        success = random.randint(
                            1, 1000) / 1000 > vuln.defenderProb
                    if success:
                        self.logger.info(
                            f"successfully compromised vehicle {v.id} vuln {vuln.id} sev {vuln.severity}")
                        v = replace(v, compromises=frozenset(
                            list(v.compromises) + [vuln]))
            vehicles.add(v)

        # calculate util change
        state = replace(state, vehicles=frozenset(vehicles))
        change = self.get_utility(state)
        utility = state.attacker_utility
        utility += change
        self.logger.debug(f"utility {utility} ({change:+d})")
        state = replace(state, attacker_utility=utility)

        # return new state
        return state


    def get_action(self, state: State) -> None:
        # Pick vehicle to attack
        candidates = [v for v in state.vehicles if v.in_platoon]
        attack = set()
        if len(candidates) == 0:
            # attack anything if platoon is empty
            candidates = state.vehicles

        if len(candidates) == 0:
            self.logger.warn("sanity check failed, no vehicles to attack")
        else:
            def eval_risk(v: Vehicle) -> float:
                return [x.severity ** 2 * x.attackerProb * (1 if not x in v.defender_choices else 1 - x.defenderProb) for x in v.attacker_choices if x not in v.compromises]
            candidates = sorted(candidates, key=eval_risk)
            for _ in range(self.attack_limit):
                if len(candidates) == 0:
                    break
                attack.add(candidates.pop())

        return AttackerAction(attack=frozenset(attack))
