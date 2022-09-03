import logging
import random
from typing import List, Set, Union
from utils import get_logger
from vehicles import Vehicle
from game import Game
from pprint import pprint

from dataclasses import dataclass



@dataclass(frozen=True)
class DefenderAction:
    monitor: Set[Vehicle]
    join: Set[Vehicle]
    kick: Set[Vehicle]

@dataclass(frozen=True)
class AttackerAction:
    attack: Set[Vehicle]

Action = Union[DefenderAction, AttackerAction]

class Agent:
    utility: int
    logger: logging.Logger
    def __init__(self, logger: logging.Logger) -> None:
        self.utility = 0
        self.logger = logger

    def get_action(self, game: Game) -> Action:
        pass

    def take_action(self, game: Game, action: Action) -> None:
        pass

    def get_utility(self, game: Game) -> int:
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
        self.utility = 0
        self.tolerance_threshold = 3
        self.max_history = 3
        self.max_size = 10
    
    def get_utility(self, game: Game) -> None:
        members = len([v for v in game.vehicles if v.in_platoon])
        compromises = -sum([s.severity for v in game.vehicles for s in v.compromises if v.in_platoon ])
        return members * 10 + compromises

    def take_action(self, game: Game, action: DefenderAction) -> float:
        if game.turn != "defender": self.logger.warn("sanity check failed, defender taking turn when game says not their turn")

        for v in action.monitor:
            self.logger.debug(f"monitoring vehicle id {v.id}")
            # The current compromises become known
            for c in v.compromises:
                if c in v.known_compromises: continue
                self.logger.info(f"discovered compromise on vehicle {v.id} vuln {c.id} sev {c.severity}")
                v.known_compromises.add(c)
        
        for v in action.kick:
            sev = sum([s.severity for s in v.known_compromises])
            self.logger.info(f"kicking vehicle {v.id} out of platoon, severity {sev} exceeded threshold {self.tolerance_threshold}")
            v.in_platoon = False

        for v in action.join:
            sev = sum([s.severity for s in v.known_compromises])
            self.logger.info(f"adding vehicle {v.id} to platoon, severity {sev}")
            v.in_platoon = True
        
        # calculate util change
        change = self.get_utility(game)
        self.utility += change
        self.logger.debug(f"utility {self.utility} ({change:+d})")
        return change

    
    def get_action(self, game: Game) -> DefenderAction:
        # pick next vehicle to monitor
        choices = [
            v for v in game.vehicles
            if v not in self.recently_monitored
        ]
        monitor=set()
        if len(choices) == 0:
            self.logger.warn("no candidates found to monitor?")
            pprint(game.vehicles)
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
        for v in [v for v in game.vehicles if v.in_platoon]:
            total = sum([s.severity for s in v.known_compromises])
            if total > self.tolerance_threshold:
                kick.add(v)

        # join low risk vehicles into the platoon
        join = set()
        member_count = len([v for v in game.vehicles if v.in_platoon])
        candidates = [
            v for v in game.vehicles
            if v.risk <= 10 \
            and sum([s.severity for s in v.known_compromises]) <= self.tolerance_threshold \
            and member_count < self.max_size
        ]
        random.shuffle(candidates)
        while member_count + len(join) < self.max_size and len(candidates) > 0:
            # take while there's room in the platoon
            join.add(candidates.pop())
        
        return DefenderAction(
            monitor=monitor,
            join=join,
            kick=kick
        )

# Very simple agent for attacker
class BasicAttackerAgent(Agent):
    def __init__(self) -> None:
        super().__init__(get_logger("BasicAttackerAgent"))
        self.attack_limit = 2

    def get_utility(self, game: Game) -> None:
        util = 0
        for v in game.vehicles:
            for c in v.compromises:
                if v.in_platoon:
                    util += c.severity * 2
                else:
                    util += c.severity
        return util

    def take_action(self, game: Game, action:AttackerAction) -> float:
        if game.turn != "attacker": self.logger.warn("sanity check failed, attacker taking turn when game says not their turn")

        for v in action.attack:
            self.logger.debug(f"attacking vehicle id {v.id}")
            for vuln in v.achoice:
                success = random.randint(1,1000) / 1000 > vuln.attackerProb
                if success and vuln in v.dchoice:
                    success = random.randint(1,1000) / 1000 > vuln.defenderProb
                if success:
                    self.logger.info(f"successfully compromised vehicle {v.id} vuln {vuln.id} sev {vuln.severity}")
                    v.compromises.add(vuln)
        
        # calculate util change
        change = self.get_utility(game)
        self.utility += change
        self.logger.debug(f"utility {self.utility} ({change:+d})")
        return change

    def get_action(self, game: Game) -> None:
        # Pick vehicle to attack
        candidates = [v for v in game.vehicles if v.in_platoon]
        attack = set()
        if len(candidates) == 0:
            # attack anything if platoon is empty
            candidates = game.vehicles

        if len(candidates) == 0:
            self.logger.warn("sanity check failed, no vehicles to attack")
        else:
            def eval_risk(v: Vehicle) -> float:
                return [x.severity ** 2 * x.attackerProb * (1 if not x in v.dchoice else 1 - x.defenderProb) for x in v.achoice if x not in v.compromises]
            candidates = sorted(candidates, key=eval_risk)
            for _ in range(self.attack_limit):
                if len(candidates) == 0: break
                attack.add(candidates.pop())

        return AttackerAction(attack=attack)
