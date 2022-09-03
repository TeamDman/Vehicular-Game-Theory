from __future__ import annotations 
from dataclasses import dataclass, replace
from typing import FrozenSet, List, Tuple, Literal, TYPE_CHECKING
from utils import get_logger
from vehicles import Vehicle, VehicleProvider
import logging
import random

if TYPE_CHECKING:
    from agents import Agent

@dataclass(frozen=True)
class State:
    vehicles: FrozenSet[Vehicle]
    defender_utility: int = 0
    attacker_utility: int = 0

class Game:
    state: State
    vehicle_provider: VehicleProvider
    attacker_agent: Agent
    defender_agent: Agent
    logger: logging.Logger
    turn: Literal["defender","attacker"]
    step_count: int
    max_vehicles: int
    def __init__(self, max_vehicles: int, vehicle_provider: VehicleProvider, defender: Agent, attacker: Agent) -> None:
        self.vehicle_provider = vehicle_provider
        self.logger = get_logger("Game")
        self.defender_agent = defender
        self.attacker_agent = attacker
        self.step_count = 0
        self.max_vehicles = max_vehicles
        self.reset()

    def reset(self) -> None:
        self.logger.info("resetting game")
        self.turn = "defender"
        self.vehicle_provider.reset()
        vehicles = set()
        for _ in range(self.max_vehicles):
            v = self.vehicle_provider.next()
            vehicles.add(v)
        self.state = State(vehicles=frozenset(vehicles))

    def step(self) -> None:
        self.logger.debug("stepping")
        self.turn = "defender" if self.turn == "attacker" else "attacker"
        if self.turn == "attacker":
            self.logger.debug(f"attacker turn begin")
            action = self.attacker_agent.get_action(self.state)
            self.state = self.attacker_agent.take_action(self.state, action)
            self.logger.debug(f"attacker turn end")
        if self.turn == "defender":
            self.logger.debug(f"defender turn begin")
            action = self.defender_agent.get_action(self.state)
            self.state = self.defender_agent.take_action(self.state, action)
            self.logger.debug(f"defender turn end")
        
        # if self.step_count % 3 == 0 and len(self.vehicles) < self.max_vehicles:
        if self.step_count % 3 == 0:
            self.logger.info("cycling out vehicles")
            cycle = 2
            removed = 0
            vehicles = list(self.state.vehicles)
            # candidates = [v for v in self.vehicles if not v.in_platoon]
            candidates = [v for v in vehicles if not v.in_platoon]
            random.shuffle(candidates)
            while removed < cycle and len(candidates) > 0:
                v = candidates.pop()
                self.logger.info(f"Vehicle {v.id} risk {v.risk} sev {[x.severity for x in v.compromises]} has left the game.")
                vehicles.remove(v)
                removed += 1
            while removed > 0:
                new = self.vehicle_provider.next()
                self.logger.info(f"Vehicle {new.id} risk {new.risk} has joined the game!")
                vehicles.append(new)
                removed -= 1
            self.state = replace(self.state, vehicles=frozenset(vehicles))
        self.step_count += 1