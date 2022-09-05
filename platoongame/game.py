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
    vehicles: Tuple[Vehicle,...]
    defender_utility: int = 0
    attacker_utility: int = 0

@dataclass
class GameConfig:
    max_vehicles: int

class Game:
    state: State
    vehicle_provider: VehicleProvider
    logger: logging.Logger
    turn: Literal["defender","attacker"]
    step_count: int
    config: GameConfig
    def __init__(self, config: GameConfig, vehicle_provider: VehicleProvider) -> None:
        self.vehicle_provider = vehicle_provider
        self.logger = get_logger("Game")
        self.step_count = 0
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.logger.info("resetting game")
        self.turn = "defender"
        self.vehicle_provider.reset()
        self.state = State(vehicles=tuple([self.vehicle_provider.next() for _ in range(self.config.max_vehicles)]))

    def step(
        self,
        attacker_agent: Agent,
        defender_agent: Agent
    ) -> None:
        self.logger.debug("stepping")
        self.turn = "defender" if self.turn == "attacker" else "attacker"
        if self.turn == "attacker":
            self.logger.debug(f"attacker turn begin")
            action = attacker_agent.get_action(self.state)
            self.state = attacker_agent.take_action(self.state, action)
            self.logger.debug(f"attacker turn end")
        if self.turn == "defender":
            self.logger.debug(f"defender turn begin")
            action = defender_agent.get_action(self.state)
            self.state = defender_agent.take_action(self.state, action)
            self.logger.debug(f"defender turn end")
        
        # if self.step_count % 3 == 0 and len(self.vehicles) < self.config.max_vehicles:
        if self.step_count % 3 == 0:
            self.logger.info("cycling out vehicles")
            cycle = 2
            remove = set()
            candidates = list([(i,v) for i,v in enumerate(self.state.vehicles) if not v.in_platoon])
            random.shuffle(candidates)
            while len(remove) < cycle and len(candidates) > 0:
                i,v = candidates.pop()
                self.logger.info(f"Vehicle {i} risk {v.risk} has left the game.")
                remove.add(i)
            vehicles = list([v for i,v in enumerate(self.state.vehicles) if i not in remove])
            for _ in range(len(remove)):
                new = self.vehicle_provider.next()
                self.logger.info(f"Vehicle with risk {new.risk} has joined the game!")
                vehicles.append(new)
            self.state = replace(self.state, vehicles=tuple(vehicles))
        self.step_count += 1