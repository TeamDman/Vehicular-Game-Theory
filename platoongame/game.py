from __future__ import annotations 
from dataclasses import dataclass, replace
import dataclasses
import json
from re import S
from typing import Dict, FrozenSet, List, Tuple, TYPE_CHECKING, Union
from models import StateTensorBatch, StateShapeData
from utils import get_logger
from vehicles import Vehicle, VehicleProvider, Vulnerability
import logging
import random
import torch
import vehicles

if TYPE_CHECKING:
    from agents import Agent


@dataclass(frozen=True)
class State:
    vehicles: Tuple[Vehicle,...]
    defender_utility: int = 0
    attacker_utility: int = 0

    def as_tensors(self, shape_data: StateShapeData) -> StateTensorBatch:
        shape = State.get_shape(shape_data, batch_size=1)
        vulns_quant = torch.zeros(shape.vulnerabilities)
        vehicles_quant = torch.zeros(shape.vehicles)
        for i, vehicle in enumerate(self.vehicles):
            vehicles_quant[0][i] = vehicle.as_tensor()
            for j, vuln in enumerate(vehicle.vulnerabilities):
                vulns_quant[0][i,j] = vuln.as_tensor()
        return StateTensorBatch(
            vulnerabilities=vulns_quant,
            vehicles=vehicles_quant,
        )

    @staticmethod
    def get_shape(shape_data: StateShapeData, batch_size: int) -> StateTensorBatch:
        return StateTensorBatch(
            vulnerabilities=(batch_size, shape_data.num_vehicles, shape_data.num_vulns, Vulnerability.get_shape()[0]),
            vehicles=(batch_size, shape_data.num_vehicles, Vehicle.get_shape()[0]),
        )


@dataclass
class GameConfig:
    max_vehicles: int = 10
    cycle_every: Union[int,None] = None
    cycle_num: int = None
    cycle_allow_platoon: bool = False

    def get_config(self) -> Dict:
        return dataclasses.asdict(self)

class Game:
    state: State
    vehicle_provider: VehicleProvider
    logger: logging.Logger
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
        self.vehicle_provider.reset()
        self.state = State(vehicles=tuple([self.vehicle_provider.next() for _ in range(self.config.max_vehicles)]))

    def step(
        self,
        attacker_agent: Agent,
        defender_agent: Agent
    ) -> None:
        self.logger.debug("stepping")
        self.logger.debug(f"attacker turn begin")
        action = attacker_agent.get_action(self.state)
        self.state = attacker_agent.take_action(self.state, action)
        self.logger.debug(f"attacker turn end")
        
        self.logger.debug(f"defender turn begin")
        action = defender_agent.get_action(self.state)
        self.state = defender_agent.take_action(self.state, action)
        self.logger.debug(f"defender turn end")
        
        self.cycle()

        self.step_count += 1

    def cycle(self) -> None:
        # if self.step_count % 3 == 0 and len(self.vehicles) < self.config.max_vehicles:
        if self.config.cycle_every is not None and self.step_count % self.config.cycle_every == 0:
            remove = set()
            candidates = list([(i,v) for i,v in enumerate(self.state.vehicles) if not v.in_platoon or self.config.cycle_allow_platoon])
            self.logger.info(f"cycling out {len(candidates)} of {self.config.cycle_num} allowed vehicles")
            random.shuffle(candidates)
            while len(remove) < self.config.cycle_num and len(candidates) > 0:
                i,v = candidates.pop()
                self.logger.info(f"Vehicle {i} risk {v.risk} has left the game.")
                remove.add(i)
            vehicles = list([v for i,v in enumerate(self.state.vehicles) if i not in remove])
            for _ in range(len(remove)):
                new = self.vehicle_provider.next()
                self.logger.info(f"Vehicle with risk {new.risk} has joined the game!")
                vehicles.append(new)
            self.state = replace(self.state, vehicles=tuple(vehicles))