from __future__ import annotations 
from dataclasses import dataclass, replace
import dataclasses
from typing import Dict, Tuple, TYPE_CHECKING, Union
from agents import AttackerAction, AttackerAgent, DefenderAction, DefenderAgent
from models import StateTensorBatch, StateShapeData
from utils import get_logger
from vehicles import Vehicle, VehicleProvider, Vulnerability
import logging
import random
import torch

if TYPE_CHECKING:
    from agents import Agent


@dataclass(frozen=True)
class StateTensorBatchShape:
    vulnerabilities: Tuple[int,int,int,int]

@dataclass(frozen=True)
class State:
    vehicles: Tuple[Vehicle,...]
    defender_utility: int = 0
    attacker_utility: int = 0

    def as_tensor_batch(self, shape_data: StateShapeData) -> StateTensorBatch:
        shape = State.get_shape(shape_data, batch_size=1)
        vulns_quant = torch.zeros(shape.vulnerabilities)
        for i, vehicle in enumerate(self.vehicles):
            for j, vuln in enumerate(vehicle.vulnerabilities):
                vulns_quant[0][i,j] = vuln.as_tensor()
        return StateTensorBatch(
            vulnerabilities=vulns_quant,
        )

    @staticmethod
    def get_shape(shape_data: StateShapeData, batch_size: int) -> StateTensorBatchShape:
        return StateTensorBatchShape(
            vulnerabilities=(batch_size, shape_data.num_vehicles, shape_data.num_vulns, Vulnerability.get_shape()[0]),
        )

    @staticmethod
    def zeros(shape_data: StateShapeData, batch_size: int) -> StateTensorBatch:
        shape = State.get_shape(shape_data, batch_size)
        return StateTensorBatch(
            vulnerabilities=torch.zeros(shape.vulnerabilities),
        )

        

@dataclass
class GameConfig:
    max_vehicles: int
    cycle_enabled: bool
    cycle_every: int
    cycle_num: int
    cycle_allow_platoon: bool

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)

class Game:
    state: State
    vehicle_provider: VehicleProvider
    logger: logging.Logger
    step: int
    config: GameConfig
    def __init__(self, config: GameConfig, vehicle_provider: VehicleProvider) -> None:
        self.vehicle_provider = vehicle_provider
        self.logger = get_logger("Game")
        self.step = 0
        self.config = config
        self.state = State(vehicles=tuple([self.vehicle_provider.next() for _ in range(self.config.max_vehicles)]))

    def take_step(
        self,
        attacker_agent: AttackerAgent,
        defender_agent: DefenderAgent
    ) -> Tuple[AttackerAction, DefenderAction]:
        self.cycle()

        self.logger.debug("stepping")
        
        self.logger.debug(f"defender turn begin")
        defender_action = defender_agent.get_action(self.state)
        self.state = defender_agent.take_action(self.state, defender_action)
        self.logger.debug(f"defender turn end")

        self.logger.debug(f"attacker turn begin")
        attacker_action = attacker_agent.get_action(self.state)
        self.state = attacker_agent.take_action(self.state, attacker_action)
        self.logger.debug(f"attacker turn end")
        
        self.step += 1

        return attacker_action, defender_action

    def cycle(self) -> None:
        if self.config.cycle_enabled and self.step % self.config.cycle_every == 0:
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