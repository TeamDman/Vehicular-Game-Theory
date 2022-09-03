import random
from typing import List, Literal
from utils import get_logger
from vehicles import Vehicle, VehicleProvider
import logging

class Game:
    vehicles: List[Vehicle]
    vehicle_provider: lambda: Vehicle
    attacker_agent: "Agent"
    defender_agent: "Agent"
    logger: logging.Logger
    turn: Literal["defender","attacker"]
    step_count: int
    max_vehicles: int
    def __init__(self, max_vehicles: int, vehicle_provider: VehicleProvider, defender: "Agent", attacker: "Agent") -> None:
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
        self.vehicles = []
        for _ in range(self.max_vehicles):
            self.vehicles.append(self.vehicle_provider.next())

    def step(self) -> None:
        self.logger.debug("stepping")
        self.turn = "defender" if self.turn == "attacker" else "attacker"
        if self.turn == "attacker":
            self.logger.debug(f"attacker turn begin")
            action = self.attacker_agent.get_action(self)
            self.attacker_agent.take_action(self, action)
            self.logger.debug(f"attacker turn end")
        if self.turn == "defender":
            self.logger.debug(f"defender turn begin")
            action = self.defender_agent.get_action(self)
            self.defender_agent.take_action(self, action)
            self.logger.debug(f"defender turn end")
        
        # if self.step_count % 3 == 0 and len(self.vehicles) < self.max_vehicles:
        if self.step_count % 3 == 0:
            self.logger.info("cycling out vehicles")
            cycle = 2
            removed = 0
            # candidates = [v for v in self.vehicles if not v.in_platoon]
            candidates = [v for v in self.vehicles if not v.in_platoon]
            random.shuffle(candidates)
            while removed < cycle and len(candidates) > 0:
                v = candidates.pop()
                self.logger.info(f"Vehicle {v.id} risk {v.risk} sev {[x.severity for x in v.compromises]} has left the game.")
                self.vehicles.remove(v)
                removed += 1
            while removed > 0:
                new = self.vehicle_provider.next()
                self.logger.info(f"Vehicle {new.id} risk {new.risk} has joined the game!")
                self.vehicles.append(new)
                removed -= 1
        self.vehicles.index
        self.step_count += 1