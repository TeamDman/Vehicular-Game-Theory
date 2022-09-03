
import json
import random
import logging
from typing import Set, Dict, List
from utils import get_logger
from vulnerability import Vulnerability

class Vehicle:
    id: int
    risk: float
    severity_chances: Dict[int,float]
    in_platoon: bool
    dchoice: List[Vulnerability]
    achoice: List[Vulnerability]
    compromises: Set[Vulnerability]
    known_compromises: Set[Vulnerability]
    def __init__(
        self,
        id,
        risk: float,
        dchoice: List[Vulnerability],
        achoice: List[Vulnerability],
        sevs: Dict[int,float],
        in_platoon: bool = False
    ) -> None:
        self.id = id
        self.risk = risk
        self.dchoice = dchoice
        self.achoice = achoice
        self.severity_chances = sevs
        self.in_platoon = in_platoon
        self.compromises = set()
        self.known_compromises = set()
    def __str__(self) -> str:
        return f"Vehicle(risk={self.risk}, sevs={self.severity_chances}, in_platoon={self.in_platoon}, compromises={self.compromises}, known_compromises={self.known_compromises})"
    def __repr__(self) -> str:
        return self.__str__()

class VehicleProvider:
    def next(self) -> Vehicle:
        pass
    def reset(self) -> None:
        pass

class JsonVehicleProvider(VehicleProvider):
    vehicles: List[Vehicle]
    seen: Set[Vehicle]
    def __init__(self, path: str) -> None:
        self.logger = get_logger("JsonVehicleProvider")
        self.seen = set()
        with open(path, "r") as f:
            loaded = json.load(f)
        self.logger.info(f"loaded {len(loaded)} vehicles")
        self.vehicles = []
        id = 0
        for v in loaded:
            self.vehicles.append(Vehicle(
                id,
                -1 * v["defender_util"],
                [Vulnerability(v["id"], v["attackerCost"], v["defenderCost"], v["attackerProb"], v["defenderProb"], v["severity"]) for v in v["dchoice"]],
                [Vulnerability(v["id"], v["attackerCost"], v["defenderCost"], v["attackerProb"], v["defenderProb"], v["severity"]) for v in v["achoice"]],
                {i:v["severity_chances"][str(i)] for i in range(1,6)}
            ))
            id += 1
    def next(self) -> Vehicle:
        candidates = [v for v in self.vehicles if v not in self.seen]
        if len(candidates) == 0: raise Exception("out of vehicles")
        rtn = random.choice(candidates)
        self.seen.add(rtn)
        return rtn

    def reset(self) -> None:
        self.logger.info("resetting json vehicle provider")
        self.i = 0
