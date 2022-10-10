from dataclasses import dataclass, field
import json
import random
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from utils import get_logger
from enum import Enum
import torch

class CompromiseState(Enum):
    NOT_COMPROMISED = 1
    COMPROMISED_UNKNOWN = 2
    COMPROMISED_KNOWN = 3


@dataclass(frozen=True)
class Vulnerability:
    prob: float = 0
    severity: int = 0
    state: CompromiseState = CompromiseState.NOT_COMPROMISED

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.prob,
            self.severity,
            self.severity * self.severity,
            # split compromised/known state apart to help model learn
            0 if self.state == CompromiseState.NOT_COMPROMISED else 0, 
            1 if self.state == CompromiseState.COMPROMISED_KNOWN else 0
        ], dtype=torch.float32)

    @staticmethod
    def get_shape():
        return (5,)

@dataclass(frozen=True)
class Vehicle:
    risk: float = 0
    vulnerabilities: Tuple[Vulnerability, ...] = field(default_factory=tuple)
    in_platoon: bool = False

    def __str__(self) -> str:
        return f"Vehicle(risk={self.risk}, in_platoon={self.in_platoon}, vulnerabilities={self.vulnerabilities})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([
            1 if self.in_platoon else 0,
            self.risk
        ], dtype=torch.float32)

    @staticmethod
    def get_shape():
        return (2,)

class VehicleProvider:
    max_vulns: int
    def next(self) -> Vehicle:
        pass

    def reset(self) -> None:
        pass


class JsonVehicleProvider(VehicleProvider):
    vehicles: List[Vehicle]
    seen: Set[Vehicle]  # track vehicles we already provided

    def __init__(self, path: str) -> None:
        self.logger = get_logger("JsonVehicleProvider")
        self.seen = set()

        # load from file
        with open(path, "r") as f:
            loaded = json.load(f)
        self.logger.info(f"loaded {len(loaded)} vehicles")

        # convert json to vehicle classes
        self.vehicles = []
        for vehicle in loaded:
            vulns = []
            for vuln in vehicle["achoice"]:
                prob = vuln["attackerProb"]
                for other in vehicle["dchoice"]:
                    if other["id"] == vuln["id"]:
                        prob *= 1-other["defenderProb"]
                vulns.append(Vulnerability(
                    prob=prob,
                    severity=vuln["severity"],
                ))

            vehicle = Vehicle(
                risk=-1 * vehicle["defender_util"],
                vulnerabilities=tuple(vulns)
            )
            self.vehicles.append(vehicle)

        ids = set()

        self.max_vulns = max([len(v.vulnerabilities) for v in self.vehicles])

    def next(self) -> Vehicle:
        # candidates = [v for v in self.vehicles if v not in self.seen]
        candidates = [v for v in self.vehicles]
        if len(candidates) == 0:
            raise Exception("out of vehicles")
        rtn = random.choice(candidates)
        self.seen.add(rtn)
        return rtn

    def reset(self) -> None:
        self.logger.info("resetting json vehicle provider")
