from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import random
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from utils import get_logger
from enum import Enum, auto
import torch

class CompromiseState(Enum):
    NOT_COMPROMISED = auto()
    COMPROMISED = auto()


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
            0 if self.state == CompromiseState.NOT_COMPROMISED else 1, 
        ], dtype=torch.float32)

    @staticmethod
    def get_shape():
        return (4,)

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

class VehicleProvider(ABC):
    max_vulns: int

    @abstractmethod
    def next(self) -> Vehicle:
        pass


class RubbishVehicleProvider(VehicleProvider):
    def __init__(self) -> None:
        self.max_vulns=1

    def next(self) -> Vehicle:
        return Vehicle(
            risk=10,
            vulnerabilities=( Vulnerability(0.5,2,CompromiseState.NOT_COMPROMISED), )
        )

class RandomVehicleProvider(VehicleProvider):
    def __init__(
        self,
        num_max_vulns: int,
        prob_mu: float,
        prob_sigma: float,
        sev_mu: float,
        sev_sigma: float,
    ) -> None:
        self.max_vulns = num_max_vulns
        self.prob_dist = torch.distributions.Normal(torch.as_tensor(prob_mu, dtype=torch.float32), torch.as_tensor(prob_sigma, dtype=torch.float32))
        self.sev_dist = torch.distributions.Normal(torch.as_tensor(sev_mu, dtype=torch.float32), torch.as_tensor(sev_sigma, dtype=torch.float32))
    
    def next(self) -> Vehicle:
        vulns = tuple(Vulnerability(
            prob=float(self.prob_dist.sample().clamp(0,1)),
            severity=int(self.sev_dist.sample().clamp(1,5)),
            state=CompromiseState.NOT_COMPROMISED,
        ) for _ in range(random.randint(0, self.max_vulns)))
        risk = sum([v.prob * v.severity ** 2 for v in vulns])
        return Vehicle(risk=risk, vulnerabilities=vulns, in_platoon=False)

class JsonVehicleProvider(VehicleProvider):
    vehicles: List[Vehicle]

    def __init__(self, path: str) -> None:
        self.logger = get_logger("JsonVehicleProvider")

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
            
        self.max_vulns = max([len(v.vulnerabilities) for v in self.vehicles])

    def next(self) -> Vehicle:
        return random.choice(self.vehicles)

