from dataclasses import dataclass, field
import json
import random
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from utils import get_logger
from enum import Enum
class CompromiseState(Enum):
    NOT_COMPROMISED = 1
    COMPROMISED_UNKNOWN = 2
    COMPROMISED_KNOWN = 3


@dataclass(frozen=True)
class Vulnerability:
    prob: float
    severity: int
    state: CompromiseState = CompromiseState.NOT_COMPROMISED


@dataclass(frozen=True)
class Vehicle:
    risk: float
    vulnerabilities: Tuple[Vulnerability, ...]
    in_platoon: bool = False

    def __str__(self) -> str:
        return f"Vehicle(risk={self.risk}, in_platoon={self.in_platoon}, vulnerabilities={self.vulnerabilities})"

    def __repr__(self) -> str:
        return self.__str__()

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
