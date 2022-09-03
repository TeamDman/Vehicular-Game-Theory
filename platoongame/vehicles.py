from dataclasses import dataclass, field
import json
import random
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from utils import get_logger


@dataclass(frozen=True)
class Vulnerability:
    id: int
    attackerCost: int
    defenderCost: int
    attackerProb: float
    defenderProb: float
    severity: int


@dataclass(frozen=True)
class Vehicle:
    id: int
    risk: float
    defender_choices: FrozenSet[Vulnerability]
    attacker_choices: FrozenSet[Vulnerability]
    in_platoon: bool = False
    compromises: FrozenSet[Vulnerability] = field(default_factory=frozenset)
    known_compromises: FrozenSet[Vulnerability] = field(default_factory=frozenset)

    def __str__(self) -> str:
        return f"Vehicle(risk={self.risk}, in_platoon={self.in_platoon}, compromises={self.compromises}, known_compromises={self.known_compromises}, defender_choices={self.defender_choices}, attacker_choices={self.attacker_choices})"

    def __repr__(self) -> str:
        return self.__str__()

class VehicleProvider:
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
        id = 0

        def get_vuln(v):
            return Vulnerability(v["id"], v["attackerCost"], v["defenderCost"], v["attackerProb"], v["defenderProb"], v["severity"])
        for v in loaded:
            v = Vehicle(
                id=id,
                risk=-1 * v["defender_util"],
                defender_choices=frozenset([get_vuln(v) for v in v["dchoice"]]),
                attacker_choices=frozenset([get_vuln(v) for v in v["achoice"]]),
            )
            self.vehicles.append(v)
            id += 1

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
