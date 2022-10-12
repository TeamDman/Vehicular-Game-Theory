from __future__ import annotations
from dataclasses import dataclass, field, _MISSING_TYPE
import dataclasses
import json
import pathlib
from statistics import mean
from agents import Agent
from game import Game, GameConfig
from utils import NoneRefersDefault
from vehicles import CompromiseState, VehicleProvider, Vulnerability
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import utils

if TYPE_CHECKING:
    from training import WolpertingerDefenderAgentTrainer, OptimizationResult

@dataclass
class EpisodeMetricsEntry:
    defender_util: float
    attacker_util: float
    compromises: int
    known_compromises: int
    compromised_overall: float
    compromised_partial: float
    platoon_severity: int
    potential_platoon_severity: int
    vehicles: int
    platoon_size: int
    epsilon_threshold: float
    platoon_risk: float
    average_platoon_risk: float

    def save(self, dir: str, step: int):
        path = pathlib.Path(dir)
        path.mkdir(exist_ok=True, parents=True)
        filename = f"{utils.get_prefix()} {step:06}.log.json"
        path = path / filename
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)



@dataclass
class OptimizationTracker:
    stats: List[OptimizationResult] = field(default_factory=list)
    def track_stats(
        self,
        optimization_results: OptimizationResult,
    ) -> None:
        self.stats.append(optimization_results)

    def set_margins(self) -> None:
        # plt.tight_layout()
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(16)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            # wspace=0.4,
            hspace=0.4,
        )

    def plot_loss(self) -> None:
        plt.subplot(9, 2, 7)
        plt.plot([x.optim.loss for x in self.stats],label="loss")
        plt.legend(loc="upper left")
        plt.title("loss")

        
    def plot(self):
        self.plot_loss()

@dataclass
class EpisodeMetricsTracker:
    stats: List[EpisodeMetricsEntry] = field(default_factory=list)

    def track_stats(self, trainer: WolpertingerDefenderAgentTrainer) -> None:
        game = trainer.game
        platoon_members = [v for v in game.state.vehicles if v.in_platoon]
        entry = EpisodeMetricsEntry(
            defender_util=game.state.defender_utility,
            attacker_util=game.state.attacker_utility,
            compromises=len([1 for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
            known_compromises=len([vuln for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN]),
            compromised_overall=len([1 for vehicle in game.state.vehicles if all([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            compromised_partial=len([1 for vehicle in game.state.vehicles if any([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            potential_platoon_severity = sum([vuln.severity for vehicle in platoon_members for vuln in vehicle.vulnerabilities]),
            platoon_severity = sum([vuln.severity for vehicle in platoon_members for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
            platoon_size=len(platoon_members),
            vehicles=len(game.state.vehicles),
            platoon_risk=sum([v.risk for v in platoon_members]),
            average_platoon_risk=0 if len(platoon_members) == 0 else mean([v.risk for v in platoon_members]),
            epsilon_threshold=trainer.get_epsilon_threshold(),
        )
        self.stats.append(entry)
        # entry.save(dir="logs", step=step)

    def plot_utilities(self) -> None:
        plt.subplot(9, 2, 1)
        plt.plot([x.defender_util for x in self.stats], label="defender")
        plt.plot([x.attacker_util for x in self.stats], label="attacker")
        plt.legend(loc="upper left")
        plt.title("accumulated utilities")

    def plot_utility_deltas(self) -> None:
        plt.subplot(9, 2, 2)
        plt.plot(np.diff([x.defender_util for x in self.stats]), label="defender", alpha=0.9)
        plt.plot(np.diff([x.attacker_util for x in self.stats]), label="attacker", alpha=0.9)
        plt.legend(loc="upper left")
        plt.title("utility deltas")

    def plot_severity(self) -> None:
        plt.subplot(9, 2, 3)
        plt.plot([x.platoon_severity for x in self.stats], label="severity", alpha=0.9)
        plt.plot([x.potential_platoon_severity for x in self.stats], label="potential severity", alpha=0.9)
        plt.legend(loc="upper left")
        plt.title("platoon severity")

    def plot_vuln_compromises(self) -> None:
        plt.subplot(9, 2, 4)
        plt.plot([x.compromises for x in self.stats], label="compromises", alpha=0.9)
        plt.plot([x.known_compromises for x in self.stats], label="known compromises", alpha=0.9)
        # plt.set_yticks(np.arange(0,1,0.1))
        plt.title("vulnerability compromises")
        plt.legend(loc="upper left")

    def plot_vulns(self) -> None:
        plt.subplot(9, 2, 5)
        plt.plot([x.compromised_overall for x in self.stats], label="overall", alpha=0.9)
        plt.plot([x.compromised_partial for x in self.stats], label="partial", alpha=0.9)
        plt.plot([x.vehicles for x in self.stats],label="vehicle count", alpha=0.9)
        # plt.set_yticks(np.arange(0,1,0.1))
        plt.title("vehicle compromises")
        plt.legend(loc="upper left")

    def plot_membership(self) -> None:
        plt.subplot(9, 2, 6)
        plt.plot([x.vehicles for x in self.stats],label="vehicle count")
        plt.plot([x.platoon_size for x in self.stats],label="platoon size")
        plt.legend(loc="upper left")
        plt.title("membership")

    def plot_platoon_risk(self) -> None:
        plt.subplot(9, 2, 8)
        plt.plot([x.platoon_risk for x in self.stats],label="risk")
        plt.legend(loc="upper left")
        plt.title("platoon risk")

    def plot_average_platoon_risk(self) -> None:
        plt.subplot(9, 2, 9)
        plt.plot([x.average_platoon_risk for x in self.stats],label="avg risk")
        plt.legend(loc="upper left")
        plt.title("average platoon risk")

    def plot_epsilon_threshold(self) -> None:
        plt.subplot(9, 2, 10)
        plt.plot([x.epsilon_threshold for x in self.stats],label="epsilon threshold")
        plt.legend(loc="upper left")
        plt.title("epsilon threshold")

    def set_margins(self) -> None:
        # plt.tight_layout()
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(16)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            # wspace=0.4,
            hspace=0.4,
        )

    def plot(self):
        self.set_margins()
        self.plot_utilities()
        self.plot_utility_deltas()
        self.plot_severity()
        self.plot_vuln_compromises()
        self.plot_vulns()
        self.plot_membership()
        self.plot_platoon_risk()
        self.plot_average_platoon_risk()
        self.plot_epsilon_threshold()