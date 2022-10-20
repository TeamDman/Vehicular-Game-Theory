from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING, List
from game import Game
from vehicles import CompromiseState
import matplotlib.pyplot as plt
import numpy as np
import utils

if TYPE_CHECKING:
    from training import OptimizationResult

@dataclass
class EpisodeMetricsEntry:
    defender_util: float
    attacker_util: float
    compromises: int
    # known_compromises: int
    num_vulns: int
    num_vehicles_fully_compromised: float
    num_vehicles_partially_compromised: float
    platoon_severity: int
    potential_platoon_severity: int
    vehicles: int
    platoon_size: int
    platoon_risk: float
    max_platoon_member_risk: float
    # todo: vulnerability severity heatmap over time

    def save(self, dir: str, step: int):
        path = pathlib.Path(dir)
        path.mkdir(exist_ok=True, parents=True)
        filename = f"{utils.get_prefix()} {step:06}.log.json"
        path = path / filename
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)


@dataclass
class TrainingMetricsEntry:
    optim: OptimizationResult
    epsilon_threshold: float
        

@dataclass
class TrainingMetricsTracker:
    stats: List[TrainingMetricsEntry] = field(default_factory=list)
    def track_stats(
        self,
        optimization_results: OptimizationResult,
        epsilon_threshold: float,
    ) -> None:
        self.stats.append(TrainingMetricsEntry(
            optim=optimization_results,
            epsilon_threshold=epsilon_threshold
        ))

    def set_margins(self) -> None:
        # plt.tight_layout()
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(10)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            # wspace=0.4,
            hspace=0.4,
        )

    def plot_loss(self) -> None:
        plt.subplot(2, 1, 1)
        plt.plot([x.optim.loss for x in self.stats],label="loss")
        plt.legend(loc="upper left")
        plt.title("loss")

    def plot_epsilon_threshold(self) -> None:
        plt.subplot(2, 1, 2)
        plt.plot([x.epsilon_threshold for x in self.stats],label="epsilon threshold")
        plt.legend(loc="upper left")
        plt.title("epsilon threshold")

        
    def plot(self):
        self.set_margins()
        self.plot_loss()
        self.plot_epsilon_threshold()

@dataclass
class EpisodeMetricsTracker:
    stats: List[EpisodeMetricsEntry] = field(default_factory=list)
    def track_stats(self, game: Game) -> None:
        platoon_members = [v for v in game.state.vehicles if v.in_platoon]
        entry = EpisodeMetricsEntry(
            defender_util=game.state.defender_utility,
            attacker_util=game.state.attacker_utility,
            compromises=len([1 for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
            # known_compromises=len([vuln for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN]),
            num_vulns=sum([len(vehicle.vulnerabilities) for vehicle in game.state.vehicles]),
            # todo: ensure partial and fully compromised vehicles don't include vehicles with no vulnerabilities, diagnose why "fully" starts gt 0
            num_vehicles_fully_compromised=len([1 for vehicle in game.state.vehicles if all([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            num_vehicles_partially_compromised=len([1 for vehicle in game.state.vehicles if any([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            potential_platoon_severity = sum([vuln.severity for vehicle in platoon_members for vuln in vehicle.vulnerabilities]),
            platoon_severity = sum([vuln.severity for vehicle in platoon_members for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
            platoon_size=len(platoon_members),
            vehicles=len(game.state.vehicles),
            platoon_risk=sum([v.risk for v in platoon_members]),
            max_platoon_member_risk=0 if len(platoon_members) == 0 else max([v.risk for v in platoon_members]),
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
        plt.plot([x.num_vulns for x in self.stats], label="num vulns", alpha=0.9)
        # plt.plot([x.known_compromises for x in self.stats], label="known compromises", alpha=0.9)
        # plt.set_yticks(np.arange(0,1,0.1))
        plt.title("vulnerability compromises")
        plt.legend(loc="upper left")

    def plot_vulns(self) -> None:
        plt.subplot(9, 2, 5)
        plt.plot([x.num_vehicles_partially_compromised for x in self.stats], label="partially", alpha=0.9)
        plt.plot([x.num_vehicles_fully_compromised for x in self.stats], label="fully", alpha=0.9)
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
        plt.subplot(9, 2, 7)
        plt.plot([x.platoon_risk for x in self.stats],label="risk")
        plt.legend(loc="upper left")
        plt.title("platoon risk")

    def plot_max_platoon_member_risk(self) -> None:
        plt.subplot(9, 2, 8)
        plt.plot([x.max_platoon_member_risk for x in self.stats],label="max risk")
        plt.legend(loc="upper left")
        plt.title("maximum platoon member risk")

    def set_margins(self) -> None:
        # plt.tight_layout()
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(16)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            # right=0.9,
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
        self.plot_max_platoon_member_risk()