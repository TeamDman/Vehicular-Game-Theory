from dataclasses import dataclass, field, _MISSING_TYPE
from agents import Agent
from game import Game, GameConfig
from utils import NoneRefersDefault
from vehicles import CompromiseState, VehicleProvider, Vulnerability
from typing import *

@dataclass
class Metrics:
    defender_util: float
    attacker_util: float
    compromises: int
    known_compromises: int
    compromised_overall: float
    compromised_partial: float
    platoon_severity: int
    vehicles: int
    platoon_size: int
# todo: add new metrics
# - Learning rate
# - Risk of the platoon

@dataclass
class Evaluator:
    attacker: Agent
    defender: Agent
    num_rounds: int
    game_config: GameConfig
    vehicle_provider: VehicleProvider
    stats: List[Metrics] = field(default_factory=list)

    def reset(self):
        self.stats = []
        self.game = Game(
            config=self.game_config,
            vehicle_provider=self.vehicle_provider
        )

    def run(self):
        self.reset()
        self.track_stats()
        for i in range(self.num_rounds):
            self.step()
    
    def step(self):
        self.game.step(
            attacker_agent=self.attacker,
            defender_agent=self.defender,
        )
        self.track_stats()

    def track_stats(self) -> None:
        self.stats.append(Metrics(
            defender_util=self.game.state.defender_utility,
            attacker_util=self.game.state.attacker_utility,
            compromises=len([1 for vehicle in self.game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
            known_compromises=len([vuln for vehicle in self.game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN]),
            compromised_overall=len([1 for vehicle in self.game.state.vehicles if all([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            compromised_partial=len([1 for vehicle in self.game.state.vehicles if any([True if vuln.state != CompromiseState.NOT_COMPROMISED else False for vuln in vehicle.vulnerabilities])]),
            platoon_severity = sum([vuln.severity for vehicle in self.game.state.vehicles for vuln in vehicle.vulnerabilities if vehicle.in_platoon and vuln.state != CompromiseState.NOT_COMPROMISED]),
            platoon_size=len([1 for v in self.game.state.vehicles if v.in_platoon]),
            vehicles=len(self.game.state.vehicles),
        ))

    def plot(self, stats: List[Metrics] = None):
        if stats is None: stats = self.stats
        import matplotlib.pyplot as plt
        import numpy as np
        # fig, axs = plt.subplots(4, figsize=(16,9), gridspec_kw={'height_ratios': [1, 2, 1, 1]})
        # fig, axs = plt.subplots(5, figsize=(10,5), gridspec_kw={'height_ratios': [1, 1, 1, 1,1]})
        fig, axs = plt.subplots(6, figsize=(8,10))
        i=0
        axs[i].plot([x.defender_util for x in self.stats], label="defender")
        axs[i].plot([x.attacker_util for x in self.stats], label="attacker")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("accumulated utilities")

        i+=1
        axs[i].plot(np.diff([x.defender_util for x in self.stats]), label="defender", alpha=0.9)
        axs[i].plot(np.diff([x.attacker_util for x in self.stats]), label="attacker", alpha=0.9)
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("utility deltas")

        i+=1
        axs[i].plot([x.platoon_severity for x in self.stats], label="severity", alpha=1)
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("platoon severity")

        i+=1
        axs[i].plot([x.compromises for x in self.stats], label="compromises", alpha=0.9)
        axs[i].plot([x.known_compromises for x in self.stats], label="known compromises", alpha=0.9)
        # axs[i].set_yticks(np.arange(0,1,0.1))
        axs[i].title.set_text("vulnerability compromises")
        axs[i].legend(loc="upper left")

        i+=1
        axs[i].plot([x.compromised_overall for x in self.stats], label="overall", alpha=0.9)
        axs[i].plot([x.compromised_partial for x in self.stats], label="partial", alpha=0.9)
        axs[i].plot([x.vehicles for x in self.stats],label="vehicle count", alpha=0.9)
        # axs[i].set_yticks(np.arange(0,1,0.1))
        axs[i].title.set_text("vehicle compromises")
        axs[i].legend(loc="upper left")

        i+=1
        axs[i].plot([x.vehicles for x in self.stats],label="vehicle count")
        axs[i].plot([x.platoon_size for x in self.stats],label="platoon size")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("membership")

        plt.tight_layout()
        plt.subplots_adjust(top=2)