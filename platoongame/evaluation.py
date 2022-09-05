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
    compromises: float
    known_compromises: float
    vehicles: int
    platoon_size: int

@dataclass
class Evaluator:
    attacker: Agent
    defender: Agent
    num_rounds: int
    game_config: GameConfig
    vehicles: VehicleProvider
    stats: List[Metrics] = field(default_factory=list)

    def run(self):
        self.stats = []
        game = Game(
            config=self.game_config,
            vehicle_provider=self.vehicles
        )
        for i in range(self.num_rounds):
            self.stats.append(Metrics(
                defender_util=game.state.defender_utility,
                attacker_util=game.state.attacker_utility,
                compromises=len([1 for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED]),
                known_compromises=len([vuln for vehicle in game.state.vehicles for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED_KNOWN]),
                platoon_size=len([1 for v in game.state.vehicles if v.in_platoon]),
                vehicles=len(game.state.vehicles),
            ))
            game.step(
                attacker_agent=self.attacker,
                defender_agent=self.defender,
            )
            game.step(
                attacker_agent=self.attacker,
                defender_agent=self.defender,
            )

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np
        # fig, axs = plt.subplots(4, figsize=(16,9), gridspec_kw={'height_ratios': [1, 2, 1, 1]})
        fig, axs = plt.subplots(4, figsize=(8,5), gridspec_kw={'height_ratios': [1, 2, 1, 1]})
        i=0
        axs[i].plot([x.defender_util for x in self.stats], label="defender")
        axs[i].plot([x.attacker_util for x in self.stats], label="attacker")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("accumulated utilities")

        i+=1
        axs[i].plot(np.diff([x.defender_util for x in self.stats]), label="defender", alpha=1)
        axs[i].plot(np.diff([x.attacker_util for x in self.stats]), label="attacker", alpha=1)
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("utility deltas")

        i+=1
        axs[i].plot([x.compromises for x in self.stats], label="compromises", alpha=1)
        axs[i].plot([x.known_compromises for x in self.stats], label="known compromises", alpha=1)
        # axs[i].set_yticks(np.arange(0,1,0.1))
        axs[i].title.set_text("compromises")
        axs[i].legend(loc="upper left")

        i+=1
        axs[i].plot([x.vehicles for x in self.stats],label="vehicles count")
        axs[i].plot([x.platoon_size for x in self.stats],label="platoon size")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("counts")

        plt.tight_layout()
        plt.subplots_adjust(top=2)