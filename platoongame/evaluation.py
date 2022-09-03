from dataclasses import dataclass, fields, _MISSING_TYPE
from agents import Agent
from game import Game
from utils import NoneRefersDefault
from vehicles import VehicleProvider
from typing import *

class Stats:
    defender_utils: List[float]
    attacker_utils: List[float]
    compromise_partial_pcts: List[float]
    compromise_overall_pcts: List[float]
    vehicle_lens: List[int]
    platoon_lens: List[int]
    def __init__(self) -> None:
        self.defender_utils = []
        self.attacker_utils = []
        self.compromise_partial_pcts = []
        self.compromise_overall_pcts = []
        self.vehicle_lens = []
        self.platoon_lens = []

class Evaluator:
    attacker: Agent
    defender: Agent
    num_rounds: int
    num_vehicles: int
    vehicles: VehicleProvider
    stats: Stats
    def __init__(
        self,
        attacker: Agent,
        defender: Agent,
        num_rounds: int,
        num_vehicles: int,
        vehicles: VehicleProvider,
        stats: Stats = None,
    ) -> None:
        self.attacker = attacker
        self.defender = defender
        self.num_rounds = num_rounds
        self.num_vehicles = num_vehicles
        self.vehicles = vehicles
        self.stats = stats

    def run(self):
        self.stats = Stats()
        game = Game(
            max_vehicles=self.num_vehicles,
            vehicle_provider=self.vehicles,
            defender=self.defender,
            attacker=self.attacker
        )
        for i in range(self.num_rounds):
            self.stats.defender_utils.append(game.defender_agent.utility)
            self.stats.attacker_utils.append(game.attacker_agent.utility)
            self.stats.compromise_partial_pcts.append(len([v for v in game.vehicles if len(v.compromises) > 0])/(len(game.vehicles) if len(game.vehicles) > 0 else 1))
            bot = sum([len(v.achoice) for v in game.vehicles])
            if bot == 0: bot = 1
            self.stats.compromise_overall_pcts.append(sum([len(v.compromises) for v in game.vehicles]) / bot)
            self.stats.vehicle_lens.append(len(game.vehicles))
            self.stats.platoon_lens.append(len([v for v in game.vehicles if v.in_platoon]))
            game.step()

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np
        # fig, axs = plt.subplots(4, figsize=(16,9), gridspec_kw={'height_ratios': [1, 2, 1, 1]})
        fig, axs = plt.subplots(4, figsize=(8,5), gridspec_kw={'height_ratios': [1, 2, 1, 1]})
        i=0
        axs[i].plot(self.stats.defender_utils, label="defender")
        axs[i].plot(self.stats.attacker_utils, label="attacker")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("accumulated utilities")

        i+=1
        axs[i].plot(np.diff(self.stats.defender_utils), label="defender", alpha=0.5)
        axs[i].plot(np.diff(self.stats.attacker_utils), label="attacker", alpha=0.5)
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("utility deltas")

        i+=1
        axs[i].plot(self.stats.compromise_overall_pcts, label="overall", alpha=0.5)
        axs[i].plot(self.stats.compromise_partial_pcts, label="partly", alpha=1)
        axs[i].set_yticks(np.arange(0,1,0.1))
        axs[i].title.set_text("fraction compromised")
        axs[i].legend(loc="upper left")

        i+=1
        axs[i].plot(self.stats.vehicle_lens,label="vehicles count")
        axs[i].plot(self.stats.platoon_lens,label="platoon size")
        axs[i].legend(loc="upper left")
        axs[i].title.set_text("counts")

        plt.tight_layout()
        plt.subplots_adjust(top=2)