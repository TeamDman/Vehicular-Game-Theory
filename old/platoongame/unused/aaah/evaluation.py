from dataclasses import dataclass
from agents import Agent, AttackerAgent, DefenderAgent, WolpertingerDefenderAgent
from game import Game, GameConfig
from metrics import EpisodeMetricsTracker
from vehicles import VehicleProvider
from memory import ReplayMemory, TransitionTensorBatch
from utils import get_device

def get_episode_metrics(
    defender_agent: DefenderAgent,
    attacker_agent: AttackerAgent,
    game_config: GameConfig,
    vehicle_provider: VehicleProvider,
    num_turns: int,
) -> EpisodeMetricsTracker:
        game = Game(
            config=game_config,
            vehicle_provider=vehicle_provider,
        )
        metrics = EpisodeMetricsTracker()
        metrics.track_stats(game)
        if isinstance(defender_agent, WolpertingerDefenderAgent):
            defender_agent.training = False # disable noise
        for _ in range(num_turns):
            game.take_step(
                attacker_agent=attacker_agent,
                defender_agent=defender_agent
            )
            metrics.track_stats(game)

        return metrics
