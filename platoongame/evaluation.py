from dataclasses import dataclass
from agents import Agent
from game import Game, GameConfig
from metrics import EpisodeMetricsTracker
from vehicles import VehicleProvider
from memory import ReplayMemory, TransitionTensorBatch
from utils import get_device

@dataclass
class ModelEvaluator:
    defender_agent: Agent
    attacker_agent: Agent
    game_config: GameConfig
    vehicle_provider: VehicleProvider
    memory: ReplayMemory
    
    def get_episode_metrics(self, num_turns: int) -> EpisodeMetricsTracker:
        game = Game(
            config=self.game_config,
            vehicle_provider=self.vehicle_provider,
        )
        metrics = EpisodeMetricsTracker()
        metrics.track_stats(game)
        for i in range(num_turns):
            game.take_step(
                attacker_agent=self.attacker_agent,
                defender_agent=self.defender_agent
            )
            metrics.track_stats(game)

        return metrics

    def sample_model_outputs(self) -> None:
        batch = TransitionTensorBatch.cat(self.memory.sample(10)).to_device(get_device())
        # batch.state.vehicles.std(dim=1)
        # batch.state.vulnerabilities.std(dim=1)
        # batch.reward

        proto_actions = self.defender_agent.actor(batch.state)
        print("action.members", proto_actions.members.sum(dim=0))
        q_values = self.defender_agent.critic(batch.state, proto_actions)
        print("q_pred", q_values)
        print("batch.reward", batch.reward)
        print("pred reward err", q_values - batch.reward)