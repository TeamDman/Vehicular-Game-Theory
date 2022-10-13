# from models import DefenderActionTensorBatch, StateTensorBatch
# from utils import get_device

# a = defender.get_action(engine.game.state)
# state = engine.game.state.as_tensors(defender.state_shape_data)
# state = StateTensorBatch(
#     vulnerabilities=state.vulnerabilities.to(get_device()),
#     vehicles=state.vehicles.to(get_device()),
# )
# print("state", state.vehicles.shape, state.vulnerabilities.shape)
# action = a.as_tensor(defender.state_shape_data)
# action = DefenderActionTensorBatch(
#     members=action.members.to(get_device()),
#     monitor=action.monitor.to(get_device()),
# )

# print("action", action.members.shape, action.monitor.shape)
# print(action)
# q_values = defender.critic(state,action) 
# print("q", q_values)
# print("actual", defender.get_utility(defender.take_action(engine.game.state, a)))
from dataclasses import dataclass
from agents import Agent
from game import Game, GameConfig
from metrics import EpisodeMetricsTracker
from vehicles import VehicleProvider
from memory import ReplayMemory, TransitionTensorBatch

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
        batch = TransitionTensorBatch.cat(self.memory.sample(10))
        batch.state.vehicles.std(dim=1)
        batch.state.vulnerabilities.std(dim=1)
        batch.reward

        proto_actions = self.defender_agent.actor(batch.state)
        print("action.members", proto_actions.members.sum(dim=0))
        q_values = self.defender_agent.critic(batch.state, proto_actions)
        print("q_pred", q_values)
        print("batch.reward", batch.reward)
        print("pred reward err", q_values - batch.reward)