# https://github.com/0xangelo/gym-cartpole-swingup/blob/master/gym_cartpole_swingup/__init__.py
from gym.envs.registration import register

register(
    id="TorchPlatoon-v0",
    entry_point="platoonenv.platoonenv:PlatoonEnv",
    max_episode_steps=100,
)