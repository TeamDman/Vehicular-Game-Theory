import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
import torch
from torch import Tensor
from typing import Literal, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import random
from ipycanvas import Canvas, hold_canvas

class InOutEnv(gym.Env):
    metadata = {
        "render_modes": ["canvas"],
        "reward_threshold": -45,
    }
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=0,high=1,shape=(10,))
        self.spec = EnvSpec(
            id="Platoon-v0",
            entry_point="platoonenv:InOutEnv",
            max_episode_steps=20,
            reward_threshold=-45,
        )
        self.render_infos = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.render_infos.clear()
        self.state = np.zeros((10,), dtype=np.float32)
        self.steps_done = 0
        return self.state.copy(), {}

    def render(self):
        vehicle_width=60
        vehicle_height=20
        padding=6
        num_vehicles = self.state.shape[0]
        width = num_vehicles * (vehicle_width + padding) + 400
        height = len(self.render_infos) * (vehicle_height + padding)
        canvas = Canvas(width=width, height=height)
        reward_total = 0
        with hold_canvas(canvas):
            y = padding
            for info in self.render_infos:
                x = padding
                for i in range(num_vehicles):
                    # draw highlight for last action
                    if i + 1 == info["action"]:
                        canvas.fill_style="red"
                        canvas.fill_rect(
                            x=x-padding/2,
                            y=y-padding/2,
                            width=vehicle_width + padding,
                            height=vehicle_height + padding,
                        )

                    # draw cube
                    if info["state"][i] == 1:
                        canvas.fill_style = "#55F"
                    else:
                        canvas.fill_style = "gray"
                    canvas.fill_rect(
                        x=x,
                        y=y,
                        width=vehicle_width,
                        height=vehicle_height,
                    )

                    x += vehicle_width + padding
                # draw reward string     
                canvas.font = "20px Consolas"
                canvas.fill_style = "black"
                reward_total += info['reward']
                canvas.fill_text(
                    text=f"action:{info['action']:02d}, reward:{info['reward']:+.03f} ({reward_total:+.03f})",
                    x = x,
                    y = y + vehicle_height - padding//2,
                )
                y += vehicle_height + padding

            
        return canvas

    def step(self, action: int):
        if action != 0:
            self.state[action-1] = 1-self.state[action-1]

        next_obs = self.state.copy()
        reward = float(-np.sum(1-self.state))
        reward /= 100.0 # scale rewards between -1 and 1 to improve PPO training
        done = False
        trunc = self.steps_done > 15
        info = {}

        # track rendering info
        self.render_infos.append({
            "action": action,
            "reward": reward,
            "state": self.state.copy(),
        })
        
        self.steps_done += 1
        return next_obs, reward, done, trunc, info

class InOutDangerEnv(gym.Env):
    metadata = {
        "render_modes": ["canvas"],
        "reward_threshold": -48,
    }
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-10,high=1,shape=(20,))
        self.spec = EnvSpec(
            id="Platoon-v1",
            entry_point="platoonenv:InOutDangerEnv",
            max_episode_steps=20,
            reward_threshold=-48
        )
        self.render_infos = []
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.render_infos.clear()
        self.state = np.zeros((10,), dtype=np.float32)
        self.values = np.zeros((10,), dtype=np.float32)
        a = self.np_random.integers(0,9)
        b = a
        while b == a:
            b = self.np_random.integers(0,9)
        self.values[a] = -10
        self.values[b] = -10

        self.steps_done = 0
        
        return np.hstack((self.state, self.values)), {}

    def render(self):
        vehicle_width=60
        vehicle_height=20
        padding=6
        num_vehicles = self.state.shape[0]
        width = num_vehicles * (vehicle_width + padding) + 400
        height = len(self.render_infos) * (vehicle_height + padding)
        canvas = Canvas(width=width, height=height)
        reward_total = 0
        with hold_canvas(canvas):
            y = padding
            for info in self.render_infos:
                x = padding
                for i in range(num_vehicles):
                    # draw highlight for last action
                    if i + 1 == info["action"]:
                        canvas.fill_style="red"
                        canvas.fill_rect(
                            x=x-padding/2,
                            y=y-padding/2,
                            width=vehicle_width + padding,
                            height=vehicle_height + padding,
                        )

                    # draw cube
                    if info["state"][i] == 1:
                        canvas.fill_style = "#55F"
                    else:
                        canvas.fill_style = "gray"
                    canvas.fill_rect(
                        x=x,
                        y=y,
                        width=vehicle_width,
                        height=vehicle_height,
                    )

                    # draw cube value
                    canvas.font = "20px Consolas"
                    canvas.fill_style = "black"
                    canvas.fill_text(
                        text=f"{info['values'][i]}",
                        x = x,
                        y = y + vehicle_height - padding//2,
                    )

                    x += vehicle_width + padding
                # draw reward string     
                canvas.font = "20px Consolas"
                canvas.fill_style = "black"
                reward_total += info['reward']
                canvas.fill_text(
                    text=f"action:{info['action']:02d}, reward:{info['reward']:+.03f} ({reward_total:+.03f})",
                    x = x,
                    y = y + vehicle_height - padding//2,
                )
                y += vehicle_height + padding

            
        return canvas

    def step(self, action: int):
        if action != 0:
            self.state[action-1] = 1-self.state[action-1]

        next_obs = np.hstack((self.state, self.values))
        reward = (-np.sum(1-self.state)) + np.sum(self.state * self.values)
        reward /= 100.0 # scale rewards between -1 and 1 to improve PPO training
        done = False
        trunc = self.steps_done >= 10
        info = {}

        # track rendering info
        self.render_infos.append({
            "action": action,
            "reward": reward,
            "state": self.state.copy(),
            "values": self.values.copy(),
        })

        self.steps_done += 1
        return next_obs, reward, done, trunc, info

class InOutValueEnv(gym.Env):
    metadata = { 
        "render_modes": ["canvas"],
    }
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-10,high=10,shape=(20,))
        self.spec = EnvSpec(
            id="Platoon-v2",
            entry_point="platoonenv:InOutValueEnv",
            max_episode_steps=20,
            reward_threshold=0, #-45
        )
        self.render_infos = []
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((10,), dtype=np.float32)
        self.values = self.np_random.integers(-10,11, size=(10,)).astype(np.float32)
        self.steps_done = 0
        self.render_infos.clear()
        return np.hstack((self.state, self.values)), {}

    def render(self):
        vehicle_width=60
        vehicle_height=20
        padding=6
        num_vehicles = self.state.shape[0]
        width = num_vehicles * (vehicle_width + padding) + 400
        height = len(self.render_infos) * (vehicle_height + padding)
        canvas = Canvas(width=width, height=height)
        reward_total = 0
        with hold_canvas(canvas):
            y = padding
            for info in self.render_infos:
                x = padding
                for i in range(num_vehicles):
                    # draw highlight for last action
                    if i + 1 == info["action"]:
                        canvas.fill_style="red"
                        canvas.fill_rect(
                            x=x-padding/2,
                            y=y-padding/2,
                            width=vehicle_width + padding,
                            height=vehicle_height + padding,
                        )

                    # draw cube
                    if info["state"][i] == 1:
                        canvas.fill_style = "#55F"
                    else:
                        canvas.fill_style = "gray"
                    canvas.fill_rect(
                        x=x,
                        y=y,
                        width=vehicle_width,
                        height=vehicle_height,
                    )

                    # draw cube value
                    canvas.font = "20px Consolas"
                    canvas.fill_style = "black"
                    canvas.fill_text(
                        text=f"{info['values'][i]}",
                        x = x,
                        y = y + vehicle_height - padding//2,
                    )

                    x += vehicle_width + padding
                # draw reward string     
                canvas.font = "20px Consolas"
                canvas.fill_style = "black"
                reward_total += info['reward']
                canvas.fill_text(
                    text=f"action:{info['action']:02d}, reward:{info['reward']:+.03f} ({reward_total:+.03f})",
                    x = x,
                    y = y + vehicle_height - padding//2,
                )
                y += vehicle_height + padding

            
        return canvas

    def step(self, action: int):
        if action != 0:
            self.state[action-1] = 1-self.state[action-1]

        next_obs = np.hstack((self.state, self.values))

        reward = 0
        # gain points using value of each member
        reward += np.sum(self.state * self.values)
        # lose 1 point for each non-member
        reward -= np.sum(1-self.state)
        # lose 10 points for each member value less than 0
        # reward -= np.sum(self.state * (self.values < 0) * 10)
        # scale reward
        reward /= 100.0 # scale rewards between -1 and 1 to improve PPO training

        done = False
        trunc = self.steps_done >= 10
        info = {}

        # track rendering info
        self.render_infos.append({
            "action": action,
            "reward": reward,
            "state": self.state.copy(),
            "values": self.values.copy(),
        })
        self.steps_done += 1

        return next_obs, reward, done, trunc, info

class InOutValueProbEnv(gym.Env):
    metadata = {
        "render_modes": ["canvas"],
        "render_fps": 10,
    }
    def __init__(
        self,
        render_mode: Optional[str] = "canvas",
        env_config={
            "num_vehicles": 10,
            "steps_before_truncation": 50,
        },
    ):
        super().__init__()
        self.spec = EnvSpec(
            id="Platoon-v3",
            entry_point="platoonenv:InOutValueProbEnv",
            max_episode_steps=env_config["steps_before_truncation"],
            # reward_threshold=0, #-45
        )
        self.num_vehicles = env_config["num_vehicles"]
        print(f"Creating env with {self.num_vehicles} vehicles")
        self.steps_before_truncation=env_config["steps_before_truncation"]
        self.action_space = spaces.Discrete(self.num_vehicles + 1) # include no-op
        self.observation_space = spaces.Box(low=-10,high=10,shape=(self.num_vehicles * 3,)) # value, member, prod
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.num_vehicles,), dtype=np.float32)
        self.values = np.around(np.clip(self.np_random.normal(-4,1, size=(self.num_vehicles,)).astype(np.float32),-10,0))
        self.probs = self.np_random.uniform(0,1, size=(self.num_vehicles,)).astype(np.float32)
        self.steps_done = 0
        self.render_infos = []
        return np.hstack((self.state, self.values, self.probs)), {}

    def step(self, action: int):
        if action != 0:
            self.state[action-1] = 1-self.state[action-1]

        attacker_action = self.np_random.integers(0, self.num_vehicles)
        prob = self.probs[attacker_action]
        roll = self.np_random.random()
        self.probs[attacker_action] = int(prob > roll)

        next_obs = np.hstack((self.state, self.values, self.probs))

        reward = 0
        # lose 1 point for each non-member
        reward -= np.sum(1-self.state)
        # lose points for each compromised member scaled by severity
        compromised = np.all((self.probs == 1, self.state == 1), axis=0)
        reward += np.sum(self.values * compromised)
        # scale reward
        reward /= 100*self.num_vehicles # scale rewards between -1 and 1 to improve PPO training

        done = False
        trunc = self.steps_done >= self.steps_before_truncation
        info = {}

        # track rendering info
        self.render_infos.append({
            "action": action,
            "attacker_action": attacker_action,
            "reward": reward,
            "state": self.state.copy(),
            "values": self.values.copy(),
            "probs": self.probs.copy(),
        })
        self.steps_done += 1

        return next_obs, reward, done, trunc, info

    def render(self):
        vehicle_width=100
        vehicle_height=20
        padding=6
        num_vehicles = self.state.shape[0]
        width = num_vehicles * (vehicle_width + padding) + 400
        height = len(self.render_infos) * (vehicle_height + padding)
        canvas = Canvas(width=width, height=height)
        reward_total = 0
        with hold_canvas(canvas):
            y = padding
            for info in self.render_infos:
                x = padding
                for i in range(num_vehicles):
                    # draw highlight for last action
                    # offset by 1 to account for no-op
                    if i + 1 == info["action"]:
                        canvas.fill_style="red"
                        canvas.fill_rect(
                            x=x-padding/2,
                            y=y-padding/2,
                            width=vehicle_width + padding,
                            height=vehicle_height + padding,
                        )
                    # draw highlight for last attacker action
                    if i == info["attacker_action"]:
                        canvas.fill_style="purple"
                        canvas.fill_rect(
                            x=x-padding/2,
                            y=y-padding/2,
                            width=vehicle_width//2 if i+1 == info["action"] else vehicle_width + padding,
                            height=vehicle_height + padding,
                        )

                    # draw cube
                    if info["state"][i] == 1:
                        # if info["probs"][i] == 1:
                        #     canvas.fill_style = "#969"
                        # else:
                        canvas.fill_style = "#55F"
                    else:
                        canvas.fill_style = "gray"
                    canvas.fill_rect(
                        x=x,
                        y=y,
                        width=vehicle_width,
                        height=vehicle_height,
                    )

                    # draw cube value
                    canvas.font = "14px Consolas"
                    if info["probs"][i] == 1:
                        canvas.fill_style = "white"
                    else:
                        canvas.fill_style = "black"

                    canvas.fill_text(
                        text=f"{info['values'][i]} | {info['probs'][i]:.2f}",
                        x = x+padding,
                        y = y + vehicle_height - padding//2,
                    )
                    x += vehicle_width + padding
                # draw reward string     
                canvas.font = "20px Consolas"
                canvas.fill_style = "black"
                reward_total += info['reward']
                canvas.fill_text(
                    text=f"action:{info['action']:02d}, reward:{info['reward']:+.03f} ({reward_total:+.03f})",
                    x = x,
                    y = y + vehicle_height - padding//2,
                )
                y += vehicle_height + padding

            
        return canvas

class InOutValueProbCyclingEnv(gym.Env):
    metadata = {
        "render_modes": ["canvas"],
        "render_fps": 10,
    }
    def __init__(
        self,
        render_mode: Optional[str] = "canvas",
        env_config={
            "num_vehicles": 10,
            "steps_before_truncation": 50,
            "cycle_interval": 1,
            "cycle_num": 1,
        },
    ):
        super().__init__()
        self.spec = EnvSpec(
            id="Platoon-v4",
            entry_point="platoonenv:InOutValueProbCyclingEnv",
            max_episode_steps=env_config["steps_before_truncation"],
            # reward_threshold=0, #-45
        )
        self.num_vehicles = env_config["num_vehicles"]
        self.cycle_interval = env_config["cycle_interval"]
        self.cycle_num = env_config["cycle_num"]
        self.steps_before_truncation=env_config["steps_before_truncation"]
        self.action_space = spaces.Discrete(self.num_vehicles + 1) # include no-op
        self.observation_space = spaces.Box(low=-10,high=10,shape=(self.num_vehicles * 3,)) # value, member, prod
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((self.num_vehicles,), dtype=np.float32)
        self.values = np.around(np.clip(self.np_random.normal(-4,1, size=(self.num_vehicles,)).astype(np.float32),-10,0))
        self.probs = self.np_random.uniform(0,1, size=(self.num_vehicles,)).astype(np.float32)
        self.steps_done = 0
        self.render_infos = []
        return np.hstack((self.state, self.values, self.probs)), {}

    def step(self, action: int):
        if action != 0:
            self.state[action-1] = 1-self.state[action-1]

        attacker_action = self.np_random.integers(0, self.num_vehicles)
        prob = self.probs[attacker_action]
        roll = self.np_random.random()
        self.probs[attacker_action] = int(prob > roll)

        next_obs = np.hstack((self.state, self.values, self.probs))

        reward = 0
        # lose 1 point for each non-member
        reward -= np.sum(1-self.state)
        # lose points for each compromised member scaled by severity
        compromised = np.all((self.probs == 1, self.state == 1), axis=0)
        reward += np.sum(self.values * compromised)
        # scale reward
        reward /= 100*self.num_vehicles # scale rewards between -1 and 1 to improve PPO training

        done = False
        trunc = self.steps_done >= self.steps_before_truncation
        info = {}



        if self.steps_done % self.cycle_interval == 0:
            # kick them out of the platoon, give them a new value, and reset their prob to simulate a new vehicle
            indices = self.np_random.choice(self.num_vehicles, self.cycle_num, replace=False)
            self.state[indices] = 0
            self.values[indices] = np.around(np.clip(self.np_random.normal(-4,1, size=(self.cycle_num,)).astype(np.float32),-10,0))
            self.probs[indices] = self.np_random.uniform(0,1, size=(self.cycle_num,)).astype(np.float32)
        else:
            indices = []

        # track rendering info
        self.render_infos.append({
            "action": action,
            "attacker_action": attacker_action,
            "reward": reward,
            "state": self.state.copy(),
            "values": self.values.copy(),
            "probs": self.probs.copy(),
            "cycled": indices
        })
        self.steps_done += 1

        return next_obs, reward, done, trunc, info

    def render(self):
        vehicle_width=100
        vehicle_height=20
        padding=6
        num_vehicles = self.state.shape[0]
        width = num_vehicles * (vehicle_width + padding) + 400
        height = len(self.render_infos) * (vehicle_height + padding)
        canvas = Canvas(width=width, height=height)
        reward_total = 0
        with hold_canvas(canvas):
            y = padding
            for info in self.render_infos:
                x = padding
                for i in range(num_vehicles):
                    # draw cube
                    if info["state"][i] == 1:
                        # if info["probs"][i] == 1:
                        #     canvas.fill_style = "#969"
                        # else:
                        canvas.fill_style = "#55F"
                    else:
                        canvas.fill_style = "gray"
                    canvas.fill_rect(
                        x=x,
                        y=y,
                        width=vehicle_width,
                        height=vehicle_height,
                    )

                    # draw highlight for last action
                    # offset by 1 to account for no-op
                    if i + 1 == info["action"]:
                        canvas.fill_style="red"
                        canvas.fill_rect(
                            x=x,
                            y=y,
                            width=10,
                            height=10,
                        )
                    # draw highlight for last attacker action
                    if i == info["attacker_action"]:
                        canvas.fill_style="purple"
                        canvas.fill_rect(
                            x=x+15,
                            y=y,
                            width=10,
                            height=10,
                        )

                    # draw highlight for cycled vehicles
                    if i in info["cycled"]:
                        canvas.fill_style="yellow"
                        canvas.fill_rect(
                            x=x+30,
                            y=y,
                            width=10,
                            height=10,
                        )

                    # draw cube value
                    canvas.font = "14px Consolas"
                    if info["probs"][i] == 1:
                        canvas.fill_style = "white"
                    else:
                        canvas.fill_style = "black"

                    canvas.fill_text(
                        text=f"{info['values'][i]} | {info['probs'][i]:.2f}",
                        x = x+padding,
                        y = y + vehicle_height - padding//2,
                    )
                    x += vehicle_width + padding
                # draw reward string     
                canvas.font = "20px Consolas"
                canvas.fill_style = "black"
                reward_total += info['reward']
                canvas.fill_text(
                    text=f"action:{info['action']:02d}, reward:{info['reward']:+.03f} ({reward_total:+.03f})",
                    x = x,
                    y = y + vehicle_height - padding//2,
                )
                y += vehicle_height + padding

            
        return canvas



# https://github.com/0xangelo/gym-cartpole-swingup/blob/master/gym_cartpole_swingup/__init__.py
from gymnasium.envs.registration import register

register(
    id="Platoon-v0",
    entry_point="platoonenv:InOutEnv",
    max_episode_steps=20,
    reward_threshold=-45,
)

register(
    id="Platoon-v1",
    entry_point="platoonenv:InOutDangerEnv",
    max_episode_steps=20,
    reward_threshold=-48
)

register(
    id="Platoon-v2",
    entry_point="platoonenv:InOutValueEnv",
    max_episode_steps=20,
    reward_threshold=500,
)

register(
    id="Platoon-v3",
    entry_point="platoonenv:InOutValueProbEnv",
    max_episode_steps=50,
    reward_threshold=500,
)

register(
    id="Platoon-v4",
    entry_point="platoonenv:InOutValueProbCyclingEnv",
    max_episode_steps=50,
    reward_threshold=500,
)

try:
    from ray.tune.registry import register_env
    register_env("Platoon-v0", lambda config: InOutEnv())
    register_env("Platoon-v1", lambda config: InOutDangerEnv())
    register_env("Platoon-v2", lambda config: InOutValueEnv())
    register_env("Platoon-v3", lambda config: InOutValueProbEnv(env_config=config))
    register_env("Platoon-v4", lambda config: InOutValueProbCyclingEnv(env_config=config))
except ImportError:
    pass
# todo: if a vuln attack fails, it should be considered failed forever (set probability to 0)Z