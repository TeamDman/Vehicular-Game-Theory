import gym
import gym.spaces
import gym.logger
import gym.error
import torch
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

# https://github.com/0xangelo/gym-cartpole-swingup/blob/master/gym_cartpole_swingup/__init__.py
from gym.envs.registration import register

register(
    id="TorchPlatoon-v0",
    entry_point="platoonenv:PlatoonEnv",
    # max_episode_steps=20,
)

@dataclass
class PlatoonEnvParams:
    num_vehicles: int = 10
    num_vulns: int = 3
    num_attack: int = 1
    attack_interval: int = 2

class MyBox(gym.spaces.Box):
    def __init__(self, low, high):
        super().__init__(low, high)

    def sample(self):
        x = super().sample()
        x[:,:,0] = x[:,:,0] > 0.5
        x[:,:,3] = x[:,:,3] > 0.5
        x[:,:,2] = x[:,:,2].round()
        return x
class PlatoonEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, render_mode: Optional[str] = None, params: Optional[PlatoonEnvParams] = None):
        gym.logger.warn("pls ignore gym complaints about observation space")
        self.params = params or PlatoonEnvParams()
        self.reward_range = (0, self.params.num_vehicles)
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.font = None
        self.clock = None
        self.isopen = True
        self.latest_reward = None
        self.render_mode = render_mode
        self.episode_reward_total = 0
        self.action_space = gym.spaces.Discrete(self.params.num_vehicles + 1) # include no-op
        self.observation_space = MyBox(
            # member, prob, sev, compromised
            low=torch.tensor((0,0.0,0,0)).repeat(self.params.num_vulns,1).repeat(self.params.num_vehicles,1,1).numpy(),
            high=torch.tensor((1,1.0,5,1)).repeat(self.params.num_vulns,1).repeat(self.params.num_vehicles,1,1).numpy(),
        )
        self.prob_dist = torch.distributions.Normal(
            # loc=torch.as_tensor(0.5),
            loc=torch.as_tensor(0.1),
            scale=torch.as_tensor(0.25),
        )
        self.sev_dist = torch.distributions.Normal(
            loc=torch.as_tensor(2.0),
            scale=torch.as_tensor(1.0),
        )


        self.reset()

    def step(self, action: int) -> Tuple[Tensor, float, bool, bool, dict]:
        if action != self.params.num_vehicles: # it not no-op
            # toggle membership for the chosen vehicle
            self.state[action,:,0] = 1-self.state[action,:,0]

        if self.current_step % self.params.attack_interval == 0:
            # prio calculated as sum of the prio of the vulns within the vehicle
            # prio = prob * severity * not_compromised + 100*is_member
            priority = (self.state[:,:,1] * self.state[:,:,2] * (1-self.state[:,:,3]) + (100*self.state[:,:,0])).sum(dim=-1)
            # identify vehicles to attack
            mask = priority.topk(self.params.num_attack).indices
            attack = self.state[mask]
            # attack proc
            roll = torch.rand(attack.shape[:-1])
            success_mask = roll > 1-attack[:,:,1]
            # update vulnerabilities successfully attacked
            success = attack[success_mask]
            success[:,3] = 1
            # copy data backwards
            attack[success_mask] = success
            self.state[mask] = attack

        # reward starts as the size of the platoon
        reward = self.state[:,:,0].sum(dim=1).count_nonzero()
        # subtract the severity of any compromised vulns within the platoon
        # is_compromised * severity * is_member
        reward -= (self.state[:,:,3] * self.state[:,:,2] * self.state[:,:,0]).sum().int()
        reward = reward.item()

        self.current_step += 1
        # done = self.current_step >= self.num_steps
        done = False
        self.latest_reward = reward
        self.episode_reward_total += reward

        return self.state.numpy(), reward, done, False, {}

    def reset(self, *, seed:Optional[int] = None, options: Optional[dict] = {}):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.current_step = 0
        self.state = torch.zeros((self.params.num_vehicles, self.params.num_vulns, 4))
        self.state[:,:,1] = self.prob_dist.sample(torch.Size((self.params.num_vehicles, self.params.num_vulns))).clamp(0,1)
        self.state[:,:,2] = self.sev_dist.sample(torch.Size((self.params.num_vehicles, self.params.num_vulns))).round().clamp(1,5)
        novulns = torch.rand(torch.Size((self.params.num_vehicles, self.params.num_vulns))) > 0.5
        self.state[novulns, :] = 0
        self.latest_reward = None
        self.episode_reward_total = 0
        return self.state.numpy(), {}
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                self.font = pygame.font.SysFont("Comic Sans MS", 30)
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        x = self.state

    
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        vehicle_size = 38
        padding = 4
        vehicles_per_row = int(self.screen_width // (vehicle_size+padding*2))
        vehicles_per_column = int(self.screen_height // (vehicle_size+padding*2))
        for i in range(self.params.num_vehicles):
            color = (0,100,255)
            solid = False
            if x[i,0,0] == 1: # in platoon
                solid = True
            if x[i,:,3].count_nonzero() > 0: # compromised
                color = (255,0,0)
            if solid:
                gfxdraw.filled_circle(
                    self.surf,
                    int((i % vehicles_per_row) * (vehicle_size+padding*2) + padding + vehicle_size/2),
                    int(i // vehicles_per_row * (vehicle_size+padding*2) + padding + vehicle_size/2),
                    vehicle_size//2,
                    color,
                )
            else:
                gfxdraw.aacircle(
                    self.surf,
                    int((i % vehicles_per_row) * (vehicle_size+padding*2) + padding + vehicle_size/2),
                    int(i // vehicles_per_row * (vehicle_size+padding*2) + padding + vehicle_size/2),
                    vehicle_size//2,
                    color,
                )
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.font is not None:
            self.screen.blit(
                source=self.font.render(
                    f"reward: {self.latest_reward or 0:+0.1f} ({self.episode_reward_total})",
                    True,
                    (0,0,0)
                ),
                dest=(0,0)
            )

        if self.render_mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            self.clock.tick(30)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen=False
