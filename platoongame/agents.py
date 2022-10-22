from __future__ import annotations
import logging
import random
from typing import Callable, Dict, Optional, Union, FrozenSet, TYPE_CHECKING
from models import StateShapeData, DefenderActionTensorBatch, AttackerActionTensorBatch, StateTensorBatch
from utils import get_logger, get_prefix
from vehicles import CompromiseState, Vehicle
from collections import deque
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
import pathlib

if TYPE_CHECKING:
    from game import Game, State


@dataclass(frozen=True)
class DefenderAction:
    members: FrozenSet[int] # binary vector, indices of corresponding vehicle

    def as_tensor_batch(self, state_shape: StateShapeData) -> DefenderActionTensorBatch:
        members = torch.zeros(state_shape.num_vehicles, dtype=torch.float32)
        members[list(self.members)] = 1

        return DefenderActionTensorBatch(
            members=members.unsqueeze(dim=0),
        )



@dataclass(frozen=True)
class AttackerAction:
    attack: FrozenSet[int] # binary vector len=|vehicles|

    def as_tensor(self, state_shape: StateShapeData):
        attack = torch.zeros(state_shape.num_vehicles, dtype=torch.float32)
        attack[list(self.attack)] = 1
        return AttackerActionTensorBatch(
            attack=attack.unsqueeze(dim=0).unsqueeze(dim=1),
        )


Action = Union[DefenderAction, AttackerAction]

###############################
#region Base stuff
###############################
class Agent(ABC):
    logger: logging.Logger

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def get_action(self, state: State) -> Action:
        pass

    @abstractmethod
    def get_random_action(self, state: State) -> Action:
        pass

    @abstractmethod
    def take_action(self, state: State, action: Action) -> State:
        pass

    @abstractmethod
    def get_utility(self, state: State) -> int:
        pass

class DefenderAgent(Agent):
    def get_utility(self, state: State) -> float:
        members = [vehicle for vehicle in state.vehicles if vehicle.in_platoon]
        compromises = sum([vuln.severity for vehicle in members for vuln in vehicle.vulnerabilities if vuln.state != CompromiseState.NOT_COMPROMISED])
        # return len(members) * 10 - compromises ** 2
        # return int(len(members) * 10 - compromises ** 1.5)
        return len(members) * 2.5 - compromises
        # return int(len(members) * 1 - compromises)

    @abstractmethod
    def get_action(self, state: State) -> DefenderAction:
        pass

    def take_action(self, state: State, action: DefenderAction) -> State:
        # create mutable copy
        vehicles = list(state.vehicles)

        for i, v in enumerate(vehicles):
            vehicles[i] = replace(v, in_platoon = i in action.members)
            # self.logger.info(f"kicking vehicle {i} out of platoon")
            # self.logger.info(f"adding vehicle {i} to platoon")
        
        self.logger.debug(f"platoon={action.members}")

        # update vehicles from mutated copy
        state = replace(state, vehicles=tuple(vehicles))

        # calculate util change
        change = self.get_utility(state)
        utility = state.defender_utility
        utility += change
        self.logger.debug(f"utility {utility} ({change:+.2f})")
        state = replace(state, defender_utility=utility)

        # return new state
        return state

    def get_random_action(self, state: State) -> DefenderAction:
        num_members = random.randint(0,len(state.vehicles))
        members = random.sample(range(len(state.vehicles)), num_members)
        return DefenderAction(members=frozenset(members))

class RandomDefenderAgent(DefenderAgent):
    def __init__(self) -> None:
        super().__init__(get_logger("RandomDefenderAgent"))

    def get_action(self, state: State) -> DefenderAction:
        return self.get_random_action(state)

class AttackerAgent(Agent):
    def get_utility(self, state: State) -> int:
        util = 0
        for vehicle in state.vehicles:
            for vuln in vehicle.vulnerabilities:
                if vuln.state != CompromiseState.NOT_COMPROMISED:
                    util += vuln.severity / (1 if vehicle.in_platoon else 4)
                # if vehicle.in_platoon and vuln.state != CompromiseState.NOT_COMPROMISED:
                #     util += vuln.severity
                # elif not vehicle.in_platoon and vuln.state == CompromiseState.COMPROMISED_UNKNOWN:
                #     util += vuln.severity / 4
        return int(util)

    @abstractmethod
    def get_action(self, state: State) -> AttackerAction:
        pass

    def take_action(self, state: State, action: AttackerAction) -> State:
        vehicles = list(state.vehicles)
        for i in action.attack:
            self.logger.debug(f"attacking vehicle {i}")
            vehicle = vehicles[i]
            for j, vuln in enumerate(vehicle.vulnerabilities):
                if vuln.state != CompromiseState.NOT_COMPROMISED: continue
                if random.random() > vuln.prob: continue
                self.logger.info(f"successfully compromised vehicle {i} vuln {j} sev {vuln.severity}")
                # new_vulns = vehicle.vulnerabilities[:j] + (replace(vuln, state=CompromiseState.COMPROMISED_UNKNOWN),) + vehicle.vulnerabilities[j+1:]
                new_vulns = vehicle.vulnerabilities[:j] + (replace(vuln, state=CompromiseState.COMPROMISED),) + vehicle.vulnerabilities[j+1:]
                vehicle = replace(vehicle, vulnerabilities=new_vulns)
            vehicles[i] = vehicle

        # calculate util change
        state = replace(state, vehicles=tuple(vehicles))
        change = self.get_utility(state)
        utility = state.attacker_utility
        utility += change
        self.logger.debug(f"utility {utility} ({change:+d})")
        state = replace(state, attacker_utility=utility)

        # return new state
        return state

    
    def get_random_action(self, state: State) -> AttackerAction:
        raise NotImplementedError()

#agent that does nothing, can act as defender or attacker
class PassiveAgent(DefenderAgent, AttackerAgent):
    def __init__(self) -> None:
        super().__init__(get_logger("PassiveAgent"))

    def get_action(self, state: State) -> None:
        return None

    def get_random_action(self, state: State) -> None:
        return None

    def take_action(self, state: State, action: None) -> State:
        return state

    def get_utility(self, state: State) -> int:
        return 0

    def __str__(self) -> str:
        return self.__class__.__name__
#endregion Base stuff

###############################
#region human design agents
###############################
class BasicDefenderAgent(DefenderAgent):
    tolerance_threshold: int

    def __init__(
        self,
        tolerance_threshold: int = 3
    ) -> None:
        super().__init__(get_logger("BasicDefenderAgent"))
    def get_action(self, state: State) -> DefenderAction:
        
        # kick risky vehicles from platoon
        kick = list()
        for i,vehicle in enumerate(state.vehicles):
            if not vehicle.in_platoon: continue
            total = sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED])
            if total > self.tolerance_threshold:
                kick.append(i)

        # join low risk vehicles into the platoon
        join = [
            i for i,vehicle in enumerate(state.vehicles)
            if not vehicle.in_platoon
            and vehicle.risk <= 10
            and sum([vuln.severity for vuln in vehicle.vulnerabilities if vuln.state == CompromiseState.COMPROMISED]) <= self.tolerance_threshold
        ]

        members = torch.tensor([1 if v.in_platoon else 0 for v in state.vehicles])
        members[kick] = 0
        members[join] = 1
        members = members.nonzero().squeeze()
        members = members.numpy()

        return DefenderAction(members=frozenset(members))

class BasicAttackerAgent(AttackerAgent):
    def __init__(
        self,
        attack_limit:int,
        attack_interval: int,
        utility_func: Optional[Callable[[AttackerAgent,State], float]] = None,
    ) -> None:
        super().__init__(get_logger("BasicAttackerAgent"))
        self.attack_limit = attack_limit
        self.attack_interval = attack_interval
        self.step = 0
        if utility_func is not None:
            self.get_utility = utility_func.__get__(self, utility_func)  # type: ignore



    def get_action(self, state: State) -> AttackerAction:
        attack = set()
        if self.step % self.attack_interval == 0:
            # Pick vehicle to attack
            candidates = list([(i,v) for i,v in enumerate(state.vehicles) if v.in_platoon])
            if len(candidates) == 0:
                # attack anything if platoon is empty
                candidates = list(enumerate(state.vehicles))

            if len(candidates) == 0:
                self.logger.warn("sanity check failed, no vehicles to attack")
            else:
                def eval_risk(v: Vehicle) -> float:
                    return sum([x.severity ** 2 * x.prob for x in v.vulnerabilities if x.state == CompromiseState.NOT_COMPROMISED])
                candidates = sorted(candidates, key=lambda x: eval_risk(x[1]))
                for _ in range(self.attack_limit):
                    if len(candidates) == 0:
                        break
                    attack.add(candidates.pop()[0])

        self.step += 1
        return AttackerAction(attack=frozenset(attack))


#endregion human design agents

###############################
#region ml agents
###############################

from utils import get_logger, get_device
from models import StateShapeData, DefenderActor, DefenderCritic
import torch
import torch.optim

#region from original deepRL author code

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.normal = torch.distributions.Normal(torch.as_tensor(0, dtype=torch.float32), torch.as_tensor(1, dtype=torch.float32))
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * torch.sqrt(torch.as_tensor(self.dt, dtype=torch.float32)) * self.normal.sample((self.size,))  # type: ignore
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros(self.size)

#endregion from original deepRL author code


class WolpertingerDefenderAgent(DefenderAgent):
    def __init__(self,
        state_shape_data: StateShapeData,
        learning_rate: float,
        num_proposals: int,
        ou_theta: float,
        ou_mu: float,
        ou_sigma: float,
        epsilon_decay_time: int,
        utility_func: Optional[Callable[[WolpertingerDefenderAgent,State], float]] = None,
    ) -> None:
        super().__init__(get_logger("WolpertingerDefenderAgent"))
        self.learning_rate = learning_rate
        self.state_shape_data = state_shape_data
        self.num_proposals = num_proposals
        # allow to monkey patch the utility function
        # helps when adjusting game balance
        if utility_func is not None:
            self.get_utility = utility_func.__get__(self, utility_func)  # type: ignore

        self.actor = DefenderActor(state_shape_data).to(get_device())
        self.actor_target = DefenderActor(state_shape_data).to(get_device())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = DefenderCritic().to(get_device())
        self.critic_target = DefenderCritic().to(get_device())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # hard update
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.random_process = OrnsteinUhlenbeckProcess(
            size=state_shape_data.num_vehicles,
            theta=ou_theta,
            mu=ou_mu,
            sigma=ou_sigma
        )
        self.collapse_noise = torch.distributions.Normal(torch.as_tensor(0, dtype=torch.float32), torch.as_tensor(0.25, dtype=torch.float32))
        self.training = False
        self.epsilon = 1
        self.epsilon_decay = 1.0 / epsilon_decay_time

    @torch.no_grad()
    def get_action(
        self,
        state_obj: State
    ) -> DefenderAction:
        # convert from state object to tensors to be fed to the actor model
        state = state_obj.as_tensor_batch(self.state_shape_data).to(get_device())
        
        # get action suggestions from the actor
        proto_actions: DefenderActionTensorBatch = self.actor(state)
        assert proto_actions.batch_size == 1, "batch size should be one here"
        if self.training:
            # apply epsilon noise
            proto_actions.members[0] += self.epsilon * self.random_process.sample().to(proto_actions.members.device)
            # decay epsilon
            self.epsilon = max(0, self.epsilon-self.epsilon_decay)

        # convert proto actions to actual actions: -lt 0 => 0, -gt 0 => 1
        actions = self.collapse_proto_actions(proto_actions)
        # copy state tensors to apply to the multiple proposed actions
        state = state.repeat(actions.batch_size)
        # grade the acctions using the critic
        action_q_values: torch.Tensor = self.critic(state, actions)
        del state

        # find the best action
        best_action_index = action_q_values.argmax().cpu()
        assert best_action_index.numel() == 1

        # convert binary vectors to vector of indices
        return DefenderAction(
            members=frozenset(actions.members[best_action_index].cpu().nonzero().flatten().numpy()),
        )
    
    def as_dict(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "num_proposals": self.num_proposals,
            "actor_hidden1_out_features": self.actor.hidden1.out_features,
            "actor_hidden2_out_features": self.actor.hidden2.out_features,
            "critic_hidden1_out_features": self.critic.hidden1.out_features,
            "critic_hidden2_out_features": self.critic.hidden2.out_features,
        }

    # Converts latent action into multiple potential actions
    def collapse_proto_actions(self, proto_actions: DefenderActionTensorBatch) -> DefenderActionTensorBatch:
        members = proto_actions.members
        assert len(members.shape) == 2 # [batch, member_binary_vectors]
        num_proposals = 5 # hyper-parameter
        batch_size, num_vehicles = members.shape
        out_batch_size = batch_size * num_proposals
        # copy members to receive different noise values
        members = members.repeat(num_proposals,1)
        # create and apply noise
        noise = self.collapse_noise.sample(torch.Size((out_batch_size, num_vehicles))).to(get_device())
        members += noise
        del noise
        # convert to binary
        zerovalue = torch.tensor(1.).to(get_device())
        members = members.heaviside(zerovalue)
        # also ensure that no-noise nearest lookup is performed
        members = torch.cat((
            members,
            proto_actions.members.heaviside(zerovalue)
        ))
        del zerovalue
        return DefenderActionTensorBatch(members=members)

    def save(self, dir: str, prefix: Optional[str] = None):
        path: pathlib.Path = pathlib.Path(dir)
        path.mkdir(parents=True, exist_ok=True)
        models = ["actor", "actor_target", "critic", "critic_target"]
        if prefix is None:
            prefix = get_prefix()
        for model in models:
            save_path = path / f"{prefix} {model}.pt"
            torch.save(getattr(self,model).state_dict(), save_path)

    def load(self, dir: str, prefix: str):
        path = pathlib.Path(dir)
        models = ["actor", "actor_target", "critic", "critic_target"]
        for model in models:
            modelpath = path / f"{prefix} {model}.pt"
            getattr(self,model).load_state_dict(torch.load(modelpath, map_location=get_device()))
 