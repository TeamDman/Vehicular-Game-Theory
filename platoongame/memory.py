from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass
import random
from typing import List, Union
from agents import Action
from game import Game, State

@dataclass(frozen=True)
class Transition:
    state: State
    action: Action
    reward: float
    next_state: Union[State, None]
    terminal: bool

class ReplayMemory(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def push(self, v: Transition) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Transition]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class DequeReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, v: Transition) -> None:
        self.memory.append(v)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# from https://github.com/ghliu/pytorch-ddpg/blob/master/memory.py#L36
class RingReplayMemory(object):
    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Transition:
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def push(self, v: Transition) -> None:
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def sample(self, batch_size: int) -> List[Transition]:
        return [self[i] for i in random.sample(range(len(self)), batch_size)]
