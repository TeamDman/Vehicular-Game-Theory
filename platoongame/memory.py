from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import random
from typing import Generic, List, Optional, TypeVar, Union
from agents import DefenderAction
from game import State
from models import DefenderActionTensorBatch, StateShapeData, StateTensorBatch
import torch

@dataclass(frozen=True)
class Transition:
    state: State
    action: DefenderAction
    reward: float
    next_state: Union[State, None]
    terminal: bool

    def as_tensor_batch(self, shape_data: StateShapeData) -> TransitionTensorBatch:
        return TransitionTensorBatch(
            state=self.state.as_tensor_batch(shape_data),
            action=self.action.as_tensor_batch(shape_data),
            reward=torch.as_tensor(self.reward, dtype=torch.float32).unsqueeze(0),
            next_state=State.zeros(shape_data, 1) if self.next_state is None else self.next_state.as_tensor_batch(shape_data),
            terminal=torch.as_tensor(self.terminal, dtype=torch.bool).unsqueeze(0),
        )
    

@dataclass(frozen=True)
class TransitionTensorBatch:
    state: StateTensorBatch
    action: DefenderActionTensorBatch
    reward: torch.Tensor #torch.float32
    next_state: StateTensorBatch
    terminal: torch.Tensor #torch.bool

    def to_device(self, device: torch.device) -> TransitionTensorBatch: 
        return TransitionTensorBatch(
            state=self.state.to(device),
            action=self.action.to_device(device),
            reward=self.reward.to(device),
            next_state=self.next_state.to(device),
            terminal=self.terminal.to(device),
        )

    @staticmethod
    def cat(items: List[TransitionTensorBatch]) -> TransitionTensorBatch:
        return TransitionTensorBatch(
            state=StateTensorBatch.cat([v.state for v in items]),
            action=DefenderActionTensorBatch.cat([v.action for v in items]),
            reward=torch.cat([v.reward for v in items]),
            next_state=StateTensorBatch.cat([v.next_state for v in items]),
            terminal=torch.cat([v.terminal for v in items]),
        )

T = TypeVar("T")
class ReplayMemory(ABC, Generic[T]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def push(self, v: T) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[T]:
        pass

    @abstractmethod
    def get_max_len(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class DequeReplayMemory(ReplayMemory, Generic[T]):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, v: T) -> None:
        self.memory.append(v)

    def sample(self, batch_size: int) -> List[T]:
        return random.sample(self.memory, batch_size)

    def get_max_len(self) -> int:
        assert self.memory.maxlen is not None
        return self.memory.maxlen

    def __len__(self) -> int:
        return len(self.memory)

# from https://github.com/ghliu/pytorch-ddpg/blob/master/memory.py#L36
class RingReplayMemory(ReplayMemory, Generic[T]):
    data: List[Optional[T]]
    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self) -> int:
        return self.length
        
    def get_max_len(self) -> int:
        return self.maxlen

    def __getitem__(self, idx) -> T:
        if idx < 0 or idx >= self.length:
            raise KeyError()
        rtn = self.data[(self.start + idx) % self.maxlen]
        assert rtn is not None
        return rtn

    def push(self, v: T) -> None:
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

    def sample(self, batch_size: int) -> List[T]:
        return [self[i] for i in random.sample(range(len(self)), batch_size)]
