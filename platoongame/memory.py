from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Our game doesn't have a "terminal state" so that simplifies things

class ReplayMemory:
    def __init__(self, capacity: int):
        pass

    def push(self, *v):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class DequeReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([],maxlen=capacity)

    def push(self, *v):
        """Save a transition"""
        self.memory.append(Transition(*v))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# By Shashank Srikanth
# https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/blob/master/memory.py#L36
class RingReplayMemory(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def push(self, *v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = Transition(*v)

    def sample(self, batch_size):
        return [self[i] for i in random.sample(range(len(self)), batch_size)]
