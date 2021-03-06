"""
    File name: replay_memory.py
    Author: Christopher Villalta
"""

import random

class ReplayMemory(object):
    """Ring buffer for more effecient experience replay buffer.

    Attributes:
        size: Capacity of the buffer.
    """

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def append(self, experience):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
