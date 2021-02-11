from collections import namedtuple
import random

class ReplayBuffer:
    def __init__(self, batch_size, size=100_000):
        self.size = size
        self.batch_size = batch_size
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.size:
            self.memory.append(experience)
        else:
            index = self.push_count % self.size
            self.memory[index] = experience

        self.push_count += 1

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def enough_sample(self):
        return len(self.memory) >= self.batch_size