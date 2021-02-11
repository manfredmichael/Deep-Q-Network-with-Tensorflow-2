from collections import namedtuple
from memory import ReplayBuffer
from strategy import Strategy

class Agent:
    def __init__(self):
        self.memory = ReplayBuffer(batch_size=64)
        self.strategy = Strategy(epsilon=1)
