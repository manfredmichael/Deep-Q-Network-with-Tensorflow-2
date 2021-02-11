import math

class Strategy:
    def __init__(self, epsilon, epsilon_min=1e-4, epsilon_dec=1e-4):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.step_count  = 0
    def get_epsilon(self):
        epsilon =  self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1 * self.step_count * self.epsilon_dec)
        self.step_count += 1
        return epsilon
