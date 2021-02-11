import math

class Strategy:
    def __init__(self, epsilon, epsilon_min=1e-4, epsilon_dec=1e-2, gamma=0.99, policy_update_step=25):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.policy_update_step = policy_update_step
        self.step_count  = 0
    def get_epsilon(self, episode):
        epsilon =  self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1 * episode * self.epsilon_dec)
        return epsilon
    def time_to_update_policy(self, step):
        return step % self.policy_update_step == 0
