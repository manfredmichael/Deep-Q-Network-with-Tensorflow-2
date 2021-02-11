from collections import namedtuple
from strategy import Strategy
from memory import ReplayBuffer
from model import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

Experience = namedtuple('Experience', ['observations', 'actions', 'rewards', 'observations_next', 'dones'])

class Agent:
    def __init__(self, obs_shape, act_shape, hidden_layers=[200, 200], optimizer=Adam(lr=0.001)):
        self.memory = ReplayBuffer(batch_size=64)
        self.strategy = Strategy(epsilon=1)
        self.optimizer = optimizer
        self.policy_model = Model(obs_shape, act_shape, hidden_layers)
        self.target_model = Model(obs_shape, act_shape, hidden_layers)
        self.obs_shape = obs_shape
        self.act_shape = act_shape

        self.update_policy()

    def update_policy(self):
        policy_variables = self.policy_model.trainable_variables
        target_variables = self.target_model.trainable_variables

        for policy_var, target_var in zip(policy_variables, target_variables):
            target_var.assign(policy_var.numpy())
    
    # STEP 1: TAKE ACTION
    def take_action(self, observation):
        epsilon = self.strategy.get_epsilon()

        # the agent takes random actions sometimes to explore environment
        if epsilon > random.random():
            return random.randrange(self.act_shape), epsilon
        else:
            return np.argmax(self.policy_model(np.expand_dims(observation, axis=0).astype(np.float32))), epsilon

    # STEP 2: SAVE THE EXPERIENCE (OBSERVATION) FROM THE ACTION TO MEMORY
    def remember(self, observation, action, observation_next, reward, done):
        self.memory.push(Experience(observation, action, observation_next, reward, done))

    # STEP 3: EVALUATE POLICY ACCORDING TO COLLECTED EXPERIENCES
    def learn(self):
        if self.memory.enough_sample():
            experiences = self.memory.sample()
            experiences = Experience(*zip(*experiences))

            observations      = np.asarray(experiences[0])
            actions           = np.asarray(experiences[1])
            rewards           = np.asarray(experiences[2])
            observations_next = np.asarray(experiences[3])
            dones             = np.asarray(experiences[4])

            q_values_next = self.target_model(observations_next.astype(np.float32))
            q_values_target = np.where(dones, rewards, rewards + self.strategy.gamma * np.max(q_values_next, axis=1))
            q_values_target = tf.convert_to_tensor(q_values_target, dtype=np.float32)

            with tf.GradientTape() as tape:
                q_values = tf.reduce_sum(self.target_model(observations.astype(np.float32)) * tf.one_hot(actions, self.act_shape), axis=1)
                loss = tf.reduce_mean(np.square(q_values_target - q_values))

            variables = self.policy_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        if self.strategy.time_to_update_policy():
            self.update_policy()