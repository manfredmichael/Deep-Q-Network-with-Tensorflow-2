import tensorflow as tf
from collections import deque
import numpy as np
import datetime


class LearningReport:
    def __init__(self, filename='agent', text_space=18, average_reward_episode=100):
        self.text_space = text_space
        self.episode_rewards = deque(maxlen=average_reward_episode)
        self.episode_reward = 0
        
        filename = filename + datetime.datetime.now().strftime(' (%d-%m-%y %H:%M:%S)')
        self.writer = tf.summary.create_file_writer(logdir='logs/{}'.format(filename))

    def format_report(self, text):
        white_space = ''.join([' ' for i in range(self.text_space - len(text))])
        return text + white_space

    def add_to_report(self, reward):
        self.episode_reward += reward

    def report_episode(self, episode, epsilon):
        self.episode_rewards.append(self.episode_reward)
        self.write_episode_report(episode, epsilon)

        episode    = self.format_report('episode: ' + str(episode))
        reward     = self.format_report('reward: ' + str(self.episode_reward))
        reward_avg = self.format_report('reward avg: ' + '{:.2f}'.format(np.mean(self.episode_rewards)))
        epsilon    = self.format_report('epsilon: ' + '{:.2f}'.format(epsilon))

        report = episode + reward + reward_avg + epsilon
        self.episode_reward = 0
        
        return report
        
    def write_episode_report(self, episode, epsilon):
        with self.writer.as_default():
            tf.summary.scalar('Episode_reward', self.episode_reward, step=episode)
            tf.summary.scalar('Running_avg_reward', np.mean(self.episode_rewards), step=episode)
            tf.summary.scalar('Epsilon', epsilon, step=episode)
