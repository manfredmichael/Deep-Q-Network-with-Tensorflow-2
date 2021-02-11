from agent import Agent
import itertools
import gym

env = gym.make('CartPole-v0')
obs_shape = env.observation_space.sample().shape
act_shape = env.action_space.n
agent = Agent(obs_shape, act_shape)

while True:
    observation = env.reset()
    done = False
    for timestep in itertools.count():
        env.render()
        action, epsilon = agent.take_action(observation)
        observation_next, reward, done, info = env.step(action)
        agent.remember(observation, action, observation_next, reward, done)
        agent.learn()

        observation = observation_next

        if done:
            break