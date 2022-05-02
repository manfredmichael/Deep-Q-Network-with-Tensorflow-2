from agent import Agent
import itertools
import gym

env = gym.make('LunarLander-v2')

obs_shape = env.observation_space.sample().shape
act_shape = env.action_space.n

agent = Agent(obs_shape, act_shape)

for episode in itertools.count():
    observation = env.reset()
    done = False
    for timestep in itertools.count():
        env.render()
        action = agent.take_action(observation)
        observation_next, reward, done, info = env.step(action)
        agent.remember(observation, action, observation_next, reward, done)
        agent.learn()

        observation = observation_next

        if done:
            print(agent.get_episode_report())
            break
