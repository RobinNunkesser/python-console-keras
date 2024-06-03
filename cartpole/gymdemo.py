import gymnasium as gym
import time

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

for step_index in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("{}: Observation {}, Reward {}, Terminated {}, Truncated {}, Info {}".format(step_index, observation, reward, terminated, truncated, info))
    time.sleep(0.2)
    if terminated:
        break
env.close()
