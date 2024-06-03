import gymnasium as gym
from dqn import DQNAgent
import numpy as np

env = gym.make('CartPole-v1',render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.load("./save/cartpole-dqn_final.weights.h5")

for e in range(10000):
    state = env.reset()
    state = np.reshape(state[0], [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, terminated, _, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if terminated:
            print("score: {}".format(time))
            break
