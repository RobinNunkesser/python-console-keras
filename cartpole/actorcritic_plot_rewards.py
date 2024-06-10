import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import ops

gamma = 0.99  # Discount factor for past rewards
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

plt.style.use('dark_background')
for reward in range(1, 500):
    if reward % 50 == 0:
        rewards_history = [1] * reward
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        plt.plot(range(0,len(returns)), returns,label="Score {}".format(reward))
plt.xlabel("Spielzeit / Belohnung")
plt.ylabel("Erwartete Belohnung")
plt.legend()
plt.tight_layout()
plt.savefig("expected_rewards.svg")
