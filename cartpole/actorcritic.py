import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import gymnasium as gym
import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuration parameters for the whole setup
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v1")  # Create the environment
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

keras.utils.plot_model(model, to_file='model_actorcritic_complete.svg', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
action_losses_history = []
critic_losses_history = []
losses_history = []
rewards_history = []
running_reward = 0
episode_count = 0


def play():
    env = gym.make("CartPole-v1",render_mode='human')
    state = env.reset()[0]
    with tf.GradientTape() as tape:
        for time in range(500):
            env.render()
            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)
            action_probs, _ = model(state)
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            state, reward, terminated, _, _ = env.step(action)
            if terminated:
                print("score: {}".format(time))
                break

while True:  # Run until solved
    state = env.reset()[0]
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Ensure that probabilities add to 1
            #new_action_probs = np.array(action_probs[0],1-action_probs[0])
            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(ops.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, terminated, _, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if terminated:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            if ret < value:
                diff = ret - value
            else:
                diff = ret - value # Different approach like  1 - value / ret is possible for positive diff
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
            )

        if episode_count % 1 == 0:
            steps = range(len(action_probs_history))
            plt.style.use('dark_background')
 #           plt.plot(steps, action_probs_history, label="ln(p(action))")
            plt.plot(steps, critic_value_history, label="Belohnungsprognose Critic")
            plt.plot(steps, returns, label="Gewichtete Zukunftsbelohnung")
            plt.plot(steps, critic_losses, label="Verlust Critic")
#            plt.plot(steps, rewards_history, label="Rewards history")
            plt.title("Belohnungsprognose Critic")
            plt.legend()
            plt.tight_layout()
            plt.savefig("critic_cartpole_e{}.svg".format(episode_count))
            plt.close()
            plt.plot(steps, action_probs_history, label="Entscheidungssicherheit Actor")
            plt.plot(steps, actor_losses, label="Verlust Actor")
            plt.title("Entscheidungssicherheit Actor")
            plt.legend()
            plt.tight_layout()
            plt.savefig("actor_cartpole_e{}.svg".format(episode_count))
            plt.close()
        # Backpropagation
        actor_losses_sum = sum(actor_losses)
        critic_losses_sum = sum(critic_losses)
        action_losses_history.append(actor_losses_sum/len(actor_losses))
        critic_losses_history.append(critic_losses_sum/len(actor_losses))
        loss_value = actor_losses_sum + critic_losses_sum
        losses_history.append(loss_value/len(actor_losses))
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        episodes = range(len(action_losses_history))
        plt.style.use('dark_background')
        plt.plot(episodes, action_losses_history, label="Verlustfunktion Action")
        plt.plot(episodes, critic_losses_history, label="Verlustfunktion Critic")
        plt.plot(episodes, losses_history, label="Verlustfunktion")
        plt.title("Verlustfunktionen")
        plt.legend()
        plt.tight_layout()
        plt.savefig("losses_cartpole.svg")
        plt.close()
        play()
        break

