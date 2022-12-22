# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
# Last update: 18th December 2022, by kajarf@kth.se
#

# Load packages
import time

import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from DQN_agent import DQNAgent
from DQN_buffer import Experience, ExperienceReplayBuffer
from util import EpsilonGreedy, running_average

# Import and initialize the discrete Lunar Laner Environment
# Switch render_mode to 'human' for visualization
scenario = 'LunarLander-v2'
env = gym.make(scenario, render_mode=None)
env.reset()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(DEVICE)

# Parameters
n_actions = env.action_space.n                  # Number of available actions
dim_state = len(env.observation_space.high)     # State dimensionality

num_episodes = 400                              # Number of episode
buffer_size = 20000                             # Size of Experience Replay Buffer
buffer_fill = buffer_size // 4                  # How much to fill buffer with random experiences
batch_size = 100                                # Size of training batch
target_update_freq = buffer_size // batch_size  # How often should target network update
discount_factor = 0.99                          # Value of the discount factor
learning_rate = 5e-4                            # Learning rate
eps_max = 0.99                                  # Max epsilon (initial value before decay)
eps_min = 0.05                                  # Min epsilon (final value after decay)
eps_decay = 0.7 * num_episodes                  # Number of epsiodes for decay time (typically 90%-95% of num_episodes)
n_ep_running_average = 50                       # Running average of 50 episodes
two_hidden_layers = True                        # Enable two hidden layer (normally one)
hidden_layer_size = 64                          # Number of neurons in hidden layer
t_max = 1000                                    # Maximum allowed number of steps

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

### Create Experience replay buffer ###
print('Creating experience replay buffer', end='\r', flush=True)
buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
buffer.fill_rand(scenario, buffer_fill)

# Agent initialization
print('Initializing agent', end='\r', flush=True)
agent = DQNAgent(dim_state,
                 n_actions,
                 h=hidden_layer_size,
                 lr=learning_rate,
                 discount=discount_factor,
                 two_layers=two_hidden_layers,
                 device=DEVICE)

# Policy strategy
epsilon = EpsilonGreedy(eps_min, eps_max, eps_decay, 'linear')

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(num_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    eps = epsilon(i)
    rands = np.random.random(t_max)

    while not done and t < t_max:

        # print(f'Episode {i=}, {t=}, {total_episode_reward=:0.2f}', end='\r', flush=True)

        # Render environment
        # env.render() will be called automatically by env.step
        # when in render_mode='human'.
        # https://stackoverflow.com/questions/73845576/whenever-i-try-to-use-env-render-for-openaigym-i-get-assertionerror
        if env.render_mode:
            time.sleep(0.01)

        # Get action
        if rands[t] > eps:
            # Take the best action
            action = agent.forward(state)
        else:
            # Take a random action
            action = env.action_space.sample()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)

        # Append experience to the buffer
        buffer.append(
            Experience(state, action, reward, next_state, done)
        )

        ## TRAINING ##

        # Sample a batch of elements
        samples = buffer.sample_batch(batch_size)

        ## MODIFICATION CER ##
        samples_mod = list(zip(*samples)) + [(state, action, reward, next_state, done)]
        samples = zip(*samples_mod)

        # Perform backward pass
        agent.backward(samples)

        # Set target network equal to main network
        if (t+1) % target_update_freq == 0:
            agent.update_target()

        # Update episode reward
        total_episode_reward += buffer[-1].reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_r = running_average(episode_reward_list, n_ep_running_average)[-1]
    avg_t = running_average(episode_number_of_steps, n_ep_running_average)[-1]
    EPISODES.set_description(
        ' - '.join([
            f'Episode {i:03}',
            f'r/t: {total_episode_reward:+05.01f}/{t:03}',
            f'Avg. r/t: {avg_r:+05.01f}/{avg_t:03}',
        ])
    )

# Close environment
env.close()

# Save network
torch.save(agent.network, 'neural-network-1.pth')
with open('dqn-parameters.txt', 'w') as f:
    s = '\n'.join([
        f'{num_episodes = }'.ljust(40)          + '# Number of episode',
        f'{buffer_size = }'.ljust(40)           + '# Size of Experience Replay Buffer',
        f'{buffer_fill = }'.ljust(40)           + '# How much to fill buffer with random experiences',
        f'{batch_size = }'.ljust(40)            + '# Size of training batch',
        f'{target_update_freq = }'.ljust(40)    + '# How often should target network update',
        f'{discount_factor = }'.ljust(40)       + '# Value of the discount factor',
        f'{learning_rate = }'.ljust(40)         + '# Learning rate',
        f'{eps_max = }'.ljust(40)               + '# Max epsilon (initial value before decay)',
        f'{eps_min = }'.ljust(40)               + '# Min epsilon (final value after decay)',
        f'{eps_decay = }'.ljust(40)             + '# Number of epsiodes for decay time (typically 90%-95% of num_episodes)',
        f'{n_ep_running_average = }'.ljust(40)  + '# Running average of 50 episodes',
        f'{two_hidden_layers = }'.ljust(40)     + '# Enable two hidden layer (normally one)',
        f'{hidden_layer_size = }'.ljust(40)     + '# Number of neurons in hidden layer',
        f'{t_max = }'.ljust(40)                 + '# Maximum allowed number of steps',
    ])
    f.write(s + '\n')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, num_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, num_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].set_ylim(-400, 400)
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, num_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, num_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].set_ylim(0, 1100)
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.savefig('dqn-performance.png')
plt.show()
