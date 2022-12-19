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
from torch import optim
from torch import nn

from DQN_agent import DQNAgent
from DQN_network import Network
from DQN_buffer import Experience, ExperienceReplayBuffer
from util import EpsilonGreedy, running_average

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(DEVICE)

# Parameters
do_render = False
N_episodes = 200                             # Number of episode
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

buf_sz = 10000
n_samples = 10
lr = 1e-1
eps_max = 0.99
eps_min = 0.05

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

### Create Experience replay buffer ###
buffer = ExperienceReplayBuffer(maximum_length=buf_sz)
buffer.fill_rand(env, buf_sz//2) # fill half, any good reasoning?

# Agent initialization
agent = DQNAgent(dim_state, n_actions, lr=lr, device=DEVICE)

# Policy strategy
epsilon = EpsilonGreedy(eps_min, eps_max, 0.9*N_episodes, 'linear')

# Set rendering mode
if do_render:
    env.render_mode('human')

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    eps = epsilon(i)
    while not done:

        # Render environment
        if do_render:
            env.render()
            time.sleep(0.01)

        # Get action
        if np.random.random() > eps:
            # Take the best action
            action = agent.forward(state)
        else:
            # Take a random action
            action = np.random.randint(0, n_actions)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)

        # Append experience to the buffer
        buffer.append(
            Experience(state, action, reward, next_state, done)
        )

        ## TRAINING ##

        # Sample a batch of 3 elements
        samples = buffer.sample_batch(n=n_samples)

        # Perform backward pass
        agent.backward(samples)

        # Update episode reward
        total_episode_reward += buffer[-1].reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Save network
torch.save(agent.network, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].set_ylim(-500, 100)
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].set_ylim(0, 500)
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
