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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
from torch import nn
from torch import optim

from DQN_network import Network

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class DQNAgent(Agent):
    """
    Args:
        n (int): Number of states
        m (int): Number of actions
    """

    def __init__(self, n, m, lr=1e-4):

        ### Create network ###
        self.network = Network(n, m)

        ### Create optimizer ###
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=lr)

    def forward(self, state: np.ndarray) -> int:

        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor(state,
                                    requires_grad=False,
                                    device=DEV,
                                    dtype=torch.float32)

        # Compute output of the network
        values = self.network(state_tensor)

        # Pick the action with greatest value
        return values.argmax().item()


    def backward(self, states, actions, rewards, next_states, dones):

        # Training process, set gradients to 0
        self.optimizer.zero_grad()

        # Compute output of the network given the states batch
        values = self.network(torch.tensor(np.array(states),
                                           requires_grad=True,
                                           dtype=torch.float32))

        # Compute loss function
        loss = nn.functional.mse_loss(
            values,
            torch.zeros_like(values, requires_grad=False)
        )

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=1.,
        )

        # Perform backward pass (backpropagation)
        self.optimizer.step()


