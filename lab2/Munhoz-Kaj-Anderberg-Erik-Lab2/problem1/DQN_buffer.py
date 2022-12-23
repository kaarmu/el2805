from collections import namedtuple, deque
import numpy as np
import gym

from DQN_agent import RandomAgent

# namedtuple is used to create a special type of tuple object. Namedtuples
# always have a specific name (like a class) and specific fields.
# In this case I will create a namedtuple 'Experience',
# with fields: state, action, reward,  next_state, done.
# Usage: for some given variables s, a, r, s, d you can write for example
# exp = Experience(s, a, r, s, d). Then you can access the reward
# field by  typing exp.reward
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def __getitem__(self, i):
        return self.buffer[i]

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)

    def fill_rand(self, scenario, n=None):

        env = gym.make(scenario)

        if n is None:
            n = self.buffer.maxlen

        n_actions = env.action_space.n                  # Number of available actions
        agent = RandomAgent(n_actions)

        state, _ = env.reset()

        for _ in range(n):

            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)

            # Append experience to the buffer
            self.append(
                Experience(state, action, reward, next_state, done)
            )

            state = env.reset()[0] if done else next_state

