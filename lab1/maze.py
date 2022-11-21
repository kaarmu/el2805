import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
# from IPython import display

def add(xs, ys):
    return tuple(map(sum, zip(xs, ys)))

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False, minotaur=False, standstill=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.minotaur                 = minotaur
        self.standstill               = standstill
        self.actions, self.amap       = self.__actions()
        self.states, self.smap         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                       random_rewards=random_rewards)
    def __actions(self):
        actions = list()
        amap = dict()

        self.moves = {
            self.STAY: (0, 0),
            self.MOVE_LEFT: (0,-1),
            self.MOVE_RIGHT: (0, 1),
            self.MOVE_UP: (-1,0),
            self.MOVE_DOWN: (1,0),
        }
        dims = [tuple(self.moves.keys())]
        if self.minotaur:
            dims *= 2 # for the minotaur
        for action in itertools.product(*dims):
            us = sum(map(self.moves.get, action), start=tuple())

            # action = (n,m)
            # actions = [(xp,yp,xm,ym), ...]
            # amap = { (xp,yp,xm,ym): i, ... }

            # {a: 0, b: 1}
            # [a, b]
            # us = c
            amap[us] = len(actions)
            # amap = {a:0, b:1, c:2}
            actions.append(us)
            # [a, b, c]
        return actions, amap

    def __states(self):
        states = list()
        smap = dict()
        dims = self.maze.shape
        if self.minotaur:
            dims += dims
        for xs in itertools.product(*map(range, dims)):
            if self.maze[xs[:2]] != 1:
                smap[xs] = len(states)
                states.append(xs)
        return states, smap

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Maze edge
        row_edge = -1, self.maze.shape[0]
        col_edge = -1, self.maze.shape[1]
        # Assume validity from start
        valid = True
        # Get states and actions for person
        xs = self.states[state][:2]
        us = self.actions[action][:2]
        # Compute the future position given current (state, action) for person
        row, col = add(xs, us)
        # Is the future position an impossible one ?
        valid &= (
            row not in row_edge and
            col not in col_edge and
            (self.maze[row,col] != 1)
        )
        ##############################
        if self.minotaur:
            # Get states and actions for person
            xs = self.states[state][2:]
            us = self.actions[action][2:]
            # Compute the future position given current (state, action) for person
            row, col = add(xs, us)
            # Is the future position an impossible one ?
            valid &= (
                row not in row_edge and
                col not in col_edge
            )
        # Based on the impossiblity check return the next state.
        if valid:
            xs = self.states[state]
            us = self.actions[action]
            xs = add(xs, us)
            return self.smap[xs]
        else:
            return state

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s, a in itertools.product(range(self.n_states), range(self.n_actions)):
            next_s = self.__move(s, a)
            if not self.minotaur:
                transition_probabilities[next_s, s, a] = 1
            elif not self.standstill and self.actions[a][2:] == self.moves[self.STAY]:
                transition_probabilities[next_s, s, a] = 0
            else:
                xs = self.states[s]
                # I know this is horrible; just wanted to try
                n_empty = 0
                for us in filter(lambda us: us[:2] == (0, 0), self.actions):
                    if not self.standstill and us[2:] == self.moves[self.STAY]:
                        continue
                    _x, _y = add(xs,us)[2:]
                    if 0 <= _x < self.maze.shape[0] and 0 <= _y < self.maze.shape[1]:
                        n_empty += 1
                transition_probabilities[next_s, s, a] = 1/n_empty if n_empty else 0

        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a)
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD
                    # Reward for hitting the minotaur
                    elif self.minotaur and self.states[s][:2] == self.states[s][2:]:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s][:2]] == 2:
                        rewards[s,a] = self.GOAL_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s][:2]]<0:
                        row, col = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a]
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a)
                     i,j = self.states[next_s]
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j]

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.smap[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                action_idxs, distrution = self.premove(s, policy[s,t])
                if sum(distrution) != 1:
                    print(list(self.action[a] for a in action_idxs), distrution)
                action_idx = np.random.choice(action_idxs, p=distrution)
                next_s = self.__move(s,action_idx)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.smap[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
        return path

    def premove(self, s, policy):
        action_p = self.actions[policy][:2]
        next_as = []
        next_ps = []
        for a, action in enumerate(self.actions):
            if action[:2] == action_p:
                next_s = self.__move(s, a)
                p = self.transition_probabilities[next_s, s, a]
                next_as.append(a)
                next_ps.append(p)
        return next_as, next_ps


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.smap)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy.astype(int)

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


    # Update the color at each frame
    for i in range(len(path)):
        # Draw Player
        grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[path[i][:2]].get_text().set_text('Player')
        if i > 0:
            if path[i] == path[i-1]:
                grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[path[i][:2]].get_text().set_text('Player is out')
            else:
                grid.get_celld()[path[i-1][:2]].set_facecolor(col_map[maze[path[i-1][:2]]])
                grid.get_celld()[path[i-1][:2]].get_text().set_text('')
        # Draw Minotaur
        if len(path[i]) > 2:
            grid.get_celld()[path[i][2:]].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[path[i][2:]].get_text().set_text('Minotaur')
            if i > 0:
                if path[i] == path[i-1]:
                    grid.get_celld()[path[i][2:]].set_facecolor(LIGHT_RED)
                    grid.get_celld()[path[i][2:]].get_text().set_text('Minotaur is out')
                else:
                    grid.get_celld()[path[i-1][2:]].set_facecolor(col_map[maze[path[i-1][2:]]])
                    grid.get_celld()[path[i-1][2:]].get_text().set_text('')
        # Display figure
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
