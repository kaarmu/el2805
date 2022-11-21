import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import functools
from IPython import display

def add(xs, ys):
    return tuple(map(sum, zip(xs, ys)))

def sub(xs, ys):
    return tuple(map(sum, zip(xs, map(lambda x: -x, ys))))

def dist(xs, ys):
    return np.hypot(*sub(xs, ys))

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

    # Maze objects

    MAZE_EMPTY          = 0
    MAZE_OBSTACLE       = 1
    MAZE_EXIT           = 2

    # States

    STATE_GAME_WON      = object()
    STATE_GAME_LOST     = object()

    STATE_POISONED      = object()

    STATE_TURN_PLAYER   = object()
    STATE_TURN_MINOTAUR = object()

    # Actions

    ACTION_MOVE_STAY           = object()
    ACTION_MOVE_LEFT           = object()
    ACTION_MOVE_RIGHT          = object()
    ACTION_MOVE_UP             = object()
    ACTION_MOVE_DOWN           = object()

    # Rewards

    REWARD_STEP         = -1
    REWARD_GOAL         = 0
    REWARD_IMPOSSIBLE   = -100

    def __init__(
        self,
        maze,
        weights=None,
        random_rewards=False,
        minotaur=False,
        standstill=False,
        poisoned=0,
        turn_based=False,
    ):
        self.maze                       = maze
        self.weights                    = weights
        self.enabled_random_rewards     = random_rewards
        self.enabled_minotaur           = minotaur
        self.enabled_standstill         = standstill
        self.enabled_turn_based         = turn_based
        self.enabled_poisoned           = bool(poisoned)
        self.poison_probability         = 1/poisoned if poisoned else 0

        if self.enabled_turn_based:
            assert self.enabled_standstill

        if self.enabled_standstill:
            assert self.enabled_minotaur

        self.actions                    = self._actions()
        self.states, self.terminals     = self._states()
        self.n_actions                  = len(self.actions)
        self.n_states                   = len(self.states)

        self.transition_probabilities   = self._transitions()
        self.rewards                    = self._rewards()

    def iter_maze(self):
        return itertools.product(*map(range, self.maze.shape))

    def iter_transitions(self):
        return itertools.product(range(self.n_states), range(self.n_actions))

    @functools.cache
    def _state2idx(self, state):
        return self.states.index(state)

    @functools.cache
    def _action2idx(self, action):
        return self.actions.index(action)

    @functools.cache
    def _is_valid_state(self, state):
        return state in self.states

    def _state2str(self, state):
        table = {
            self.STATE_GAME_WON: 'won',
            self.STATE_GAME_LOST: 'lost',
            self.STATE_POISONED: 'poisoned',
            self.STATE_TURN_MINOTAUR: 'minotaur',
            self.STATE_TURN_PLAYER: 'player',
        }
        return (
            table[state] if state in table else
            '{}'.format(*state) if len(state) == 1 else
            '{} {}'.format(*state) if len(state) == 2 else
            '{} {} {}'.format(*state[:2], self._state2str(state[2]))
        )

    def _action2move(self, action):
        return {
            self.ACTION_MOVE_STAY:  (+0, +0),
            self.ACTION_MOVE_LEFT:  (+0, -1),
            self.ACTION_MOVE_RIGHT: (+0, +1),
            self.ACTION_MOVE_UP:    (-1, +0),
            self.ACTION_MOVE_DOWN:  (+1, +0),
        }.get(action, (0, 0))

    def _action2str(self, action):
        return {
            self.ACTION_MOVE_STAY:  'stays',
            self.ACTION_MOVE_LEFT:  'moves left',
            self.ACTION_MOVE_RIGHT: 'moves right',
            self.ACTION_MOVE_UP:    'moves up',
            self.ACTION_MOVE_DOWN:  'moves down',
        }[action]

    def _states(self):
        """
        State ::= Tuple
                | STATE_GAME_WON
                | STATE_GAME_LOST
                | STATE_POISONED

        Tuple ::= (Coord)               # if not enabled_minotaur
                | (Coord, Coord)        # if enabled_minotaur
                | (Coord, Coord, Turn)  # if enabled_turn_based

        Coord ::= (Int, Int)            # rows, cols

        Turn ::= STATE_TURN_PLAYER      # if enabled_turn_based
               | STATE_TURN_MINOTAUR    # if enabled_turn_based

        Terminals ::= STATE_GAME_WON
                    | STATE_GAME_LOST
        """
        states = []
        terminals = []

        states.append(self.STATE_GAME_WON)
        states.append(self.STATE_GAME_LOST)
        states.append(self.STATE_POISONED)

        terminals.append(self.STATE_GAME_WON)
        terminals.append(self.STATE_GAME_LOST)

        # Add all combinations of player, minotaur positions and turns
        # Maze for player
        args = [self.iter_maze()]
        # Maze for minotaur
        args += [self.iter_maze()] if self.enabled_minotaur else []
        # Turn
        turns = (self.STATE_TURN_MINOTAUR, self.STATE_TURN_PLAYER)
        args += [turns] if self.enabled_turn_based else []

        # Add tuple states (player state, minotaur state, turn state)
        states += (
            state
            for state in itertools.product(*args)
            if self.maze[state[0]] != self.MAZE_OBSTACLE
        )

        return states, terminals

    def _actions(self):
        """
        Action ::= ACTION_MOVE_STAY
                 | ACTION_MOVE_LEFT
                 | ACTION_MOVE_RIGHT
                 | ACTION_MOVE_UP
                 | ACTION_MOVE_DOWN
        """

        actions = [
            self.ACTION_MOVE_STAY,
            self.ACTION_MOVE_LEFT,
            self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_UP,
            self.ACTION_MOVE_DOWN,
        ]

        return actions


    def _transitions(self):
        """Computes the transition probabilities for every state action pair."""
        # Initialize the transition probailities tensor (S,S,A)

        dims = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dims)

        # Compute the transition probabilities.
        for i_s, i_a in self.iter_transitions():
            state = self.states[i_s]
            action = self.actions[i_a]

            next_states = self._apply(state, action)

            for next_state in next_states:

                # probability will become 0
                if not self._is_valid_state(next_state):
                    continue

                # get index of next state
                j_s = self._state2idx(next_state)

                # terminal state
                if len(next_states) == 1 and next_state in self.terminals:
                    transition_probabilities[j_s, i_s, i_a] = 1

                # probability will become 0
                elif self.enabled_turn_based and state[2] == state[2]:
                    continue

                elif self.enabled_poisoned:
                    if state is self.STATE_POISONED:
                        transition_probabilities[j_s, i_s, i_a] = (
                            1 if next_state is self.STATE_GAME_LOST else 0
                        )
                    else:
                        transition_probabilities[j_s, i_s, i_a] = (
                            self.poison_probability if next_state is self.STATE_POISONED else
                            (1 - self.poison_probability) / (len(next_states) - 1)
                        )

                else:
                    transition_probabilities[j_s, i_s, i_a] = 1/len(next_states)

        return transition_probabilities

    def _rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if self.weights is None:
            for i_s, i_a in self.iter_transitions():
                state = self.states[i_s]
                action = self.actions[i_a]
                for next_state in self._apply(state, action):

                    # Terminal states
                    if state is self.STATE_GAME_WON:
                        rewards[i_s, i_a] = self.REWARD_GOAL
                    elif state is self.STATE_GAME_LOST:
                        rewards[i_s, i_a] = self.REWARD_IMPOSSIBLE
                    # Reward for hitting a wall and more
                    elif not self._is_valid_state(next_state):
                        rewards[i_s, i_a] = self.REWARD_IMPOSSIBLE
                    else:
                        rewards[i_s, i_a] = self.REWARD_STEP

                    if self.enabled_random_rewards and self.maze[state[0]] < 0:
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[next_state[0]])) * rewards[i_s, i_a]
                        # With probability 0.5 the reward is
                        r2 = rewards[i_s, i_a]
                        # The average reward
                        rewards[i_s, i_a] = 0.5*r1 + 0.5*r2

        # If the weights are described by a weight matrix
        else:
            for i_s, i_a in self.iter_transitions():
                state = self.states[i_s]
                action = self.actions[i_a]

                for next_state in self._apply(state, action):
                    # Simply put the reward as the weights o the next state.
                    j_yp, j_xp = next_state[0]
                    rewards[i_s, i_a] = self.weights[j_yp][j_xp]

        return rewards

    def _peek_surrounding(self, coord):
        """
        Args:
            Coord

        Returns:
            list[Coord] for all nearby coordinates (reachable in next move) that are empty
            list[Coord] for all nearby coordinates (reachable in next move) that are occupied
            list[Coord] for all nearby coordinates (reachable in next move) that are outside
        """

        actions = [
            # self.ACTION_MOVE_STAY,
            self.ACTION_MOVE_LEFT,
            self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_UP,
            self.ACTION_MOVE_DOWN,
        ]

        empty, occupied, outside = [], [], []

        for move in map(self._action2move, actions):
            row, col = add(coord, move)
            is_outside = not (-1 < row < self.maze.shape[0] and -1 < col < self.maze.shape[1])
            is_occupied = not is_outside and self.maze[row, col] != self.MAZE_OBSTACLE
            bin = outside if is_outside else occupied if is_occupied else empty
            bin.append((row, col))

        return empty, occupied, outside

    def _apply(self, state, action):
        """
        Args:
            State
            Action

        Returns:
            list[State] for all possible end states after applying action
        """

        # Terminal states
        if state in self.terminals:
            return [state]

        if state is self.STATE_POISONED:
            return [self.STATE_GAME_LOST]

        next_states = []

        player = state[0]
        move = self._action2move(action)
        next_player = add(player, move)

        for exit_state in zip(*np.where(self.maze == self.MAZE_EXIT)):
            if player == exit_state:
                return [self.STATE_GAME_WON]

        if self.enabled_minotaur:
            minotaur = state[1]
            empty, occupied, _ = self._peek_surrounding(minotaur)
            all_next_minotaur = empty + occupied

            if player == minotaur:
                return [self.STATE_GAME_LOST]

            if self.enabled_standstill:
                all_next_minotaur.append(minotaur)

            if self.enabled_turn_based:
                turn = state[2]
                next_turn = {
                    self.STATE_TURN_PLAYER:   self.STATE_TURN_MINOTAUR,
                    self.STATE_TURN_MINOTAUR: self.STATE_TURN_PLAYER,
                }[turn]

                for next_minotaur in all_next_minotaur:
                    if next_player == next_minotaur:
                        next_states.append(self.STATE_GAME_LOST)
                    else:
                        next_states.append((next_player, next_minotaur, next_turn))
            else:
                for next_minotaur in all_next_minotaur:
                    if next_player == next_minotaur:
                        next_states.append(self.STATE_GAME_LOST)
                    else:
                        next_states.append((next_player, next_minotaur))
        else:
            next_states.append((next_player,))

        if self.enabled_poisoned:
            next_states.append(self.STATE_POISONED)

        return next_states

    def _move(self, state, policy):
        """Makes a step in the maze, given a current position."""
        i_s = self._state2idx(state)
        i_a = policy
        # if s := sum(self.transition_probabilities[:, i_s, i_a]):
        #     print(s, set(map(self._action2str,
        #                       map(self.actions.__getitem__,
        #                           np.where(self.transition_probabilities[:, i_s, :] != 0)[1]))),
        #           self._action2str(self.actions[i_a]),
        #           self.transition_probabilities[:, i_s, i_a][self.transition_probabilities[:, i_s, i_a] != 0],
        #         )
        j_s = np.random.choice(range(self.n_states),
                               p=self.transition_probabilities[:, i_s, i_a])
        return self.states[j_s]

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t, state = 0, start
            # Add the starting state in the maze to the path
            path.append(state)
            while t < horizon-1:
                path.append(state)
                i_s = self._state2idx(state)
                t += 1
                state = self._move(state, policy[i_s, t])
        if method == 'ValIter':
            # Initialize current state and time
            t, state, i_s = 1, start, self._state2idx(start)
            # Loop while state is not a terminal state
            while state not in self.terminals:
                path.append(state)
                state = self._move(state, policy[i_s])
                i_s = self._state2idx(state)
                t +=1
            # Add terminal state
            path.append(state)

        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        # print('The mapping of the states:')
        # print(self.smap)
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

def animate_solution(env, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows,cols = env.maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[env.maze[j,i]] for i in range(cols)] for j in range(rows)]

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
    for i, state in enumerate(path):

        j = None
        while state in env.terminals:
            j = i if j is None else j
            j, state = j-1, path[j-1]

        # Final animation
        if i+1 == len(path):
            # Draw Player
            grid.get_celld()[state[0]].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[state[0]].get_text().set_text('Player')
            # Draw Minotaur
            if env.enabled_minotaur:
                grid.get_celld()[state[1]].set_facecolor(LIGHT_RED)
                grid.get_celld()[state[1]].get_text().set_text('Minotaur is out')

        # Running animation
        else:
            # Draw Player
            grid.get_celld()[state[0]].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[state[0]].get_text().set_text('Player')
            # Draw Minotaur
            if env.enabled_minotaur:
                grid.get_celld()[state[1]].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[state[1]].get_text().set_text('Minotaur')

        # Display figure
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.8)

        # Reset state unless final animation
        if i+1 != len(path):
            # Reset Player
            grid.get_celld()[state[0]].set_facecolor(col_map[env.maze[state[0]]])
            grid.get_celld()[state[0]].get_text().set_text('')
            if env.enabled_minotaur:
                grid.get_celld()[state[1]].set_facecolor(col_map[env.maze[state[1]]])
                grid.get_celld()[state[1]].get_text().set_text('')

