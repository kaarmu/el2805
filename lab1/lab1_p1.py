import numpy as np
import maze as mz

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
])
# with the convention
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# Create an environment maze
env = mz.Maze(maze, minotaur=True)
# env.show()

#
# Dynamic Programming
#

if True:

    # Finite horizon
    horizon = 20
    # Solve the MDP problem with dynamic programming
    V, policy= mz.dynamic_programming(env,horizon);

    # Simulate the shortest path starting from position A
    method = 'DynProg';
    start  = (0,0);
    path = env.simulate(start, policy, method);

    # Show the shortest path
    mz.animate_solution(maze, path)

#
# Value Iteration
#

if False:

    # Discount Factor
    gamma   = 0.95;
    # Accuracy treshold
    epsilon = 0.0001;
    V, policy = mz.value_iteration(env, gamma, epsilon)

    method = 'ValIter';
    start  = (0,0);
    path = env.simulate(start, policy, method)

    # Show the shortest path
    mz.animate_solution(maze, path)

