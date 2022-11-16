import numpy as np
import maze as mz
import matplotlib.pyplot as plt

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0],
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
results = []
num_runs = 100
method = 'DynProg';
start  = (0,0,6,5) if env.minotaur else (0,0);

for horizon in range(15, 18):
    print('Horizon: ', horizon)
    # Solve the MDP problem with dynamic programming
    V, policy= mz.dynamic_programming(env,horizon);
    print('DynProg done')
    # Simulate the shortest path starting from position A
    
    # path = env.simulate(start, policy, method);

    # Show the shortest path
    # mz.animate_solution(maze, path)

    num_escapes = 0
    for i in range(0,num_runs):
        path = env.simulate(start, policy, method);
        for state in path:
            if state[:2] == state[2:]:
                break # Minotaur caught you
            elif list(state[:2]) == [6,5]:
                num_escapes += 1
                break # Player escaped!
    print("Escaped {} times out of {} runs with horizon {}".format(num_escapes, num_runs, horizon))

    results.append((horizon, num_escapes/num_runs))

plt.plot(*zip(*results))
plt.show()

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

