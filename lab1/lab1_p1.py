import itertools
import maze2 as mz
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

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

def DynProg(
    env,
    start,
    horizons=range(1, 31),
    num_runs=10_000,
    show_results=False,
    animate=False,
    animate_track=False,
    animate_time_step=0.75,
):

    assert num_runs
    method = 'DynProg';
    results = []

    for horizon in horizons:

        print(f'> Start DynProg with {horizon=}')
        _, policy = mz.dynamic_programming(env,horizon);

        print(f'| Starting {num_runs} runs')
        num_escapes = 0
        for _ in range(num_runs):
            path = env.simulate(start, policy, method);
            for state in path:
                if state is env.STATE_GAME_LOST:
                    break # Minotaur caught you
                elif state is env.STATE_GAME_WON:
                    num_escapes += 1
                    break # Player escaped!

        print(f'| Escaped {num_escapes} times')
        results.append((horizon, num_escapes/num_runs))

        if animate:
            print('| Animating solution of last run')
            mz.animate_solution(env, path, time_step=animate_time_step, track=animate_track)

    if show_results:
        plt.figure()
        plt.scatter(*zip(*results))


def ValIter(
    env,
    start,
    gammas=(0.95,),
    epsilons=(1e-4,),
    num_runs=10_000,
    show_results=False,
    animate=False,
    animate_track=False,
    animate_time_step=0.75,
):

    assert num_runs
    method = 'ValIter'
    results = []

    for gamma, epsilon in itertools.product(gammas, epsilons):

        print(f'> Start ValIter with {gamma=} and {epsilon=}')
        _, policy = mz.value_iteration(env, gamma, epsilon)

        print(f'| Starting {num_runs} runs')
        num_escapes = 0
        for _ in range(num_runs):
            path = env.simulate(start, policy, method)
            for state in path:
                if state is env.STATE_GAME_LOST:
                    break # Minotaur caught you
                elif state is env.STATE_GAME_WON:
                    num_escapes += 1
                    break # Player escaped!

        print(f'| Escaped {num_escapes} times')
        results.append(num_escapes/num_runs)

        if animate:
            print('| Animating solution of last run')
            mz.animate_solution(env, path, time_step=animate_time_step, track=animate_track)

    if show_results:
        plt.figure()
        plt.scatter(*zip(*enumerate(results)))

