"""

Problem 1 of lab 1 in EL2805.

The subproblems (a-h) can be enabled/disabled by switching the if-statements to True/False.

Authors: 
    - Kaj Munhoz Arfvidsson (980213-4032)
    - Erik Andeberg ()
"""

import maze2 as mz
from lab1_p1 import maze, DynProg, ValIter

import matplotlib.pyplot as plt


##
## Problem 1.a)
## 

# Theory only


##
## Problem 1.b)
## 

# Theory only


##
## Problem 1.c)
## 

if False:

    ## Show plot with minotaur in the way
    
    # Create an environment maze
    env = mz.Maze(maze, minotaur=True, minotaur_moveable=False)

    # Create start state
    start = ((0, 0), (2,0))

    DynProg(
        env,
        start,
        horizons=range(17, 18),
        num_runs=1,
        animate=True,
        animate_track=True,
        animate_time_step=0,
        show_results=False,
    )

    plt.title('Minotaur in the way')
    plt.show()

    ## Show plot with minotaur not in the way
    
    # Create an environment maze
    env = mz.Maze(maze, minotaur=True, minotaur_moveable=False)

    # Create start state
    start = ((0, 0), (2,1))

    DynProg(
        env,
        start,
        horizons=range(17, 18),
        num_runs=1,
        animate=True,
        animate_track=True,
        animate_time_step=0,
        show_results=False,
    )

    plt.title('Minotaur not in the way')
    plt.show()


##
## Problem 1.d)
## 

if False:

    ## Show plot with minotaur in walk-only mode
    
    # Create an environment maze
    env = mz.Maze(maze, minotaur=True, standstill=False)

    # Create start state
    start = ((0, 0), (6,5))

    DynProg(
        env,
        start,
        horizons=range(1, 30),
        num_runs=10_000,
        show_results=True,
    )

    plt.title('Minotaur in walk-only mode')
    plt.ylabel('Success rate')
    plt.xlabel('Horizon [s]')
    plt.show()

    ## Show plot with minotaur with standstill enabled
    
    # Create an environment maze
    env = mz.Maze(maze, minotaur=True, standstill=True)

    # Create start state
    start = ((0, 0), (6,5))

    DynProg(
        env,
        start,
        horizons=range(1, 30),
        num_runs=10_000,
        show_results=True,
    )

    plt.title('Minotaur with standstill enabled')
    plt.ylabel('Success rate')
    plt.xlabel('Horizon [s]')
    plt.show()


##
## Problem 1.e)
## 

# Theory only


##
## Problem 1.f)
## 

if False:

    # Create an environment maze
    env = mz.Maze(maze, minotaur=True, poisoned=30)

    # Create start state
    start = ((0, 0), (6,5))

    ValIter(
        env,
        start,
        num_runs=10_000,
    )


##
## Problem 1.g)
## 

# Theory only


##
## Problem 1.h)
##

# Theory only


