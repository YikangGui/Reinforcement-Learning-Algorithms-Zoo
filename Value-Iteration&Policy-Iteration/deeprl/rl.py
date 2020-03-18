# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import lake_envs as lake_env
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import copy


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    count = 0
    while True:
        delta = 0
        count += 1
        value_func_new = np.zeros(env.nS)
        for i in range(env.nS):
            v = value_func[i]
            value_func_new[i] = sum([(env.P[i][policy[i]][0][2] + gamma * value_func[env.P[i][policy[i]][0][1]])])
            delta = max(delta, abs(v - value_func_new[i]))
        value_func = copy.deepcopy(value_func_new)
        if delta < tol or count >= max_iterations:
            return value_func, count


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    count = 0
    while True:
        delta = 0
        count += 1
        for i in range(env.nS):
            v = value_func[i]
            value_func[i] = sum([(env.P[i][policy[i]][0][2] + gamma * value_func[env.P[i][policy[i]][0][1]])])
            delta = max(delta, abs(v - value_func[i]))
        if delta < tol or count >= max_iterations:
            return value_func, count


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    count = 0
    while True:
        count += 1
        delta = 0
        indice = list(range(env.nS))
        np.random.shuffle(indice)
        for i in indice:
            v = value_func[i]
            value_func[i] = sum([(env.P[i][policy[i]][0][2] + gamma * value_func[env.P[i][policy[i]][0][1]])])
            delta = max(delta, abs(v - value_func[i]))
        if delta < tol or count >= max_iterations:
            return value_func, count


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for i in range(env.nS):
        old_action = policy[i]
        policy[i] = np.argmax([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)])
        if policy[i] != old_action:
            policy_stable = False
    return policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    # value_func = np.zeros(env.nS)
    iterations_improvement = 0
    total_iterations_evaluation = 0
    while True:
        iterations_improvement += 1
        value_func, iterations_evaluation = evaluate_policy_sync(env, gamma, policy)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        total_iterations_evaluation += iterations_evaluation
        if policy_stable or iterations_improvement >= max_iterations:
            return policy, value_func, iterations_improvement, total_iterations_evaluation


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    # value_func = np.zeros(env.nS)
    count = 0
    total_iterations_evaluation = 0
    while True:
        count += 1
        value_func, iterations = evaluate_policy_async_ordered(env, gamma, policy)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        total_iterations_evaluation += iterations
        if policy_stable or count >= max_iterations:
            return policy, value_func, count, total_iterations_evaluation


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    # value_func = np.zeros(env.nS)
    count = 0
    total_iterations_evaluation = 0
    while True:
        count += 1
        value_func, iterations = evaluate_policy_async_randperm(env, gamma, policy)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        total_iterations_evaluation += iterations
        if policy_stable or count >= max_iterations:
            return policy, value_func, count, total_iterations_evaluation


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    policy = []
    count = 0
    while True:
        count += 1
        delta = 0
        value_func_new = np.zeros(env.nS)
        for i in range(env.nS):
            v = value_func[i]
            value_func_new[i] = max([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)])
            delta = max(abs(v - value_func_new[i]), delta)
        value_func = copy.deepcopy(value_func_new)
        if delta < tol or count >= max_iterations:
            break
    for i in range(env.nS):
        policy.append(np.argmax([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)]))
    policy = np.array(policy)
    return value_func, count, policy


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    policy = []
    count = 0
    while True:
        count += 1
        delta = 0
        for i in range(env.nS):
            v = value_func[i]
            value_func[i] = max([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)])
            delta = max(abs(v - value_func[i]), delta)
        if delta < tol or count >= max_iterations:
            break
    for i in range(env.nS):
        policy.append(np.argmax([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)]))
    policy = np.array(policy)
    return value_func, count, policy


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    policy = []
    count = 0
    while True:
        count += 1
        delta = 0
        indice = list(range(env.nS))
        np.random.shuffle(indice)
        for i in indice:
            v = value_func[i]
            value_func[i] = max(
                [env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)])
            delta = max(abs(v - value_func[i]), delta)
        if delta < tol or count >= max_iterations:
            break
    for i in range(env.nS):
        policy.append(
            np.argmax([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)]))
    policy = np.array(policy)
    return value_func, count, policy


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    policy = []
    count = 0
    while True:
        count += 1
        delta = 0
        if env.nS == 16:
            indice = [5, 0, 1, 2, 6, 10, 9, 8, 4, 3, 7, 11, 15, 14, 13, 12]
        elif env.nS == 64:
            indice = [57, 56, 48, 49,50, 58, 40, 41,42, 43, 51, 59, 32, 33, 34, 35, 36, 44, 52, 60, 24, 25, 26, 27, 28,
                      29, 37, 45, 53, 61, 62, 54, 46, 38, 30, 22, 21, 20, 19, 18, 17, 16, 8, 9, 10, 11, 12, 13, 14, 15,
                      23, 31, 39, 47, 55, 63, 0, 1, 2, 3, 4, 5, 6, 7]
        else:
            raise ValueError("Incorrect env!")
        assert sum(indice) == sum(range(env.nS))
        for i in indice:
            v = value_func[i]
            value_func[i] = max(
                [env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)])
            delta = max(abs(v - value_func[i]), delta)
        if delta < tol or count >= max_iterations:
            break
    for i in range(env.nS):
        policy.append(
            np.argmax([env.P[i][action][0][2] + gamma * value_func[env.P[i][action][0][1]] for action in range(4)]))
    policy = np.array(policy)
    return value_func, count, policy


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 2.2 & 2.6

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])
    
    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)
    

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)
    
    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))
    
    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 2.3 & 2.5

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6)) 
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]), 
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1], 
                xticklabels = np.arange(1, env.nrow+1))
    plt.savefig('heatmap.png')
    plt.show()
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None


if __name__ == '__main__':
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    policy_pi_sync, value_func_pi_sync, iterations_improvement_pi_sync, iterations_evaluation_pi_sync = policy_iteration_sync(env, 0.9)

    policy_pi_async_order, value_func_pi_async_order, iterations_improvement_pi_async_order, iterations_evaluation_pi_async_order = policy_iteration_async_ordered(env, 0.9)

    policy_pi_async_rand, value_func_pi_async_rand, iterations_improvement_pi_async_rand, iterations_evaluation_pi_async_rand = policy_iteration_async_randperm(env, 0.9)

    value_func_vi_sync, count_vi_sync, policy_vi_sync = value_iteration_sync(env, 0.9)

    value_func_vi_async_order, count_vi_async_order, policy_vi_async_order = value_iteration_async_ordered(env, 0.9)

    value_func_vi_async_rand, count_vi_async_rand, policy_vi_async_rand = value_iteration_async_randperm(env, 0.9)

    value_func_vi_async_custom, count_vi_async_custom, policy_vi_async_custom = value_iteration_async_custom(env, 0.9)


