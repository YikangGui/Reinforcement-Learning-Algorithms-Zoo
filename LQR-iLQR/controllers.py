"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg
import deeprl_hw6
import gym
import matplotlib.pyplot as plt


def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = x.copy()
    x_1, _, _, _ = env.step(u, dt)
    return (x_1 - x) / dt


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """

    x_dot = np.zeros((4, 4))
    x_copy = x.copy()
    for dim in range(4):
        tmp1 = np.zeros(4)
        tmp1[dim] = delta
        tmp1 += x_copy
        f1 = simulate_dynamics(env, tmp1, u)

        tmp2 = np.zeros(4)
        tmp2[dim] = - delta
        tmp2 += x_copy
        f2 = simulate_dynamics(env, tmp2, u)
        x_dot[:, dim] = (f1 - f2) / 2 / delta
    return x_dot


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will need to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    u_dot = np.zeros((4, 2))
    u_copy = u.copy()
    for dim in range(2):
        tmp1 = np.zeros(2)
        tmp1[dim] = delta
        tmp1 += u_copy
        f1 = simulate_dynamics(env, x, tmp1)

        tmp2 = np.zeros(2)
        tmp2[dim] = - delta
        tmp2 += u_copy
        f2 = simulate_dynamics(env, x, tmp2)
        u_dot[:, dim] = (f1 - f2) / 2 / delta
    return u_dot


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    Q = env.Q
    R = env.R
    x = env.state
    u = np.zeros(2)
    A = approximate_A(sim_env, x, u)
    B = approximate_B(sim_env, x, u)

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.dot(np.linalg.pinv(R), np.dot(B.T, P))
    uu = - np.dot(K, x - env.goal)
    return uu


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    print('LQR Test...')
    env = gym.make('TwoLinkArm-v0')
    sim_env = gym.make('TwoLinkArm-v0')
    rewards = 0
    steps = 0
    state_list, action_list = [env.state], []

    done = False
    while not done:
        action = calc_lqr_input(env, sim_env)
        nxt_state, reward, done, _ = env.step(action)
        rewards += reward
        steps += 1
        state_list.append(nxt_state)
        action_list.append(action)
        print('===============================================')
        print(f'Step: {steps}, Reward: {round(rewards, 4)}')
        print(f'State: {nxt_state}')
        print(f'Action: {action}')
        print()
    print(f'Total steps: {steps}, reward: {rewards}')

    states = np.asanyarray(state_list)
    x1 = states[:, 0]
    x2 = states[:, 1]
    v1 = states[:, 2]
    v2 = states[:, 3]

    plt.figure(3)
    plt.plot(list(range(len(x1))), x1)
    plt.plot(list(range(len(x2))), x2)
    plt.xlabel('iterations')
    plt.ylabel('position')
    plt.legend(['coordinate1', 'coordinate2'])
    plt.savefig('./LQR_plot/LQR_position.png')

    plt.figure(4)
    plt.plot(list(range(len(v1))), v1)
    plt.plot(list(range(len(v2))), v2)
    plt.xlabel('iterations')
    plt.ylabel('velocity')
    plt.legend(['velocity1', 'velocity2'])
    plt.savefig('./LQR_plot/LQR_velocity.png')

    actions = np.asanyarray(action_list)
    np.save(actions)
    u1 = actions[:, 0]
    u2 = actions[:, 1]

    plt.figure(5)
    plt.plot(list(range(len(u1))), u1)
    plt.plot(list(range(len(u2))), u2)
    plt.xlabel('iterations')
    plt.ylabel('action')
    plt.legend(['action of dim1', 'action of dim2'])
    plt.savefig('./LQR_plot/LQR_action.png')



