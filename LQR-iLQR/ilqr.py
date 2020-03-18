"""LQR, iLQR and MPC."""

from  controllers import approximate_B, approximate_A
import numpy as np
import gym
import deeprl_hw6
import time
import matplotlib.pyplot as plt
import copy

np.random.seed(1024)
EPS_CONVERGE = 1e-4
FINAL_COST_WEIGHT = int(1e4)
INTER_COST_WEIGHT = 1
tN = 100
dt = 1e-3
MPC = False


def inv_stable(M, lamb=1e-2):
    """Inverts matrix M in a numerically stable manner.

    This involves looking at the eigenvector (i.e., spectral) decomposition of the
    matrix, and (1) removing any eigenvectors with non-positive eingenvalues, and
    (2) adding a constant to all eigenvalues.
    """
    M_evals, M_evecs = np.linalg.eig(M)
    M_evals[M_evals < 0] = 0.0
    M_evals += lamb
    M_inv = np.dot(M_evecs,
                   np.dot(np.diag(1.0 / M_evals), M_evecs.T))
    return M_inv


def simulate_dynamics_next(env, x, u):
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

    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    # env.render()
    # next_state, _, _, _ = env.step(u, dt=1e-5)
    next_state, _, _, _ = env.step(u, dt=dt)
    return next_state
    # return np.zeros(x.shape)


def cost_inter(env, x, u):
    """intermediate cost function

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

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """

    l = np.sum(u ** 2) * INTER_COST_WEIGHT
    l_x = np.zeros(4) * INTER_COST_WEIGHT
    l_xx = np.zeros((4, 4)) * INTER_COST_WEIGHT
    l_u = 2 * u * INTER_COST_WEIGHT
    l_uu = 2 * np.eye(2) * INTER_COST_WEIGHT
    l_ux = np.zeros((2, 4)) * INTER_COST_WEIGHT
    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    l = np.sum((x - env.goal) ** 2) * FINAL_COST_WEIGHT
    l_x = 2 * (x - env.goal) * FINAL_COST_WEIGHT
    l_xx = 2 * np.eye(4) * FINAL_COST_WEIGHT
    return l, l_x, l_xx


def simulate(env, x0, U, dt=1e-5):
    env.state = x0.copy()
    tN = U.shape[0]
    X = np.zeros((tN, 4))
    X[0] = x0.copy()
    cost = 0
    for i in range(tN - 1):
        # X[i+1] = simulate_dynamics_next(env, X[i], U[i])
        x_, _, _, _ = env.step(U[i])
        X[i+1] = x_.copy()
        l, _, _, _, _, _ = cost_inter(env, X[i], U[i])
        # cost += l * dt
        cost += l
    l_f, _, _ = cost_final(env, X[-1])
    # cost += l_f * dt
    cost += l_f
    return X, cost
    # return None


#  Reference: https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
def calc_ilqr_input(env, sim_env, tN=100, max_iter=1e5):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    x0 = env.state
    sim_env.state = x0.copy()

    dof = 2
    num_states = 4
    cost_list = []

    U = np.zeros((tN - 1, dof))
    # U = np.random.uniform(-10, 10, (tN - 1, dof))

    X, cost = simulate(sim_env, x0, U)
    cost_list.append(cost)
    for ii in range(int(max_iter)):

        f_x = np.zeros((tN - 1, num_states, num_states))
        f_u = np.zeros((tN - 1, num_states, dof))
        l = np.zeros((tN, 1))
        l_x = np.zeros((tN, num_states))
        l_xx = np.zeros((tN, num_states, num_states))
        l_u = np.zeros((tN, dof))
        l_uu = np.zeros((tN, dof, dof))
        l_ux = np.zeros((tN, dof, num_states))

        for t in range(tN - 1):
            A = approximate_A(sim_env, X[t], U[t])
            B = approximate_B(sim_env, X[t], U[t])
            # A = approximate_A_discrete(sim_env, X[t], U[t])
            # B = approximate_B_discrete(sim_env, X[t], U[t])
            f_x[t] = np.eye(num_states) + A * dt
            f_u[t] = B * dt

            l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t] = cost_inter(sim_env, X[t], U[t])
            l[t] *= dt
            l_x[t] *= dt
            l_xx[t] *= dt
            l_u[t] *= dt
            l_uu[t] *= dt
            l_ux[t] *= dt
        l[-1], l_x[-1], l_xx[-1] = cost_final(sim_env, X[-1])
        l[-1] *= dt
        l_x[-1] *= dt
        l_xx[-1] *= dt

        V_x = l_x[-1].copy()  # dV / dx
        V_xx = l_xx[-1].copy()  # d^2 V / dx^2
        k = np.zeros((tN - 1, dof))
        K = np.zeros((tN - 1, dof, num_states))

        for t in range(tN - 2, -1, -1):

            Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
            Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

            Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t]))
            Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
            Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

            Q_uu_inv = np.linalg.pinv(Q_uu + 1e-2 * np.eye(2))
            # Q_uu_inv = inv_stable(Q_uu)

            k[t] = -np.dot(Q_uu_inv, Q_u)
            K[t] = -np.dot(Q_uu_inv, Q_ux)

            V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
            V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))
        U_ = np.zeros((tN - 1, dof))
        x_ = x0.copy()  # 7a)

        for t in range(tN - 1):
            U_[t] = U[t] + k[t] + np.dot(K[t], x_ - X[t])
            x_ = simulate_dynamics_next(sim_env, x_, U_[t])

        Xnew, costnew = simulate(sim_env, x0, U_)
        # print(cost)
        # print(costnew)
        cost_list.append(costnew)

        X = np.copy(Xnew)  
        U = np.copy(U_)  
        oldcost = np.copy(cost)
        cost = np.copy(costnew)

        if abs(oldcost - cost) < EPS_CONVERGE:
            break
    # print(cost)
    return U, cost, cost_list


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    env = gym.make('TwoLinkArm-v0')
    sim_env = gym.make('TwoLinkArm-v0')

    env.seed(1024)
    sim_env.seed(1024)

    rewards = 0
    steps = 0
    cost_list, reward_list, state_list, action_list = [], [], [env.state], []
    done = False

    print('===============================================')
    print(f'Configurations:')
    print(f'tN = {tN}')
    print(f'eps_converge = {EPS_CONVERGE}')
    print(f'final_cost_weight = {FINAL_COST_WEIGHT}')
    print(f'MPC = {MPC}')
    print('===============================================\n')
    print('iLQR Test Starts...\n')

    total_start_time = time.time()

    while not done:
        start_time = time.time()
        U, cost, cost_list = calc_ilqr_input(env, sim_env, tN=tN)
        if not MPC:
            for action in U:
                nxt_state, reward, done, _ = env.step(action)

                cost_list.append(cost)
                reward_list.append(reward)
                state_list.append(nxt_state)
                action_list.append(action)

                rewards += reward
                steps += 1
                print('===============================================')
                print(f'Step: {steps}, Reward: {round(rewards, 4)}, Time: {round(time.time() - start_time, 2)}s')
                print(f'State: {nxt_state}')
                print(f'Action: {action}')
                print()
            break
        else:
            action = U[0]
            nxt_state, reward, done, _ = env.step(action)

            cost_list.append(cost)
            reward_list.append(reward)
            state_list.append(nxt_state)
            action_list.append(action)

            rewards += reward
            steps += 1
            print('===============================================')
            print(f'Step: {steps}, Reward: {round(rewards, 4)}, Time: {round(time.time() - start_time, 2)}s')
            print(f'State: {nxt_state}')
            print(f'Action: {action}')
            print()

    print(f'Total steps: {steps}, reward: {round(rewards, 4)}, total time: {round(time.time() - total_start_time, 2)}s')

    plt.figure(1)
    # cost_list = np.load('./iLQR_plot/cost_%s_%s_%s.npy' % (MPC, tN, FINAL_COST_WEIGHT))
    plt.plot(list(range(len(cost_list))), cost_list)
    plt.xlabel('iterations')
    plt.ylabel('total cost')
    plt.tight_layout()
    plt.savefig('./iLQR_plot/iLQR_cost_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))

    plt.figure(2)
    # reward_list = np.load('./iLQR_plot/reward_%s_%s_%s.npy' % (MPC, tN, FINAL_COST_WEIGHT))
    plt.plot(list(range(len(reward_list))), np.cumsum(reward_list))
    plt.xlabel('iterations')
    plt.ylabel('total reward')
    plt.tight_layout()
    plt.savefig('./iLQR_plot/iLQR_reward_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))

    states = np.asanyarray(state_list)
    # states = np.load('./iLQR_plot/states_%s_%s_%s.npy' % (MPC, tN, FINAL_COST_WEIGHT))
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
    plt.savefig('./iLQR_plot/iLQR_position_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))

    plt.figure(4)
    plt.plot(list(range(len(v1))), v1)
    plt.plot(list(range(len(v2))), v2)
    plt.xlabel('iterations')
    plt.ylabel('velocity')
    plt.legend(['velocity1', 'velocity2'])
    plt.savefig('./iLQR_plot/iLQR_velocity_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))

    actions = np.asanyarray(action_list)
    # actions = np.load('./iLQR_plot/actions_%s_%s_%s.npy' % (MPC, tN, FINAL_COST_WEIGHT))
    u1 = actions[:, 0]
    u2 = actions[:, 1]

    plt.figure(5)
    plt.plot(list(range(len(u1))), u1)
    plt.plot(list(range(len(u2))), u2)
    plt.xlabel('iterations')
    plt.ylabel('action')
    plt.legend(['action of dim1', 'action of dim2'])
    plt.savefig('./iLQR_plot/iLQR_action_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))

    plt.figure(6)
    plt.plot(list(range(len(cost_list))), cost_list)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.tight_layout()
    plt.savefig('./iLQR_plot/iLQR_single_cost_%s_%s_%s.png' % (MPC, tN, FINAL_COST_WEIGHT))
