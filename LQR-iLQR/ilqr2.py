"""LQR, iLQR and MPC."""

# from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import gym
import deeprl_hw6


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
    x_new, _, _, _ = env.step(u)
    return x_new
    # return np.zeros(x.shape)


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
    A = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
        # x_perturb = x.copy()
        # x_perturb[i] += delta
        # A[:, i] = simulate_dynamics(env, x_perturb, u, dt)
        x_perturb = x.copy()
        x_perturb[i] -= delta
        x_next0 = simulate_dynamics_next(env, x_perturb, u)
        x_perturb[i] += 2 * delta
        x_next1 = simulate_dynamics_next(env, x_perturb, u)
        A[:, i] = (x_next1 - x_next0) / (2 * delta)
    return A


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
      The command to test. You will ned to perturb this.
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
    B = np.zeros((x.shape[0], u.shape[0]))
    for i in range(len(u)):
        # u_perturb = u.copy()
        # u_perturb[i] += delta
        # B[:, i] = simulate_dynamics(env, x, u_perturb, dt)
        u_perturb = u.copy()
        u_perturb[i] -= delta
        x_next0 = simulate_dynamics_next(env, x, u_perturb)
        u_perturb[i] += 2 * delta
        x_next1 = simulate_dynamics_next(env, x, u_perturb)
        B[:, i] = (x_next1 - x_next0) / (2 * delta)
    return B


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
    l = np.sum(u ** 2)
    l_x = np.zeros(x.shape[0])
    l_xx = np.zeros((x.shape[0], x.shape[0]))
    l_u = 2 * u
    l_uu = 2 * np.eye(u.shape[0])
    l_ux = np.zeros((u.shape[0], x.shape[0]))
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
    l = np.sum((x - env.goal) ** 2) * 1e6
    l_x = 1e6 * 2 * (x - env.goal)
    l_xx = 1e6 * 2 * np.eye(x.shape[0])
    return l, l_x, l_xx


def simulate(env, x0, U):
    env.state = x0.copy()
    X = np.zeros((U.shape[0] + 1, x0.shape[0]))
    X[0] = x0.copy()
    for i in range(U.shape[0]):
        x, _, _, _ = env.step(U[i])
        X[i + 1] = x.copy()
    return X


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
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
    ###
    # initialization
    x0 = env.state.copy()
    U = np.zeros((tN, 2))  # ???
    # U = np.random.uniform(-100, 100, (tN, 2)) 
    # U = 1000*np.ones((tN, 2)) 
    X = simulate(sim_env, x0, U)
    j_old = 0
    # run max_iter for convergence
    for iter in range(int(max_iter)):
        # calculate A, B, l
        A_N = np.zeros((tN, 4, 4))
        B_N = np.zeros((tN, 4, 2))
        l_N = np.zeros((tN, 1))
        l_x_N = np.zeros((tN, 4))
        l_xx_N = np.zeros((tN, 4, 4))
        l_u_N = np.zeros((tN, 2))
        l_uu_N = np.zeros((tN, 2, 2))
        l_ux_N = np.zeros((tN, 2, 4))
        for t in range(tN - 1):
            A_N[t] = approximate_A(sim_env, X[t], U[t])
            B_N[t] = approximate_B(sim_env, X[t], U[t])
            l_N[t], l_x_N[t], l_xx_N[t], l_u_N[t], l_uu_N[t], l_ux_N[t] = cost_inter(sim_env, X[t], U[t])
        l_N[-1], l_x_N[-1], l_xx_N[-1] = cost_final(sim_env, X[-1])
        # initialize V, K, k
        V = l_N[-1].copy()
        V_x = l_x_N[-1].copy()
        V_xx = l_xx_N[-1].copy()
        k = np.zeros((tN, 2))
        K = np.zeros((tN, 2, 4))
        # backward pass for tN
        for t in range(tN - 2, -1, -1):
            # Q
            Q_x = l_x_N[t] + np.dot(A_N[t].T, V_x)
            Q_u = l_u_N[t] + np.dot(B_N[t].T, V_x)
            Q_xx = l_xx_N[t] + np.dot(A_N[t].T, np.dot(V_xx, A_N[t]))
            Q_ux = l_ux_N[t] + np.dot(B_N[t].T, np.dot(V_xx, A_N[t]))
            Q_uu = l_uu_N[t] + np.dot(B_N[t].T, np.dot(V_xx, B_N[t]))
            Q_uu = Q_uu + np.eye(2) * 0.1
            # k&K
            k[t] = -np.dot(np.linalg.pinv(Q_uu), Q_u)
            K[t] = -np.dot(np.linalg.pinv(Q_uu), Q_ux)
            V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
            V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))
        # forward pass for tN
        # update x and u

        U_new = np.zeros((tN, 2))
        x_new = x0.copy()
        for t in range(tN - 1):
            U_new[t] = U[t] + k[t] + np.dot(K[t], x_new - X[t])
            x_new = simulate_dynamics_next(sim_env, x_new, U_new[t])
            # X[t+1] = x_new.copy()
        U = U_new
        X = simulate(sim_env, x0, U)

        # x_new = x0.copy()
        # for t in range(tN-1):
        #   u_new = U[t] + k[t] + np.dot(K[t], x_new-X[t])
        #   x_new = X[t+1] + np.dot(A_N[t], x_new-X[t+1]) + np.dot(B_N[t], u_new-U[t])
        #   U[t] = u_new
        #   X[t+1] = x_new

        if np.abs(np.sum(l_N) - j_old) < 0.0001:
            break
        j_old = np.sum(l_N)

        if iter % 100 == 0:
            print("oops")
    return U


if __name__ == '__main__':
    env = gym.make('TwoLinkArm-v0')
    sim_env = gym.make('TwoLinkArm-v0')

    env.seed(1024)
    sim_env.seed(1024)

    U = calc_ilqr_input(env, sim_env)
