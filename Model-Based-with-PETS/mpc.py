import os
import tensorflow as tf
import numpy as np
import gym
import copy
import time
import envs
import multiprocessing
from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1024)
random.seed(1024)

EPSILON = 1e-8


class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            # assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        # raise NotImplementedError
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.mu, self.sigma, self.goal = None, None, None

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        length = state.shape[0]
        # pusher_x, pusher_y = state[:, 0], state[:, 1]
        box_x, box_y = state[:, 2], state[:, 3]
        # goal_x, goal_y = np.tile(self.goal[0], (length, 1)), np.tile(self.goal[1], (length, 1))

        pusher = state[:, 0:2]
        box = state[:, 2:4]
        goal = np.tile(self.goal, (length, 1))
        goal_x, goal_y = goal[:, 0], goal[:, 1]

        d_box = np.linalg.norm(pusher - box, axis=1, ord=2)
        d_goal = np.linalg.norm(box - goal, axis=1, ord=2)


        # pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        # box_goal = np.array([goal_x - box_x, goal_y - box_y])
        # d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        # d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / (box_y + EPSILON) - goal_x / (goal_y + EPSILON))
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max([d_box - 0.4, np.zeros(len(d_box))], axis=0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # TODO: write your code here
        # states.shape = (200, 6, 8)
        # actions.shape = (200, 2)
        a = np.tile(actions, (6, 1, 1)).transpose(1, 0, 2)
        # return self.model.main(np.concatenate((states, np.tile(actions, (self.num_particles, 1))), axis=1),
        #                        None, train_mode=False)
        return self.model.main(np.concatenate((states, a), axis=2).reshape(-1, 10), None, train_mode=False).reshape(self.popsize, self.num_particles, self.state_dim)

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here

        # return [self.env.get_nxt_state(states[i], actions) for i in range(self.num_particles)]
        return np.array([[self.env.get_nxt_state(states[j][i], actions[j]) for i in range(self.num_particles)] for j in range(self.popsize)])

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5, save_fig=False):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        # TODO: write your code here

        s, a, s_ = [], [], []

        for i in range(len(obs_trajs)):
            s.extend(obs_trajs[i].tolist()[:-1])
            a.extend(acs_trajs[i].tolist())
            s_.extend(obs_trajs[i].tolist()[1:])
        assert len(s) == len(s_) == len(a)

        s = np.array(s)
        a = np.array(a)
        s_ = np.array(s_)
        length = len(s)
        print('Training Corpus: %d' % length)

        rmse_list, loss_list = [], []

        for e in range(epochs):
            # print('{} / {}'.format(e, epochs))
            indices = list(range(length))
            random.shuffle(indices)

            s = s[indices]
            a = a[indices]
            s_ = s_[indices]

            rmse_tmp, loss_tmp = [], []
            for batch in range(int(length/128) - 1):

                # tmp = self.model.sess.run(self.model.output, {self.model.input_state: s[batch*128:(batch+1)*128, :8],
                #                                               self.model.input_action: a[batch*128:(batch+1)*128],
                #                                               self.model.state_: s_[batch*128:(batch+1)*128, :8]})
                mse, loss = self.model.main(np.concatenate((s[batch*128:(batch+1)*128, :8], a[batch*128:(batch+1)*128]), axis=1),
                                s_[batch*128:(batch+1)*128, :8])
                rmse_tmp.append(np.sqrt(mse))
                loss_tmp.append(loss)
            rmse_list.append(np.mean(rmse_tmp))
            loss_list.append(np.mean(loss_tmp))
            # self.model.lr *= 0.9993

        if save_fig:
            plt.figure(1)
            plt.plot(range(len(rmse_list)), rmse_list)
            plt.xlabel('Epochs')
            plt.ylabel("RMSE")
            plt.savefig('rmse.png')

            plt.figure(2)
            plt.plot(range(len(loss_list)), loss_list)
            plt.xlabel('Epochs')
            plt.ylabel("Gauss Loss")
            plt.savefig('GLoss.png')
        return np.mean(rmse_list), np.mean(loss_list)

        # print('Training Completed!')
        # raise NotImplementedError

    def reset(self):
        # TODO: write your code here
        # raise NotImplementedError
        # self.mu = [np.zeros(2) for _ in range(self.plan_horizon)]
        # self.sigma = [np.eye(2) * 0.5 for _ in range(self.plan_horizon)]
        self.mu = np.zeros(self.plan_horizon * self.action_dim)
        self.sigma = np.ones(self.plan_horizon * self.action_dim) * 0.5
        # self.mu = np.zeros(self.plan_horizon * self.action_dim)
        # self.sigma = np.eye(self.plan_horizon * self.action_dim) * 0.5

    def act(self, state, t, noisy=False):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        # TODO: write your code here
        # raise NotImplementedError
        losses_list, actions_list = [], []
        for i in range(self.max_iters):
            # ts1 = time.time()
            # for m in range(self.popsize):
            #     actions, losses = self.generate_trajectory(state)
            #     losses_list.append(np.mean(losses))
            #     actions_list.append(actions.flatten())

            # input_pop = np.tile(state, (self.popsize, 1))
            # action_loss = list(map(self.generate_trajectory, input_pop))
            # for m in range(self.popsize):
            #     losses_list.append(np.mean(action_loss[m][1]))
            #     actions_list.append(action_loss[m][0].flatten())

            actions_list, losses_list = self.generate_trajectory(state)

            if not self.use_random_optimizer:
                elites_idx = np.argpartition(losses_list, self.num_elites)
                elites = np.array(actions_list)[elites_idx[:self.num_elites]]

                self.mu = np.mean(elites, axis=0)
                self.sigma = np.var(elites, axis=0)

        if self.use_random_optimizer:
            idx = np.argmin(losses_list)
            if self.use_mpc:
                action = np.array(actions_list[idx]).reshape(self.plan_horizon, self.action_dim)[0]
                pass
            else:
                action = np.array(actions_list[idx]).reshape(self.plan_horizon, self.action_dim)
            return action

        if self.use_mpc:
            action = copy.deepcopy(self.mu[[0, 1]])
            if noisy:
                self.mu = np.zeros(self.plan_horizon * self.action_dim)
            else:
                self.mu = np.append(self.mu[2:], [0, 0])
        else:
            action = copy.deepcopy(self.mu.reshape(self.plan_horizon, self.action_dim))
            self.mu *= 0.3  # 0.25-0.86, 0.3-0.92

        # self.sigma = np.eye(self.plan_horizon * self.action_dim) * 0.5
        self.sigma = np.ones(self.plan_horizon * self.action_dim) * 0.5

        return action

    def generate_trajectory(self, state):
        self.goal = state[-2:].copy()
        # actions = np.array(list(map(np.random.normal, self.mu, self.sigma))).reshape(self.plan_horizon, self.action_dim)
        actions = np.array([np.array(list(map(np.random.normal, self.mu, self.sigma))).reshape(self.plan_horizon, self.action_dim) for _ in range(self.popsize)])
        # actions.shape = (200, 5, 2)

        # state_matrix = np.zeros((self.num_particles, self.plan_horizon + 1, 8))
        state_matrix = np.zeros((self.popsize, self.num_particles, self.plan_horizon + 1, 8))
        # state_matrix.shape = (200, 6, 6, 8)

        # state_matrix[:, 0] = np.array([state[:8]] * self.num_particles)
        state_matrix[:, :, 0] = np.array([state[:8]] * self.num_particles)

        for t in range(self.plan_horizon):
            # action = actions[t]
            action = actions[:, t]  # (200, 2)

            # state_ = self.predict_next_state(state_matrix[:, t], action)
            state_ = self.predict_next_state(state_matrix[:, :, t], action)

            # state_matrix[:, t+1] = state_
            state_matrix[:, :, t+1] = state_

        # losses = list(map(self.obs_cost_fn, state_matrix))
        losses = np.array([np.mean(list(map(self.obs_cost_fn, state_matrix[i]))) for i in range(self.popsize)])
        # actions = actions.reshape(self.popsize, self.plan_horizon * self.action_dim).tolist()
        actions = actions.reshape(self.popsize, self.plan_horizon * self.action_dim).tolist()
        return actions, losses


if __name__ == '__main__':
    # env = gym.make('Pushing2D-v1')
    # states = [env.reset() for _ in range(10000)]
    # actions = [np.ones(2) for _ in range(10000)]
    # st1 = time.time()
    # [env.get_nxt_state(states[i], actions[i]) for i in range(10000)]
    # st2 = time.time()
    # list(map(env.get_nxt_state, states, actions))
    # st3 = time.time()
    # l = []
    # for i in range(10000):
    #     l.append(env.get_nxt_state(states[i], actions[i]))
    # st4 = time.time()
    # print(st2-st1)
    # print(st3-st2)
    # print(st4-st3)

    # states = [np.random.random((21, 10)) for _ in range(1000)]
    # actions = [np.random.random((20, 2)) for _ in range(1000)]
    #
    # data = {'state': [], 'action': [], 'nxt_state': []}

    # def p(data, states, actions, i):
    #     data['state'].append(states[i].tolist()[:-1])
    #     data['action'].append(actions[i].tolist())
    #     data['nxt_state'].append(states[i].tolist()[1:])
    #
    #
    # num_cores = multiprocessing.cpu_count()
    # inputs = range(len(states))
    # print('begin')
    # ts1 = time.time()
    # Parallel(n_jobs=num_cores)(delayed(p)(data, states, actions, i) for i in inputs)
    # ts2 = time.time()
    # print(ts2 - ts1)

    # ts3 = time.time()
    # for i in inputs:
    #     data['state'].extend(states[i].tolist()[:-1])
    #     data['action'].extend(actions[i].tolist())
    #     data['nxt_state'].extend(states[i].tolist()[1:])
    # ts4 = time.time()
    # data['state'] = [states[i].tolist()[:-1] for i in inputs]
    # data['action'] = [actions[i].tolist() for i in inputs]
    # data['nxt_state'] = [states[i].tolist()[1:] for i in inputs]
    # print(ts4 - ts3, time.time() - ts4)

    env = gym.make('Pushing2D-v1')
    #
    # states = [np.random.random(10) for _ in range(10000)]
    # actions = [np.random.random(2) for _ in range(10000)]
    #
    # ts1 = time.time()
    # for i in range(10000):
    #     env.get_nxt_state(states[i], actions[i])
    # ts2 = time.time()
    # [env.get_nxt_state(states[i], actions[i]) for i in range(10000)]
    # ts3 = time.time()
    # print(ts2-ts1, ts3-ts2)

    mpc = MPC(env, 5, None, 200, 20, 5, 1)
    state1 = env.reset()
    state2, _, _, _ = env.step([1, 0.5])
    s = np.vstack((state1, state2))
    mpc.goal = state1[-2:]
    print(mpc.obs_cost_fn(s))


