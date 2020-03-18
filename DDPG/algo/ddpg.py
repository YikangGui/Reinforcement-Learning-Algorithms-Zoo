import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from operator import itemgetter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf.compat.v1.disable_eager_execution()

LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.98  # reward discount
TAU = 0.05  # soft replacement
BUFFER_SIZE = 100000  # 20000 can get converge
BATCH_SIZE = 128  # 1024 can get -20
STD = 2
SIGMA_DECAY = 0.9999
EPSILON = 0.1
ENV_HIGH = 1
RENDER = False

POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
NOISE_CLIP_DECAY = 1

tf.random.set_seed(1024)
random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1225)


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.sigma_decay = SIGMA_DECAY
        self.min_sigma = 0.1
        self.epsilon = epsilon
        self.epsilon_decay = 0.99995
        self.min_epsilon = 0.05

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            self.sigma = max(self.sigma * self.sigma_decay, self.min_sigma)
            # random = np.random.normal(self.mu, self.sigma)
            return np.random.normal(action, self.sigma)
        else:
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            # self.sigma = self.sigma * self.sigma_decay
            self.sigma = max(self.sigma * self.sigma_decay, self.min_sigma)
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG_agent(object):
    def __init__(self, a_dim, s_dim):
        self.buffer = np.zeros((BUFFER_SIZE, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'states')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'next_states')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'rewards')
        self.D = tf.compat.v1.placeholder(tf.float32, [None, 1], 'dones')

        self.A = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'actions')

        self.mu = self.actor(self.S)
        q = self.critic(self.S, self.mu)
        # q_test = self.critic(self.S, self.A, reuse=True)
        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self.actor(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self.critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        # a_loss = - tf.reduce_mean(q_test)  # maximize the q

        # self.atrain = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE_ACTOR).minimize(a_loss, var_list=a_params)
        self.grad = tf.gradients(q, a_params)
        # self.grad = tf.gradients(q_test, a_params)

        q_target = tf.stop_gradient(self.R + GAMMA * self.D * q_)
        td_error = tf.losses.mean_squared_error(q_target, q)
        # td_error = tf.losses.mean_squared_error(q_target, q_test)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE_CRITIC).minimize(td_error, var_list=c_params)

        with tf.control_dependencies(target_update):
            # TODO TEST
            self.atrain = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE_ACTOR).minimize(a_loss, var_list=a_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def actor(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.compat.v1.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 400, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.compat.v1.layers.dense(net, 64, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def critic(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.compat.v1.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, 400], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, 400], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, 400], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.compat.v1.layers.dense(net, 400, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.compat.v1.layers.dense(net, 400, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)

    def get_action(self, s):
        return self.sess.run(self.mu, {self.S: s[None]})[0]

    def train(self):
        indices = np.random.choice(BUFFER_SIZE, size=BATCH_SIZE)
        bt = self.buffer[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 2: -self.s_dim - 1]
        bs_ = bt[:, -self.s_dim - 1: -1]
        bd = bt[:, -1:]

        # self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.mu: ba, self.R: br, self.S_: bs_, self.D: bd})
        self.sess.run(self.atrain, {self.S: bs})

        # self.sess.run(self.ctrain, {self.S: bs, self.A: ba, self.R: br, self.S_: bs_, self.D: bd})
        return self.sess.run(self.grad, {self.S: bs})

    def store(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % BUFFER_SIZE
        self.buffer[index, :] = transition
        self.pointer += 1


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        self.env = env
        self.outfile = outfile_name
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        # self.lr = 3e-4
        self.std = STD
        self.gamma = GAMMA
        self.epsilon = EPSILON

        self.sess = tf.compat.v1.Session()
        # tf.keras.backend.set_session(self.sess)
        self.agent = DDPG_agent(action_dim, state_dim)

        self.soft_epsilon = EpsilonNormalActionNoise(0, self.std, self.epsilon)
        # raise NotImplementedError

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(state)
                a_t = self.agent.get_action(state)
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                state = new_s
                # s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            # if i < 9:
            #     plt.subplot(3, 3, i + 1)
            #     s_vec = np.array(s_vec)
            #     pusher_vec = s_vec[:, :2]
            #     puck_vec = s_vec[:, 2:4]
            #     goal_vec = s_vec[:, 4:]
            #     plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
            #     plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
            #     plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
            #     plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
            #     plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
            #                      color='g' if success else 'r')
            #     plt.xlim([-1, 6])
            #     plt.ylim([-1, 6])
            #     if i == 0:
            #         plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
            #     if i == 8:
            #         # Comment out the line below to disable plotting.
            #         # pass
            #         plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        test_episode = []
        test_mean = []
        test_std = []
        episode_grad = []
        success_count = 0
        if hindsight:
            eval_episode = 100
        else:
            eval_episode = 10
        print('Initializing Buffer...')
        while self.agent.pointer <= BUFFER_SIZE:
            state = self.env.reset()
            tmp_states = [state]
            tmp_actions = []
            use_HER = False
            done = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(state)
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)

                self.agent.store(state, action, reward, next_state, not done)

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                if hindsight and use_HER:
                    self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

                state = next_state

        for i in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            tmp_states = [state]
            tmp_actions = []
            grad_metric_list = [0]
            use_HER = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(state)
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                self.agent.store(state, action, reward, next_state, not done)

                if self.agent.pointer > BUFFER_SIZE:
                    grad = self.agent.train()
                    grad_metric = abs(np.mean([np.mean(i) for i in grad]))
                    grad_metric_list.append(grad_metric)

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                state = next_state
                step += 1

            if hindsight and use_HER:
                self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

            # Logging
            print("Episode %d: Total reward = %d | sigma = %f | mean grad = %f" % (
            i, total_reward, self.soft_epsilon.sigma, np.mean(grad_metric_list)))
            # episode_grad.extend(grad_metric_list[1:])
            episode_grad.append(np.mean(grad_metric_list[1:]))

            if i % 100 == 0:
                successes, mean_rewards, std_rewards = self.evaluate(eval_episode)
                if successes >= 0.95:
                    success_count += 1
                else:
                    success_count = 0
                test_episode.append(i)
                test_mean.append(mean_rewards)
                test_std.append(std_rewards)
                print('Evaluation: success = %.2f; return = %.2f; success count = %d' % (
                successes, mean_rewards, success_count))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f, %.2f\n" % (successes, mean_rewards, std_rewards))
                if success_count > 250:
                    break

        np.save(str(hindsight) + '_mean.npy', np.array(test_mean))
        np.save(str(hindsight) + '_std.npy', np.array(test_std))
        np.save(str(hindsight) + '_episode.npy', np.array(test_episode))
        np.save(str(hindsight) + '_grad.npy', np.array(episode_grad))

        plt.figure(1)
        plt.errorbar(test_episode, test_mean, test_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel('Test reward mean')
        plt.savefig(str(hindsight) + '_mean.png')

        plt.figure(2)
        plt.plot(range(len(episode_grad)), episode_grad)
        plt.savefig(str(hindsight) + '_grad.png')

    def add_hindsight_replay_experience(self, states, actions, buffer):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of actions.
        """
        # raise NotImplementedError
        assert len(states) == len(actions) + 1
        self.env.apply_hindsight(states, actions, buffer)


class TD3_agent(object):
    def __init__(self, a_dim, s_dim):
        self.buffer = np.zeros((20000, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.step = 0
        self.policy_freq = 5
        self.policy_noise = 0.2 * ENV_HIGH
        self.noise_clip = 0.5 * ENV_HIGH
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'states')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 'next_states')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'rewards')
        self.D = tf.compat.v1.placeholder(tf.float32, [None, 1], 'dones')
        # self.A = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'actions')
        self.e = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'noise')

        self.mu = self.actor(self.S)
        # current_q1, current_q2 = self.critic_1(self.S, self.A), self.critic_2(self.S, self.A)
        current_q1, current_q2 = self.critic_1(self.S, self.mu), self.critic_2(self.S, self.mu)
        # current_q1, current_q2 = self.critic(self.S, self.mu)

        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c1_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Critic_1')
        c2_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Critic_2')

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c1_params + c2_params)]

        mu_ = self.actor(self.S_, reuse=True, custom_getter=ema_getter)
        # mu_ = self.actor(self.S_, reuse=True)

        max_action = ENV_HIGH
        A_ = tf.clip_by_value(mu_ + self.e, clip_value_min=-max_action, clip_value_max=max_action)

        target_q1, target_q2 = self.critic_1(self.S_, A_, reuse=True, custom_getter=ema_getter), self.critic_2(self.S_,
                                                                                                               A_,
                                                                                                               reuse=True,
                                                                                                               custom_getter=ema_getter)
        # target_q1, target_q2 = self.critic(self.S_, A_, reuse=True, custom_getter=ema_getter)

        target_q = tf.minimum(target_q1, target_q2)

        q = self.critic_1(self.S, self.mu, reuse=True)
        # q, _ = self.critic(self.S, self.mu, reuse=True)
        a_loss = - tf.reduce_mean(q)

        target_q = tf.stop_gradient(self.R + GAMMA * self.D * target_q)
        td_error = 0.5 * (tf.losses.mean_squared_error(target_q, current_q1) + tf.losses.mean_squared_error(target_q,
                                                                                                            current_q2))
        self.ctrain = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(td_error, var_list=c1_params + c2_params)
        self.c_grad = tf.gradients(td_error, c1_params + c2_params)

        with tf.control_dependencies(target_update):
            self.atrain = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(a_loss, var_list=a_params)
        self.a_grad = tf.gradients(a_loss, a_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def actor(self, s, reuse=None, custom_getter=None):
        # trainable = True if reuse is None else False
        trainable = True
        with tf.compat.v1.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.compat.v1.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 256, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net = tf.compat.v1.layers.dense(net, 64, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def critic_1(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # trainable = True
        with tf.compat.v1.variable_scope('Critic_1', reuse=reuse, custom_getter=custom_getter):
            C1_w1_s = tf.compat.v1.get_variable('C1_w1_s', [self.s_dim, 400], trainable=trainable)
            C1_w1_a = tf.compat.v1.get_variable('C1_w1_a', [self.a_dim, 400], trainable=trainable)
            C1_b1 = tf.compat.v1.get_variable('C1_b1', [1, 400], trainable=trainable)
            C1_net = tf.nn.relu(tf.matmul(s, C1_w1_s) + tf.matmul(a, C1_w1_a) + C1_b1)
            # concat = tf.concat([s, a], 1)
            # C1_w1 = tf.compat.v1.get_variable('C1_w1', [self.s_dim + self.a_dim, 256], trainable=trainable)
            # C1_b1 = tf.compat.v1.get_variable('C1_b1', [1, 256], trainable=trainable)
            # C1_net = tf.nn.relu(tf.matmul(concat, C1_w1) + C1_b1)
            C1_net = tf.compat.v1.layers.dense(C1_net, 256, activation=tf.nn.relu, name='C1_l2', trainable=trainable)
        return tf.compat.v1.layers.dense(C1_net, 1, trainable=trainable)

    def critic_2(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # trainable = True
        with tf.compat.v1.variable_scope('Critic_2', reuse=reuse, custom_getter=custom_getter):
            C2_w1_s = tf.compat.v1.get_variable('C2_w1_s', [self.s_dim, 400], trainable=trainable)
            C2_w1_a = tf.compat.v1.get_variable('C2_w1_a', [self.a_dim, 400], trainable=trainable)
            C2_b1 = tf.compat.v1.get_variable('C2_b1', [1, 400], trainable=trainable)
            C2_net = tf.nn.relu(tf.matmul(s, C2_w1_s) + tf.matmul(a, C2_w1_a) + C2_b1)
            # concat = tf.concat([s, a], 1)
            # C2_w1 = tf.compat.v1.get_variable('C2_w1', [self.s_dim + self.a_dim, 256], trainable=trainable)
            # C2_b1 = tf.compat.v1.get_variable('C2_b1', [1, 256], trainable=trainable)
            # C2_net = tf.nn.relu(tf.matmul(concat, C2_w1) + C2_b1)
            C2_net = tf.compat.v1.layers.dense(C2_net, 256, activation=tf.nn.relu, name='C2_l2', trainable=trainable)
        return tf.compat.v1.layers.dense(C2_net, 1, trainable=trainable)

    def critic(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # trainable = True
        with tf.compat.v1.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            # C1_w1_s = tf.compat.v1.get_variable('C1_w1_s', [self.s_dim, 400], trainable=trainable)
            # C1_w1_a = tf.compat.v1.get_variable('C1_w1_a', [self.a_dim, 400], trainable=trainable)
            # C1_b1 = tf.compat.v1.get_variable('C1_b1', [1, 400], trainable=trainable)
            # C1_net = tf.nn.relu(tf.matmul(s, C1_w1_s) + tf.matmul(a, C1_w1_a) + C1_b1)
            concat_1 = tf.concat([s, a], 1)
            C1_w1 = tf.compat.v1.get_variable('C1_w1', [self.s_dim + self.a_dim, 256], trainable=trainable)
            C1_b1 = tf.compat.v1.get_variable('C1_b1', [1, 256], trainable=trainable)
            C1_net = tf.nn.relu(tf.matmul(concat_1, C1_w1) + C1_b1)
            C1_net = tf.compat.v1.layers.dense(C1_net, 256, activation=tf.nn.relu, name='C1_l2', trainable=trainable)
            # C2_w1_s = tf.compat.v1.get_variable('C2_w1_s', [self.s_dim, 400], trainable=trainable)
            # C2_w1_a = tf.compat.v1.get_variable('C2_w1_a', [self.a_dim, 400], trainable=trainable)
            # C2_b1 = tf.compat.v1.get_variable('C2_b1', [1, 400], trainable=trainable)
            # C2_net = tf.nn.relu(tf.matmul(s, C2_w1_s) + tf.matmul(a, C2_w1_a) + C2_b1)
            # concat_2 = tf.concat([s, a], 1)
            C2_w1 = tf.compat.v1.get_variable('C2_w1', [self.s_dim + self.a_dim, 256], trainable=trainable)
            C2_b1 = tf.compat.v1.get_variable('C2_b1', [1, 256], trainable=trainable)
            C2_net = tf.nn.relu(tf.matmul(concat_1, C2_w1) + C2_b1)
            C2_net = tf.compat.v1.layers.dense(C2_net, 256, activation=tf.nn.relu, name='C2_l2', trainable=trainable)
        return tf.compat.v1.layers.dense(C2_net, 1, trainable=trainable), tf.compat.v1.layers.dense(C1_net, 1,
                                                                                                    trainable=trainable)

    def train(self):
        indices = np.random.choice(BUFFER_SIZE, size=BATCH_SIZE)
        bt = self.buffer[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 2: -self.s_dim - 1]
        bs_ = bt[:, -self.s_dim - 1: -1]
        bd = bt[:, -1:]

        # print(bt[:10])

        e = np.random.normal(size=(BATCH_SIZE, self.a_dim), loc=0, scale=self.policy_noise)
        e = np.clip(e, -self.noise_clip, self.noise_clip)
        self.noise_clip *= 0.9999
        # print(e)

        self.sess.run(self.ctrain, {self.S: bs, self.mu: ba, self.R: br, self.S_: bs_, self.D: bd, self.e: e})
        if self.step % self.policy_freq == 0:
            self.sess.run(self.atrain, {self.S: bs})
        self.step += 1
        return self.sess.run(self.a_grad, {self.S: bs}), self.sess.run(self.c_grad,
                                                                       {self.S: bs, self.mu: ba, self.R: br,
                                                                        self.S_: bs_, self.D: bd, self.e: e})

    def get_action(self, s):
        return self.sess.run(self.mu, {self.S: s[None]})[0]

    def store(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % BUFFER_SIZE
        self.buffer[index, :] = transition
        self.pointer += 1


class TD3(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        self.env = env
        self.outfile = outfile_name
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        # self.lr = 3e-4
        self.std = 2
        self.gamma = GAMMA
        self.epsilon = EPSILON

        self.sess = tf.compat.v1.Session()
        self.agent = TD3_agent(action_dim, state_dim)

        self.soft_epsilon = EpsilonNormalActionNoise(0, self.std, self.epsilon)
        # raise NotImplementedError

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(state)
                a_t = self.agent.get_action(state)
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                state = new_s
                # s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i + 1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        test_episode = []
        test_mean = []
        test_std = []
        episode_grad = []
        success_count = 0
        if hindsight:
            eval_episode = 100
        else:
            eval_episode = 10
        print('Initializing Buffer...')
        while self.agent.pointer <= BUFFER_SIZE:
            state = self.env.reset()
            tmp_states = [state]
            tmp_actions = []
            use_HER = False
            done = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(state)
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)

                self.agent.store(state, action, reward, next_state, not done)

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                if hindsight and use_HER:
                    self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

                state = next_state

        for i in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            tmp_states = [state]
            tmp_actions = []
            grad_metric_list = [0]
            use_HER = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(state)
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                self.agent.store(state, action, reward, next_state, not done)

                if self.agent.pointer > BUFFER_SIZE:
                    a_grad, c_grad = self.agent.train()
                    # print(a_grad)
                    # print()
                    # print(c_grad)

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                state = next_state
                step += 1

            if hindsight and use_HER:
                self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

            # Logging
            print("Episode %d: Total reward = %d | sigma = %f" % (
                i, total_reward, self.soft_epsilon.sigma))
            episode_grad.extend(grad_metric_list[1:])

            if i % 100 == 0:
                successes, mean_rewards, std_rewards = self.evaluate(eval_episode)
                if successes >= 0.95:
                    success_count += 1
                else:
                    success_count = 0
                test_episode.append(i)
                test_mean.append(mean_rewards)
                test_std.append(std_rewards)
                print('Evaluation: success = %.2f; return = %.2f; success count = %d' % (
                    successes, mean_rewards, success_count))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f, %.2f\n" % (successes, mean_rewards, std_rewards))
                if success_count > 15:
                    break

        np.save(str(hindsight) + '_mean.npy', np.array(test_mean))
        np.save(str(hindsight) + '_std.npy', np.array(test_std))
        np.save(str(hindsight) + '_episode.npy', np.array(test_episode))
        np.save(str(hindsight) + '_grad.npy', np.array(episode_grad))

        plt.figure(1)
        plt.errorbar(test_episode, test_mean, test_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel('Test reward mean')
        plt.savefig(str(hindsight) + '_mean.png')

        plt.figure(2)
        plt.plot(range(len(episode_grad)), episode_grad)
        plt.savefig(str(hindsight) + '_grad.png')

    def add_hindsight_replay_experience(self, states, actions, buffer):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of actions.
        """
        # raise NotImplementedError
        assert len(states) == len(actions) + 1
        self.env.apply_hindsight(states, actions, buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1_s = nn.Linear(state_dim, 400)
        self.l1_a = nn.Linear(action_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 1)

        # Q2 architecture
        self.l4_s = nn.Linear(state_dim, 400)
        self.l4_a = nn.Linear(action_dim, 400)
        self.l5 = nn.Linear(400, 400)
        self.l6 = nn.Linear(400, 1)

    def forward(self, state, action):
        # sa = torch.cat([state, action], 1)
        q1_s = self.l1_s(state)
        q1_a = self.l1_a(action)
        q1 = torch.add(q1_s, q1_a)

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # sa = torch.cat([state, action], 1)
        q2_s = self.l4_s(state)
        q2_a = self.l4_a(action)
        q2 = torch.add(q2_s, q2_a)

        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        q1_s = self.l1_s(state)
        q1_a = self.l1_a(action)
        q1 = torch.add(q1_s, q1_a)

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_Torch_agent(object):

    def __init__(self, action_dim, state_dim):

        self.max_action = ENV_HIGH
        self.discount = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOISE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.noise_clip_decay = NOISE_CLIP_DECAY
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)

        self.buffer = np.zeros((20000, state_dim * 2 + action_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.step = 0

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def store(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % BUFFER_SIZE
        self.buffer[index, :] = transition
        self.pointer += 1

    def train(self):
        self.step += 1

        indices = np.random.choice(BUFFER_SIZE, size=BATCH_SIZE)

        bt = self.buffer[indices, :]
        state = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        action = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(device)
        reward = torch.FloatTensor(bt[:, -self.s_dim - 2: -self.s_dim - 1]).to(device)
        next_state = torch.FloatTensor(bt[:, -self.s_dim - 1: -1]).to(device)
        done = torch.FloatTensor(bt[:, -1:]).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = 1 / 2 * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.step % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.noise_clip = max(self.noise_clip * self.noise_clip_decay, 0.05)


class TD3_Torch(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        self.env = env
        self.outfile = outfile_name
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.std = 2
        self.gamma = GAMMA
        self.epsilon = EPSILON

        self.agent = TD3_Torch_agent(action_dim, state_dim)

        self.soft_epsilon = EpsilonNormalActionNoise(0, self.std, self.epsilon)
        # raise NotImplementedError

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.agent.get_action(s_t)
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                # state = new_s
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i + 1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        test_episode = []
        test_mean = []
        test_std = []
        episode_grad = []
        success_count = 0
        if hindsight:
            eval_episode = 100
        else:
            eval_episode = 10

        print('Initializing Buffer...')
        while self.agent.pointer <= BUFFER_SIZE:
            state = self.env.reset()
            tmp_states = [state]
            tmp_actions = []
            use_HER = False
            done = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(np.array(state))
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)

                self.agent.store(state, action, reward, next_state, not done)

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                if hindsight and use_HER:
                    self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

                state = next_state

        for i in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            tmp_states = [state]
            tmp_actions = []
            grad_metric_list = [0]
            use_HER = False

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.agent.get_action(np.array(state))
                action = self.soft_epsilon(action)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                self.agent.store(state, action, reward, next_state, not done)

                if self.agent.pointer > BUFFER_SIZE:
                    self.agent.train()

                if np.linalg.norm(state[2:4] - next_state[2:4]) > 0:
                    use_HER = True

                if hindsight:
                    tmp_actions.append(action)
                    tmp_states.append(next_state)

                state = next_state
                step += 1

            if hindsight and use_HER:
                self.add_hindsight_replay_experience(tmp_states, tmp_actions, self.agent)

            # Logging
            print("Episode %d: Total reward = %d | sigma = %f" % (
                i, total_reward, self.soft_epsilon.sigma))
            episode_grad.extend(grad_metric_list[1:])

            if i % 100 == 0:
                successes, mean_rewards, std_rewards = self.evaluate(eval_episode)
                if successes >= 0.95:
                    success_count += 1
                else:
                    success_count = 0
                test_episode.append(i)
                test_mean.append(mean_rewards)
                test_std.append(std_rewards)
                print('Evaluation: success = %.2f; return = %.2f; success count = %d' % (
                    successes, mean_rewards, success_count))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f, %.2f\n" % (successes, mean_rewards, std_rewards))
                if success_count > 15:
                    break

        np.save(str(hindsight) + '_mean_TD3.npy', np.array(test_mean))
        np.save(str(hindsight) + '_std_TD3.npy', np.array(test_std))
        np.save(str(hindsight) + '_episode_TD3.npy', np.array(test_episode))
        np.save(str(hindsight) + '_grad_TD3.npy', np.array(episode_grad))

        plt.figure(1)
        plt.errorbar(test_episode, test_mean, test_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel('Test reward mean')
        plt.savefig(str(hindsight) + '_mean_TD3.png')

        plt.figure(2)
        plt.plot(range(len(episode_grad)), episode_grad)
        plt.savefig(str(hindsight) + '_grad_TD3.png')

    def add_hindsight_replay_experience(self, states, actions, buffer):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of actions.
        """
        # raise NotImplementedError
        assert len(states) == len(actions) + 1
        self.env.apply_hindsight(states, actions, buffer)
