import matplotlib.pyplot as plt
import numpy as np
import gym
import envs
import os.path as osp
import time
import tensorflow as tf
from collections import deque

from agent import Agent, RandomPolicy
from mpc import MPC
from model import PENN

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8

LOG_DIR = './data'

np.random.seed(1)


class ExperimentGTDynamics(object):
    def __init__(self, env_name='Pushing2D-v1', mpc_params=None):
        self.env = gym.make(env_name)
        self.env.seed(1024)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env, self.env.control_noise != 0.)
        # Does not need model
        self.warmup = False
        mpc_params['use_gt_dynamics'] = True
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name='Pushing2D-v1', num_nets=1, mpc_params=None):
        self.env = gym.make(env_name)
        self.env.seed(1024)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        mpc_params['use_gt_dynamics'] = False
        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)
        self.random_policy_no_mpc = RandomPolicy(len(self.env.action_space.sample()))

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            print('Test episode {}'.format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """ Train a single probabilistic model using a random policy """

        samples = []
        for i in range(num_episodes):
            samples.append(self.agent.sample(self.task_horizon, self.random_policy_no_mpc))

        obs = [sample["obs"] for sample in samples]
        ac = [sample["ac"] for sample in samples]
        rewards = [sample["rewards"] for sample in samples]

        self.cem_policy.train(
            obs,
            ac,
            rewards,
            epochs=num_epochs,
            save_fig=True
        )
        return samples

    def train(self, num_train_epochs, num_episodes_per_epoch, evaluation_interval, samples_):
        """ Jointly training the model and the policy """

        samples = deque(maxlen=100)
        samples.extend(samples_)
        print('Sample size:', len(samples))
        rmse_list, loss_list = [], []
        for i in range(num_train_epochs):
            print("####################################################################")
            print("Starting training epoch %d." % (i + 1))

            # samples = []
            for j in range(num_episodes_per_epoch):
                samples.append(
                    self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                )
            # print("Rewards obtained:", [sample["reward_sum"] for sample in samples])

            rmse, loss = self.cem_policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
                epochs=5,
                save_fig=False
            )
            rmse_list.append(rmse)
            loss_list.append(loss)

            if (i + 1) % evaluation_interval == 0:
                self.model.saver.save(self.model.sess, './model/v1/1.3/model_' + str(i) + '.ckpt')

                avg_return, avg_success = self.test(20, optimizer='cem')
                print('Test success CEM + MPC:', avg_success)
                avg_return, avg_success = self.test(20, optimizer='random')
                print('Test success Random + MPC:', avg_success)

        plt.figure(3)
        plt.plot(range(len(rmse_list)), rmse_list)
        plt.xlabel('Epochs')
        plt.ylabel("RMSE")
        plt.savefig('rmse_.png')

        plt.figure(4)
        plt.plot(range(len(loss_list)), loss_list)
        plt.xlabel('Epochs')
        plt.ylabel("Gauss Loss")
        plt.savefig('GLoss_.png')


def test_cem_gt_dynamics(num_episode=10):
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('CEM PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()

    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('CEM+MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()

    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('CEM PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()

    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('CEM+MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()

    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    print('Random PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()

    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    print('Random+MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    print()


def train_single_dynamics(num_test_episode=50):
    num_nets = 1
    num_episodes = 1000  # 1000
    num_epochs = 100  # 100
    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)

    # exp.model.model.load_weights('./model/model_good.h5')
    # exp.model.model.load_weights('./model/model.h5')
    # exp.model.saver.restore(exp.model.sess, "./model/v1/1.2/model.ckpt")

    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    # exp.model.saver.save(exp.model.sess, "./model/v1/1.2/model.ckpt")
    # exp.model.model.save_weights('./model/model.h5')

    avg_reward, avg_success = exp.test(num_test_episode, optimizer='cem')
    print('CEM+MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    avg_reward, avg_success = exp.test(num_test_episode, optimizer='random')
    print('Random+MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))


def train_pets():
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 50
    num_episodes_per_epoch = 1

    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    samples = exp.model_warmup(num_episodes=100, num_epochs=10)
    exp.train(num_train_epochs=num_epochs,
              num_episodes_per_epoch=num_episodes_per_epoch,
              evaluation_interval=evaluation_interval,
              samples_=samples)


if __name__ == "__main__":
    test_cem_gt_dynamics(50)
    train_single_dynamics(50)
    train_pets()
