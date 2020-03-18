import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import backend as K
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers
from keras import initializers
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from reinforce import Reinforce

np.random.seed(1234)
tf.set_random_seed(1234)

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, env, lr, critic_lr, n=20, actor_dims=[64, 24, 16], critic_dims=[20, 20, 20], render=False):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.nS = env.observation_space.shape[0]
        self.nA = env.action_space.n
        self.render = render
        self.n = n
        self.actor_lr = lr
        self.critic_lr = critic_lr
        self.actor_output = env.action_space.n
        self.output_dim = self.actor_output
        self.actor_input = env.observation_space.shape[0]
        self.model_save_path = './a2c/model/'
        self.model = Sequential()
        self.critic_model = Sequential()
        self.build_network_actor(self.actor_input, self.actor_output, actor_dims, self.actor_lr)
        self.build_network_critic(self.actor_input, critic_dims, self.critic_lr)

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def build_network_actor(self, input_dim, output_dim, hidden_dims, lr):
        self.model.add(Dense(hidden_dims[0], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu', input_shape=(input_dim,)))
        self.model.add(Dense(hidden_dims[1], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu'))
        self.model.add(Dense(hidden_dims[2], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu'))
        self.model.add(Dense(output_dim, kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.adam(lr))
        print("Actor...")
        print(self.model.get_config())
        # print(self.model.summary())
        print('lr: ', lr)
        print()

    def build_network_actor_atari(self, input_dim, output_dim, hidden_dims, lr):

        pass

    def build_network_critic(self, input_dim, hidden_dims, lr):
        self.critic_model.add(Dense(hidden_dims[0], activation='relu', input_shape=(input_dim,)))
        self.critic_model.add(Dense(hidden_dims[1], activation='relu'))
        self.critic_model.add(Dense(hidden_dims[2], activation='relu'))
        self.critic_model.add(Dense(1, activation='linear'))
        self.critic_model.compile(loss='MSE', optimizer=optimizers.adam(lr))
        print('Critic...')
        print(self.critic_model.get_config())
        # print(self.critic_model.summary())
        print('lr: ', lr)
        print()

    def save_model_wegihts(self, file_name, time_stamp):
        self.model.save_weights(self.model_save_path + file_name + '_' + str(time_stamp)+'.h5')

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states, actions, rewards_origin = self.generate_episode(env, self.render)
        rewards = np.array(rewards_origin) / 100

        Rts = []
        pred = self.critic_model.predict((np.array(states))).flatten()
        for t in range(len(rewards) - 1, -1, -1):
            if t + self.n >= len(rewards):
                v_end = 0
            else:
                v_end = pred[t + self.n]
            Rt = np.power(gamma, self.n) * v_end + sum([np.power(gamma, k) * rewards[t+k] if t + k < len(rewards) else 0
                                                        for k in range(0, self.n)])
            Rts.append(Rt)
        Rts.reverse()
        # print(Rts)
        v = np.array(Rts) - self.critic_model.predict(np.array(states)).flatten()

        self.model.train_on_batch(np.array(states), np.array(actions), sample_weight=v)
        self.critic_model.train_on_batch(np.array(states), Rts)
        return rewards_origin


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    print('v0')
    print(tf.__version__)
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    gamma = 0.99
    plot_path = './a2c/plot/'

    # Create the environment.
    # env = gym.make('Breakout-v0')
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')

    env.seed(1234)

    # TODO: Create the model.
    for n in [1, 100, 20, 50]:
        print(n, "Steps A2C...")
        if n == 1:
            agent = A2C(env, lr, critic_lr, n, critic_dims=[32, 32, 32])
        else:
            agent = A2C(env, lr, critic_lr, n)
        test_count, success = 0, 1
        test_mean, test_std, test_episode, test_rewards = [], [], [], []
        test = False
        reward_150 = False

        # TODO: Train the model using A2C and plot the learning curves.
        scores = deque(maxlen=100)
        for t in range(1, num_episodes):
            rewards = agent.train(env, gamma)
            reward = sum(rewards)
            scores.append(reward)
            print('episode {} | reward: {} | ave reward: {} | length : {}'.format(t, np.round(reward, 4),
                                                                                  np.round(np.mean(scores), 4),
                                                                                  len(rewards)))

            if np.mean(scores) >= 150 and not reward_150:
                print("Reward achieve 150...")
                print("Changing lr...")
                agent.save_model_wegihts(str(np.round(np.mean(scores), 4)), time.time())
                if n == 1:
                    RL_tmp = A2C(env, 5e-4, 3e-4, n, critic_dims=[32, 32, 32])
                else:
                    RL_tmp = A2C(env, 5e-4, 3e-4, n)
                RL_tmp.model.set_weights(agent.model.get_weights())
                RL_tmp.critic_model.set_weights((agent.critic_model.get_weights()))
                agent = RL_tmp
                reward_150 = True

            if t % 500 == 0:
                for _ in range(100):
                    reward = sum(agent.generate_episode(env)[2])
                    test_rewards.append(reward)
                test_rewards_mean = np.mean(test_rewards)
                test_rewards_std = np.std(test_rewards)
                test_mean.append(test_rewards_mean)
                test_std.append(test_rewards_std)
                print("Test | average reward {} | std {}".format(np.round(test_rewards_mean, 4),
                                                                 np.round(test_rewards_std, 4)))
                test_episode.append(t)
                test_rewards = []

            if np.mean(scores) >= 201 and not test:
                print("Training Completed!")
                print("Testing...")
                agent.save_model_wegihts(str(np.round(np.mean(scores), 4)), time.time())
                if n == 1:
                    RL_tmp = A2C(env, 0, 0, n, critic_dims=[32, 32, 32])
                else:
                    RL_tmp = A2C(env, 0, 0, n)
                RL_tmp.model.set_weights(agent.model.get_weights())
                RL_tmp.critic_model.set_weights((agent.critic_model.get_weights()))
                agent = RL_tmp
                test = True

            if test_count == 2000:
                print("Test Completed!")
                success += 1
                break

            if test:
                test_count += 1

        np.save(str(n)+'_mean.npy', np.array(test_mean))
        np.save(str(n)+'_std.npy', np.array(test_std))
        np.save(str(n)+'_episode.npy', np.array(test_episode))

        plt.figure(n)
        plt.errorbar(test_episode, test_mean, test_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel('Test reward mean')
        plt.savefig(plot_path + str(n) + '_mean.png')


if __name__ == '__main__':
    main(sys.argv)
