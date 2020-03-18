import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import initializers
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

np.random.seed(1225)
tf.set_random_seed(1225)


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, lr, gamma, render=False, model_save_path='./pg/model/', hidden_dims=[64, 32, 16]):
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.gamma = gamma
        self.render = render
        self.model = Sequential()
        self.build_network(self.input_dim, self.output_dim, hidden_dims, self.lr)
        self.model_save_path = model_save_path
        print("lr :", self.lr)

    def build_network(self, input_dim, output_dim, hidden_dims, lr):
        self.model.add(Dense(hidden_dims[0], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu', input_shape=(input_dim,)))
        self.model.add(Dense(hidden_dims[1], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu'))
        self.model.add(Dense(hidden_dims[2], kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='relu'))
        self.model.add(Dense(output_dim, kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.adam(lr))
        print(self.model.get_config())
        print(self.model.summary())

    def save_model_wegihts(self, file_name, time_stamp):
        self.model.save_weights(self.model_save_path + file_name + '_' + str(time_stamp)+'.h5')

    def load_model_wegihts(self, file_name):
        self.model.load_weights(self.model_save_path+file_name)

    def train(self, env, gamma=1.0, train_mode=True):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env, self.render)
        actions_onehot = np_utils.to_categorical(actions, num_classes=self.output_dim)
        Gt = [0]
        for i in range(len(rewards)-1, -1, -1):
            Gt.append(rewards[i] + gamma * Gt[-1])
        Gt.pop(0)
        Gt.reverse()
        Gt = (Gt - np.mean(Gt)) / (np.std(Gt))
        # Gt = (Gt - np.mean(Gt)) / np.var(Gt)
        y = np.array(Gt)[None].T * actions_onehot
        x = np.array(states)
        if train_mode:
            # self.model.fit(x, -y, batch_size=128, verbose=0)
            self.model.train_on_batch(x, np.array(actions), sample_weight=Gt)
            # self.model.train_on_batch(x, -y)
        return rewards

    def generate_episode(self, env, render=False, test=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        done = False
        state = env.reset()
        while not done:
            if render:
                env.render()
            states.append(state)
            if not test:
                action = np.random.choice(range(self.output_dim), p=np.squeeze(self.model.predict(np.array(state)[None])))
            else:
                action = np.argmax(self.model.predict(np.array(state)[None]))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            rewards.append(reward)
            actions.append(action)
        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=40000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="The learning rate.")
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
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    gamma = args.gamma
    render = args.render
    plot_path = './pg/plot/'
    success = 0

    for timestamp in range(5):
        test_mean = []
        test_std = []
        test_rewards = []
        test_episode = []

        # Create the environment.
        env = gym.make('Breakout-v0')
        # env = gym.make('LunarLander-v2')
        # env = gym.make('CartPole-v0')
        env.seed(1225 + timestamp)

        # TODO: Create the model.
        RL = Reinforce(env, lr, gamma, render)
        test = False
        test_count = 0
        reward_150 = False

        # TODO: Train the model using REINFORCE and plot the learning curve.
        scores = deque(maxlen=100)
        for t in range(1, num_episodes):
            rewards = RL.train(env, gamma=RL.gamma)
            reward = sum(rewards)
            scores.append(reward)
            print('episode {} | reward: {} | ave reward: {} | length : {}'.format(t, np.round(reward, 4),
                                                                                  np.round(np.mean(scores), 4),
                                                                                  len(rewards)))
            if t % 500 == 0:
                for _ in range(100):
                    reward = sum(RL.generate_episode(env)[2])
                    test_rewards.append(reward)
                test_rewards_mean = np.mean(test_rewards)
                test_rewards_std = np.std(test_rewards)
                test_mean.append(test_rewards_mean)
                test_std.append(test_rewards_std)
                print("Test | average reward {} | std {}".format(np.round(test_rewards_mean, 4), np.round(test_rewards_std, 4)))
                test_episode.append(t)
                test_rewards = []

            if np.mean(scores) >= 200 and not test:
                print("Training Completed!")
                print("Testing...")
                RL.save_model_wegihts(str(np.round(np.mean(scores), 4)), time.time())
                RL_tmp = Reinforce(env, 0, 0.99, render)
                RL_tmp.model.set_weights(RL.model.get_weights())
                RL = RL_tmp
                test = True

            if np.mean(scores) >= 150 and not reward_150:
                print("Reward achieve 150...")
                print("Changing lr...")
                RL.save_model_wegihts(str(np.round(np.mean(scores), 4)), time.time())
                RL_tmp = Reinforce(env, 1e-4, 0.99, render)
                RL_tmp.model.set_weights(RL.model.get_weights())
                RL = RL_tmp
                reward_150 = True

            if test_count == 2000:
                print("Test Completed!")
                success += 1
                break

            if test:
                test_count += 1

        plt.figure(timestamp)
        plt.errorbar(test_episode, test_mean, test_std, capsize=2)
        plt.xlabel('Episode')
        plt.ylabel('Test reward')
        plt.savefig(plot_path + str(timestamp) + 'mean.png')
        np.save(str(timestamp)+'_mean.npy', np.array(test_mean))
        np.save(str(timestamp)+'_std.npy', np.array(test_std))
        np.save(str(timestamp)+'_episode.npy', np.array(test_episode))

    print("Success {} times".format(success))


if __name__ == '__main__':
    main(sys.argv)
