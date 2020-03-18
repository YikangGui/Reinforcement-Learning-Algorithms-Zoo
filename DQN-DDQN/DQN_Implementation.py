#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr=0.001):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.lr = lr
        self.env = env
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.shape[0]
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(self.nS,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.nA, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=lr))
        print('lerning rate:', lr)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.model.save_weights(suffix)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        # e.g.: torch.save(self.model.state_dict(), model_file)
        self.model.load_model(model_file)

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        # e.g.: self.model.load_state_dict(torch.load(model_file))
        self.model.load_weights(weight_file)
        print("Loaded model with \'%s\'!" % weight_file)


class Replay_Memory():

    def __init__(self, burn_in, memory_size=600000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # Hint: you might find this useful:
        # 		collections.deque(maxlen=memory_size)
        self.memory = deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        mini_batch = random.sample(self.memory, batch_size)
        return mini_batch

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.extend(transition.tolist())


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, lr=None, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.render = render
        self.epsilon = 0.5
        self.epsilon_decay_linear = (0.5 - 0.05) / 100000
        self.epsilon_decay = 0.99995
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.shape[0]
        self.batch_size = 64
        self.min_epsilon = 0.001
        self.evaluation_frequency = 20

        if environment_name == 'MountainCar-v0':
            self.gamma = 1
            self.lr = 0.0005
            self.epsilon = 1
            self.epsilon_decay = 0.9
            self.episodes = 1000
            self.expect_reward = -110

        elif environment_name == 'CartPole-v0':
            self.gamma = 0.99
            self.lr = 0.0001
            self.epsilon = 1
            self.epsilon_decay = 0.99
            self.episodes = 2000
            self.expect_reward = 199
        else:
            raise ValueError('Wrong Environment Name!')

        self.local = QNetwork(self.env, self.lr)
        self.target = QNetwork(self.env)
        self.experiency_memory = Replay_Memory(self.batch_size)
        self.burn_in_memory()
        self.targets_f = []
        self.td_error = 0

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand() < self.epsilon:
            return random.randrange(self.nA)
        return np.argmax(q_values[0])

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values[0])

    def act(self, state):
        return self.epsilon_greedy_policy(self.local.model.predict(state[None]))

    def train(self, mode, train=True):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        mini_batches = self.experiency_memory.sample_batch(self.batch_size)

        states = []
        next_states = []
        rewards = []
        actions = []
        dones = []
        if train:
            for index, (state, action, next_state, reward, done) in enumerate(mini_batches):
                states.append(state)
                next_states.append(next_state)
                rewards.append(reward)
                actions.append(action)
                if not done:
                    dones.append(index)

            if mode == 'DQN':
                targets = np.array(reward) + self.gamma * np.amax(self.target.model.predict(np.array(next_states)), axis=1)
                self.targets_f = self.local.model.predict(np.array(states))
                v_st = copy.deepcopy(self.targets_f)
                for i in range(self.batch_size):
                    self.targets_f[i][actions[i]] = rewards[i]
                for done in dones:
                    self.targets_f[done][actions[done]] = targets[done]
                self.td_error = np.sum(abs(self.targets_f - v_st)) / v_st.shape[0]
                history = self.local.model.fit(np.array(states), self.targets_f, epochs=1, verbose=0)
                return history.history['loss'][0], self.td_error

            elif mode == 'DDQN':
                action_by_local = np.argmax(self.local.model.predict(np.array(next_states)), axis=1)
                targets = self.target.model.predict(np.array(next_states))
                targets = np.array(reward) + self.gamma * np.array([x[y] for x, y in zip(targets, action_by_local)])
                self.targets_f = self.local.model.predict(np.array(states))
                v_st = copy.deepcopy(self.targets_f)
                for i in range(self.batch_size):
                    self.targets_f[i][actions[i]] = rewards[i]
                for done in dones:
                    self.targets_f[done][actions[done]] = targets[done]
                self.td_error = np.sum(abs(self.targets_f - v_st)) / v_st.shape[0]
                history = self.local.model.fit(np.array(states), self.targets_f, epochs=1, verbose=0)
                return history.history['loss'][0], self.td_error
            else:
                raise ValueError("Wrong Mode!!!")
        else:
            return -1, -1

    def update_target_model(self):
        self.target.model.set_weights(self.local.model.get_weights())

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        if model_file != None:
            self.target.model.load_weights(model_file)

        rewards = []
        for _ in range(self.evaluation_frequency):
            state = self.env.reset()
            done = False
            while not done:
                action = self.greedy_policy(self.target.model.predict(np.array(state)[None]))
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                rewards.append(reward)
        return sum(rewards) / self.evaluation_frequency

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        while len(self.experiency_memory.memory) < self.experiency_memory.burn_in:
            state = self.env.reset()
            action = np.argmax(self.local.model.predict(state[None]))
            next_state, reward, done, _ = self.env.step(action)
            self.experiency_memory.memory.append((state, action, next_state, reward, done))

    def remember(self, memory):
        self.experiency_memory.memory.append(memory)


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
	# Usage:
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.greedy_policy(agent.target.model.predict(np.array(state)[None]))
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    # env_names = ['CartPole-v0', 'MountainCar-v0']
    env_names = ['MountainCar-v0']

    modes = ['DQN', 'DDQN']
    # modes = ['DQN']

    # model_weight = './model/MountainCar-v0/MountainCar-v0_episode_2900.h5'
    # agent.local.load_model_weights(model_weight)
    # agent.target.load_model_weights(model_weight)
    # episode_start = int(model_weight.split('_')[-1].split('.')[0])
    # agent.batch_size = 32
    # agent.lr = 0.0001
    # agent.epsilon = 0.1

    plt_num = 0
    for mode in modes:
        for env_name in env_names:
            env = gym.make(env_name)
            agent = DQN_Agent(env_name)
            episode_start = 0
            train_flag = True
            reassign_lr = True

            reward_window = deque(maxlen=100)
            reward_train, ave_reward_train, td_error_train, mean_test_reward_train, mean_test_reward_x = [], [], [], [], []
            best_reward = -200
            evaluation_episode = 0
            evaluation_flag = False
            timestamp = time.time()
            save_path = "./model/%s/%s/%s" % (env_name, mode, str(timestamp))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for episode in range(episode_start, 10000):
                state = env.reset()
                done = False
                rewards = []
                losses, td_errors = [], []
                while not done:
                    # if episode % 50 == 0:
                    # 	env.render()
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.remember((state, action, next_state, reward, done))
                    state = next_state
                    loss, td_error = agent.train(mode, train_flag)
                    rewards.append(reward)
                    losses.append(loss)
                    td_errors.append(td_error)

                    if done:
                        reward = sum(rewards)
                        agent.update_target_model()
                        reward_window.append(reward)
                        reward_train.append(reward)
                        ave_reward_train.append(np.mean(reward_window))
                        td_error_train.append(np.mean(td_errors))
                        if train_flag:
                            print('episode: {} | reward: {} | average reward:{:.3f} | mean q: {:.3f} | epsilon: {:.3f} | '
                              'TD error: {:.3f} | loss: {:.3f}'.format(episode, reward, np.mean(reward_window), np.mean(agent.targets_f),
                                                    agent.epsilon, np.mean(td_errors), np.mean(losses)))
                        else:
                            print('episode: {} | reward: {} | average reward:{:.3f} '.format(episode, reward, np.mean(reward_window)))
                        if reward > best_reward:
                            best_reward = reward
                            agent.target.save_model_weights(
                                './model/%s/%s/%s/%s_best_%d_episode_%d.h5' % (env_name, mode, str(timestamp), env_name, reward, episode))
                            break
                        if episode % 50 == 0:
                            agent.target.save_model_weights('./model/%s/%s/%s/%s_episode_%d.h5' % (env_name, mode, str(timestamp), env_name, episode))
                        break
                agent.epsilon = max(agent.epsilon_decay * agent.epsilon, agent.min_epsilon)

                if episode % 100 == 0:
                    mean_test_reward = agent.test()
                    mean_test_reward_train.append(mean_test_reward)
                    mean_test_reward_x.append(episode)
                    print("test mean reward: {} at episode {}".format(mean_test_reward, episode))

                # if episode % int(agent.episodes/3) == 0:
                #     test_video(agent, env, episode)

                if evaluation_flag or np.mean(reward_window) > agent.expect_reward:
                    if reassign_lr:
                        print('Reassigning lr!')
                        lr = 1e-10
                        train_flag = False
                        agent.local = QNetwork(env, lr)
                        agent.local.model.set_weights(agent.target.model.get_weights())
                        reassign_lr = False
                    if evaluation_episode >= agent.episodes:
                        print("Complete training and testing!")
                        break
                    evaluation_flag = True
                    evaluation_episode += 1

            plt.figure(plt_num)
            plt.plot(mean_test_reward_x, mean_test_reward_train)
            plt.savefig('average test reward_%s_%s.png' % (mode, env_name))
            plt.show()
            plt_num += 1

            plt.figure(plt_num)
            plt.plot(range(episode + 1), td_error_train)
            plt.savefig('td_error_%s_%s.png' % (mode, env_name))
            plt.show()
            plt_num += 1


if __name__ == '__main__':
    main(sys.argv)

