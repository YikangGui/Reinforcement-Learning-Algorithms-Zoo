import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from util import ZFilter
import gym
import envs
import random

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1024)
np.random.seed(1024)
random.seed(1024)


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.compat.v1.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        # tf.compat.v1.keras.backend.set_session(self.sess)

        # Log variance bounds
        # self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        # self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        self.max_logvar = tf.constant(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.constant(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # TODO write your code here
        # Create and initialize your model
        self.env = gym.make('Pushing2D-v1')
        self.env.seed(1024)

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.model = self.create_network()
        self.input_state = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.input_action = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim], 'action')
        self.state_ = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim], 'next_state')

        self.input = tf.concat([self.input_state, self.input_action], axis=1)

        model_weights = self.model.trainable_variables

        self.output = self.model(inputs=self.input)

        self.mean, logvar = self.get_output(self.output)
        self.var_ = tf.exp(logvar)
        # self.s_output = tf.stop_gradient(tf.compat.v1.random.normal(shape=[6, 8], mean=self.mean, stddev=tf.sqrt(self.var_)))
        # self.s_output = tf.compat.v1.random.normal(shape=[6, 8])

        var = tf.linalg.diag(self.var_)
        mean = tf.expand_dims(self.mean, 1)
        y = tf.expand_dims(self.state_, 1)

        self.mse = tf.stop_gradient(tf.reduce_mean(tf.losses.mse(mean, y)))
        self.loss = tf.reduce_mean(0.5 * tf.squeeze((y - mean) @ var @ tf.transpose(y - mean, perm=[0, 2, 1]))
                                   + 0.5 * tf.linalg.logdet(var))
        #
        self.model_train = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=model_weights)

        if self.num_nets > 1:
            self.model2 = self.create_network()

            model2_weights = self.model2.trainable_variables

            self.output2 = self.model2(inputs=self.input)

            self.mean2, logvar2 = self.get_output(self.output2)
            self.var_2 = tf.exp(logvar2)
            # self.s_output2 = tf.random.normal(shape=[6, 8], mean=self.mean2, stddev=tf.sqrt(self.var_2))

            var2 = tf.linalg.diag(self.var_2)
            mean2 = tf.expand_dims(self.mean2, 1)

            self.mse2 = tf.stop_gradient(tf.reduce_mean(tf.losses.mse(mean2, y)))
            self.loss2 = tf.reduce_mean(0.5 * tf.squeeze((y - mean2) @ var2 @ tf.transpose(y - mean2, perm=[0, 2, 1]))
                                        + 0.5 * tf.linalg.logdet(var2))
            #
            self.model2_train = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss2, var_list=model2_weights)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(inputs=I, outputs=O)
        return model

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        # TODO: write your code here
        # raise NotImplementedError
        for e in range(epochs):

            pass

    def main(self, inputs, labels, train_mode=True):
        if train_mode:
            # with tf.GradientTape() as tape:
            #     output = self.model(inputs)
            #     mean, logvar = self.get_output(output)
            #     var_ = tf.exp(logvar)
            #     var = tf.linalg.diag(var_)
            #     # mean1 = tf.expand_dims(mean, 1)
            #     mean = tf.reshape(mean, (mean.shape[0], 1, mean.shape[1]))
            #     # y1 = tf.expand_dims(labels, 1)
            #     y = tf.reshape(tf.Variable(labels, dtype=tf.float32), mean.shape)
            #
            #     loss = tf.reduce_mean(0.5 * tf.squeeze((y - mean) @ var @ tf.transpose(y - mean, perm=[0, 2, 1]))
            #                           + 0.5 * tf.linalg.logdet(var))
            #     print(loss)
            # grad = tape.gradient(loss, self.model.trainable_variables)
            # self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            if self.num_nets == 1:
                mse, loss, _ = self.sess.run([self.mse, self.loss, self.model_train], {self.input_state: inputs[:, :8], self.input_action: inputs[:, 8:], self.state_: labels})
                return mse, loss
            else:
                mse1, mse2, loss1, loss2, _, _ = self.sess.run([self.mse, self.mse2, self.loss, self.loss2, self.model_train, self.model2_train], {self.input_state: inputs[:, :8], self.input_action: inputs[:, 8:], self.state_: labels})
                return np.mean([mse1, mse2]), np.mean([loss1, loss2])
            # print(loss[0])
        else:
            # output = self.model(inputs)
            # mean, logvar = self.get_output(output)
            # var = np.exp(logvar)

            # output = self.sess.run(self.output, {self.input: inputs})
            # var = self.sess.run(tf.exp(logvar))
            # s_ = [np.random.normal(mean.numpy(), var.numpy()) for _ in range(num_particles)]
            if self.num_nets == 1:
                mean, var = self.sess.run([self.mean, self.var_], {self.input: inputs})
                s_ = np.random.normal(mean, var)
                # s_ = self.sess.run(self.s_output, {self.input: inputs})
                return s_
            else:
                mean1, var1, mean2, var2 = self.sess.run([self.mean, self.var_, self.mean2, self.var_2], {self.input: inputs})
                s1 = np.random.normal(mean1, var1)
                s2 = np.random.normal(mean2, var2)
                # s1, s2 = self.sess.run([self.s_output, self.s_output2], {self.input: inputs})
                assert s1.shape == s2.shape
                s_ = np.array([s1[i] if random.random() > 0.5 else s2[i] for i in range(s1.shape[0])])
                return s_

    def test(self):
        state = np.array([1., 1, 2, 2, 0, 0, 0, 0])
        action = np.array([0.5, 0.75])
        self.state__ = self.env.get_nxt_state(state, action)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, "./model/test/test.ckpt")

        print(self.sess.run(self.output, feed_dict={self.input_state: state[:8][None], self.input_action: action[None],
                                                    self.state_: self.state__[:8][None]}))

        for _ in range(300):
            self.sess.run([self.loss, self.model_train], feed_dict={self.input_state: state[:8][None], self.input_action: action[None],
                                                   self.state_: self.state__[:8][None]})

        print(self.sess.run(self.output, feed_dict={self.input_state: state[:8][None], self.input_action: action[None],
                                                    self.state_: self.state__[:8][None]}))
        # saver.save(self.sess, "./model/test/test.ckpt")
        # for _ in range(300):
        #     self.main(np.concatenate((state[:8][None], action[None]), axis=1), self.state__[:8][None])

    # TODO: Write any helper functions that you need


if __name__ == '__main__':
    penn = PENN(1, 8, 2, 1e-3)
    penn.test()
    print(1)
    # env = gym.make('Pushing2D-v1')
    # sess = tf.compat.v1.Session()
    # state = env.reset()
    # action = np.array([0.5, 0.75])
    # state_ = env.get_nxt_state(state, action)
    #
    # output = sess.run(penn.output, feed_dict={penn.input_state: state[:8][None], penn.input_action: action[None], penn.state_: state_[:8][None]})
