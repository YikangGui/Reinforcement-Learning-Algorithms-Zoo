import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
import numpy as np
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras import Model
from keras import utils as np_utils
from .CriticNetwork import CriticNetwork

HIDDEN1_UNITS = 256
HIDDEN2_UNITS = 128
HIDDEN3_UNITS = 64

np.random.seed(1024)

def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    # state_input = Input(shape=[state_size])
    # raise NotImplementedError
    model = Sequential()
    model.add(Dense(HIDDEN1_UNITS, activation='relu', input_shape=(state_size,)))
    model.add(Dense(HIDDEN2_UNITS, activation='relu'))
    model.add(Dense(HIDDEN3_UNITS, activation='relu'))
    model.add(Dense(action_size, activation='tanh'))
    model = Model(inputs=model.input, outputs=model.output)
    return model


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, actor_lr, critic_lr, gamma):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tf.Variable(tau, dtype=tf.dtypes.float32)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.model = create_actor_network(self.state_size, self.action_size)
        self.target = create_actor_network(self.state_size, self.action_size)
        self.target.set_weights(self.model.get_weights())
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic = CriticNetwork(sess, self.state_size, self.action_size, self.batch_size, self.tau, self.critic_lr)
        self.target_params = self.target.trainable_variables
        self.model_params = self.model.trainable_variables

    def critic_loss(self, states, action, next_states, reward, done):
        y_ = self.critic.model([states, action])
        mu_prime = self.target(next_states)
        # print(reward)
        # print(next_states)
        # print()
        y = np.array(reward) + self.gamma * np.array(done) * self.critic.target([next_states, mu_prime])
        # print(tf.reduce_mean(tf.losses.MSE(y, y_)))
        # return tf.reduce_mean(tf.losses.MSE(y, y_))
        return tf.losses.MSE(y, y_)

    def critic_train(self, states, action, next_states, reward, done):
        with tf.GradientTape() as t:
            loss = self.critic_loss(states, action, next_states, reward, done)
            # print(loss)
            grads = t.gradient(loss, self.critic.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.critic.model.trainable_variables))

    def train(self, states, action, next_states, reward, done):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            action: a batched numpy array storing the actions.
            next_states: a batched numpy array storing the next_state.
            reward: a batched numpy array storing the reward.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """
        with tf.GradientTape() as tape:
            # states = tf.Variable(states, dtype=tf.dtypes.float32)
            mu = self.model(states)
            Q = - self.critic.model([states, mu])
            # print(Q)
            # Q = - Q
            # Q = - tf.reduce_mean(Q)
            # Q = tf.reduce_mean(Q)
            # print(Q)
        actor_grad = (tape.gradient(Q, self.model.trainable_variables))
        # print(Q.shape)
        # print(actor_grad[0].shape)
        # print(actor_grad)
        # loss = -tf.reduce_mean(tf.multiply(action_grads, actor_grad))
        self.opt.apply_gradients(zip(actor_grad, self.model.trainable_variables))
        self.critic_train(states, action, next_states, reward, done)
        self.update_target()
        return self.critic.gradients(states, action)
        # raise NotImplementedError

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        # print(tf.trainable_variables())
        # print(self.target_params)
        updated_params = [((1 - self.tau) * self.target_params[i] + self.tau * self.model_params[i]).numpy() for i in range(len(self.target_params))]
        # print(updated_params)
        self.target.set_weights(updated_params)
        # print('Actor updated!')
        # self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
        #                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
        # for i in range(len(self.target_network_params))
        # self.ema.apply(self.model.get_weights(), self.target.get_weights())
        # self.target.set_weights(self.tau * self.model.get_weights() + (1 - self.tau) * self.target.get_weights())
        self.critic.update_target()
        # raise NotImplementedError
