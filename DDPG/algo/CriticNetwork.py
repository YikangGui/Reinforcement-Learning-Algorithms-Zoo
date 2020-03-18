import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam, SGD

# from tensorflow.keras.layers import Dense, Input, Concatenate
# from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 256
HIDDEN2_UNITS = 128


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    # raise NotImplementedError
    # state_input = Input(shape=(state_size,))
    # action_input = Input(shape=(action_size,))
    # output1 = Dense(HIDDEN1_UNITS, activation='relu')(state_input)
    # output2 = Dense(HIDDEN2_UNITS, activation='relu')(output1)
    #
    # output3 = Dense(64, activation='relu')(action_input)
    #
    # model_concat = concatenate([output2, output3])
    # model_concat = Dense(1, activation='linear')(model_concat)
    # model = Model(inputs=[state_input, action_input], outputs=model_concat)
    # return model

    # model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    # model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    # return model, state_input, action_input

    model1 = Sequential()
    model1.add(Dense(32, activation='linear', input_shape=(state_size,)))
    # model1.add(Dense(HIDDEN2_UNITS, activation='relu'))

    model2 = Sequential()
    model2.add(Dense(32, activation='linear', input_shape=(action_size,)))
    # model2.add(Dense(32, activation='relu'))

    model_concat = concatenate([model1.output, model2.output])
    model_concat = Dense(HIDDEN1_UNITS, activation='relu')(model_concat)
    model_concat = Dense(HIDDEN2_UNITS, activation='relu')(model_concat)
    model_concat = Dense(1, activation='linear')(model_concat)
    model = Model(inputs=[model1.input, model2.input], outputs=model_concat)
    return model


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        # raise NotImplementedError
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tf.Variable(tau, dtype=tf.dtypes.float32)
        self.lr = learning_rate
        self.model = create_critic_network(self.state_size, self.action_size, self.lr)
        self.target = create_critic_network(self.state_size, self.action_size, self.lr)
        self.target.set_weights(self.model.get_weights())
        self.opt = Adam(self.lr)
        self.gamma = 0.9
        self.target_params = self.target.trainable_variables
        self.model_params = self.model.trainable_variables
        # self.sess = sess
        # self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        # raise NotImplementedError
        # print(actions)
        # print()
        # print(states)
        # actions = tf.Variable(actions, dtype=tf.dtypes.float32)
        # states = tf.Variable(states, dtype=tf.dtypes.float32)
        # q = self.model.predict([states, actions])
        # print(1111)
        with tf.GradientTape() as tape:
            actions = tf.Variable(actions, dtype=tf.dtypes.float32)
            states = tf.Variable(states, dtype=tf.dtypes.float32)
            # tape.watch(actions)
            # print(actions)
            # print(states)
            # print(self.model.summary())
            q = self.model([states, actions])
        return tape.gradient(q, actions)

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        # raise NotImplementedError
        self.target.set_weights([(1 - self.tau) * self.target_params[i] + self.tau * self.model_params[i] for i in range(len(self.target_params))])
        # print('Critic updated!')
