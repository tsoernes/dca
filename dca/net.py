import numpy as np
import tensorflow as tf

# Neighbors2
# (3,3)
# min row: 1
# max row: 5
# min col: 1
# max col: 5
# (4,3)
# min row: 2
# max row: 6
# min col: 1
# max col: 5
# So it might be a good idea to have 4x4 filters,
# as that would cover all neighs2
#
# Padding with 0's is the natural choice since that would be
# equivalent to having empty cells outside of grid
#
# For a policy network, i.e. with actions [0, 1, ..., n_channels-1]
# corresponding to the probability of assigning the different channels,
# how can the network know, or be trained to know, that some actions
# are illegal/unavailable?


class Net:
    def __init__(self, logger, n_in, n_out, alpha, gamma,
                 *args, **kwargs):
        self.logger = logger
        tf.reset_default_graph()
        # Use ADAM, not rmsprop or sdg
        # learning rate decay not critical (but possible)
        # to do with adam.

        # consider batch norm [ioffe and szegedy, 2015]
        # batch norm is inserted after fully connected or convolutional
        # layers and before nonlinearity

        # possible data prep: set unused channels to -1,
        # OR make it unit gaussian. refer to alphago paper -- did they prep
        # the board? did they use the complete board as input to any of their
        # nets, or just features?

        # sanity checks:
        # - double check that loss is sane
        # for softmax classifier: print loss, should be roughly:
        # "-log(1/n_classes)"
        # - make sure that it's possible to overfit
        # (ie loss of nearly 0, when no regularization)
        # a very small portion (eg <20 samples) of the training data
        #
        # On finding learning rate:
        # start very small (eg 1e-6), make sure it's barely changing
        # or decreasing very slowly.
        # If cost is NaN or inf, learning rate is too high
        #
        # on tuning hyperparams:
        # if cost goes over 3x original cost, break out early
        #
        # big gap between train and test accuracy:
        # overfitting. reduce net size or increase regularization
        # no gap: increase net size
        #
        # debugging nets: track ratio of weight updates/weight magnitues
        # should be somewhere around 0.001 or so. if too high, decrease
        # learning rate, if too log (like 1e-6), increase lr.

        self.alpha = alpha
        self.gamma = gamma

        self.inputs1 = tf.placeholder(shape=[1, n_in], dtype=tf.bool)
        self.W = tf.Variable(tf.random_uniform([n_in, n_out], 0, 0.01))
        # One q-value for each of the possible actions
        self.Qout = tf.matmul(self.inputs1, self.W)
        # self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.td_err = tf.placeholder(shape=[1, n_out], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.td_err))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.updateModel = trainer.minimize(loss)
        # NOTE This is a STATE->QVALS network, where QVALS
        # is the q-values for each possible action
        # The current strat code utilize (STATE, ACTION) -> QVAL
        # One possible way to circumvent this is to make
        # get_qval() run a standard feedworward pass,
        # and just return the q-value for the relevant
        # index. Remember to optimize where possible,
        # so that multiple passes with the same input
        # are not made through the network just to get
        # the qval for different actions.

        # NOTE remember to call sess.close()
        self.sess = tf.session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def forward(self, state):
        """
        Forward pass. Given an input, such as a feature vector
        or the whole state, return the output of the network.
        """
        # inp = self.grid.state[cell]
        qvals = self.sess.run(
                self.Qout,
                feed_dict={self.inputs1: state})  # self.inputs1?
        # the action should be executed here
        # so that reward and next_state can be observed
        return qvals

    def backward(self, state, td_err):
        # Obtain maxQ' and set our target value for chosen action.
        # Train our network using target and predicted Q values
        _, W1 = self.sess.run(
            [self.updateModel, self.W],
            feed_dict={self.inputs1: state, self.td_err: td_err})
        # should W be set to W1?

    def weight_init(self):
        inp = None
        hidden_layer_sizes = [0]
        Hs = {}
        for i in range(hidden_layer_sizes):
            X = inp if i == 0 else Hs[i-1]
            fan_in = X.shape[1]
            fan_out = hidden_layer_sizes[i]
            # init according to [He et al. 2015]
            # fan_in: number input
            W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)
            H = np.dot(X, W)
            Hs[i] = H

    def save(self, filenam):
        """
        Save parameters to disk
        """
        pass


class PGNet(Net):
    """
    Policy gradient net
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RSValNet(Net):
    """
    Input is coordinates and number of used channels.
    Output is a state value.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RSPolicyNet(Net):
    """
    Input is coordinates and number of used channels.
    Output is a vector with probability for each channel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
