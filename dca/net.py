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
    def __init__(self, logger, n_in, n_out, alpha,
                 *args, **kwargs):
        self.n_out = n_out
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

        self.inputs = tf.placeholder(shape=[1, n_in], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([n_in, n_out], 0, 0.01))
        # One q-value for each of the possible actions
        self.Qout = tf.matmul(self.inputs, self.W)
        self.argmaxQ = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.targets = tf.placeholder(shape=[1, n_out], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.targets - self.Qout))
        # GradientDescentOptimizer is designed to use a constant learning rate.
        # The best is probably to use AdamOptimizer which is out-of-the-box
        # adaptive, i.e. it controls the learning rate in some way.
        trainer = tf.train.AdamOptimizer(learning_rate=alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        self.updateModel = trainer.minimize(loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def forward(self, state):
        """
        Forward pass. Given an input, such as a feature vector
        or the whole state, return the output of the network.
        """
        action, qvals = self.sess.run(
                [self.argmaxQ, self.Qout],
                feed_dict={self.inputs: state})
        return action[0], qvals[0]

    def backward(self, state, targets):
        """
        Back-propagation
        """
        targets.shape = (1, 70)
        # Obtain maxQ' and set our target value for chosen action.
        # Train our network using target and predicted Q values
        _, W1 = self.sess.run(
            [self.updateModel, self.W],
            feed_dict={self.inputs: state, self.targets: targets})
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
