import numpy as np
import tensorflow as tf

from eventgen import CEvent
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


class Net2:
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


class Net:
    def __init__(self, logger, n_out, alpha,
                 *args, **kwargs):
        self.n_out = n_out
        self.logger = logger
        tf.reset_default_graph()

        self.inputs = tf.placeholder(shape=[1, 7, 7, 71], dtype=tf.float32)
        # conv1 = tf.layers.conv2d(
        #     inputs=self.inputs,
        #     filters=70,
        #     kernel_size=4,
        #     strides=2,
        #     padding="same",  # pad with 0's
        #     activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling2d(
        #     inputs=conv1, pool_size=2, strides=1)
        # conv2 = tf.layers.conv2d(
        #     inputs=pool1,
        #     filters=140,
        #     kernel_size=4,
        #     padding="same",
        #     activation=tf.nn.relu)
        # # Dense Layer
        # conv2_flat = tf.reshape(conv2, [-1, 3 * 3 * 140])
        # dense = tf.layers.dense(
        #         inputs=conv2_flat, units=256, activation=tf.nn.relu)

        # TODO verify that linear neural net performs better than random.
        # Perhaps reducing call rates will increase difference between
        # fixed/random and a good alg, thus making testing nets easier.
        # If so then need to retest sarsa-strats and redo hyperparam opt.
        inputs_flat = tf.reshape(self.inputs, [-1, 7*7*71])
        self.Qout = tf.layers.dense(inputs=inputs_flat, units=70)
        self.argmaxQ = tf.argmax(self.Qout, axis=1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.targets = tf.placeholder(shape=[1, n_out], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.targets - self.Qout))
        # GradientDescentOptimizer is designed to use a constant learning rate.
        # The best is probably to use AdamOptimizer which is out-of-the-box
        # adaptive, i.e. it controls the learning rate in some way.
        # NOTE Should the rate of learning rate decrease be set for alpha?
        # trainer = tf.train.AdamOptimizer(learning_rate=alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        trainer = tf.train.RMSPropOptimizer(learning_rate=alpha)
        self.updateModel = trainer.minimize(loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def forward(self, features):
        """
        Forward pass. Given an input, such as a feature vector
        or the whole state, return the output of the network.
        """
        action, qvals = self.sess.run(
                [self.argmaxQ, self.Qout],
                feed_dict={self.inputs: features})
        return action[0], qvals[0]

    def forward_ch(self, ce_type, state, cell, features, neighs2=None):
        """
        Get the argmin (for END events) or argmax (for NEW events) of
        the q-values for the given features. Only valid channels
        are taken the argmin/argmax over.
        """
        tf_state = tf.placeholder(shape=[7, 7, 70], dtype=tf.bool)
        tf_cell = tf.placeholder(shape=[2], dtype=tf.int32)
        tf_neighs2 = tf.placeholder(dtype=tf.in32)

        zero = tf.constant(False, dtype=tf.bool)
        region = tf.gather_nd(tf_state, tf_cell)
        q_flat = tf.reshape(self.Qout, [-1])

        notzero = tf.not_equal(region, zero)
        tf_inuse = tf.reshape(tf.where(notzero), [-1])
        tf_inuse_q = tf.gather(q_flat, tf_inuse)
        inuse_isEmpty = tf.equal(tf.size(tf_inuse_q), tf.constant(0))
        argmin_q = tf.cond(
            inuse_isEmpty,
            lambda: tf.constant(-1, dtype=tf.int64),
            lambda: tf_inuse[tf.argmin(tf_inuse_q)])

        # NOTE This is not correct. Does not check that the chs
        # are not in use in neighboring cells.
        iszero = tf.equal(region, zero)
        tf_free = tf.reshape(tf.where(iszero), [-1])
        tf_free_q = tf.gather(q_flat, tf_free)
        free_isEmpty = tf.equal(tf.size(tf_inuse_q), tf.constant(0))
        argmax_q = tf.cond(
            free_isEmpty,
            lambda: tf.constant(-1, dtype=tf.int64),
            lambda: tf_free[tf.argmax(tf_free_q)])

        if ce_type == CEvent.END:
            ch, qvals = self.sess.run(
                [argmin_q, q_flat],
                feed_dict={tf_state: state, tf_cell: cell,
                           self.inputs: features})
        else:
            ch, qvals = self.sess.run(
                [argmax_q, q_flat],
                feed_dict={tf_state: state, tf_cell: cell,
                           self.inputs: features, tf_neighs2: neighs2})
        if ch == -1:
            ch = None
        return ch, qvals

    def neighbors2(self, cell):
        """
        Returns a list with indices of neighbors within a radius of 2,
        not including self
        """
        idxs = []

        zero = tf.constant(0)
        row = cell[0]
        col = cell[1]
        r_low = tf.max(0, row - 2)
        r_hi = tf.min(self.rows - 1, row + 2)
        c_low = tf.max(0, col - 2)
        c_hi = tf.min(self.cols - 1, col + 2)
        k = 7
        at = tf.transpose(
            tf.meshgrid(tf.range(k), tf.range(k), indexing="ij"),
            perm=[1, 2, 0])

        cross1 = tf.cond(
            tf.equal(tf.mod(col, 2), zero),
            tf.subtract(row, 2),
            tf.add(row, 2))
        cross2 = tf.cond(
            tf.equal(tf.mod(col, 2), zero),
            tf.add(row, 2),
            tf.subtract(row, 2))

        for r in range(r_low, r_hi + 1):
            for c in range(c_low, c_hi + 1):
                if not ((r, c) == (row, col) or
                        (r, c) == (cross1, col - 2) or
                        (r, c) == (cross1, col - 1) or
                        (r, c) == (cross1, col + 1) or
                        (r, c) == (cross1, col + 2) or
                        (r, c) == (cross2, col - 2) or
                        (r, c) == (cross2, col + 2)):
                    idxs.append((r, c))
        return idxs

    # def get_free_chs(self, cell):
    #     """
    #     Find the channels that are free in 'cell' and all of
    #     its neighbors by bitwise ORing all their allocation maps
    #     """
    #     neighs = self.neighbors2(*cell)
    #     alloc_map = np.bitwise_or(
    #         self.state[cell], self.state[neighs[0]])
    #     for n in neighs[1:]:
    #         alloc_map = np.bitwise_or(alloc_map, self.state[n])
    #     free = np.where(alloc_map == 0)[0]
    #     return free

    def backward(self, state, targets):
        """
        Back-propagation
        """
        targets.shape = (1, 70)
        # Obtain maxQ' and set our target value for chosen action.
        # Train our network using target and predicted Q values
        self.sess.run(
            self.updateModel,
            feed_dict={self.inputs: state, self.targets: targets})

    def weight_init(self):
        raise NotImplementedError
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
