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
                 neighs2, neighs2mask, rows=7, cols=7,
                 *args, **kwargs):
        self.rows = rows
        self.cols = cols
        self.n_out = n_out
        self.logger = logger
        tf.reset_default_graph()
        self.neighs2 = tf.constant(neighs2)
        self.neighs2mask = tf.constant(neighs2mask)
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
        # TODO Try encoding grid as +1/-1 instead of +1/0
        inputs_flat = tf.reshape(self.inputs, [-1, 7 * 7 * 71])
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
        trainer = tf.train.AdamOptimizer(learning_rate=alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=alpha)
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

    def forward_ch(self, ce_type, state, cell, features):
        """
        Get the argmin (for END events) or argmax (for NEW/HOFF events) of
        the q-values for the given features. Only valid channels
        are taken argmin/argmax over.
        """
        tf_ce_type = tf.placeholder(shape=[1], dtype=tf.int32)
        tf_state = tf.placeholder(shape=[7, 7, 70], dtype=tf.bool)
        tf_cell = tf.placeholder(shape=[2], dtype=tf.int32)
        # tf_epsilon = tf.placeholder(shape=[1], dtype=tf.float32)

        region = tf.gather_nd(tf_state, tf_cell)
        q_flat = tf.reshape(self.Qout, [-1])

        # NOTE When everything is a lambda, conds evaluate lazily,
        # which prevents the calculation of free channels for end-events
        # and vice-versa.

        # Get the minimally valued channel that's in use
        tf_inuse_chs = lambda: tf.where(region)

        # Get the maximally valued channel that's free in this cell
        # and its neighbors within a radius of 2
        tf_neighs2i = lambda: self.neighbors2(cell)
        tf_neighs2 = lambda: tf.gather_nd(tf_state, tf_neighs2i())
        tf_free_neighs = lambda: tf.reduce_any(tf_neighs2(), axis=0)
        tf_free = lambda: tf.logical_or(region, tf_free_neighs())
        tf_free_chs = lambda: tf.where(tf.logical_not(tf_free()))

        tf_chs = lambda: tf.cond(
            tf.equal(tf_ce_type[0], 1),
            tf_inuse_chs,  # END event
            tf_free_chs)  # NEW/HOFF event
        isempty = lambda: tf.equal(tf.size(tf_chs()), 0)
        tf_valid_qs = lambda: tf.gather_nd(q_flat, tf_chs())

        tf_ch_min = lambda: tf.cond(
            isempty(),
            lambda: tf.constant(-1, dtype=tf.int64),
            lambda: tf_chs()[tf.argmin(tf_valid_qs())])

        tf_ch_max = lambda: tf.cond(
            isempty(),
            lambda: tf.constant(-1, dtype=tf.int64),
            lambda: tf_chs()[tf.argmax(tf_valid_qs())])

        tf_ch = tf.cond(
            tf.equal(tf_ce_type[0], 1),
            tf_ch_min,
            tf_ch_max)

        print(ce_type)
        ch, qvals = self.sess.run(
            [tf_ch, q_flat],
            {tf_state: state, tf_cell: cell,
             self.inputs: features,
             tf_ce_type: [ce_type.value]})

        # e-greedy
        # probs = lambda: tf.log([tf.ones(n_valid)/n_valid])
        # e_greedy_select = tf.cond(
        #     tf.less(tf.random_uniform([1]), tf.epsilon),
        #     lambda: tf_chs[tf.multinomial(probs)[0][0]],
        #     lambda: argm_q)
        # probs = tf.log([tf.ones(70)/70])# this shouldnt be 70 but n valid chs
        # sample = tf.multinomial(probs, 1)
        # ch = chs[sample[0][0]]

        if ch == -1:
            ch = None
        return ch, qvals

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
            X = inp if i == 0 else Hs[i - 1]
            fan_in = X.shape[1]
            fan_out = hidden_layer_sizes[i]
            # init according to [He et al. 2015]
            # fan_in: number input
            W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)
            H = np.dot(X, W)
            Hs[i] = H

    def neighbors2(self, cell):
        return tf.boolean_mask(
            tf.gather_nd(self.tneighs2, cell),
            tf.gather_nd(self.tneighs2mask, cell))

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
