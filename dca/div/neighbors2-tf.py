import numpy as np
import tensorflow as tf


def all_neighs(self):
    # neighs = tf.zeros((self.rows, self.cols, 19, 2))
    # mask = None
    mesh = tf.transpose(
        tf.meshgrid(tf.range(self.rows), tf.range(self.cols), indexing="ij"), [1, 2, 0])
    res = tf.map_fn(self.neighbors2, mesh)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run([res])
        print(res)
        return res


def n2run(self):
    mesh = tf.transpose(
        tf.meshgrid(tf.range(self.rows), tf.range(self.cols), indexing="ij"), [1, 2, 0])
    tf_cell = mesh[1][1]
    fn = self.neighbors2(tf_cell)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run([fn])
        print(res)
        return res


def forward_ch():
    # in init
    # self.neighs2 = tf.constant(neighs2)
    # self.neighs2mask = tf.constant(neighs2mask)

    def neighbors2(self, cell):
        return tf.boolean_mask(
            tf.gather_nd(self.neighs2, cell), tf.gather_nd(self.neighs2mask, cell))

    def forward_ch(self, ce_type, grid, cell, features):
        """
        Get the argmin (for END events) or argmax (for NEW/HOFF events) of
        the q-values for the given features. Only valid channels
        are taken argmin/argmax over.
        """
        tf_ce_type = tf.placeholder(shape=[1], dtype=tf.int32)
        tf_grid = tf.placeholder(shape=[7, 7, 70], dtype=tf.bool)
        tf_cell = tf.placeholder(shape=[2], dtype=tf.int32)
        # tf_epsilon = tf.placeholder(shape=[1], dtype=tf.float32)

        region = tf.gather_nd(tf_grid, tf_cell)
        q_flat = tf.reshape(self.Qout, [-1])

        # NOTE When everything is a lambda, conds evaluate lazily,
        # which prevents the calculation of free channels for end-events
        # and vice-versa.

        # Get the minimally valued channel that's in use
        tf_inuse_chs = lambda: tf.where(region)

        # Get the maximally valued channel that's free in this cell
        # and its neighbors within a radius of 2
        tf_neighs2i = lambda: self.neighbors2(cell)
        tf_neighs2 = lambda: tf.gather_nd(tf_grid, tf_neighs2i())
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

        tf_ch = tf.cond(tf.equal(tf_ce_type[0], 1), tf_ch_min, tf_ch_max)

        ch, qvals = self.sess.run([tf_ch, q_flat], {
            tf_grid: grid,
            tf_cell: cell,
            self.inputs: features,
            tf_ce_type: [ce_type.value]
        })

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

    def e_greedy(self):
        neighbors2i = tf.placeholder()
        # NOTE May not work corrctly with batches
        region = tf.gather_nd(self.grid, self.cell)
        q_flat = tf.reshape(self.q_vals, [-1])

        # Get the minimally valued channel that's in use
        inuse_chs = tf.where(region)

        # Get the maximally valued channel that's free in this cell
        # and its neighbors within a radius of 2
        tf_neighs2i = self.neighbors2(self.cell)
        tf_neighs2 = tf.gather_nd(self.grid, tf_neighs2i())
        tf_free_neighs = tf.reduce_any(tf_neighs2(), axis=0)
        tf_free = tf.logical_or(region, tf_free_neighs())
        tf_free_chs = tf.where(tf.logical_not(tf_free()))

        tf_chs = lambda: tf.cond(
            tf.equal(tf_ce_type[0], 1),
            tf_inuse_chs,  # END event
            tf_free_chs)  # NEW/HOFF event
        isempty = tf.equal(tf.size(inuse_chs()), 0)
        tf_valid_qs = lambda: tf.gather_nd(q_flat, tf_chs())
        # NOTE e-greedy should not be used because it does not
        # select from valid actions only.
        batch_size = tf.shape(self.grid)[0]
        random_actions = tf.random_uniform(
            tf.stack([batch_size]), maxval=self.n_channels, dtype=tf.int64)
        choose_random = tf.random_uniform(
            tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.epsilon
        # Epsilon-greedy action
        self.q_eps_amax = tf.where(choose_random, random_actions, self.q_amax)
