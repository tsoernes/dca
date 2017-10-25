    def all_neighs(self):
        # neighs = tf.zeros((self.rows, self.cols, 19, 2))
        # mask = None
        mesh = tf.transpose(tf.meshgrid(tf.range(self.rows),
                                        tf.range(self.cols),
                                        indexing="ij"), [1, 2, 0])
        res = tf.map_fn(self.neighbors2, mesh)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            res = sess.run([res])
            print(res)
            return res

    def n2run(self):
        mesh = tf.transpose(tf.meshgrid(tf.range(self.rows),
                                        tf.range(self.cols),
                                        indexing="ij"), [1, 2, 0])
        tf_cell = mesh[1][1]
        fn = self.neighbors2(tf_cell)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            res = sess.run([fn])
            print(res)
            return res

    def neighbors2(self, cell):
        """
        Returns a list with indices of neighbors within a radius of 2,
        not including self
        TODO Generate all neighs on init and store in a [7x7xMAX_Nx2] array,
        where MAX_N is the maximum number of neighbors. Create a boolean mask
        over the array, gather from it.
        """
        row = cell[0]
        col = cell[1]
        r_low = tf.maximum(0, row - 2)
        r_hi = tf.minimum(self.rows - 1, row + 2)
        c_low = tf.maximum(0, col - 2)
        c_hi = tf.minimum(self.cols - 1, col + 2)

        shape = tf.stack([self.rows + 2, self.cols + 2])
        oh_idxs = tf.get_variable(
            name="oh_idxs",
            shape=[self.rows + 2, self.cols + 2],
            dtype=tf.bool,
            initializer=tf.zeros_initializer(dtype=tf.bool),
            validate_shape=False)
        # oh_idxs = oh_idxs[r_low: r_hi + 1, c_low: c_hi + 1].assign(True)
        oh_idxs = oh_idxs[1][0].assign([True])
        oh_idxs[r_low+2, c_low+1].assign(True)
        oh_idxs[r_low+1, c_low+2].assign(True)
        oh_idxs[r_low, c_low].assign(True)
        # tf.scatter_nd_update(oh_idxs, [[r_low, c_low]], [True])

        cross1 = tf.cond(
            tf.equal(tf.mod(col, 2), 0),
            lambda: tf.subtract(row, 2),
            lambda: tf.add(row, 2))
        cross2 = tf.cond(
            tf.equal(tf.mod(col, 2), 0),
            lambda: tf.add(row, 2),
            lambda: tf.subtract(row, 2))

        oh_idxs[row, col].assign(False)
        oh_idxs[cross1, col - 2].assign(False)
        oh_idxs[cross1, col - 1].assign(False)
        oh_idxs[cross1, col + 1].assign(False)
        oh_idxs[cross1, col + 2].assign(False)
        oh_idxs[cross2, col - 2].assign(False)
        oh_idxs[cross2, col + 2].assign(False)

        idxs = tf.cast(tf.transpose(tf.where(oh_idxs)), tf.int32)
        return idxs

def forward_ch():
    # in init
    self.neighs2 = tf.constant(neighs2)
    self.neighs2mask = tf.constant(neighs2mask)
    def neighbors2(self, cell):
        return tf.boolean_mask(
            tf.gather_nd(self.neighs2, cell),
            tf.gather_nd(self.neighs2mask, cell))

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

