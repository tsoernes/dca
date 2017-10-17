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
