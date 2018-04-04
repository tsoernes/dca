import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_cells


class SinghQNet(Net):
    def __init__(self, pp, logger, frepshape):
        self.name = "SinghNet"
        # self.frep_depth = pp['n_channels'] + 1 if frep_depth is None else frep_depth
        self.pre_conv = pp['pre_conv']
        self.grid_inp = pp['singh_grid']
        self.frepshape = frepshape
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _build_net(self, inp, name):
        with tf.variable_scope('model/' + name) as scope:
            pad = tf.keras.layers.ZeroPadding2D((1, 1))
            out = pad(inp)
            conv2 = tf.keras.layers.LocallyConnected2D(
                filters=70,
                kernel_size=3,
                padding="valid",
                kernel_initializer=self.kern_init_conv(),
                use_bias=self.pp['conv_bias'],
                activation=None)(out)
            value = tf.layers.dense(
                inputs=conv2,
                units=1,
                kernel_initializer=self.kern_init_dense(),
                use_bias=False,
                name="value")
            assert (value.shape[-1] == 1)
            advantages = tf.layers.dense(
                inputs=conv2,
                units=self.n_channels,
                use_bias=False,
                kernel_initializer=self.kern_init_dense(),
                name="advantages")
            q_vals = value + (
                advantages - tf.reduce_mean(advantages, axis=1, keep_dims=True))
            trainable_vars = get_trainable_vars(scope)
            print(q_vals.shape)
            return q_vals, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        gridshape = [None, self.rows, self.cols, self.n_channels]
        oh_cellshape = [None, self.rows, self.cols, 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.grids = tf.placeholder(tf.bool, gridshape, "grids")
        self.elig = tf.placeholder(tf.bool, gridshape, "elig")
        self.cells = tf.placeholder(tf.int32, [None, 2], "cell")
        self.oh_cells = tf.placeholder(tf.bool, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(tf.int32, [None], "ch")
        self.q_targets = tf.placeholder(tf.float32, [None], "value_target")

        oh_cells = tf.cast(self.oh_cells, tf.float32)
        grids = tf.cast(self.grids, tf.float32)
        elig = tf.cast(self.elig, tf.float32)
        nrange = tf.range(tf.shape(self.freps)[0], name="cellrange")
        ncells = tf.concat([tf.expand_dims(nrange, axis=1), self.cells], axis=1)
        numbered_chs = tf.stack([nrange, self.chs], axis=1, name="cellstack")

        net_inputs = tf.concat(
            [self.freps, oh_cells, grids], axis=3, name="frep_cell_inp")
        conv, online_vars = self._build_net(net_inputs, name="q_networks/online")
        self.online_q_vals = tf.gather_nd(conv, ncells)

        # Online Q-value for given ch
        self.online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)
        self.online_q_max = tf.reduce_max(self.online_q_vals, axis=0)

        free_conv = elig * conv
        # For each batch, the mean of the maximum eligible q-values of each cell.
        self.q_target_out = tf.reduce_mean(tf.reduce_max(free_conv, axis=3), axis=[1, 2])
        self.err = self.q_targets - self.online_q_selected
        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.q_targets, predictions=self.online_q_selected)
        return self.loss, online_vars

    def forward(self, grids, freps, cells):
        values = self.sess.run(
            self.online_q_vals,
            feed_dict={
                self.grids: grids,
                self.freps: freps,
                self.cells: cells,
                self.oh_cells: prep_data_cells(cells),
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.squeeze(values, axis=1)
        return vals

    def backward(self, grids, freps, cells, chs, rewards, next_grids, next_elig,
                 next_freps, next_cells, discount):
        # next_value = self.sess.run(
        #     self.online_q_selected, {
        #         self.freps: next_freps,
        #         self.oh_cells: prep_data_cells(next_cells),
        #         self.chs: next_chs
        #     })[0]
        next_value = self.sess.run(
            self.q_target_out,
            {
                self.grids: next_grids,
                self.elig: next_elig,
                self.freps: next_freps,
                # self.cells: [next_cells],
                self.oh_cells: prep_data_cells(next_cells),
            })
        assert next_value.shape == (1, )
        value_target = rewards + discount * next_value[0]
        data = {
            self.grids: grids,
            self.freps: freps,
            self.cells: cells,
            self.oh_cells: prep_data_cells(cells),
            self.chs: chs,
            self.q_targets: [value_target]
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
