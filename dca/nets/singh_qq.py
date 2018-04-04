import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_cells


class SinghQQNet(Net):
    def __init__(self, pp, logger, frepshape):
        self.name = "SinghNet"
        # self.frep_depth = pp['n_channels'] + 1 if frep_depth is None else frep_depth
        self.pre_conv = pp['pre_conv']
        self.grid_inp = pp['singh_grid']
        self.frepshape = frepshape
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _build_net(self, inp, name):
        with tf.variable_scope('model/' + name) as scope:
            if self.pp['dueling_qnet']:
                value = tf.layers.dense(
                    inputs=inp,
                    units=1,
                    kernel_initializer=self.kern_init_dense(),
                    use_bias=False,
                    name="value")
                assert (value.shape[-1] == 1)
                advantages = tf.layers.dense(
                    inputs=inp,
                    units=self.n_channels,
                    use_bias=False,
                    kernel_initializer=self.kern_init_dense(),
                    name="advantages")
                q_vals = value + (
                    advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
                print("Dueling q-out shape:", q_vals.shape)
            else:
                q_vals = tf.layers.dense(
                    inputs=inp,
                    units=self.n_channels,
                    use_bias=False,
                    kernel_initializer=self.kern_init_dense(),
                    name="qvals")
            trainable_vars = get_trainable_vars(scope)
            return q_vals, trainable_vars

    def build(self):
        # gridshape = [None, self.rows, self.cols, self.n_channels]
        oh_cellshape = [None, self.rows, self.cols, 1]
        self.frep = tf.placeholder(tf.int32, (None, *self.frepshape), "feature_reps")
        # self.grid = tf.placeholder(tf.bool, gridshape, "grids")
        self.oh_cell = tf.placeholder(tf.bool, oh_cellshape, "oh_cell")
        self.q_target = tf.placeholder(tf.float32, [None], "value_target")
        # self.elig = tf.placeholder(tf.bool, gridshape, "elig")
        # self.cells = tf.placeholder(tf.int32, [None, 2], "cell")
        self.ch = tf.placeholder(tf.int32, [None], "ch")

        frep = tf.cast(self.frep, tf.float32)
        oh_cell = tf.cast(self.oh_cell, tf.float32)
        # grid = tf.cast(self.grid, tf.float32)
        # elig = tf.cast(self.elig, tf.float32)
        nrange = tf.range(tf.shape(self.frep)[0], name="cellrange")
        # ncells = tf.concat([tf.expand_dims(nrange, axis=1), self.cells], axis=1)
        numbered_ch = tf.stack([nrange, self.ch], axis=1, name="cellstack")

        # if self.grid_inp:
        #     self.grid = tf.placeholder(
        #         tf.bool, [None, self.rows, self.cols, 2 * self.n_channels], "grid")
        #     grid = tf.cast(self.grid, tf.float32)
        #     top_inp = tf.concat([grid, frep], axis=3)
        # else:
        #     top_inp = frep
        top_inp = tf.concat([frep, oh_cell], axis=3)

        self.online_q_vals, online_vars = self._build_net(
            tf.layers.flatten(top_inp), name="q_networks/online")

        # Online Q-value for given ch
        self.online_q_selected = tf.gather_nd(self.online_q_vals, numbered_ch)
        self.online_q_max = tf.reduce_max(self.online_q_vals, axis=0)

        # For each batch, the mean of the maximum eligible q-values of each cell.
        self.err = self.q_target - self.online_q_selected
        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.q_target, predictions=self.online_q_selected)
        return self.loss, online_vars

    def forward(self, grids, freps, cells):
        data = {
            self.frep: freps,
            # self.cell: cells,
            self.oh_cell: prep_data_cells(cells),
        }
        # if self.grid_inp:
        #     data[self.grid] = grids
        values = self.sess.run(
            self.online_q_vals,
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, *, grids, freps, cells, chs, value_targets, **kwargs):
        data = {
            # self.grid: grids,
            self.frep: freps,
            self.oh_cell: prep_data_cells(cells),
            self.ch: chs,
            self.q_target: value_targets
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
