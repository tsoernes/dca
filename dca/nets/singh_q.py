import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (copy_net_op, get_trainable_vars, prep_data_cells,
                        scale_freps_big)


class SinghQNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def _build_net(self, inp, name):
        with tf.variable_scope('model/' + name) as scope:
            conv = tf.layers.Conv2D(
                filters=70,
                kernel_size=2,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.conv_regularizer,
                use_bias=self.pp['conv_bias'],
                activation=self.act_fn)
            out = conv.apply(inp)
            conv2 = tf.keras.layers.LocallyConnected2D(
                filters=70,
                kernel_size=1,
                padding="valid",
                kernel_initializer=self.kern_init_dense(),
                use_bias=self.pp['conv_bias'],
                activation=None)(out)
            # q_vals = tf.layers.dense(
            #     inputs=tf.layers.flatten(inp),
            #     units=self.n_channels,
            #     kernel_initializer=tf.zeros_initializer(),
            #     kernel_regularizer=None,
            #     bias_initializer=tf.zeros_initializer(),
            #     use_bias=False,
            #     activation=None,
            #     name="vals")
            trainable_vars = get_trainable_vars(scope)
            return conv2, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        oh_cellshape = [None, self.rows, self.cols, 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.cells = tf.placeholder(tf.int32, [None, 2], "cell")
        self.oh_cells = tf.placeholder(tf.bool, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(tf.int32, [None], "ch")
        self.q_targets = tf.placeholder(tf.float32, [None], "value_target")

        oh_cells = tf.cast(self.oh_cells, tf.float32)

        nrange = tf.range(tf.shape(self.freps)[0], name="cellrange")
        ncells = tf.concat([tf.expand_dims(nrange, axis=1), self.cells], axis=1)
        numbered_chs = tf.stack([nrange, self.chs], axis=1, name="cellstack")

        net_inputs = tf.concat(self.freps, axis=3, name="frep_cell_inp")
        conv, online_vars = self._build_net(net_inputs, name="q_networks/online")
        print(conv.shape)
        self.online_q_vals = tf.gather_nd(conv, ncells)

        # Online Q-value for given ch
        self.online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)
        self.online_q_max = tf.reduce_max(self.online_q_vals, axis=0)

        self.q_target_out = tf.expand_dims(
            tf.reduce_mean(tf.reduce_max(conv, axis=3)), axis=0)
        self.err = self.q_targets - self.online_q_selected
        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.q_targets, predictions=self.online_q_selected)
        return self.loss, online_vars

    def forward(self, freps, cells):
        values = self.sess.run(
            self.online_q_vals,
            feed_dict={
                self.freps: freps,
                self.cells: cells,
                # self.oh_cells: prep_data_cells(cells),
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, cells, chs, rewards, next_freps, next_cells, discount,
                 next_chs):
        # next_value = self.sess.run(
        #     self.online_q_selected, {
        #         self.freps: next_freps,
        #         self.oh_cells: prep_data_cells(next_cells),
        #         self.chs: next_chs
        #     })[0]
        next_value = self.sess.run(
            self.q_target_out,
            {
                self.freps: next_freps,
                # self.cells: [next_cells],
                # self.oh_cells: prep_data_cells(next_cells),
            })[0]
        assert next_value.shape == ()
        value_target = rewards + discount * next_value
        data = {
            self.freps: freps,
            self.cells: cells,
            # self.oh_cells: prep_data_cells(cells),
            self.chs: chs,
            self.q_targets: [value_target]
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
