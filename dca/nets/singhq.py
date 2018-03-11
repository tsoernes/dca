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
            q_vals = tf.layers.dense(
                inputs=tf.layers.flatten(inp),
                units=self.n_channels,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=True,
                activation=None,
                name="vals")
            trainable_vars = get_trainable_vars(scope)
            return q_vals, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        oh_cellshape = [None, self.rows, self.cols, 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.oh_cells = tf.placeholder(tf.bool, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(tf.int32, [None], "ch")
        self.q_targets = tf.placeholder(tf.float32, [None], "value_target")

        cells = tf.cast(self.oh_cells, tf.float32)
        nrange = tf.range(tf.shape(self.freps)[0], name="cellrange")
        numbered_chs = tf.stack([nrange, self.chs], axis=1, name="cellstack")

        if self.pp['scale_freps']:
            freps = scale_freps_big(self.freps)
        else:
            freps = self.freps

        net_inputs = tf.concat([freps, cells], axis=3, name="frep_cell_inp")
        self.online_q_vals, online_vars = self._build_net(
            net_inputs, name="q_networks/online")
        # Keep searate weights for target Q network
        target_q_vals, target_vars = self._build_net(net_inputs, name="q_networks/target")
        # copy_online_to_target should be called periodically to creep
        # weights in the target Q-network towards the online Q-network
        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # Maximum valued ch from online network
        self.online_q_amax = tf.argmax(
            self.online_q_vals, axis=1, name="online_q_amax", output_type=tf.int32)
        # Target Q-value for greedy channel as selected by online network
        numbered_q_amax = tf.stack([nrange, self.online_q_amax], axis=1)
        self.target_q_max = tf.gather_nd(target_q_vals, numbered_q_amax)
        # Target Q-value for given ch
        self.target_q_selected = tf.gather_nd(target_q_vals, numbered_chs)
        # Online Q-value for given ch
        online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)

        self.err = self.q_targets - online_q_selected
        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.q_targets, predictions=online_q_selected)
        return self.loss, online_vars

    def forward(self, freps, cells):
        values = self.sess.run(
            self.online_q_vals,
            feed_dict={
                self.freps: freps,
                self.oh_cells: prep_data_cells(cells),
            },
            options=self.options,
            run_metadata=self.run_metadata)
        # print(values.shape)
        vals = np.reshape(values, [-1])
        return vals

    def _double_q_target(self, freps, cells) -> [float]:
        data = {self.oh_cells: prep_data_cells(cells), self.freps: freps}
        qvals = self.sess.run(self.target_q_max, data)
        return qvals

    def backward(self, freps, cells, chs, rewards, next_freps, next_cells, next_val, discount):
        # next_value = self._double_q_target(next_freps, next_cells)[0]
        value_target = rewards + discount * next_val
        data = {
            self.freps: freps,
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
