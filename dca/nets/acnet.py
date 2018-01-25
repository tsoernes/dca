from typing import List, Tuple

import numpy as np
import tensorflow as tf

import nets.utils as nutils
from nets.net import Net


class ACNet(Net):
    def __init__(self, *args, **kwargs):
        """
        """
        self.max_grad_norm = 40.0
        super().__init__(name="ACNet", *args, **kwargs)
        self.pp['n_step'] = 10

    def _build_net(self, grid, cell, name):
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=70,
                kernel_size=5,
                padding="same",
                activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)
            stacked = tf.concat([conv2, cell], axis=3)
            hidden = tf.layers.dense(
                tf.layers.flatten(stacked), units=256, activation=tf.nn.relu)

            # Output layers for policy and value estimations
            policy = tf.layers.dense(
                hidden,
                units=self.n_channels,
                activation=tf.nn.softmax,
                kernel_initializer=nutils.normalized_columns_initializer(0.01),
                bias_initializer=None)
            value = tf.layers.dense(
                hidden,
                units=1,
                activation=None,
                kernel_initializer=nutils.normalized_columns_initializer(1.0),
                bias_initializer=None)
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {
            var.name[len(scope.name):]: var
            for var in trainable_vars
        }
        return policy, value, trainable_vars_by_name

    def build(self):
        gridshape = [None, self.pp['rows'], self.pp['cols'], self.n_channels]
        # TODO Convert to onehot in TF
        cellshape = [None, self.pp['rows'], self.pp['cols'], 1]  # Onehot
        self.grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="grid")
        self.cell = tf.placeholder(
            shape=cellshape, dtype=tf.float32, name="cell")
        self.action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="action")

        # These are not currently in use, but
        # could perhaps be if stop-gradient is used, and rewards are inputted
        self.next_grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="next_grid")
        self.next_cell = tf.placeholder(
            shape=cellshape, dtype=tf.float32, name="next_cell")
        self.next_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="next_action")

        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        self.policy, self.value, _ = self._build_net(
            self.grid, self.cell, name="ac_network/online")

        action_oh = tf.one_hot(
            self.action, self.pp['n_channels'], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * action_oh, [1])

        # TODO Perhaps these should be 'reduce_mean' instead.
        self.value_loss = tf.reduce_sum(
            tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_sum(
            tf.log(self.responsible_outputs) * self.advantages)
        self.loss = 0.25 * self.value_loss + self.policy_loss - self.entropy * 0.01
        self.do_train = self._build_default_trainer()

    def forward(self, grid, cell) -> Tuple[List[float], float]:
        a_dist, val = self.sess.run(
            [self.policy, self.value],
            feed_dict={
                self.grid: nutils.prep_data_grids(grid),
                self.cell: nutils.prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        return a_dist[0], val[0, 0]

    def backward(self, grids, cells, vals, actions, rewards, next_grid,
                 next_cell) -> float:
        # Estimated value after trajectory, V(S_t+n)
        bootstrap_val = self.sess.run(
            self.value,
            feed_dict={
                self.grid: nutils.prep_data_grids(next_grid),
                self.cell: nutils.prep_data_cells(next_cell)
            })
        rewards_plus = np.asarray(rewards + [bootstrap_val])
        discounted_rewards = nutils.discount(rewards_plus, self.gamma)[:-1]
        value_plus = np.asarray(vals + [bootstrap_val])
        advantages = nutils.discount(
            rewards + self.gamma * value_plus[1:] - value_plus[:-1],
            self.gamma)

        data = {
            self.grid: nutils.prep_data_grids(np.array(grids)),
            self.cell: nutils.prep_data_cells(cells),
            self.target_v: discounted_rewards,
            self.action: actions,
            self.advantages: advantages
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"Invalid loss: {loss}")
        return loss
