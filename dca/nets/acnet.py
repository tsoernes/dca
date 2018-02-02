from typing import List, Tuple

import numpy as np
import tensorflow as tf

import nets.utils as nutils
from nets.net import Net


class ACNet(Net):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(name="ACNet", *args, **kwargs)
        self.pp['n_step'] = 10

    def _build_net(self, grid, cell, name):
        base_net = self._build_base_net(grid, cell, name)
        with tf.variable_scope(name) as scope:
            hidden = tf.layers.dense(base_net, units=128, activation=tf.nn.relu)
            # Output layers for policy and value estimations
            policy = tf.layers.dense(
                hidden,
                units=self.n_channels,
                activation=tf.nn.softmax,
                kernel_initializer=nutils.normalized_columns_initializer(0.01))
            value = tf.layers.dense(
                hidden,
                units=1,
                activation=None,
                kernel_initializer=nutils.normalized_columns_initializer(1.0))
            trainable_vars_by_name = self._get_trainable_vars(scope)
        return policy, value, trainable_vars_by_name

    def build(self):
        gridshape = [None, self.pp['rows'], self.pp['cols'], self.n_channels]
        # TODO Convert to onehot in TF
        cellshape = [None, self.pp['rows'], self.pp['cols'], 1]  # Onehot
        self.grid = tf.placeholder(shape=gridshape, dtype=tf.float32, name="grid")
        self.cell = tf.placeholder(shape=cellshape, dtype=tf.float32, name="cell")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

        self.value_target = tf.placeholder(shape=[None], dtype=tf.float32)
        # 'psi' may be a) The return R_t (reward from time t onward), optionally
        # with a baseline i.e. [R_t - b(s_t)], b) state-action value fn q(s,a),
        # c) TD error [r_t + v_{t+1}(s) - v_t(s)], d) the advantage fn
        # [q(s,a) - v(s)], e) the GAE estimator etc. See schulman2016.
        self.psi = tf.placeholder(shape=[None], dtype=tf.float32)

        self.policy, self.value, trainable_vars = self._build_net(
            self.grid, self.cell, name="ac_network/online")

        action_oh = tf.one_hot(self.action, self.pp['n_channels'], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * action_oh, [1])

        self.value_loss = self.pp['vf_coeff'] * tf.losses.mean_squared_error(
            tf.squeeze(self.value), self.value_target)
        self.entropy = self.pp[
            'entropy_coeff'] * -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_mean(self.psi * tf.log(self.responsible_outputs))
        self.loss = self.value_loss + self.policy_loss - self.entropy
        self.do_train = self._build_default_trainer(trainable_vars)

    def forward(self, grid, cell) -> Tuple[List[float], float]:
        a_dist, val = self.sess.run(
            [self.policy, self.value],
            feed_dict={
                self.grid: nutils.prep_data_grids(grid),
                self.cell: nutils.prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        assert val.shape == (1, 1)
        assert a_dist.shape == (1, self.n_channels)
        return a_dist[0], val[0, 0]

    def _backward(self, data):
        _, loss, pg_loss, vf_loss, entropy = self.sess.run(
            [self.do_train, self.loss, self.policy_loss, self.value_loss, self.entropy],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return (loss, pg_loss, vf_loss, entropy)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell) -> float:
        # TODO Save and pass 'val' from earlier forward pass
        val = self.sess.run(
            self.value,
            feed_dict={
                self.grid: nutils.prep_data_grids(grid),
                self.cell: nutils.prep_data_cells(cell)
            })[0]
        next_val = self.sess.run(
            self.value,
            feed_dict={
                self.grid: nutils.prep_data_grids(next_grid),
                self.cell: nutils.prep_data_cells(next_cell)
            })[0]
        target_val = reward + self.gamma * next_val
        advantage = target_val - val

        data = {
            self.grid: nutils.prep_data_grids(np.array(grid)),
            self.cell: nutils.prep_data_cells(cell),
            self.value_target: target_val,
            self.action: [ch],
            self.psi: advantage
        }
        return self._backward(data)

    def backward_gae(self, grids, cells, vals, chs, rewards, next_grid,
                     next_cell) -> float:
        """Generalized Advantage Estimation"""
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
            rewards + self.gamma * value_plus[1:] - value_plus[:-1], self.gamma)

        data = {
            self.grid: nutils.prep_data_grids(np.array(grids)),
            self.cell: nutils.prep_data_cells(cells),
            self.value_target: discounted_rewards,
            self.action: chs,
            self.psi: advantages
        }
        return self._backward(data)
