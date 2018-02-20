import numpy as np
import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.utils import copy_net_op, prep_data_cells, prep_data_grids


def build(self):
    depth = self.n_channels * 2 if self.pp['grid_split'] else self.n_channels
    gridshape = [None, self.pp['rows'], self.pp['cols'], depth]
    oh_cellshape = [None, self.pp['rows'], self.pp['cols'], 1]  # Onehot
    self.grid = tf.placeholder(boolean, gridshape, "grid")
    gridf = tf.cast(self.grid, float32)
    self.cell = tf.placeholder(int32, [None, 2], "cell")
    self.oh_cell = tf.placeholder(float32, oh_cellshape, "oh_cell")
    self.ch = tf.placeholder(int32, [None], "ch")
    self.reward = tf.placeholder(float32, [None], "reward")
    self.next_grid = tf.placeholder(boolean, gridshape, "next_grid")
    next_gridf = tf.cast(self.next_grid, float32)
    self.next_oh_cell = tf.placeholder(float32, oh_cellshape, "next_oh_cell")
    self.next_ch = tf.placeholder(int32, [None], "next_ch")
    # Allows for passing in varying gamma, e.g. beta discount
    self.tf_gamma = tf.placeholder(float32, [1], "gamma")

    self.online_q_vals, online_vars = self._build_net(
        gridf, self.oh_cell, name="q_networks/online")
    # Keep separate weights for target Q network
    target_q_vals, target_vars = self._build_net(
        next_gridf, self.next_oh_cell, name="q_networks/target")
    # copy_online_to_target should be called periodically to creep
    # weights in the target Q-network towards the online Q-network
    self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                             self.pp['net_creep_tau'])

    # Maximum valued ch from online network
    self.online_q_amax = tf.argmax(self.online_q_vals, axis=1, name="online_q_amax")
    # Maximum Q-value for given next state
    # Q-value for given ch
    self.online_q_selected = tf.reduce_sum(
        self.online_q_vals * tf.one_hot(self.ch, self.n_channels),
        axis=1,
        name="online_q_selected")

    # Target Q-value for given next ch
    self.target_q_selected = tf.reduce_sum(
        target_q_vals * tf.one_hot(self.next_ch, self.n_channels),
        axis=1,
        name="target_q_selected")
    self.next_q = self.target_q_selected

    self.q_target = self.reward + self.tf_gamma * self.next_q

    self.loss = tf.losses.mean_squared_error(
        labels=tf.stop_gradient(self.q_target), predictions=self.online_q_selected)
    return online_vars


def backward(self, grids, cells, chs, rewards, next_grids, next_cells, next_chs=None):
    """
    If 'next_chs' are specified, do SARSA update,
    else greedy selection (Q-Learning).
    If 'next_q', do supervised learning.
    """
    gamma = self.gamma  # Not using beta-discount; use fixed constant
    p_next_grids = prep_data_grids(next_grids, self.pp['grid_split'])
    p_next_cells = prep_data_cells(next_cells)
    next_chs = self.sess.run(
        self.online_q_amax,
        feed_dict={
            self.grid: p_next_grids,
            self.oh_cell: p_next_cells
        })
    data = {
        self.grid: prep_data_grids(grids, self.pp['grid_split']),
        self.oh_cell: prep_data_cells(cells),
        self.ch: chs,
        self.reward: rewards,
        self.next_grid: p_next_grids,
        self.next_oh_cell: p_next_cells,
        self.next_ch: next_chs,
        self.tf_gamma: [gamma]
    }
    _, loss, lr = self.sess.run(
        [self.do_train, self.loss, self.lr],
        feed_dict=data,
        options=self.options,
        run_metadata=self.run_metadata)
    return loss, lr
