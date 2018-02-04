import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import prep_data_grids


class DistQNet(Net):
    def __init__(self, name="QNet", *args, **kwargs):
        """
        Lagging QNet. If 'max_next_action', perform greedy
        Q-learning updates, else SARSA updates.
        """
        super().__init__(name=name, *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_net(self, grid, name):
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=self.n_channels,
                kernel_size=4,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,  # Default setting
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            flat = tf.layers.flatten(conv2)
            q_vals = tf.layers.dense(
                inputs=flat,
                units=self.n_channels,
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                name="q_vals")
            trainable_vars_by_name = self._get_trainable_vars(scope)
        return q_vals, trainable_vars_by_name

    def build(self):
        depth = self.n_channels * 2 if self.pp['grid_split'] else self.n_channels
        gridshape = [None, 5, 5, depth]
        self.grid = tf.placeholder(shape=gridshape, dtype=tf.float32, name="grid")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self.next_grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="next_grid")
        self.next_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="next_action")

        self.online_q_vals = []
        self.online_q_amax = []
        self.loss = []
        self.train = []
        for r in range(self.pp['rows']):
            r_qvals, r_loss, r_q_amax, r_train = [], [], [], [], []
            for c in range(self.pp['cols']):
                qvals, online_vars = self._build_net(
                    self.grid, name=f"q_networks/online/{r.c}")
                q_amax = tf.argmax(qvals, axis=1, name=f"online_q_amax/{r.c}")
                q_selected = tf.reduce_sum(
                    qvals * tf.one_hot(self.action, self.n_channels),
                    axis=1,
                    name=f"online_q_selected/{r.c}")
                target_q_selected = tf.reduce_sum(
                    qvals * tf.one_hot(self.next_action, self.n_channels),
                    axis=1,
                    name="target_q_selected")
                q_target = self.reward + self.gamma * target_q_selected
                loss = tf.losses.mean_squared_error(
                    labels=tf.stop_gradient(q_target), predictions=q_selected)
                do_train = self._build_default_trainer(online_vars)
                r_qvals.append(qvals)
                r_q_amax.append(q_amax)
                r_loss.append(loss)
                r_train.append(do_train)
            self.online_q_vals.append(r_qvals)
            self.online_q_amax.append(r_q_amax)
            self.loss.append(r_loss)
            self.train.append(r_train)

    def forward(self, grid, cell):
        q_vals = self.sess.run(
            [self.online_q_vals[cell[0]][cell[1]]],
            feed_dict={
                self.grid:
                prep_data_grids(
                    grid, cell, neg=self.pp['grid_neg'], split=self.pp['grid_split']),
            },
            options=self.options,
            run_metadata=self.run_metadata)
        q_vals = np.reshape(q_vals, [-1])
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def backward(self, grids, cells, actions, rewards, next_grids, next_cells):
        p_next_grids = prep_data_grids(next_grids, next_cells, self.pp['grid_neg'],
                                       self.pp['grid_split'])
        next_actions = self.sess.run(
            self.online_q_amax[next_cells[0]][next_cells[1]],
            feed_dict={
                self.grid: p_next_grids
            })
        data = {
            self.grid:
            self.prep_data_grids(grids, cells, self.pp['grid_neg'],
                                 self.pp['grid_split']),
            self.action:
            actions,
            self.reward:
            rewards,
            self.next_grid:
            p_next_grids,
            self.next_action:
            next_actions
        }
        _, loss = self.sess.run(
            [self.do_train[cells[0]][cells[1]], self.loss[cells[0]][cells[1]]],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss

    def prep_data_grids(self, grids, cells, neg=False, split=True):
        """
        neg: Represent empty channels as -1 instead of 0
        split: Double the depth and represent empty channels as 1 on separate layer
        """
        assert type(grids) == np.ndarray
        assert not (neg and split), "Can't have both options"
        if grids.ndim == 3:
            grids = np.expand_dims(grids, axis=0)
        assert grids.shape[1:] == (7, 7, 70)
        if type(cells) is tuple:
            cells = [cells]
        # TODO Take out local part
        if neg:
            grids = grids.astype(np.int8)
            # Make empty cells -1 instead of 0.
            # Temporarily convert to int8 to save memory
            grids = grids * 2 - 1
        elif split:
            sgrids = np.zeros((len(grids), 7, 7, 140), dtype=np.bool)
            sgrids[:, :, :, :70] = grids
            sgrids[:, :, :, 70:] = np.invert(grids)
            grids = sgrids
        grids = grids.astype(np.float16)
        return grids
