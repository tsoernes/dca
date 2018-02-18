import numpy as np
import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.net import Net
from nets.utils import (copy_net_op, get_trainable_vars, prep_data_cells,
                        prep_data_grids)


class QNet(Net):
    def __init__(self, name="QNet", *args, **kwargs):
        """
        Lagging QNet. If 'max_next_action', perform greedy
        Q-learning updates, else SARSA updates.
        """
        super().__init__(name=name, *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_net(self, grid, cell, name):
        base_net = self._build_base_net(grid, cell, name)
        with tf.variable_scope('model/' + name) as scope:
            if self.pp['dueling_qnet']:
                h1 = base_net
                # h1 = tf.layers.dense(
                #     inputs=base_net,
                #     units=140,
                #     kernel_initializer=self.kern_init_dense(),
                #     use_bias=False,
                #     name="h1")
                value = tf.layers.dense(
                    inputs=h1,
                    units=1,
                    kernel_initializer=self.kern_init_dense(),
                    use_bias=False,
                    name="value")
                advantages = tf.layers.dense(
                    inputs=h1,
                    units=self.n_channels,
                    use_bias=False,
                    kernel_initializer=self.kern_init_dense(),
                    name="advantages")
                # Avg. dueling supposedly more stable than max according to paper
                # Max Dueling
                # q_vals = value + (advantages - tf.reduce_max(
                #     advantages, axis=1, keepdims=True))
                # Average Dueling
                q_vals = value + (
                    advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
                if "online" in name:
                    self.advantages = advantages
                if "target" in name:
                    self.value = value
            else:
                q_vals = tf.layers.dense(
                    inputs=base_net,
                    units=self.n_channels,
                    kernel_initializer=self.kern_init_dense(),
                    kernel_regularizer=self.regularizer,
                    use_bias=False,
                    name="q_vals")
            trainable_vars_by_name = get_trainable_vars(scope)
        return q_vals, trainable_vars_by_name

    def build(self):
        depth = self.n_channels * 2 if self.pp['grid_split'] else self.n_channels
        gridshape = [None, self.pp['rows'], self.pp['cols'], depth]
        oh_cellshape = [None, self.pp['rows'], self.pp['cols'], 1]  # Onehot
        self.grid = tf.placeholder(boolean, gridshape, "grid")
        gridf = tf.cast(self.grid, float32)
        self.cell = tf.placeholder(int32, [None, 2], "cell")
        self.oh_cell = tf.placeholder(float32, oh_cellshape, "oh_cell")
        self.action = tf.placeholder(int32, [None], "action")
        self.reward = tf.placeholder(float32, [None], "reward")
        self.next_grid = tf.placeholder(boolean, gridshape, "next_grid")
        next_gridf = tf.cast(self.next_grid, float32)
        self.next_oh_cell = tf.placeholder(float32, oh_cellshape, "next_oh_cell")
        self.next_action = tf.placeholder(int32, [None], "next_action")
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

        # Maximum valued action from online network
        self.online_q_amax = tf.argmax(self.online_q_vals, axis=1, name="online_q_amax")
        # Maximum Q-value for given next state
        # Q-value for given action
        self.online_q_selected = tf.reduce_sum(
            self.online_q_vals * tf.one_hot(self.action, self.n_channels),
            axis=1,
            name="online_q_selected")

        # Target Q-value for given next action
        self.target_q_selected = tf.reduce_sum(
            target_q_vals * tf.one_hot(self.next_action, self.n_channels),
            axis=1,
            name="target_q_selected")
        if self.pp['dueling_qnet']:
            # WHAT?
            # self.next_q = tf.squeeze(self.value)
            self.next_q = self.target_q_selected
        elif self.pp['train_net']:
            self.next_q = tf.placeholder(shape=[None], dtype=float32, name="qtarget")
        else:
            self.next_q = self.target_q_selected

        self.q_target = self.reward + self.tf_gamma * self.next_q

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_target), predictions=self.online_q_selected)
        # # Write out statistics to file
        # with tf.name_scope("summaries"):
        #     tf.summary.scalar("loss", self.loss)
        #     # tf.summary.scalar("grad_norm", grad_norms)
        #     tf.summary.histogram("qvals", self.online_q_vals)
        # self.summaries = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(self.log_path + '/train',
        #                                           self.sess.graph)
        # self.eval_writer = tf.summary.FileWriter(self.log_path + '/eval')
        return online_vars

    def forward(self, grid, cell, ce_type):
        if self.pp['dueling_qnet']:
            q_vals_op = self.advantages
        else:
            q_vals_op = self.online_q_vals
        q_vals = self.sess.run(
            [q_vals_op],
            feed_dict={
                self.grid: prep_data_grids(grid, split=self.pp['grid_split']),
                self.oh_cell: prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        q_vals = np.reshape(q_vals, [-1])
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def backward(self,
                 grids,
                 cells,
                 actions,
                 rewards,
                 next_grids,
                 next_cells,
                 next_actions=None,
                 next_q=None,
                 gamma=None):
        """
        If 'next_actions' are specified, do SARSA update,
        else greedy selection (Q-Learning).
        If 'next_q', do supervised learning.
        """
        if gamma is None:
            gamma = self.gamma  # Not using beta-discount; use fixed constant
        data = {
            self.grid: prep_data_grids(grids, self.pp['grid_split']),
            self.oh_cell: prep_data_cells(cells),
            self.action: actions,
            self.reward: rewards,
        }
        if next_q is not None:
            # Pass explicit qval when supervised training on e.g. qtable
            data[self.next_q] = next_q
        else:
            p_next_grids = prep_data_grids(next_grids, self.pp['grid_split'])
            p_next_cells = prep_data_cells(next_cells)
            if next_actions is None:
                # TODO Can't this be done in a single pass
                next_actions = self.sess.run(
                    self.online_q_amax,
                    feed_dict={
                        self.grid: p_next_grids,
                        self.oh_cell: p_next_cells
                    })
            data.update({
                self.next_grid: p_next_grids,
                self.next_oh_cell: p_next_cells,
                self.next_action: next_actions,
                self.tf_gamma: [gamma]
            })
        _, loss, lr = self.sess.run(
            [self.do_train, self.loss, self.lr],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr
