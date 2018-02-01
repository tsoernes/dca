import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import copy_net_op, prep_data_cells, prep_data_grids


class QNet(Net):
    def __init__(self, max_next_action, *args, **kwargs):
        """
        Lagging QNet. If 'max_next_action', perform greedy
        Q-learning updates, else SARSA updates.
        """
        self.max_next_action = max_next_action
        name = "QLearnNet" if max_next_action else "SarsaNet"
        super().__init__(name=name, *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_net(self, grid, name):
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=self.n_channels,
                kernel_size=5,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,  # Default setting
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=4,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=70,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            q_vals = tf.layers.conv2d(
                inputs=conv3,
                filters=70,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            trainable_vars_by_name = self._get_trainable_vars(scope)
            return q_vals, trainable_vars_by_name

    def build(self):
        gridshape = [None, self.pp['rows'], self.pp['cols'], self.n_channels]
        cellshape = [None, self.pp['rows'], self.pp['cols'], 1]  # Onehot
        self.grid = tf.placeholder(shape=gridshape, dtype=tf.float32, name="grid")
        self.cell = tf.placeholder(shape=[None, 3], dtype=tf.int32, name="cell")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self.next_grid = tf.placeholder(shape=gridshape, dtype=tf.float32, name="next_grid")
        self.next_action = tf.placeholder(shape=[None], dtype=tf.int32, name="next_action")

        online_q_vals_all, online_vars = self._build_net(self.grid, name="q_networks/online")
        # Keep separate weights for target Q network
        target_q_vals_all, target_vars = self._build_net(self.next_grid, name="q_networks/target")
        # copy_online_to_target should be called periodically to creep
        # weights in the target Q-network towards the online Q-network
        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # self.online_q_vals = tf.squeeze(
        #     tf.gather_nd(online_q_vals_all, self.cell), axis=3)
        self.online_q_vals = tf.gather_nd(online_q_vals_all, self.cell)
        # Maximum valued action from online network
        self.online_q_amax = tf.argmax(self.online_q_vals, axis=1, name="online_q_amax")
        # Maximum Q-value for given next state
        # Q-value for given action
        self.online_q_selected = tf.reduce_sum(
            self.online_q_vals * tf.one_hot(self.action, self.n_channels),
            axis=1,
            name="online_q_selected")

        # Target Q-value for given next action
        # (for SARSA and eligibile Q-learning)
        # self.target_q = tf.squeeze(
        #     tf.reduce_mean(target_q_vals_all, axis=1, name="target_next_q"),
        #     axis=1)
        self.target_q = tf.reduce_mean(target_q_vals_all, axis=[1, 2, 3], name="target_next_q")
        self.q_target = self.reward + self.gamma * self.target_q

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_target), predictions=self.online_q_selected)
        self.do_train = self._build_default_trainer(online_vars)

    def forward(self, grid, cell):
        q_vals_op = self.online_q_vals
        q_vals = self.sess.run(
            [q_vals_op],
            feed_dict={
                self.grid: prep_data_grids(grid, empty_neg=self.pp['empty_neg']),
                self.cell: [(0, *cell)]
            },
            options=self.options,
            run_metadata=self.run_metadata)
        q_vals = np.reshape(q_vals, [-1])
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"

        return q_vals

    def backward(self, grids, cells, actions, rewards, next_grids, next_cells, next_actions=None):
        """
        If 'next_actions' are specified, do SARSA update,
        else greedy selection (Q-Learning)
        """
        p_next_grids = prep_data_grids(next_grids, self.pp['empty_neg'])
        data = {
            self.grid: prep_data_grids(grids, self.pp['empty_neg']),
            self.cell: [(0, *cells)],
            self.action: actions,
            self.reward: rewards,
            self.next_grid: p_next_grids,
        }
        if next_actions is not None:
            data[self.next_action] = next_actions
        else:
            na = self.sess.run(
                self.online_q_amax,
                feed_dict={
                    self.grid: p_next_grids,
                    self.cell: [(0, *next_cells)]
                })
            data[self.next_action] = na
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss
