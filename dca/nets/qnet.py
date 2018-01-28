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

    def _build_qnet(self, grid, cell, name):
        base_net = self._build_base_net(grid, cell, name)
        with tf.variable_scope(name) as scope:
            if self.pp['dueling_qnet']:
                value = tf.layers.dense(
                    inputs=base_net,
                    units=1,
                    kernel_initializer=self.kern_init_dense(),
                    name="value")
                advantage = tf.layers.dense(
                    inputs=base_net,
                    units=self.n_channels,
                    kernel_initializer=self.kern_init_dense(),
                    name="advantage")
                # Max Dueling
                # self.q_vals= value + (advantage -
                #     tf.reduce_max(advantage, axis=1, keep_dims=True))
                # Average Dueling
                q_vals = value + (advantage - tf.reduce_mean(
                    advantage, axis=1, keep_dims=True))
            else:
                q_vals = tf.layers.dense(
                    inputs=base_net,
                    units=self.n_channels,
                    kernel_initializer=self.kern_init_dense(),
                    kernel_regularizer=self.regularizer,
                    name="q_vals")
            trainable_vars_by_name = self._get_trainable_vars(scope)
        return q_vals, trainable_vars_by_name

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
        self.reward = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward")
        self.next_grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="next_grid")
        self.next_cell = tf.placeholder(
            shape=cellshape, dtype=tf.float32, name="next_cell")
        self.next_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="next_action")

        # Keep separate weights for target Q network
        self.online_q_vals, online_vars = self._build_qnet(
            self.grid, self.cell, name="q_networks/online")
        target_q_vals, target_vars = self._build_qnet(
            self.next_grid, self.next_cell, name="q_networks/target")

        # copy_online_to_target should be called periodically to creep
        # weights in the target Q-network towards the online Q-network
        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # Maximum valued action from online network
        # Not in use
        # self.online_q_amax = tf.argmax(
        #     self.online_q_vals, axis=1, name="online_q_amax")
        # Maximum Q-value for given next state
        self.target_q_max = tf.reduce_max(
            target_q_vals, axis=1, name="target_q_max")
        # Q-value for given action
        self.online_q_selected = tf.reduce_sum(
            self.online_q_vals * tf.one_hot(self.action, self.n_channels),
            axis=1,
            name="online_q_selected")
        # Target Q-value for given next action
        self.target_q_selected = tf.reduce_sum(
            target_q_vals * tf.one_hot(self.next_action, self.n_channels),
            axis=1,
            name="target_next_q_selected")

        if self.max_next_action:
            # Target for Q-learning
            next_q = self.target_q_max
        else:
            # Target for SARSA and eligibile Q-learning
            next_q = self.target_q_selected
        self.q_target = self.reward + self.gamma * next_q

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_target),
            predictions=self.online_q_selected)
        self.do_train = self._build_default_trainer(online_vars)
        # Write out statistics to file
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            # tf.summary.scalar("grad_norm", grad_norms)
            tf.summary.histogram("qvals", self.online_q_vals)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_path + '/train',
                                                  self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(self.log_path + '/eval')

    def forward(self, grid, cell):
        q_vals = self.sess.run(
            [self.online_q_vals],
            feed_dict={
                self.grid: prep_data_grids(
                    grid, empty_neg=self.pp['empty_neg']),
                self.cell: prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        assert q_vals[0].shape == (1, self.n_channels)
        q_vals = np.reshape(q_vals, [-1])
        return q_vals

    def backward(self,
                 grids,
                 cells,
                 actions,
                 rewards,
                 next_grids,
                 next_cells,
                 next_actions=None):
        """
        If 'next_actions' are specified, do SARSA update,
        else greedy selection (Q-Learning)
        """
        data = {
            self.grid: prep_data_grids(grids, self.pp['empty_neg']),
            self.cell: prep_data_cells(cells),
            self.action: actions,
            self.reward: rewards,
            self.next_grid: prep_data_grids(next_grids, self.pp['empty_neg']),
            self.next_cell: prep_data_cells(next_cells),
        }
        if next_actions is not None:
            data[self.next_action] = next_actions
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"Invalid loss: {loss}")
        return loss
