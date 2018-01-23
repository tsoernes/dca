import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import prep_data_cells, prep_data_grids


class QNet(Net):
    def __init__(self, max_next_action, *args, **kwargs):
        """
        Lagging QNet. If 'max_next_action', perform greedy
        Q-learning updates, else SARSA updates.
        """
        self.max_next_action = max_next_action
        super().__init__(name="QNet", *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_qnet(self, grid, cell, name):
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=70,
                kernel_size=4,
                padding="same",
                activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)
            stacked = tf.concat([conv2, cell], axis=3)
            conv2_flat = tf.layers.flatten(stacked)
            q_vals = tf.layers.dense(
                inputs=conv2_flat, units=70, name="q_vals")
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {
            var.name[len(scope.name):]: var
            for var in trainable_vars
        }
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

        copy_ops = [
            target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()
        ]
        # copy_online_to_target should be called periodically to copy
        # weights from online Q-network to target Q-network
        self.copy_online_to_target = tf.group(*copy_ops)

        # Maximum valued action from online network
        self.online_q_amax = tf.argmax(
            self.online_q_vals, axis=1, name="online_q_amax")
        # Maximum Q-value for given next state
        self.target_q_max = tf.reduce_max(
            target_q_vals, axis=1, name="target_q_max")
        # Q-value for given action
        self.online_q_selected = tf.reduce_sum(
            self.online_q_vals * tf.one_hot(self.action,
                                            self.pp['n_channels']),
            axis=1,
            name="online_q_selected")
        # Target Q-value for given next action
        self.target_q_selected = tf.reduce_sum(
            target_q_vals * tf.one_hot(self.next_action,
                                       self.pp['n_channels']),
            axis=1,
            name="target_q_selected_next")

        if self.max_next_action:
            # Target for Q-learning
            next_actionval = self.target_q_max
        else:
            # Target for SARSA
            next_actionval = self.target_q_selected
        self.q_target = self.reward + self.gamma * next_actionval

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_target),
            predictions=self.online_q_selected)

        # trainer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=self.l_rate)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=self.l_rate)
        trainer = tf.train.MomentumOptimizer(
            learning_rate=self.l_rate, momentum=0.95)

        # self.max_grad_norm = 100000
        # gradients, trainable_vars = zip(*trainer.compute_gradients(
        #     self.loss, var_list=online_vars))
        # clipped_grads, grad_norms = tf.clip_by_global_norm(
        #     gradients, self.max_grad_norm)
        # self.do_train = trainer.apply_gradients(
        #     zip(clipped_grads, trainable_vars))
        self.do_train = trainer.minimize(
            self.loss, var_list=online_vars)
        # Write out statistics to file
        # with tf.name_scope("summaries"):
        #     tf.summary.scalar("learning_rate", self.l_rate)
        #     tf.summary.scalar("loss", self.loss)
        #     tf.summary.scalar("grad_norm", grad_norms)
        #     tf.summary.histogram("qvals", self.online_q_vals)
        # self.summaries = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(self.log_path + '/train',
        #                                           self.sess.graph)
        # self.eval_writer = tf.summary.FileWriter(self.log_path + '/eval')

    def forward(self, grid, cell):
        q_vals = self.sess.run(
            [self.online_q_vals],
            feed_dict={
                self.grid: prep_data_grids(grid),
                self.cell: prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
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
            self.grid: prep_data_grids(grids),
            self.cell: prep_data_cells(cells),
            self.action: actions,
            self.reward: rewards,
            self.next_grid: prep_data_grids(next_grids),
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
