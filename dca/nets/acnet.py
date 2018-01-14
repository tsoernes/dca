import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.net import Net
from nets.utils import normalized_columns_initializer


class ACNet(Net):
    def __init__(self, max_next_action=True, *args, **kwargs):
        """
        Lagging QNet. If 'max_next_action', perform greedy
        Q-learning updates, else SARSA updates.
        """
        self.max_next_action = max_next_action
        super().__init__(name="ACNet", *args, **kwargs)

    def _build_net(self, grid, cell, name):
        with tf.variable_scope(name) as scope:
            conv1 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=grid,
                num_outputs=70,
                kernel_size=5,
                padding='SAME')
            conv2 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=conv1,
                num_outputs=70,
                kernel_size=4,
                padding='SAME')
            stacked = tf.concat([conv2, cell], axis=3)
            hidden = slim.fully_connected(
                slim.flatten(stacked), 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(grid)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=state_in,
                sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            policy = slim.fully_connected(
                rnn_out,
                70,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            value = slim.fully_connected(
                rnn_out,
                1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
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
        self.reward = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward")
        self.next_grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="next_grid")
        self.next_cell = tf.placeholder(
            shape=cellshape, dtype=tf.float32, name="next_cell")
        self.next_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="next_action")

        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        self.policy, self.value, trainable_vars = self._build_net(
            self.grid, self.cell, name="ac_network")
        target_q_vals, target_vars = self._build_qnet(
            self.next_grid, self.next_cell, name="q_networks/target")

        self.value_loss = 0.5 * tf.reduce_sum(
            tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_sum(
            tf.log(self.responsible_outputs) * self.advantages)
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        # trainer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.MomentumOptimizer(
        #     learning_rate=self.alpha, momentum=0.95)
        gradients = tf.gradients(self.loss, trainable_vars)
        clipped_grads, self.grad_norms = tf.clip_by_global_norm(
            gradients, 40.0)

        # Apply gradients to network
        self.do_train = trainer.apply_gradients(
            zip(clipped_grads, trainable_vars))

        # Write out statistics to file
        with tf.name_scope("summaries"):
            tf.summary.scalar("learning_rate", self.alpha)
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("qvals", self.online_q_vals)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_path + '/train',
                                                  self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(self.log_path + '/eval')

    def forward(self, grid, cell):
        q_vals = self.sess.run(
            [self.online_q_vals],
            feed_dict={
                self.grid: self.prep_data_grids(grid),
                self.cell: self.prep_data_cells(cell)
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
            self.grid: self.prep_data_grids(grids),
            self.cell: self.prep_data_cells(cells),
            self.action: actions,
            self.reward: rewards,
            self.next_grid: self.prep_data_grids(next_grids),
            self.next_cell: self.prep_data_cells(next_cells),
        }
        if next_actions:
            data[self.next_action] = next_actions
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"Invalid loss: {loss}")
        return loss
