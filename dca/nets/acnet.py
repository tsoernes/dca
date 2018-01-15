import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import nets.utils as nutils
from nets.net import Net


class ACNet(Net):
    def __init__(self, *args, **kwargs):
        """
        """
        self.max_grad_norm = 40.0
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

            # Output layers for policy and value estimations
            policy = slim.fully_connected(
                hidden,
                70,
                activation_fn=tf.nn.softmax,
                weights_initializer=nutils.normalized_columns_initializer(
                    0.01),
                biases_initializer=None)
            value = slim.fully_connected(
                hidden,
                1,
                activation_fn=None,
                weights_initializer=nutils.normalized_columns_initializer(1.0),
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

        self.policy, self.value, trainable_vars = self._build_net(
            self.grid, self.cell, name="ac_network")
        target_q_vals, target_vars = self._build_qnet(
            self.next_grid, self.next_cell, name="q_networks/target")

        action_oh = tf.one_hot(
            self.actions, self.pp['n_channels'], dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * action_oh, [1])

        # TODO Perhaps these should be 'reduce_mean' instead.
        self.value_loss = tf.reduce_sum(
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
            gradients, self.max_grad_norm)
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
        a_dist, val = self.sess.run(
            [self.policy, self.value],
            feed_dict={
                self.grid: nutils.prep_data_grids(grid),
                self.cell: nutils.prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a, val

    def backward(self,
                 grids,
                 cells,
                 actions,
                 rewards,
                 next_grids,
                 next_cells,
                 next_actions=None):
        data = {
            self.grid: nutils.prep_data_grids(grids),
            self.cell: nutils.prep_data_cells(cells),
            self.action: actions,
            self.reward: rewards,
            self.next_grid: nutils.prep_data_grids(next_grids),
            self.next_cell: nutils.prep_data_cells(next_cells),
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
