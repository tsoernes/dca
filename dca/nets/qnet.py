import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.net import Net
from nets.utils import (copy_net_op, get_trainable_vars, prep_data_cells,
                        prep_data_grids)


class QNet(Net):
    def __init__(self, name, *args, **kwargs):
        """
        Lagging Double QNet. Can do supervised learning, Q-Learning, SARSA.
        Optionally duelling architecture.
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
        self.grids = tf.placeholder(boolean, gridshape, "grid")
        self.oh_cells = tf.placeholder(float32, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(int32, [None], "ch")
        # self.next_grids = tf.placeholder(boolean, gridshape, "next_grid")
        # self.oh_next_cells = tf.placeholder(float32, oh_cellshape, "next_oh_cell")
        self.q_targets = tf.placeholder(float32, [None], "qtarget")

        fgrids = tf.cast(self.grids, float32)
        # next_fgrids = tf.cast(self.next_grids, float32)
        nrange = tf.range(tf.shape(self.grids)[0])
        # numbered_chs = [[0, ch0], [1, ch1], [2, ch2], ...]
        numbered_chs = tf.stack([nrange, self.chs], axis=1)

        self.online_q_vals, online_vars = self._build_net(
            fgrids, self.oh_cells, name="q_networks/online")
        # Keep separate weights for target Q network
        target_q_vals, target_vars = self._build_net(
            fgrids, self.oh_cells, name="q_networks/target")
        # target_q_vals, target_vars = self._build_net(
        #     next_fgrids, self.oh_next_cells, name="q_networks/target")
        # copy_online_to_target should be called periodically to creep
        # weights in the target Q-network towards the online Q-network
        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # Maximum valued ch from online network
        self.online_q_amax = tf.argmax(
            self.online_q_vals, axis=1, name="online_q_amax", output_type=int32)
        # Target Q-value for greedy channel as selected by online network
        numbered_q_amax = tf.stack([nrange, self.online_q_amax], axis=1)
        self.target_q_max = tf.gather_nd(target_q_vals, numbered_q_amax)
        # Q-value for given ch
        online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)

        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_targets), predictions=online_q_selected)
        return online_vars

    def forward(self, grid, cell, ce_type):
        if self.pp['dueling_qnet']:
            q_vals_op = self.advantages
        else:
            q_vals_op = self.online_q_vals
        q_vals = self.sess.run(
            [q_vals_op],
            feed_dict={
                self.grids: prep_data_grids(grid, split=self.pp['grid_split']),
                self.oh_cells: prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        q_vals = q_vals[0][0]
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def _backward(self, data):
        _, loss, lr = self.sess.run(
            [self.do_train, self.loss, self.lr],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr

    def backward_supervised(self, grids, cells, chs, q_targets):
        data = {
            self.grids: prep_data_grids(grids, self.pp['grid_split']),
            self.oh_cells: prep_data_cells(cells),
            self.chs: chs,
            self.q_targets: q_targets,
        }
        return self._backward(data)

    def backward_max_next(self, grids, cells, chs, rewards, next_grids, next_cells,
                          gamma):
        next_qvals = self.sess.run(
            self.target_q_max,
            feed_dict={
                self.grids: prep_data_grids(next_grids, self.pp['grid_split']),
                self.oh_cells: prep_data_cells(next_cells)
            })
        q_targets = rewards + gamma * next_qvals
        return self.backward_supervised(grids, cells, chs, q_targets)

    def backward_nstep(self, grid, cell, ch, rewards, next_grid, next_cell):
        """
        Update Q(grid, cell, ch) = Q(grid, cell, ch) + net_lr *
            [reward[0] + gamma*reward[1] + (gamma**2)*reward[2] + ...
             + (gamma**n)*Q(next_grid, next_cell, next_max_ch) - Q(grid, cell, ch)]
        where n=len(rewards), and next_max_ch is argmax q from online net
        """

        # Find bootstrap value, i.e. Q(Stn, argmax(Stn, A; Wo); Wt)
        # where Stn: state at time t+n; Wo/Wt: online/target network
        q_next = self.sess.run(
            self.target_q_max,
            feed_dict={
                self.grids: prep_data_grids(next_grid, self.pp['grid_split']),
                self.oh_cells: prep_data_cells(next_cell)
            })
        q_target = q_next
        for reward in rewards[::-1]:
            q_target = reward + self.gamma * q_target

        data = {
            self.grids: prep_data_grids(grid, self.pp['grid_split']),
            self.oh_cells: prep_data_cells(cell),
            self.chs: [ch],
            self.q_target: q_target
        }
        return self._backward(data)
