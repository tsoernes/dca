import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.net import Net
from nets.utils import copy_net_op, get_trainable_vars, prep_data_grids


class BigHeadQNet(Net):
    def __init__(self, name, *args, **kwargs):
        """
        Lagging Double QNet. Can do supervised learning, Q-Learning, SARSA.
        Optionally duelling architecture.
        """
        super().__init__(name=name, *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_base_net(self, grid, ncell, name):
        with tf.variable_scope('model/' + name) as scope:
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
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            q_vals = tf.gather_nd(conv3, ncell)
            trainable_vars_by_name = get_trainable_vars(scope)
            return q_vals, trainable_vars_by_name

    def build(self):
        depth = self.n_channels * 2 if self.grid_split else self.n_channels
        gridshape = [None, self.rows, self.cols, depth]
        self.grids = tf.placeholder(boolean, gridshape, "grid")
        nrange = tf.range(tf.shape(self.grids)[0])
        self.cells = tf.placeholder(int32, [None, 2], "cell")
        cells = tf.concat([tf.expand_dims(nrange, axis=1), self.cells], axis=1)
        self.chs = tf.placeholder(int32, [None], "ch")
        self.q_targets = tf.placeholder(float32, [None], "qtarget")
        self.weights = 1

        grids_f = tf.cast(self.grids, float32)
        # numbered_chs: [[0, ch0], [1, ch1], [2, ch2], ..., [n, ch_n]]
        numbered_chs = tf.stack([nrange, self.chs], axis=1)

        # Create online and target networks
        self.online_q_vals, online_vars = self._build_base_net(
            grids_f, cells, name="q_networks/online")
        target_q_vals, target_vars = self._build_base_net(
            grids_f, cells, name="q_networks/target")

        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # Maximum valued ch from online network
        self.online_q_amax = tf.argmax(
            self.online_q_vals, axis=1, name="online_q_amax", output_type=int32)
        # Target Q-value for greedy channel as selected by online network
        numbered_q_amax = tf.stack([nrange, self.online_q_amax], axis=1)
        self.target_q_max = tf.gather_nd(target_q_vals, numbered_q_amax)
        # Target Q-value for given ch
        self.target_q_selected = tf.gather_nd(target_q_vals, numbered_chs)
        # Online Q-value for given ch
        online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)

        self.td_err = self.q_targets - online_q_selected
        self.loss = tf.losses.mean_squared_error(
            labels=self.q_targets, predictions=online_q_selected, weights=self.weights)
        return online_vars

    def forward(self, grid, cell, ce_type, frep=None):
        data = {
            self.grids: prep_data_grids(grid, split=self.grid_split),
        }
        data[self.cells] = [cell]
        if frep is not None:
            data[self.freps] = [frep]
        if self.pp['dueling_qnet']:
            q_vals_op = self.advantages
        else:
            q_vals_op = self.online_q_vals
        q_vals = self.sess.run(
            q_vals_op, data, options=self.options, run_metadata=self.run_metadata)
        q_vals = q_vals[0]
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def _backward(self, data) -> (float, float):
        _, loss, lr, td_err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.td_err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, td_err

    def backward_supervised(self, grids, cells, chs, q_targets, freps=None, weights=None):
        data = {
            self.grids: prep_data_grids(grids, self.grid_split),
            self.chs: chs,
            self.q_targets: q_targets,
        }
        data[self.cells] = [cells]
        if freps is not None:
            data[self.freps] = freps
        if weights is not None:
            data[self.weights] = weights
        return self._backward(data)

    def _double_q_target(self, grids, cells, freps=None, target_chs=None) -> [float]:
        """Find bootstrap value, i.e. Q(Stn, A; Wt).
        where Stn: state at time t+n
              A: target_chs, if specified, else argmax(Q(Stn, a; Wo))
              n: usually 1, unless n-step Q-learning
              Wo/Wt: online/target network"""
        data = {
            self.grids: prep_data_grids(grids, self.grid_split),
        }
        data[self.cells] = [cells]
        if target_chs is None:
            # Greedy Q-Learning
            target_q = self.target_q_max
        else:
            # SARSA or Eligible Q-learning
            target_q = self.target_q_selected
            data[self.chs] = target_chs
        if freps is not None:
            data[self.freps] = freps
        qvals = self.sess.run(target_q, data)
        return qvals

    def backward(self,
                 grids,
                 cells,
                 chs,
                 rewards,
                 next_grids,
                 next_cells,
                 gamma,
                 freps=None,
                 next_freps=None,
                 next_chs=None,
                 weights=None) -> (float, float):
        """
        Supports n-step learning where (grids, cells) is from time t
        and (next_grids, next_cells) is from time t+n
        Support greedy action selection if 'next_chs' is None
        Feature representations (freps) of grids are optional
        """
        next_qvals = self._double_q_target(next_grids, next_cells, next_freps, next_chs)
        q_targets = next_qvals
        for reward in rewards[::-1]:
            q_targets = reward + gamma * q_targets
        return self.backward_supervised(grids, cells, chs, q_targets, freps, weights)
