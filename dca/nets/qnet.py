import numpy as np
import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.net import Net
from nets.utils import (copy_net_op, discount, get_trainable_vars,
                        prep_data_cells, prep_data_grids)


class QNet(Net):
    def __init__(self, name, *args, **kwargs):
        """
        Lagging Double QNet. Can do supervised learning, Q-Learning, SARSA.
        Optionally duelling architecture.
        """
        super().__init__(name=name, *args, **kwargs)
        self.sess.run(self.copy_online_to_target)

    def _build_net(self, grid, frep, cell, name):
        base_net = self._build_base_net(grid, frep, cell, name)
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
        depth = self.n_channels * 2 if self.grid_split else self.n_channels
        gridshape = [None, self.rows, self.cols, depth]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        oh_cellshape = [None, self.rows, self.cols, 1]  # Onehot
        self.grids = tf.placeholder(boolean, gridshape, "grid")
        self.freps = tf.placeholder(int32, frepshape, "frep")
        self.oh_cells = tf.placeholder(boolean, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(int32, [None], "ch")
        self.q_targets = tf.placeholder(float32, [None], "qtarget")

        grids_f = tf.cast(self.grids, float32)
        freps_f = tf.cast(self.freps, float32)
        oh_cells_f = tf.cast(self.oh_cells, float32)
        mult1 = np.ones(frepshape[1:], np.float32)
        mult1[:, :, :-1] /= 43
        mult1[:, :, -1] /= 70
        tmult1 = tf.constant(mult1)
        freps_f = freps_f * tmult1
        nrange = tf.range(tf.shape(self.grids)[0])
        # numbered_chs: [[0, ch0], [1, ch1], [2, ch2], ..., [n, ch_n]]
        numbered_chs = tf.stack([nrange, self.chs], axis=1)

        self.online_q_vals, online_vars = self._build_net(
            grids_f, freps_f, oh_cells_f, name="q_networks/online")
        # Keep separate weights for target Q network
        target_q_vals, target_vars = self._build_net(
            grids_f, freps_f, oh_cells_f, name="q_networks/target")
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
        # Target Q-value for given ch
        self.target_q_selected = tf.gather_nd(target_q_vals, numbered_chs)
        # Online Q-value for given ch
        online_q_selected = tf.gather_nd(self.online_q_vals, numbered_chs)
        # Sum of squares difference between the target and prediction Q values.
        if self.pp['huber_loss'] is not None:
            self.loss = tf.losses.huber_loss(
                labels=self.q_targets,
                predictions=online_q_selected,
                reduction=tf.losses.Reduction.MEAN,
                delta=self.pp['huber_loss'])
        else:
            self.loss = tf.losses.mean_squared_error(
                labels=self.q_targets, predictions=online_q_selected)

        # td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        # errors = U.huber_loss(td_error)
        # weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # tf.losses.huber_loss
        return online_vars

    def forward(self, grid, frep, cell, ce_type):
        if self.pp['dueling_qnet']:
            q_vals_op = self.advantages
        else:
            q_vals_op = self.online_q_vals
        data = {
            self.grids: prep_data_grids(grid, split=self.grid_split),
            self.oh_cells: prep_data_cells(cell)
        }
        if frep is not None:
            data[self.freps] = [frep]
        q_vals = self.sess.run(
            [q_vals_op], data, options=self.options, run_metadata=self.run_metadata)
        q_vals = q_vals[0][0]
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def _backward(self, data) -> (float, float):
        _, loss, lr = self.sess.run(
            [self.do_train, self.loss, self.lr],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr

    def backward_supervised(self, grids, freps, cells, chs, q_targets):
        data = {
            self.grids: prep_data_grids(grids, self.grid_split),
            self.oh_cells: prep_data_cells(cells),
            self.chs: chs,
            self.q_targets: q_targets,
        }
        if freps is not None:
            data[self.freps] = freps
        return self._backward(data)

    def _double_q_target(self, grids, freps, cells, target_chs=None) -> [float]:
        """Find bootstrap value, i.e. Q(Stn, A; Wt).
        where Stn: state at time t+n
              A: target_chs, if specified, else argmax(Q(Stn, a; Wo))
              Wo/Wt: online/target network"""
        data = {
            self.grids: prep_data_grids(grids, self.grid_split),
            self.oh_cells: prep_data_cells(cells)
        }
        if target_chs is None:
            target_q = self.target_q_max
        else:
            target_q = self.target_q_selected
            data[self.chs] = target_chs
        if freps is not None:
            data[self.freps] = freps
        qvals = self.sess.run(target_q, data)
        return qvals

    def backward(self, grids, freps, cells, chs, rewards, next_grids, next_freps,
                 next_cells, next_chs, gamma) -> (float, float):
        """
        Supports n-step learning where (grids, cells) is from time t
        and (next_grids, next_cells) is from time t+n
        Support greedy action selection if 'next_chs' is None
        """
        next_qvals = self._double_q_target(next_grids, next_freps, next_cells, next_chs)
        q_targets = next_qvals
        for reward in rewards[::-1]:
            q_targets = reward + gamma * q_targets
        return self.backward_supervised(grids, freps, cells, chs, q_targets)

    def backward_multi_nstep(self, grids, cells, chs, rewards, next_grid, next_cell,
                             next_ch, gamma) -> (float, float):
        """
        Multi n-step. Train on n-step, (n-1)-step, (n-2)-step, ..., 1-step returns
        """
        next_qvals = self._double_q_target(next_grid, next_cell, next_ch)
        rewards_plus = np.asarray(rewards + [next_qvals])
        # [(r0 + g*r1 + g**2*r2 +..+ g**n*q_n), (r1 + g*r2 +..+ g**(n-1)*q_n), ..,
        # (r(n-1) + g*q_n)] where g: gamma, q_n: next_qval (bootstrap val)
        q_targets = discount(rewards_plus, gamma)[:-1]
        return self.backward_supervised(grids, cells, chs, q_targets)

    def backward_gae(self, grids, cells, chs, rewards, next_grid, next_cell, next_ch,
                     gamma) -> (float, float):
        """Generalized Advantage Estimation"""
        next_qvals = self._double_q_target(next_grid, next_cell, next_ch)
        vals = self.sess.run(
            self.target_q_max, {
                self.grids: prep_data_grids(grids, self.grid_split),
                self.oh_cells: prep_data_cells(cells),
                self.chs: chs
            })
        value_plus = np.zeros((len(vals) + 1))
        value_plus[:len(vals)] = vals
        value_plus[-1] = next_qvals
        advantages = discount(rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)
        return self.backward_supervised(grids, cells, chs, advantages)
