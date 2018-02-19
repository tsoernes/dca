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
        self.grids = tf.placeholder(boolean, gridshape, "grid")
        self.oh_cells = tf.placeholder(float32, oh_cellshape, "oh_cell")
        self.chs = tf.placeholder(int32, [None], "ch")
        self.rewards = tf.placeholder(float32, [None], "reward")
        self.next_grids = tf.placeholder(boolean, gridshape, "next_grid")
        self.oh_next_cells = tf.placeholder(float32, oh_cellshape, "next_oh_cell")
        self.next_chs = tf.placeholder(int32, [None], "next_ch")
        # Allows for passing in varying gamma, e.g. beta discount
        self.tf_gamma = tf.placeholder(float32, [1], "gamma")
        fgrids = tf.cast(self.grids, float32)
        next_fgrids = tf.cast(self.next_grids, float32)
        chs_range = tf.range(tf.shape(self.chs)[0])
        next_chs_range = tf.range(tf.shape(self.next_chs)[0])
        # numb_chs = [[0, ch0], [1, ch1], [2, ch2], ..] where ch0=chs[0]
        numb_chs = tf.stack([chs_range, self.chs], axis=1)
        numb_next_chs = tf.stack([next_chs_range, self.next_chs], axis=1)

        self.online_q_vals, online_vars = self._build_net(
            fgrids, self.oh_cells, name="q_networks/online")
        # Keep separate weights for target Q network
        target_q_vals, target_vars = self._build_net(
            next_fgrids, self.oh_next_cells, name="q_networks/target")
        # copy_online_to_target should be called periodically to creep
        # weights in the target Q-network towards the online Q-network
        self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                 self.pp['net_creep_tau'])

        # Maximum valued ch from online network
        self.online_q_amax = tf.argmax(self.online_q_vals, axis=1, name="online_q_amax")
        # Q-value for given ch
        self.online_q_selected = tf.gather_nd(self.online_q_vals, numb_chs)
        # Target Q-value for given next ch
        self.target_q_selected = tf.gather_nd(target_q_vals, numb_next_chs)

        # TODO This part is abit messy. Not sure if duel target is correct either.
        if self.pp['dueling_qnet']:
            # WHAT?
            # self.next_q = tf.squeeze(self.value)
            self.next_q = self.target_q_selected
        elif self.pp['train_net']:
            self.next_q = tf.placeholder(shape=[None], dtype=float32, name="next_q")
        else:
            self.next_q = self.target_q_selected

        if self.pp['strat'].lower() == "nqlearnnet":
            self.q_target = tf.placeholder(shape=[None], dtype=float32, name="qtarget")
        else:
            self.q_target = self.rewards + self.tf_gamma * self.next_q

        # Sum of squares difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_target), predictions=self.online_q_selected)
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

    def n_step_backward(self, grid, cell, ch, rewards, next_grid, next_cell):
        """
        Update Q(grid, cell, ch) = Q(grid, cell, ch) + net_lr *
            [reward[0] + gamma*reward[1] + (gamma**2)*reward[2] + ...
             + (gamma**n)*Q(next_grid, next_cell, next_max_ch) - Q(grid, cell, ch)]
        where n=len(rewards), and next_max_ch is argmax q from online net
        """
        p_next_grids = prep_data_grids(next_grid, self.pp['grid_split'])
        p_next_cells = prep_data_cells(next_cell)

        next_ch = self.sess.run(
            self.online_q_amax,
            feed_dict={
                self.grids: p_next_grids,
                self.oh_cells: p_next_cells
            })
        q_next = self.sess.run(
            self.target_q_selected,
            feed_dict={
                self.next_grids: p_next_grids,
                self.oh_next_cells: p_next_cells,
                self.next_chs: next_ch
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
        _, loss, lr = self.sess.run(
            [self.do_train, self.loss, self.lr],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr

    def backward(self,
                 grids,
                 cells,
                 chs,
                 rewards,
                 next_grids,
                 next_cells,
                 next_chs=None,
                 next_q=None,
                 gamma=None):
        """
        If 'next_chs' are specified, do SARSA update,
        else greedy selection (Q-Learning).
        If 'next_q', do supervised learning.
        """
        if gamma is None:
            gamma = self.gamma  # Not using beta-discount; use fixed constant
        data = {
            self.grids: prep_data_grids(grids, self.pp['grid_split']),
            self.oh_cells: prep_data_cells(cells),
            self.chs: chs,
            self.rewards: rewards,
        }
        if next_q is not None:
            # Pass explicit qval when supervised training on e.g. qtable
            data[self.next_q] = next_q
        else:
            p_next_grids = prep_data_grids(next_grids, self.pp['grid_split'])
            p_next_cells = prep_data_cells(next_cells)
            if next_chs is None:
                # TODO Can't this be done in a single pass
                next_chs = self.sess.run(
                    self.online_q_amax,
                    feed_dict={
                        self.grids: p_next_grids,
                        self.oh_cells: p_next_cells
                    })
            data.update({
                self.next_grids: p_next_grids,
                self.oh_next_cells: p_next_cells,
                self.next_chs: next_chs,
                self.tf_gamma: [gamma]
            })
        _, loss, lr = self.sess.run(
            [self.do_train, self.loss, self.lr],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr
