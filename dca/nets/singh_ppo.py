import numpy as np
import tensorflow as tf

from eventgen import CEvent
from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_cells


class PPOSinghNet(Net):
    def __init__(self, pre_conv=False, double_net=False, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        self.pre_conv = pre_conv
        self.double_net = double_net
        super().__init__(name=self.name, *args, **kwargs)
        self.weight_beta = self.pp['weight_beta']
        self.weight_beta_decay = self.pp['weight_beta_decay']
        self.avg_reward = 0

    def _build_vnet(self, freps, name):
        with tf.variable_scope('model/' + name) as scope:
            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.dense_regularizer,
                use_bias=False,
                activation=None)
            value = value_layer.apply(tf.layers.flatten(freps))
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            # NOTE TODO either gotta have 7x7x70 outputs, or input cell
            # also gotta think about a hidden layer before value/policy
            trainable_vars = get_trainable_vars(scope)
        return value, trainable_vars

    def _build_pnet(self, freps, name):
        with tf.variable_scope('model/' + name) as scope:
            # policy = tf.keras.layers.LocallyConnected2D(
            #     filters=70,
            #     kernel_size=1,
            #     padding="valid",
            #     kernel_initializer=tf.zeros_initializer(),
            #     use_bias=self.pp['conv_bias'],
            #     activation=None)(freps)
            # print(policy.shape)
            policy_layer = tf.layers.Dense(
                units=70,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.dense_regularizer,
                use_bias=False,
                activation=None)
            policy = policy_layer.apply(tf.layers.flatten(freps))
            # self.weight_vars.append(policy_layer.kernel)
            # self.weight_names.append(policy_layer.name)
            trainable_vars = get_trainable_vars(scope)
        return policy, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None], "value_target")
        oh_cellshape = [None, self.rows, self.cols, 1]
        self.cells = tf.placeholder(tf.bool, oh_cellshape, "oh_cell")
        cells = tf.cast(self.cells, tf.float32)
        # self.cells = tf.placeholder(tf.int32, [None, 2], "cell")
        self.action = tf.placeholder(tf.int32, [None], "action")
        self.policy_in = tf.placeholder(tf.float32, [70], "pol_in")
        self.old_neglogpac = tf.placeholder(tf.float32, [None], "old_neglogpac")
        inp = tf.concat([self.freps, cells], axis=3)
        self.value, online_vf_vars = self._build_vnet(self.freps, "online-vf")
        self.policy, online_pg_vars = self._build_pnet(inp, "online-pg")
        # nrange = tf.range(tf.shape(self.freps)[0], name="cellrange")
        # ncells = tf.concat([tf.expand_dims(nrange, axis=1), self.cells], axis=1)
        # self.policy = tf.gather_nd(self.conv_policy, ncells)

        self.err = self.value_target - self.value
        self.vf_loss = tf.losses.mean_squared_error(
            labels=tf.expand_dims(self.value_target, axis=1), predictions=self.value)

        CLIPRANGE = 0.2
        self.neglogpac_out = self.neglogp(self.policy_in, self.action)
        self.neglogpac = self.neglogp(self.policy, self.action)
        ratio = tf.exp(self.old_neglogpac - self.neglogpac)
        pg_losses = -self.value_target * ratio
        pg_losses2 = -self.value_target * tf.clip_by_value(ratio, 1.0 - CLIPRANGE,
                                                           1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # entropy = self.entropy(self.policy)
        # pgnet_loss = pg_loss + 0.01 * entropy
        # trainer = tf.train.AdamOptimizer(learning_rate=self.pp, epsilon=1e-5)
        trainer = tf.train.GradientDescentOptimizer(
            learning_rate=1e-6)  #self.pp['net_lr'])
        grads = trainer.compute_gradients(pg_loss, online_pg_vars)
        self.do_train_pg = trainer.apply_gradients(grads)
        return self.vf_loss, online_vf_vars

    @staticmethod
    def entropy(logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    @staticmethod
    def neglogp(logits, x):
        one_hot_actions = tf.one_hot(x, logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=one_hot_actions)

    def forward_action(self, frep, cell, ce_type, chs):
        # u = tf.random_uniform(tf.shape(self.policy))
        # self.sample_action = tf.argmax(elig_policy - tf.log(-tf.log(u)), axis=-1)
        policy = self.sess.run(self.policy, {
            self.freps: [frep],
            self.cells: prep_data_cells(cell),
        })[0]

        # u = np.random.uniform(policy.shape)
        # policy_ent = policy - np.log(-np.log(u))
        # NOTE TODO should this be argmin for END?
        if ce_type == CEvent.END:
            idx = np.argmin(policy[chs])
        else:
            idx = np.argmax(policy[chs])
        ch = chs[idx]
        neglogpac = self.sess.run(self.neglogpac_out, {
            self.policy_in: policy,
            self.action: [ch]
        })
        return ch, neglogpac

    def get_neglogpac(self, frep, cell, ch):
        policy = self.sess.run(self.policy, {
            self.freps: [frep],
            self.cells: prep_data_cells(cell),
        })[0]
        neglogpac = self.sess.run(self.neglogpac_out, {
            self.policy_in: policy,
            self.action: [ch]
        })
        return neglogpac

    def forward_value(self, freps):
        value = self.sess.run(
            self.value,
            feed_dict={
                self.freps: freps
            },
        ).squeeze()
        return value

    def backward(self, step, buf, n_step):
        # TODO:
        # - collect nsteps of data. 16-128
        # - train noptepochs consecutive times on pg net. 4
        # next_values = self.sess.run(
        #     self.value, feed_dict={
        #         self.freps: [e.next_frep for e in buf]
        #     }).squeeze()
        value_target = step.reward - self.avg_reward + step.next_val
        loss, lr, err = self.backward_vf([step.frep], [value_target])
        if len(buf) != n_step:
            return loss, lr, err
        # np.random.shuffle(buf)
        next_values = np.array([e.next_val for e in buf])
        rewards = [e.reward for e in buf]
        value_targets = rewards + next_values - self.avg_reward
        freps = [e.frep for e in buf]
        cells = [e.cell for e in buf]
        neglogpacs = [e.neglogpac for e in buf]
        chs = [e.ch for e in buf]
        for _ in range(4):
            self.sess.run(
                [self.do_train_pg], {
                    self.freps: freps,
                    self.cells: prep_data_cells(cells),
                    self.value_target: value_targets,
                    self.old_neglogpac: neglogpacs,
                    self.action: chs
                })
        return loss, lr, err

    def backward_vf(self, freps, value_target):
        data = {self.freps: freps, self.value_target: value_target}
        _, loss, lr, err = self.sess.run([self.do_train, self.vf_loss, self.lr, self.err],
                                         data)
        self.avg_reward += self.weight_beta * np.mean(err)
        return loss, lr, err
