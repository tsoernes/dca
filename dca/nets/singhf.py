import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_optimizer_by_name, get_trainable_vars


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)
        self.sess.run(self.copy)

    def build(self):
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.nfreps = tf.placeholder(tf.float32, frepshape, "feature_reps2")
        self.rewards = tf.placeholder(tf.float32, [None, 1], "rewards")
        self.gamma = tf.placeholder(tf.float32, [None, 1], "gamma")

        with tf.variable_scope('model/online' + self.name) as scope:
            self.value_o = tf.layers.dense(
                inputs=tf.layers.flatten(self.freps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None,
                name="vals")
            online_vars = get_trainable_vars(scope)
        with tf.variable_scope('model/target/' + self.name) as scope:
            self.value_t = tf.layers.dense(
                inputs=tf.layers.flatten(self.nfreps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None,
                name="vals")
            target_vars = get_trainable_vars(scope)

        target_val = self.rewards + self.gamma * self.value_t
        self.err = target_val - self.value_o
        self.loss = tf.losses.mean_squared_error(
            labels=target_val, predictions=self.value_o)
        if self.pp['net_lr_decay'] < 1:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.pp['net_lr'], global_step,
                                                       10000, self.pp['net_lr_decay'])
        else:
            global_step = None
            learning_rate = tf.constant(self.pp['net_lr'])
        trainer = get_optimizer_by_name(self.pp['optimizer'], learning_rate)
        gradients_o, trainable_vars = zip(
            *trainer.compute_gradients(self.loss, var_list=online_vars))
        gradients_t, _ = zip(*trainer.compute_gradients(self.loss, var_list=target_vars))
        full_grads = (gradients_t[0] + gradients_o[0], )
        print(full_grads[0].shape)
        print(gradients_t[0].shape)
        self.do_train2 = trainer.apply_gradients(
            zip(full_grads, trainable_vars), global_step=global_step)

        copy_ops = []
        for var_name, target_var in target_vars.items():
            online_val = online_vars[var_name].value()
            op = target_var.assign(online_val)
            copy_ops.append(op)
        self.copy = tf.group(*copy_ops)
        return online_vars

    def forward(self, freps):
        values = self.sess.run(
            self.value_o,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, gamma):
        data = {
            self.freps: freps,
            self.nfreps: next_freps,
            self.rewards: [rewards],
            self.gamma: [[gamma]]
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train2, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.sess.run(self.copy)
        return loss, lr, err
