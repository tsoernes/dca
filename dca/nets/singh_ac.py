import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (copy_net_op, get_trainable_vars,
                        normalized_columns_initializer, scale_freps_big)


class ACSinghNet(Net):
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
        self.avg_reward = [0]

    def _build_net(self, freps, name):
        with tf.variable_scope('model/' + name) as scope:
            if self.pre_conv:
                dense_inp = self.add_conv_layer(freps, self.pp['conv_nfilters'][0],
                                                self.pp['conv_kernel_sizes'][0])
            else:
                dense_inp = freps
            h = self.add_dense_layer(dense_inp, 70, normalized_columns_initializer(0.01))
            value = self.add_dense_layer(h, 1, normalized_columns_initializer(0.01))
            policy = self.add_dense_layer(h, 70, normalized_columns_initializer(0.01),
                                          tf.nn.softmax)
            trainable_vars = get_trainable_vars(scope)
            # Output layers for policy and value estimations
        return value, policy, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.weights = tf.placeholder(tf.float32, [None, 1], "weight")

        freps = scale_freps_big(self.freps) if self.pp['scale_freps'] else self.freps
        self.value, self.policy, online_vars = self._build_net(freps, "online")

        self.err = self.value_target - self.value
        if self.pp['huber_loss'] is not None:
            # Linear when loss is above delta and squared difference below
            self.loss = tf.losses.huber_loss(
                labels=self.value_target,
                predictions=self.value,
                delta=self.pp['huber_loss'])
        else:
            self.loss = tf.losses.mean_squared_error(
                labels=self.value_target, predictions=self.value, weights=self.weights)
        return self.loss, online_vars

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward_supervised(self, freps, value_target, weight=1, *args, **kwargs):
        data = {
            self.freps: freps,
            self.value_target: value_target,
            self.weights: [[weight]]
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if self.pp['avg_reward']:
            self.avg_reward += self.weight_beta * err[0][0]
            # self.weight_beta *= self.weight_beta_decay
            # print(self.avg_reward)
        return loss, lr, err

    def backward(self, freps, rewards, next_freps, discount=None, weight=1):
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})
        if self.pp['avg_reward']:
            value_target = rewards + next_value - self.avg_reward
        else:
            value_target = rewards + discount * next_value
        return self.backward_supervised(freps, value_target, weight)
