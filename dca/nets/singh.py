import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import copy_net_op, get_trainable_vars, scale_freps_big


class SinghNet(Net):
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

    def _build_net(self, freps, name):
        with tf.variable_scope('model/' + name) as scope:
            if self.pre_conv:
                dense_inp = self.add_conv_layer(freps, self.pp['conv_nfilters'][0],
                                                self.pp['conv_kernel_sizes'][0])
            else:
                dense_inp = freps
            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=self.dense_regularizer,
                use_bias=False,
                activation=None)
            value = value_layer.apply(tf.layers.flatten(dense_inp))
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            trainable_vars = get_trainable_vars(scope)
        return value, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.weights = tf.placeholder(tf.float32, [None, 1], "weight")

        freps = scale_freps_big(self.freps) if self.pp['scale_freps'] else self.freps
        self.value, online_vars = self._build_net(freps, "online")
        if self.double_net:
            self.target_value, target_vars = self._build_net(freps, "target")
            self.copy_online_to_target = copy_net_op(online_vars, target_vars,
                                                     self.pp['net_creep_tau'])

        self.err = self.value_target - self.value
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
            self.avg_reward += self.weight_beta * err
            self.weight_beta *= self.weight_beta_decay
        return loss, lr, err

    def backward(self, freps, rewards, next_freps, discount=None, weight=1):
        if self.double_net:
            target_val = self.target_value
        else:
            target_val = self.value
        next_value = self.sess.run(target_val, feed_dict={self.freps: next_freps})
        if self.pp['avg_reward']:
            value_target = rewards - self.avg_reward + next_value
        else:
            value_target = rewards + discount * next_value
        return self.backward_supervised(freps, value_target, weight)
