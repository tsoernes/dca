import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, scale_freps_big


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)
        self.weight_beta = self.pp['weight_beta']
        self.avg_reward = 0

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.weights = tf.placeholder(tf.float32, [None, 1], "weight")

        if self.pp['scale_freps']:
            freps = scale_freps_big(self.freps)
        else:
            freps = self.freps
        with tf.variable_scope('model/' + self.name) as scope:
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(freps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                use_bias=False,
                activation=None,
                name="vals")
            online_vars = get_trainable_vars(scope)

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
        return loss, lr, err

    def backward(self, freps, rewards, next_freps, gamma=None, weight=1):
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})
        if self.pp['avg_reward']:
            value_target = rewards - self.avg_reward + next_value
        else:
            value_target = rewards + gamma * next_value
        return self.backward_supervised(freps, value_target, weight)
