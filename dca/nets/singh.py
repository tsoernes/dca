import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, scale_freps


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")

        if self.pp['scale_freps']:
            freps = scale_freps(self.freps)
        else:
            freps = self.freps
        with tf.variable_scope('model/' + self.name) as scope:
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(freps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=True,
                activation=None,
                name="vals")
            online_vars = get_trainable_vars(scope)

        self.err = self.value_target - self.value
        self.loss = tf.losses.mean_squared_error(
            labels=self.value_target, predictions=self.value)
        return online_vars

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, gamma):
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})
        value_target = rewards + gamma * next_value
        # TODO NOTE TODO IS this really the correct reward, and the
        # correct target
        # if next_value[0] != next_val:
        #     print(next_value, next_val)
        # value_target = reward + self.gamma * next_val
        data = {self.freps: freps, self.value_target: value_target}
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
