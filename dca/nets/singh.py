import numpy as np
import tensorflow as tf

from nets.net import Net


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        self.freps = tf.placeholder(
            shape=[None, self.pp['rows'], self.pp['cols'], self.n_channels + 1],
            # shape=[None, self.n_channels + 1],
            dtype=tf.float32,
            name="feature_representations")
        self.value_target = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name="value_target")

        inp = tf.layers.flatten(self.freps)
        with tf.variable_scope(self.name) as scope:
            self.value = tf.layers.dense(
                inputs=inp,
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=True,
                activation=None,
                name="vals")
            online_vars = self._get_trainable_vars(scope)

        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.value_target), predictions=self.value)
        self.do_train = self._build_default_trainer(online_vars)

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, next_freps, value_target):
        # next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})
        # TODO NOTE TODO IS this really the correct reward, and the
        # correct target
        # if next_value[0] != next_val:
        #     print(next_value, next_val)
        # value_target = reward + self.gamma * next_val
        data = {
            self.freps: freps,
            self.value_target: value_target,
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss
