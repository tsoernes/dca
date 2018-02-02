import numpy as np
import tensorflow as tf

from nets.net import Net


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "VNet"
        super().__init__(name="VNet", *args, **kwargs)

    def build(self):
        self.freps = tf.placeholder(
            shape=[None, self.pp['rows'], self.pp['cols'], self.n_channels + 1],
            dtype=tf.float32,
            name="feature_representations")
        self.value_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="value_target")

        # inp = tf.layers.flatten(
        #     tf.concat(
        #         [tf.expand_dims(self.n_free_self, axis=3), self.n_used_neighs],
        #         axis=3))
        inp = tf.layers.flatten(self.freps)
        with tf.variable_scope(self.name) as scope:
            self.value = tf.layers.dense(
                inputs=inp,
                units=1,
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.regularizer,
                name="vals")
            online_vars = self._get_trainable_vars(scope)

        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.value_target), predictions=self.value)
        self.do_train = self._build_default_trainer(online_vars)

    # def forward(self, n_free_self, n_used_neighs):
    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={
                self.freps: freps
                # self.n_free_self: n_free_self,
                # self.n_used_neighs: n_used_neighs,
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    # def backward(self, n_free_self, n_used_neighs, reward, next_n_free_self,
    #              next_n_used_neighs):
    def backward(self, freps, reward, next_freps):
        next_value = self.sess.run(
            self.value,
            feed_dict={
                self.freps: next_freps
                # self.n_free_self: next_n_free_self,
                # self.n_used_neighs: next_n_used_neighs,
            })
        value_target = reward + self.gamma * next_value
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
