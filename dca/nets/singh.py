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
        gridshape = [None, self.pp['rows'], self.pp['cols'], self.n_channels]
        self.grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="grid")
        self.value_target = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name="value_target")
        self.n_used_neighs = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="grid")
        n_free_self = tf.expand_dims(
            tf.count_nonzero(self.grid, axis=3, dtype=tf.float32), axis=3)

        inp = tf.layers.flatten(
            tf.concat([n_free_self, self.n_used_neighs], axis=3))
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

    def forward(self, grids, n_used_neighs):
        values = self.sess.run(
            self.value,
            feed_dict={
                self.grid: grids,
                self.n_used_neighs: n_used_neighs,
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, grid, n_used_neighs, reward, next_grid,
                 next_n_used_neighs):
        next_value = self.sess.run(
            self.value,
            feed_dict={
                self.grid: next_grid,
                self.n_used_neighs: next_n_used_neighs,
            })
        value_target = reward + self.gamma * next_value
        data = {
            self.grid: grid,
            self.n_used_neighs: n_used_neighs,
            self.value_target: value_target,
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss
