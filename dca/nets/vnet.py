import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import copy_net_op, prep_data_grids


class VNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        super().__init__(name="VNet", *args, **kwargs)

    def _build_net(self, grid, name):
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=self.n_channels,
                kernel_size=4,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=self.n_channels,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                activation=self.act_fn)
            flat = tf.layers.flatten(conv2)
            hidden = tf.layers.dense(
                inputs=flat,
                units=50,
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.regularizer,
                activation=self.act_fn,
                name="hidden")
            value = tf.layers.dense(
                inputs=flat,
                units=1,
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.regularizer,
                name="value")
            trainable_vars_by_name = self._get_trainable_vars(scope)
        return value, trainable_vars_by_name

    def build(self):
        gridshape = [None, self.pp['rows'], self.pp['cols'], self.n_channels]
        self.grid = tf.placeholder(
            shape=gridshape, dtype=tf.float32, name="grid")
        self.value_target = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name="value_target")

        self.value, online_vars = self._build_net(self.grid, "online_vnet")
        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.value_target), predictions=self.value)
        self.do_train = self._build_default_trainer(online_vars)

    def forward(self, grids):
        values = self.sess.run(
            self.value,
            feed_dict={
                self.grid: prep_data_grids(
                    grids, empty_neg=self.pp['empty_neg']),
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, grid, reward, next_grid):
        next_value = self.sess.run(
            self.value,
            feed_dict={
                self.grid: prep_data_grids(next_grid, self.pp['empty_neg'])
            })
        value_target = reward + self.gamma * next_value
        data = {
            self.grid: prep_data_grids(grid, self.pp['empty_neg']),
            self.value_target: value_target,
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss
