import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_grids


class AfterstateNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        # depth = self.n_channels * 2 if self.pp['grid_split'] else self.n_channels
        depth = self.n_channels + 1
        self.grids = tf.placeholder(
            tf.float32, [None, self.pp['rows'], self.pp['cols'], depth], "grids")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")

        if self.pp['scale_freps']:
            frepshape = [None, self.rows, self.cols, self.n_channels + 1]
            mult1 = np.ones(frepshape[1:], np.float32)  # Scaling feature reps
            mult1[:, :, :-1] /= 43
            mult1[:, :, -1] /= 70
            inp = self.grids * tf.constant(mult1)
        else:
            inp = self.grids

        with tf.variable_scope('model/' + self.name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=inp,
                filters=140,
                kernel_size=5,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.act_fn)
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(conv2),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None,
                name="vals")
            online_vars = get_trainable_vars(scope)

        self.err = self.value_target - self.value
        self.loss = tf.losses.mean_squared_error(
            labels=self.value_target, predictions=self.value)
        return online_vars

    def forward(self, grids):
        values = self.sess.run(
            self.value,
            # feed_dict={self.grids: prep_data_grids(grids, self.pp['grid_split'])},
            feed_dict={self.grids: grids},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, grids, next_grids, value_target, gamma):
        data = {
            # self.grids: prep_data_grids(grids, self.pp['grid_split']),
            self.grids: grids,
            self.value_target: value_target
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
