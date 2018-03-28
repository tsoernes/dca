import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_grids, scale_freps_big


class SinghNet(Net):
    def __init__(self, pp, logger, big_freps=False):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        self.pre_conv = pp['pre_conv']
        self.grid_inp = pp['singh_grid']
        self.depth = 3 * 70 + 1 if big_freps else 70 + 1
        self.frepshape = [pp['rows'], pp['cols'], self.depth]
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _build_net(self, inp, name):
        with tf.variable_scope('model/' + name) as scope:
            if self.pre_conv:
                # [filter_height, filter_width, in_channels, channel_multiplier]
                # filters = tf.Variable(
                #     tf.random_normal((2, 2, 70 * 3 + 1, 1), mean=0.2, stddev=0.2))
                filters = tf.ones((3, 3, self.depth, 1)) * 0.1
                conv = tf.nn.depthwise_conv2d(
                    inp, filters, strides=[1, 3, 3, 1], padding='SAME')
                print(inp.shape, conv.shape)
                dense_inp = tf.nn.relu(conv)
                # pad = tf.keras.layers.ZeroPadding2D((1, 1))
                # out = pad(inp)
                # dense_inp = tf.keras.layers.LocallyConnected2D(
                #     filters=80,
                #     kernel_size=3,
                #     padding="valid",
                #     kernel_initializer=self.kern_init_conv(),
                #     use_bias=self.pp['conv_bias'],
                #     activation=None)(out)
                # dense_inp = self.add_conv_layer(inp, 80, 5)
            else:
                dense_inp = inp
            value_layer = tf.layers.Dense(
                units=1,
                # kernel_initializer=tf.zeros_initializer(),
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.dense_regularizer,
                use_bias=False,
                activation=None)
            value = value_layer.apply(tf.layers.flatten(dense_inp))
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            trainable_vars = get_trainable_vars(scope)
        return value, trainable_vars

    def build(self):
        self.freps = tf.placeholder(tf.float32, [None, *self.frepshape], "feature_reps")
        self.grids = tf.placeholder(
            tf.bool, [None, self.rows, self.cols, 2 * self.n_channels], "grid")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.weights = tf.placeholder(tf.float32, [None, 1], "weight")

        freps = scale_freps_big(self.freps) if self.pp['scale_freps'] else self.freps
        if self.grid_inp:
            net_inp = tf.concat([tf.cast(self.grids, tf.float32), freps], axis=3)
        else:
            net_inp = freps
        self.value, online_vars = self._build_net(net_inp, "online")

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

    def forward(self, freps, grids=None):
        data = {
            self.freps: freps,
        }
        if self.grid_inp:
            data[self.grids] = prep_data_grids(grids, self.grid_split)
        values = self.sess.run(
            self.value, data, options=self.options, run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward_supervised(self,
                            freps,
                            value_target,
                            weights=[1],
                            grids=None,
                            *args,
                            **kwargs):
        weights = np.expand_dims(weights, axis=1)
        data = {self.freps: freps, self.value_target: value_target, self.weights: weights}
        if self.grid_inp:
            data[self.grids] = prep_data_grids(grids, self.grid_split)
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err

    def backward(self,
                 freps,
                 rewards,
                 next_freps,
                 grids=None,
                 next_grids=None,
                 discount=None,
                 weights=[1]):
        data = {self.freps: next_freps}
        if self.grid_inp:
            data[self.grids] = prep_data_grids(next_grids, self.grid_split)
        next_value = self.sess.run(self.value, data)
        # print(next_value, next_value.shape)
        rewards = np.expand_dims(rewards, axis=1)
        value_target = rewards + discount * next_value
        # print(value_target, value_target.shape, rewards)
        return self.backward_supervised(grids, freps, value_target, weights)
