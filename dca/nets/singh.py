import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as k  # noqa

from nets.convlayers import SeparableSplit  # noqa
from nets.convlayers import InPlaneSplit, InPlaneSplitLocallyConnected2D
from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_grids


class SinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        self.name = "SinghNet"
        # self.frep_depth = pp['n_channels'] + 1 if frep_depth is None else frep_depth
        self.pre_conv = pp['pre_conv']
        self.grid_inp = pp['singh_grid']
        self.frepshape = frepshape
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _pre_conv(inp, name):
        with tf.variable_scope('model/' + name):
            print(inp.shape)
            # [filter_height, filter_width, in_channels, channel_multiplier]
            # filters = tf.Variable(tf.ones((3, 3, self.depth, 1)) * 0.1)
            # conv = tf.nn.depthwise_conv2d(
            #     inp, filters, strides=[1, 1, 1, 1], padding='SAME')
            # dense_inp = tf.nn.relu(conv)

            dense_inp = SeparableSplit(
                kernel_size=3, stride=1, use_bias=False, padding="VALID").apply(
                    inp, True)
            # c1 = InPlaneSplit(
            #     kernel_size=3, stride=1, use_bias=False, padding="VALID").apply(
            #         inp, False)
            # dense_inp = InPlaneSplit(
            #     kernel_size=3, stride=1, use_bias=False, padding="VALID").apply(
            #         c1, True)

            # lconv = k.layers.LocallyConnected2D(filters=70, kernel_size=3)
            # dense_inp = lconv(inp)

            # dense_inp = self.add_conv_layer(inp, 70, 3, padding="same", use_bias=False)
            # dense_inp = self.add_conv_layer(conv1, 3 * 70, 3)
            # TODO: Try with bias
            # pad = tf.keras.layers.ZeroPadding2D((1, 1))
            # out = pad(inp)
            # inplane_loc = InPlaneSplitLocallyConnected2D(
            #     kernel_size=[3, 3],
            #     strides=[3, 3],
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.constant_initializer(0.1))
            # self.dense_inp = dense_inp = inplane_loc(inp)
            print(dense_inp.shape)
        return dense_inp

    def _build_net(self, top_inp, name):
        dense_inp = self.pre_conv(top_inp, name) if self.pre_conv else top_inp
        with tf.variable_scope('model/' + name) as scope:
            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None)
            value = value_layer.apply(tf.layers.flatten(dense_inp))
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            trainable_vars = get_trainable_vars(scope)
        return value, trainable_vars

    def _build_inputs(self):
        self.frep = tf.placeholder(tf.int32, [None, *self.frepshape], "feature_rep")
        self.value_target = tf.placeholder(tf.float32, [None], "value_target")
        # Weighting the loss of each sample. If a single value is given, it weighs all samples
        self.weight = tf.placeholder(tf.float32, [None], "weight")
        frep = tf.cast(self.frep, tf.float32)
        if self.grid_inp:
            self.grid = tf.placeholder(
                tf.bool, [None, self.rows, self.cols, 2 * self.n_channels], "grid")
            grid = tf.cast(self.grid, tf.float32)
            top_inp = tf.concat([grid, frep], axis=3)
        else:
            top_inp = frep
        return top_inp

    def build(self):
        top_inp = self._build_inputs()
        value, online_vars = self._build_net(top_inp, "online")
        self.value = tf.squeeze(value, 1)
        self.err = self.value_target - value
        if self.pp['huber_loss'] is not None:
            # Linear when loss is above delta and squared difference below
            self.loss = tf.losses.huber_loss(
                labels=self.value_target, predictions=value, delta=self.pp['huber_loss'])
        else:
            self.loss = tf.losses.mean_squared_error(
                labels=self.value_target, predictions=self.value, weights=self.weight)
        return self.loss, online_vars

    def forward(self, freps, grids=None):
        data = {self.frep: freps}
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        values = self.sess.run(
            self.value, data, options=self.options, run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, *, freps, value_targets, grids=None, weights=[1], **kwargs):
        data = {self.frep: freps, self.value_target: value_targets, self.weight: weights}
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        _, loss, lr, errs = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        errs = np.squeeze(errs, 1)
        return loss, lr, errs
