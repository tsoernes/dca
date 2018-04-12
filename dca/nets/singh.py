import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as k  # noqa

# yapf: disable
from nets.convlayers import (DepthwiseConv2D, InPlaneSplit, SeparableConv2D, # noqa
                             InPlaneSplitLocallyConnected2D, SeparableSplit)
# yapf: enable
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

    def _build_pre_conv(self, inp, name):
        with tf.variable_scope('model/' + name):
            print(inp.shape)
            # WORKS very well in start, but block prob diverges rapidly around 80k iters
            # [filter_height, filter_width, in_channels, channel_multiplier]
            # filters = tf.Variable(self.kern_init_conv()(
            #     (self.pp['conv_kernel_sizes'][0], self.pp['conv_kernel_sizes'][0],
            #      self.depth, 1)))
            # conv = tf.nn.depthwise_conv2d(
            #     inp, filters, strides=[1, 1, 1, 1], padding='SAME')
            # dense_inp = tf.nn.relu(conv)

            # dense_inp = SeparableSplit(
            #     kernel_size=3,
            #     stride=1,
            #     use_bias=False,
            #     padding="VALID",
            #     kernel_initializer=self.kern_init_conv).apply(inp, True)

            dense_inp = SeparableConv2D(
                kernel_size=3,
                stride=1,
                padding="VALID",
                kernel_initializer=self.kern_init_conv).apply(inp)

            # c1 = InPlaneSplit(
            #     kernel_size=3, stride=1, use_bias=False, padding="VALID").apply(
            #         inp, False)
            # dense_inp = InPlaneSplit(
            #     kernel_size=3, stride=1, use_bias=False, padding="VALID").apply(
            #         c1, True)

            # lconv = k.layers.LocallyConnected2D(
            #     filters=self.pp['conv_nfilters'][0],
            #     kernel_size=self.pp['conv_kernel_sizes'][0],
            #     activation=tf.nn.relu,
            #     kernel_initializer=self.kern_init_conv(),
            #     use_bias=self.pp['conv_bias'])
            # dense_inp = lconv(inp)

            # dense_inp = self.add_conv_layer(
            #     inp,
            #     filters=self.pp['conv_nfilters'][0],
            #     kernel_size=self.pp['conv_kernel_sizes'][0],
            #     padding="same",
            #     use_bias=self.pp['conv_bias'])
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
        dense_inp = self._build_pre_conv(top_inp, name) if self.pre_conv else top_inp
        with tf.variable_scope('model/' + name) as scope:
            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=self.kern_init_dense,
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
            grid_depth = 2 * self.n_channels
            self.grid = tf.placeholder(tf.bool, [None, self.rows, self.cols, grid_depth],
                                       "grid")
            grid = tf.cast(self.grid, tf.float32)
            top_inp = tf.concat([grid, frep], axis=3)
            self.depth = self.frepshape[-1] + grid_depth
        else:
            top_inp = frep
            self.depth = self.frepshape[-1]
        return top_inp

    def build(self):
        top_inp = self._build_inputs()
        value, online_vars = self._build_net(top_inp, "online")
        self.value = tf.squeeze(value, 1)
        self.err = self.value_target - value
        self.loss = self.default_loss(
            pred=self.value, target=self.value_target, weight=self.weight)
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
