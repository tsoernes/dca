import tensorflow as tf

from nets.convlayers import InPlaneSplit, SeparableSplit, separable_conv2d  # noqa
from nets.qnet import QNet
from nets.utils import NominalInitializer, get_trainable_vars  # noqa


class BigHeadQNet(QNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self, top_inp, ncells, name):
        with tf.variable_scope('model/' + name) as scope:
            # print(top_inp.shape)
            # conv1 = self.add_conv_layer(top_inp, self.pp['conv_nfilters'][0],
            #                             self.pp['conv_kernel_sizes'][0])
            # conv1 = SeparableSplit(
            # conv1 = InPlaneSplit(
            #     kernel_size=self.pp['conv_kernel_sizes'][0],
            #     stride=1,
            #     use_bias=self.pp['conv_bias'],
            #     padding="SAME",
            #     kernel_initializer=self.kern_init_conv()).apply(top_inp)
            # conv2 = InPlaneSplit(
            #     kernel_size=self.pp['conv_kernel_sizes'][1],
            #     stride=1,
            #     use_bias=self.pp['conv_bias'],
            #     padding="SAME",
            #     kernel_initializer=self.kern_init_conv()).apply(conv1)
            # conv = separable_conv2d(
            #     inp=top_inp,
            #     kernel_size=self.pp['conv_kernel_sizes'][0],
            #     stride=1,
            #     padding="SAME",
            #     kernel_initializer=self.kern_init_conv())
            pad = tf.keras.layers.ZeroPadding2D((1, 1))
            out = pad(top_inp)
            conv1 = tf.keras.layers.LocallyConnected2D(
                filters=70,
                kernel_size=self.pp['conv_kernel_sizes'][0],
                padding="valid",
                kernel_initializer=self.kern_init_dense(),
                use_bias=False,
                activation=tf.nn.relu)(out)
            pad = tf.keras.layers.ZeroPadding2D((1, 1))
            out = pad(conv1)
            conv3 = tf.keras.layers.LocallyConnected2D(
                filters=70,
                kernel_size=self.pp['conv_kernel_sizes'][-1],
                padding="valid",
                kernel_initializer=self.kern_init_dense(),
                use_bias=False,
                activation=tf.nn.relu)(out)
            print(conv3.shape)
            q_vals = tf.gather_nd(conv3, ncells)
            trainable_vars_by_name = get_trainable_vars(scope)
        return q_vals, trainable_vars_by_name
