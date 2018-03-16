import tensorflow as tf

from nets.qnet import QNet
from nets.utils import NominalInitializer, get_trainable_vars


class BigHeadQNet(QNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_base_net(self, top_inp, ncells, name):
        with tf.variable_scope('model/' + name) as scope:
            print(top_inp.shape)
            conv1 = tf.layers.conv2d(
                inputs=top_inp,
                filters=70,
                kernel_size=4,
                padding="same",
                # kernel_initializer=NominalInitializer(10, 100),
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=4,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.act_fn)
            # conv2 = tf.layers.conv2d(
            #     inputs=conv1,
            #     filters=70,
            #     kernel_size=3,
            #     padding="same",
            #     kernel_initializer=self.kern_init_conv(),
            #     kernel_regularizer=self.regularizer,
            #     use_bias=False,
            #     activation=self.act_fn)
            conv3 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                # kernel_initializer=NominalInitializer(10, 100),
                kernel_regularizer=self.regularizer,
                use_bias=False,
                activation=self.act_fn)
            # print(conv3.shape)
            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=70,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                bias_initializer=NominalInitializer(10, 100),
                activation=self.act_fn)
            # NOTE TODO WHY is conv3 shape = (1,1,70,70) where
            # first 70 is conv1 filters, second is conv3 filters?
            # Should it not be (1, 7, 7, 70)
            # Need to read up on/write down conv weight shapes vs output shapes
            q_vals = tf.gather_nd(conv4, ncells)
        return q_vals
