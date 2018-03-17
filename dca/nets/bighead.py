import tensorflow as tf

from nets.qnet import QNet
from nets.utils import NominalInitializer, get_trainable_vars


class BigHeadQNet(QNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_base_net(self, top_inp, ncells, name):
        with tf.variable_scope('model/' + name) as scope:
            # print(top_inp.shape)
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
            conv3 = tf.keras.layers.LocallyConnected2D(
                filters=70,
                kernel_size=1,
                padding="valid",
                kernel_initializer=NominalInitializer(0.1, 1.1),
                use_bias=False,
                activation=None)(conv2)
            q_vals = tf.gather_nd(conv3, ncells)
        return q_vals
