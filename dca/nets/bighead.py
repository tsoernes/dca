import tensorflow as tf

from nets.qnet import QNet
from nets.utils import get_trainable_vars


class BigHeadQNet(QNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_base_net(self, top_inp, ncells, name):
        with tf.variable_scope('model/' + name) as scope:
            conv1 = tf.layers.conv2d(
                inputs=top_inp,
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
                use_bias=True,
                activation=self.act_fn)
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=70,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            q_vals = tf.gather_nd(conv3, ncells)
        return q_vals
