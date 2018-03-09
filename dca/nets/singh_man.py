import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, get_trainable_vars


class ManSinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Manual implementation
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")

        with tf.variable_scope('model/' + self.name) as scope:
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(self.freps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                use_bias=False,
                activation=None,
                name="vals")
            online_vars = tuple(get_trainable_vars(scope).values())
        self.grads = [(tf.placeholder(tf.float32, [3479, 1]), online_vars[0])]

        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        self.do_train = trainer.apply_gradients(self.grads, global_step=global_step)
        return None, None

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, discount=None):
        value = self.sess.run(self.value, feed_dict={self.freps: freps})[0][0]
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})[0][0]
        td_err = rewards[0] + discount * next_value - value
        frep_colvec = np.reshape(freps[0], [-1, 1])
        grad = -2 * td_err * frep_colvec  # Gradient for MSE
        data = {self.freps: freps, self.grads[0][0]: grad}
        lr, _ = self.sess.run([self.lr, self.do_train], data)
        assert not np.isnan(td_err) or not np.isinf(td_err)
        return td_err**2, lr, td_err
