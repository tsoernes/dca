import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (build_default_trainer, get_trainable_vars,
                        scale_freps_big)


class TDLSinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        True online td lambda
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)
        self.z = np.zeros((self.rows * self.cols * (self.n_channels + 1), 1))
        self.lmbda = self.pp['lambda']
        self.v_old = 0

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        d = self.rows * self.cols * (self.n_channels + 1)
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.grads = tf.placeholder(tf.float32, [d, 1], "grad_corr")

        if self.pp['scale_freps']:
            freps = scale_freps_big(self.freps)
        else:
            freps = self.freps
        with tf.variable_scope('model/' + self.name) as scope:
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(freps),
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None,
                name="vals")
            online_vars = tuple(get_trainable_vars(scope).values())
        self.grads = [(tf.placeholder(tf.float32, [d, 1]), online_vars[0])]

        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        self.do_train = trainer.apply_gradients(self.grads, global_step=global_step)
        return None, None

    def forward(self, freps, grids=None):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, discount, weights):
        assert len(freps) == 1  # Hard coded for one-step
        value = self.sess.run(self.value, feed_dict={self.freps: freps})[0, 0]
        next_value, lr = self.sess.run(
            [self.value, self.lr], feed_dict={self.freps: next_freps})
        next_value = next_value[0]
        frep_colvec = np.reshape(freps[0], [-1, 1])
        # next_frep_colvec = np.reshape(next_freps[0], [-1, 1])

        td_err = rewards[0] + discount * next_value - value

        dot = np.dot(self.z.T, frep_colvec)
        self.z = discount * self.lmbda * self.z + (
            1 - lr * discount * self.lmbda * dot) * frep_colvec
        z_colvec = np.reshape(self.z, [-1, 1])
        grad = (td_err + value - self.v_old) * z_colvec - (
            value - self.v_old) * frep_colvec
        data = {self.grads[0][0]: -grad}
        lr, _ = self.sess.run(
            [self.lr, self.do_train],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.v_old = next_value
        return td_err**2, lr, td_err
