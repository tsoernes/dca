from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (build_default_trainer, get_trainable_vars,
                        scale_freps_big)


class TDCSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction
        """
        self.name = "TDCNet"
        self.frepshape = frepshape
        self.wdim = reduce(mul, frepshape)
        super().__init__(name=self.name, pp=pp, logger=logger)
        self.grad_beta = self.pp['grad_beta']
        self.weights = np.zeros((self.wdim, 1))

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.grads = tf.placeholder(tf.float32, [self.wdim, 1], "grad_corr")

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
        self.grads = [(tf.placeholder(tf.float32, [self.wdim, 1]), online_vars[0])]

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

    def backward(self,
                 *,
                 freps,
                 rewards,
                 next_freps,
                 discount,
                 weights,
                 avg_reward=None,
                 **kwargs):
        # NOTE can possible take in val, next_val here as theyre already known
        assert len(freps) == 1  # Hard coded for one-step
        value = self.sess.run(self.value, feed_dict={self.freps: freps})[0, 0]
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})[0, 0]
        if avg_reward is None:
            td_err = rewards[0] + discount * next_value - value
        else:
            td_err = rewards[0] - avg_reward + next_value - value

        frep_colvec = np.reshape(freps[0], [-1, 1])
        next_frep_colvec = np.reshape(next_freps[0], [-1, 1])
        # dot is inner product and therefore a scalar
        dot = np.dot(frep_colvec.T, self.weights)
        if avg_reward is None:
            grad = -2 * weights[0] * (
                td_err * frep_colvec - discount * next_frep_colvec * dot)
        else:
            grad = -2 * weights[0] * (
                td_err * frep_colvec + avg_reward - next_frep_colvec * dot)
        data = {self.grads[0][0]: grad}
        lr, _ = self.sess.run(
            [self.lr, self.do_train],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.weights += self.grad_beta * (td_err - dot) * frep_colvec
        return td_err**2, lr, td_err
