import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (build_default_trainer, get_trainable_vars,
                        prep_data_grids)
from utils import prod


class TDCSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction
        """
        self.name = "TDCNet"
        self.grid_inp = pp['singh_grid']
        self.frepshape = frepshape
        if self.grid_inp:
            self.wdim = prod(frepshape) + prod([7, 7, 2 * pp['n_channels']])
        else:
            self.wdim = prod(frepshape)
        super().__init__(name=self.name, pp=pp, logger=logger)
        self.grad_beta = self.pp['grad_beta']
        self.grad_beta_decay = 1 - self.pp['grad_beta_decay']
        # Weights is a column vector of same length as flatten net input
        self.weights = np.zeros((self.wdim, 1))

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        self.frep = tf.placeholder(tf.int32, [None, *self.frepshape], "feature_reps")
        self.grads = tf.placeholder(tf.float32, [self.wdim, 1], "grad_corr")

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

        with tf.variable_scope('model/' + self.name) as scope:
            self.value = tf.layers.dense(
                inputs=tf.layers.flatten(top_inp),
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
        data = {self.frep: freps}
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        values = self.sess.run(
            self.value, data, options=self.options, run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward_supervised(self, *, freps, value_targets, **kwargs):
        raise NotImplementedError
        value = self.sess.run(self.value, feed_dict={self.frep: freps})[0, 0]
        frep_colvec = np.reshape(freps[0], [-1, 1])
        grad = -2 * (value_targets[0] - value) * frep_colvec
        data = {self.grads[0][0]: grad}
        lr, _ = self.sess.run([self.lr, self.do_train], feed_dict=data)

    def backward(self,
                 *,
                 freps,
                 rewards,
                 next_freps,
                 discount,
                 weights,
                 avg_reward=None,
                 grids=None,
                 next_grids=None,
                 **kwargs):
        # NOTE can possible take in val, next_val here as theyre already known
        assert len(freps) == 1  # Hard coded for one-step
        data1 = {self.frep: freps}
        data2 = {self.frep: next_freps}
        if self.grid_inp:
            pgrids = prep_data_grids(grids, self.grid_split)
            pnext_grids = prep_data_grids(next_grids, self.grid_split)
            data1[self.grid] = pgrids
            data2[self.grid] = pnext_grids
        value = self.sess.run(self.value, feed_dict=data1)[0, 0]
        next_value = self.sess.run(self.value, feed_dict=data2)[0, 0]
        if avg_reward is None:
            td_err = rewards[0] + discount * next_value - value
        else:
            td_err = rewards[0] - avg_reward + next_value - value

        if self.grid_inp:
            # print(pgrids[0].shape, freps[0].shape)
            inp = np.dstack((pgrids[0], freps[0]))
            next_inp = np.dstack((pnext_grids[0], next_freps[0]))
            inp_colvec = np.reshape(inp, [-1, 1])
            next_inp_colvec = np.reshape(next_inp, [-1, 1])
        else:
            inp_colvec = np.reshape(freps[0], [-1, 1])
            next_inp_colvec = np.reshape(next_freps[0], [-1, 1])
        # dot is inner product and therefore a scalar
        dot = np.dot(inp_colvec.T, self.weights)
        if avg_reward is None:
            grad = -2 * weights[0] * (
                td_err * inp_colvec - discount * next_inp_colvec * dot)
        else:
            grad = -2 * weights[0] * (
                td_err * inp_colvec + avg_reward - next_inp_colvec * dot)
        lr, _ = self.sess.run(
            [self.lr, self.do_train],
            feed_dict={self.grads[0][0]: grad},
            options=self.options,
            run_metadata=self.run_metadata)
        self.weights += self.grad_beta * (td_err - dot) * inp_colvec
        self.grad_beta *= self.grad_beta_decay
        return td_err**2, lr, td_err
