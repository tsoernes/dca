import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, prep_data_grids
from utils import prod


class ManSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        Manual gradients
        """
        self.name = "SinghNet"
        self.grid_inp = pp['singh_grid']
        self.frepshape = frepshape
        if self.grid_inp:
            self.wdim = prod(frepshape) + prod([7, 7, 2 * pp['n_channels']])
        else:
            self.wdim = prod(frepshape)
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.frep = tf.placeholder(tf.int32, [None, *self.frepshape], "feature_reps")
        self.next_frep = tf.placeholder(tf.int32, [None, *self.frepshape],
                                        "nfeature_reps")
        self.grads = tf.placeholder(tf.float32, [self.wdim, 1], "grad_corr")

        frep = tf.cast(self.frep, tf.float32)
        next_frep = tf.cast(self.next_frep, tf.float32)
        if self.grid_inp:
            gridshape = [None, self.rows, self.cols, 2 * self.n_channels]
            self.grid = tf.placeholder(tf.bool, gridshape, "grid")
            self.next_grid = tf.placeholder(tf.bool, gridshape, "next_grid")
            grid = tf.cast(self.grid, tf.float32)
            next_grid = tf.cast(self.next_grid, tf.float32)
            net_inp = tf.concat([grid, frep], axis=3)
            next_net_inp = tf.concat([next_grid, next_frep], axis=3)
        else:
            net_inp = frep
            next_net_inp = next_frep

        hidden = tf.Variable(tf.zeros(shape=(self.wdim, 1)), name="hidden")
        self.value = tf.matmul(tf.layers.flatten(net_inp), hidden)
        self.next_value = tf.matmul(tf.layers.flatten(next_net_inp), hidden)
        self.grads = [(tf.placeholder(tf.float32, [self.wdim, 1]), hidden)]

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

    def _get_vals_inps(self, freps, next_freps, grids, next_grids):
        data = {self.frep: freps, self.next_frep: next_freps}
        if self.grid_inp:
            pgrids = prep_data_grids(grids, self.grid_split)
            pnext_grids = prep_data_grids(next_grids, self.grid_split)
            data[self.grid] = pgrids
            data[self.next_grid] = pnext_grids
        val, next_val, lr = self.sess.run(
            [self.value, self.next_value, self.lr], feed_dict=data)
        value, next_value = val[0, 0], next_val[0, 0]

        if self.grid_inp:
            # print(pgrids[0].shape, freps[0].shape)
            inp = np.dstack((pgrids[0], freps[0]))
            next_inp = np.dstack((pnext_grids[0], next_freps[0]))
            inp_colvec = np.reshape(inp, [-1, 1])
            next_inp_colvec = np.reshape(next_inp, [-1, 1])
        else:
            inp_colvec = np.reshape(freps[0], [-1, 1])
            next_inp_colvec = np.reshape(next_freps[0], [-1, 1])
        return value, next_value, inp_colvec, next_inp_colvec, lr

    def backward(self,
                 *,
                 freps,
                 rewards,
                 next_freps,
                 discount=None,
                 weights=[1],
                 avg_reward=None,
                 grids=None,
                 next_grids=None,
                 **kwargs):
        assert len(freps) == 1  # Hard coded for one-step

        value, next_value, inp_colvec, next_inp_colvec, _ = self._get_vals_inps(
            freps, next_freps, grids, next_grids)

        td_err = rewards[0] + discount * next_value - value
        grad = -2 * td_err * inp_colvec  # Gradient for MSE
        data = {self.grads[0][0]: grad}
        lr, _ = self.sess.run([self.lr, self.do_train], data)
        assert not np.isnan(td_err) or not np.isinf(td_err)
        return td_err**2, lr, td_err
