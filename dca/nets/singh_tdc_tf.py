import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, prep_data_grids
from utils import prod


class TFTDCSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction, TF impl
        """
        self.name = "TF_TDC"
        self.grad_beta = pp['grad_beta']
        self.grad_beta_decay = 1 - pp['grad_beta_decay']
        self.frepshape = [None, *frepshape]
        self.grid_inp = pp['singh_grid']
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.frep = tf.placeholder(tf.int32, self.frepshape, "feature_reps")
        self.next_frep = tf.placeholder(tf.int32, self.frepshape, "next_feature_reps")
        self.avg_reward = tf.placeholder(tf.float32, [], "avg_reward")
        self.reward = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.ph_grad_beta = tf.placeholder(tf.float32, [], "grad_beta")
        self.imp_weight = tf.placeholder(tf.float32, [], "importance_weight")

        frep = tf.cast(self.frep, tf.float32)
        next_frep = tf.cast(self.next_frep, tf.float32)
        if self.grid_inp:
            gridshape = [None, self.rows, self.cols, self.n_channels * 2]
            wdim = prod(self.frepshape[1:]) + prod(gridshape[1:])
            self.grid = tf.placeholder(tf.bool, gridshape, "grid")
            self.next_grid = tf.placeholder(tf.bool, gridshape, "next_grid")
            grid = tf.cast(self.grid, tf.float32)
            next_grid = tf.cast(self.next_grid, tf.float32)
            net_inp = tf.concat([grid, frep], axis=3)
            next_net_inp = tf.concat([next_grid, next_frep], axis=3)
        else:
            wdim = prod(self.frepshape[1:])  # Number of parameters in neural net
            net_inp, next_net_inp = frep, next_frep

        net_inp_rv = tf.layers.flatten(net_inp)  # x_t  Row vector
        next_net_inp_rv = tf.layers.flatten(next_net_inp)  # x_{t+1}  Row vector
        net_inp_cv = tf.transpose(net_inp_rv)  # Col vector
        next_net_inp_cv = tf.transpose(next_net_inp_rv)  # Col vector

        self.weights = tf.Variable(tf.zeros(shape=(wdim, 1)), name="gradweights")  # w_t
        hidden = tf.Variable(tf.zeros(shape=(wdim, 1)), name="dense")  # theta_t
        self.value = tf.matmul(net_inp_rv, hidden)
        next_value = tf.matmul(next_net_inp_rv, hidden)

        td_err = self.reward - self.avg_reward + self.discount * next_value - self.value
        self.loss_grad = self.default_loss_grad(td_err)
        dot = tf.matmul(net_inp_rv, self.weights)
        # Multiply by 2 to get equivalent magnitude to MSE
        # Multiply by -1 because SGD-variants inverts grads
        grads = (-2 * self.loss_grad) * net_inp_cv - (2 * self.avg_reward) + (
            2 * self.discount * dot) * next_net_inp_cv
        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        self.do_train = trainer.apply_gradients(
            [(self.imp_weight * grads, hidden)], global_step=global_step)

        # Update nn weights before grad corr weights
        with tf.control_dependencies([self.do_train]):
            diff = (self.ph_grad_beta * self.imp_weight *
                    (self.loss_grad - dot)) * net_inp_cv
            self.update_weights = self.weights.assign_add(diff)

        return None, None

    def forward(self, freps, grids):
        data = {self.frep: freps}
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        values = self.sess.run(
            self.value,
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

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
        assert discount is not None or avg_reward is not None
        assert weights is not None
        assert weights[0] is not None
        if avg_reward is not None:
            discount = 1
        else:
            avg_reward = 0

        data = {
            self.frep: freps,
            self.next_frep: next_freps,
            self.reward: rewards,
            self.discount: [discount],
            self.avg_reward: avg_reward,
            self.ph_grad_beta: self.grad_beta,
            self.imp_weight: weights[0]
        }
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
            data[self.next_grid] = prep_data_grids(next_grids, self.grid_split)
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.loss_grad, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.grad_beta *= self.grad_beta_decay
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
