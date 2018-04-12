import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer
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
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.frep = tf.placeholder(tf.int32, self.frepshape, "feature_reps")
        self.next_frep = tf.placeholder(tf.int32, self.frepshape, "next_feature_reps")
        self.avg_reward = tf.placeholder(tf.float32, [], "avg_reward")
        self.reward = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.ph_grad_beta = tf.placeholder(tf.float32, [], "grad_beta")

        freps_rowvec = tf.layers.flatten(tf.cast(self.frep, tf.float32))
        next_freps_rowvec = tf.layers.flatten(tf.cast(self.next_frep, tf.float32))
        freps_colvec = tf.transpose(freps_rowvec)  # x_t
        next_freps_colvec = tf.transpose(next_freps_rowvec)  # x_{t+1}

        d = prod(self.frepshape[1:])
        self.weights = tf.Variable(tf.zeros(shape=(d, 1)), name="gradweights")  # w_t
        hidden = tf.Variable(tf.zeros(shape=(d, 1)), name="dense")
        self.value = tf.matmul(freps_rowvec, hidden)
        next_value = tf.matmul(next_freps_rowvec, hidden)

        td_err = self.reward - self.avg_reward + self.discount * next_value - self.value
        self.loss_grad = self.default_loss_grad(td_err)
        dot = tf.matmul(freps_rowvec, self.weights)
        # Multiply by 2 to get equivalent magnitude to MSE
        # Multiply by -1 because SGD-variants inverts grads
        grads = (-2 * self.loss_grad) * freps_colvec - (2 * self.avg_reward) + (
            2 * self.discount * dot) * next_freps_colvec
        grads_and_vars = [(grads, hidden)]
        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([self.do_train]):
            diff = (self.ph_grad_beta * (self.loss_grad - dot)) * freps_colvec
            self.update_weights = self.weights.assign_add(diff)

        return None, None

    def forward(self, freps, grids):
        values = self.sess.run(
            self.value,
            feed_dict={self.frep: freps},
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
                 weights=None,
                 avg_reward=None,
                 **kwargs):
        assert len(freps) == 1  # Hard coded for one-step
        assert discount is not None or avg_reward is not None
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
            self.ph_grad_beta: self.grad_beta
        }
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.loss_grad, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.grad_beta *= self.grad_beta_decay
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
