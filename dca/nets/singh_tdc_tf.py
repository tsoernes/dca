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
        self.name = "SinghNet"
        self.grad_beta = pp['grad_beta']
        self.grad_beta_decay = 1 - pp['grad_beta_decay']
        self.frepshape = [None, *frepshape]
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.freps = tf.placeholder(tf.int32, self.frepshape, "feature_reps")
        self.next_freps = tf.placeholder(tf.int32, self.frepshape, "next_feature_reps")
        self.avg_reward = tf.placeholder(tf.float32, [], "avg_reward")
        self.rewards = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.ph_grad_beta = tf.placeholder(tf.float32, [], "grad_beta")

        freps_rowvec = tf.layers.flatten(tf.cast(self.freps, tf.float32))
        next_freps_rowvec = tf.layers.flatten(tf.cast(self.next_freps, tf.float32))
        freps_colvec = tf.transpose(freps_rowvec)  # x_t
        next_freps_colvec = tf.transpose(next_freps_rowvec)  # x_{t+1}

        d = prod(self.frepshape[1:])
        self.weights = tf.Variable(tf.zeros(shape=(d, 1)), name="gradweights")  # w_t
        hidden = tf.Variable(tf.zeros(shape=(d, 1)), name="dense")
        self.value = tf.matmul(freps_rowvec, hidden)
        next_value = tf.matmul(next_freps_rowvec, hidden)

        self.td_err = self.rewards - self.avg_reward + self.discount * next_value - self.value
        dot = tf.matmul(freps_rowvec, self.weights)
        # Multiply by 2 to get equivalent magnitude to MSE
        # Multiply by -1 because SGD-variants inverts grads
        grads = (-2 * self.td_err) * freps_colvec - (2 * self.avg_reward) + (
            2 * self.discount * dot) * next_freps_colvec
        grads_and_vars = [(grads, hidden)]
        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([self.do_train]):
            diff = (self.ph_grad_beta * (self.td_err - dot)) * freps_colvec
            self.update_weights = self.weights.assign_add(diff)

        return None, None

    def forward(self, freps, grids):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, *, freps, rewards, next_freps,
                 discount=None, weights=None, avg_reward=None, **kwargs):
        assert len(freps) == 1  # Hard coded for one-step
        assert discount is not None or avg_reward is not None
        if avg_reward is not None:
            discount = 1
        else:
            avg_reward = 0

        data = {
            self.freps: freps,
            self.next_freps: next_freps,
            self.rewards: rewards,
            self.discount: [discount],
            self.avg_reward: avg_reward,
            self.ph_grad_beta: self.grad_beta
        }
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.td_err, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.grad_beta *= self.grad_beta_decay
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
