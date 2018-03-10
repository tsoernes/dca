import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, get_trainable_vars


class TFTDCSinghNet(Net):
    def __init__(self, pp, logger):
        """
        TD0 with Gradient correction
        """
        self.name = "SinghNet"
        self.weight_beta = pp['weight_beta']
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        d = self.rows * self.cols * (self.n_channels + 1)
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.next_freps = tf.placeholder(tf.float32, frepshape, "next_feature_reps")
        self.rewards = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.grads = tf.placeholder(tf.float32, [d, 1], "grad_corr")
        freps_rowvec = tf.layers.flatten(self.freps)
        next_freps_rowvec = tf.layers.flatten(self.next_freps)
        freps_colvec = tf.transpose(freps_rowvec)
        next_freps_colvec = tf.transpose(next_freps_rowvec)

        self.weights = tf.Variable(
            tf.zeros(shape=(self.rows * self.cols * (self.n_channels + 1), 1)))

        with tf.variable_scope('model/' + self.name) as scope:
            dense = tf.layers.Dense(
                units=1,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(),
                use_bias=False,
                activation=None,
                name="vals")
            self.value = dense.apply(freps_rowvec)
            next_value = dense.apply(next_freps_rowvec)
            online_vars = tuple(get_trainable_vars(scope).values())

        trainer, self.lr, global_step = build_default_trainer(**self.pp)

        self.td_err = self.rewards + self.discount * next_value - self.value
        dot = tf.matmul(freps_rowvec, self.weights)
        # Multiply by 2 to get equivalent magnitude to MSE
        # Multiply by -1 because SGD-variants invert grads
        grads = -2 * (
            self.td_err * freps_colvec - self.discount * next_freps_colvec * dot)
        grads_and_vars = [(grads, online_vars[0])]
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([self.do_train]):
            diff = self.weight_beta * (self.td_err - dot) * freps_colvec
            self.update_weights = self.weights.assign_add(diff)

        return None, None

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, discount):
        assert len(freps) == 1  # Hard coded for one-step

        data = {
            self.freps: freps,
            self.next_freps: next_freps,
            self.rewards: rewards,
            self.discount: [discount]
        }
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.td_err[0, 0], self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return td_err**2, lr, td_err
