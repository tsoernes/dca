import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, get_trainable_vars


class GTD2SinghNet(Net):
    def __init__(self, *args, **kwargs):
        self.name = "GTD2"
        super().__init__(name=self.name, *args, **kwargs)
        self.weight_beta = self.pp['weight_beta']
        self.weights = np.zeros((self.rows * self.cols * (self.n_channels + 1), 1))

    def build(self):
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.next_freps = tf.placeholder(tf.float32, frepshape, "next_feature_reps")
        self.rewards = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.dot = tf.placeholder(tf.float32, [None, 1], "dot")
        freps_rowvec = tf.layers.flatten(self.freps)
        next_freps_rowvec = tf.layers.flatten(self.next_freps)
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
            self.next_value = dense.apply(next_freps_rowvec)
            # online_vars = tuple(get_trainable_vars(scope).values())
            online_vars = get_trainable_vars(scope)

        self.td_err = self.rewards + self.discount * self.next_value - self.value

        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        grads, trainable_vars = zip(*trainer.compute_gradients(self.td_err, online_vars))
        # grads = grads * self.dot  #
        grads = [grad * self.dot for grad in grads]
        self.do_train = trainer.apply_gradients(
            zip(grads, trainable_vars), global_step=global_step)
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

        frep_colvec = np.reshape(freps[0], [-1, 1])
        # dot is inner product and therefore a scalar
        dot = np.dot(frep_colvec.T, self.weights)

        data = {
            self.freps: freps,
            self.next_freps: next_freps,
            self.rewards: rewards,
            self.discount: [discount],
            self.dot: dot,
        }
        td_err, lr, _ = self.sess.run(
            [self.td_err, self.lr, self.do_train],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.weights += self.weight_beta * (td_err - dot) * frep_colvec
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
