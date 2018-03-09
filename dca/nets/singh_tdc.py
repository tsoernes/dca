import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import (get_optimizer_by_name, get_trainable_vars,
                        scale_freps_big)


class TDCSinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        TD0 with Gradient correction
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)
        self.weight_beta = self.pp['weight_beta']
        self.weights = np.zeros((self.rows * self.cols * (self.n_channels + 1), 1))

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.grads = tf.placeholder(tf.float32, [3479, 1], "grad_corr")

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
        self.grads = [(tf.placeholder(tf.float32, [3479, 1]), online_vars[0])]

        if self.pp['net_lr_decay'] < 1:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.pp['net_lr'], global_step,
                                                       10000, self.pp['net_lr_decay'])
        else:
            global_step = None
            learning_rate = tf.constant(self.pp['net_lr'])
        self.lr = learning_rate

        trainer = get_optimizer_by_name(self.pp['optimizer'], learning_rate)
        self.do_train2 = trainer.apply_gradients(self.grads, global_step=global_step)

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, frep, reward, next_frep, gamma):
        value = self.sess.run(self.value, feed_dict={self.freps: [frep]})[0, 0]
        next_value = self.sess.run(self.value, feed_dict={self.freps: [next_frep]})[0, 0]
        td_err = reward + gamma * next_value - value

        frep_colvec = np.reshape(frep, [-1, 1])
        next_frep_colvec = np.reshape(next_frep, [-1, 1])
        dot = np.dot(frep_colvec.T, self.weights)
        # dot should be a inner product and therfore result in a scalar
        s = np.reshape(dot, [-1]).shape
        assert s == (1, ), s
        grad = td_err * frep_colvec - gamma * np.dot(next_frep_colvec, dot)
        data = {
            self.freps: [frep],
            self.grads[0][0]: -2 * grad
        }  # yapf:disable
        lr, _ = self.sess.run(
            [self.lr, self.do_train2],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.weights += self.weight_beta * (td_err - dot) * frep_colvec
        return td_err**2, lr, td_err
