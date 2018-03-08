import numpy as np
import tensorflow as tf

from nets.net import Net


class LSTDNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [self.rows, self.cols, self.n_channels + 1]
        self.frep = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.next_frep = tf.placeholder(tf.float32, frepshape, "next_feature_reps")
        frepf = tf.reshape(self.frep, [-1, 1])
        next_frepf = tf.reshape(self.next_frep, [-1, 1])
        self.gamma = tf.placeholder(tf.float32, [1], "gamma")
        self.reward = tf.placeholder(tf.float32, [1], "reward")
        d = self.rows * self.cols * (self.n_channels + 1)
        a_inv = tf.Variable(0.1 * tf.ones(shape=(d, d), dtype=tf.float32))
        b = tf.Variable(tf.zeros(shape=(d, 1), dtype=tf.float32))
        k = frepf - self.gamma * next_frepf
        v = tf.matmul(tf.transpose(a_inv), k)
        c1 = tf.matmul(tf.matmul(tf.transpose(a_inv), frepf), tf.transpose(v))
        c2 = 1 + tf.matmul(tf.transpose(v), frepf)
        self.update_a_inv = a_inv.assign_sub(c1 / c2)
        self.update_b = b.assign_add(self.reward * frepf)
        theta = tf.matmul(a_inv, b)

        self.freps = tf.placeholder(tf.float32, [None, *frepshape], "feature_reps")
        # This is suspect. Why must freps be [d, X] here and [X, d] above?
        frepsf = tf.reshape(self.freps, [d, -1])
        self.value = tf.matmul(tf.transpose(theta), frepsf)

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, frep, reward, next_frep, gamma):
        self.sess.run(
            self.update_a_inv,
            feed_dict={
                self.frep: frep,
                self.next_frep: next_frep,
                self.gamma: [gamma]
            })
        self.sess.run(self.update_b, feed_dict={self.frep: frep, self.reward: [reward]})
