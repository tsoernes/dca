import numpy as np
import tensorflow as tf

from nets.net import Net


class LSTDSinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Least Squares Temporal Difference
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        frepshape = [self.rows, self.cols, self.n_channels + 1]
        self.frep = tf.placeholder(tf.float32, frepshape, "feature_rep")
        self.next_frep = tf.placeholder(tf.float32, frepshape, "next_feature_rep")
        self.discount = tf.placeholder(tf.float32, [1], "discount")
        self.reward = tf.placeholder(tf.float32, [1], "reward")
        # Make column vectors of feature representations
        frepf = tf.reshape(self.frep, [-1, 1])
        next_frepf = tf.reshape(self.next_frep, [-1, 1])

        d = self.rows * self.cols * (self.n_channels + 1)
        diag = self.pp['alpha'] * tf.ones(shape=(d), dtype=tf.float32)
        a_inv = tf.Variable(tf.diag(diag))
        b = tf.Variable(tf.zeros(shape=(d, 1), dtype=tf.float32))

        # Update by least-squares
        k = frepf - self.discount * next_frepf
        v = tf.matmul(tf.transpose(a_inv), k)
        c1 = tf.matmul(tf.matmul(a_inv, frepf), tf.transpose(v))
        c2 = 1 + tf.matmul(tf.transpose(v), frepf)
        self.update_a_inv = a_inv.assign_sub(c1 / c2)
        self.update_b = b.assign_add(self.reward * frepf)

        # For feed-forward phase
        self.freps = tf.placeholder(tf.float32, [None, *frepshape], "feature_reps")
        theta = tf.matmul(a_inv, b)
        # Not entirely sure that this part is correct
        frepsf = tf.layers.flatten(self.freps)
        self.value = tf.matmul(frepsf, theta)

        return None, None

    def forward(self, freps):
        vals = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        return np.squeeze(vals, axis=1)

    def backward(self, freps, rewards, next_freps, discount):
        frep, reward, next_frep = freps[0], rewards[0], next_freps[0]
        self.sess.run(
            [self.update_a_inv, self.update_b],
            feed_dict={
                self.frep: frep,
                self.next_frep: next_frep,
                self.discount: [discount],
                self.reward: [reward]
            })
        # LSTD has no loss, learning rate, error, ..
        return 0, 0, 0
