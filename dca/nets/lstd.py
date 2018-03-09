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
        frepshape = [self.rows, self.cols, self.n_channels + 1]
        self.frep = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.next_frep = tf.placeholder(tf.float32, frepshape, "next_feature_reps")
        self.gamma = tf.placeholder(tf.float32, [1], "gamma")
        self.reward = tf.placeholder(tf.float32, [1], "reward")
        frepf = tf.reshape(self.frep, [-1, 1])
        next_frepf = tf.reshape(self.next_frep, [-1, 1])

        d = self.rows * self.cols * (self.n_channels + 1)
        diag = 10000 * tf.ones(shape=(d), dtype=tf.float32)
        a_inv = tf.Variable(tf.diag(diag))
        b = tf.Variable(tf.zeros(shape=(d, 1), dtype=tf.float32))

        k = frepf - self.gamma * next_frepf
        v = tf.matmul(tf.transpose(a_inv), k)
        c1 = tf.matmul(tf.matmul(a_inv, frepf), tf.transpose(v))
        c2 = 1 + tf.matmul(tf.transpose(v), frepf)
        self.update_a_inv = a_inv.assign_sub(c1 / c2)
        self.update_b = b.assign_add(self.reward * frepf)
        theta = tf.matmul(a_inv, b)
        self.value = tf.matmul(tf.transpose(theta), frepf)

        return None, None

    def forward(self, freps):
        values = []
        for i in range(len(freps)):
            v = self.sess.run(
                self.value,
                feed_dict={self.frep: freps[i]},
                options=self.options,
                run_metadata=self.run_metadata)
            values.append(v[0][0])
        return np.array(values)

    def backward(self, freps, rewards, next_freps, gamma):
        frep, reward, next_frep = freps[0], rewards[0], next_freps[0]
        self.sess.run(
            self.update_a_inv,
            feed_dict={
                self.frep: frep,
                self.next_frep: next_frep,
                self.gamma: [gamma]
            })
        self.sess.run(self.update_b, feed_dict={self.frep: frep, self.reward: [reward]})
