import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_optimizer_by_name, get_trainable_vars, scale_freps


class SinghNet(Net):
    def __init__(self, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        super().__init__(name=self.name, *args, **kwargs)

    def build(self):
        frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.nfreps = tf.placeholder(tf.float32, frepshape, "feature_reps2")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.grads = tf.placeholder(tf.float32, [None, 10339, 1], "value_target")

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
            online_vars = get_trainable_vars(scope)

        self.err = self.value_target - self.value
        self.loss = tf.losses.mean_squared_error(
            labels=self.value_target, predictions=self.value)
        if self.pp['net_lr_decay'] < 1:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.pp['net_lr'], global_step,
                                                       10000, self.pp['net_lr_decay'])
        else:
            global_step = None
            learning_rate = tf.constant(self.pp['net_lr'])
        trainer = get_optimizer_by_name(self.pp['optimizer'], learning_rate)
        self.gradients_out, trainable_vars = zip(
            *trainer.compute_gradients(self.loss, var_list=online_vars))
        full_grads = (-(self.grads[0] - self.gradients_out[0]), )
        self.do_train2 = trainer.apply_gradients(
            zip(full_grads, trainable_vars), global_step=global_step)
        return online_vars

    def forward(self, freps):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, freps, rewards, next_freps, gamma):
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})
        value_target = rewards + gamma * next_value
        # TODO NOTE TODO IS this really the correct reward, and the
        # correct target
        # if next_value[0] != next_val:
        #     print(next_value, next_val)
        # value_target = reward + self.gamma * next_val
        next_g = self.sess.run(
            self.gradients_out,
            feed_dict={
                self.freps: next_freps,
                self.value_target: value_target
            })

        data = {self.freps: freps, self.value_target: value_target, self.grads: next_g}
        _, loss, lr, err = self.sess.run(
            [self.do_train2, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        return loss, lr, err
