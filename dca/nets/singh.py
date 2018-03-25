import numpy as np
import tensorflow as tf

from nets.net import Net
from nets.utils import get_trainable_vars, prep_data_grids, scale_freps_big


class SinghNet(Net):
    def __init__(self, pp, *args, **kwargs):
        """
        Afterstate value net
        """
        self.name = "SinghNet"
        self.pre_conv = pp['pre_conv']
        super().__init__(name=self.name, pp=pp, *args, **kwargs)
        self.weight_beta = self.pp['weight_beta']
        self.weight_beta_decay = self.pp['weight_beta_decay']
        self.avg_reward = [0]

    def _build_net(self, freps, name):
        with tf.variable_scope('model/' + name) as scope:
            if self.pre_conv:
                pad = tf.keras.layers.ZeroPadding2D((1, 1))
                out = pad(freps)
                dense_inp = tf.keras.layers.LocallyConnected2D(
                    filters=70,
                    kernel_size=3,
                    padding="valid",
                    kernel_initializer=self.kern_init_conv(),
                    use_bias=self.pp['conv_bias'],
                    activation=None)(out)
            else:
                dense_inp = freps
            value_layer = tf.layers.Dense(
                units=1,
                # kernel_initializer=tf.zeros_initializer(),
                kernel_initializer=self.kern_init_dense(),
                kernel_regularizer=self.dense_regularizer,
                use_bias=False,
                activation=None)
            value = value_layer.apply(tf.layers.flatten(dense_inp))
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            trainable_vars = get_trainable_vars(scope)
        return value, trainable_vars

    def build(self):
        # frepshape = [None, self.rows, self.cols, self.n_channels * 3 + 1]
        frepshape = [None, self.rows, self.cols, self.n_channels + 1]
        self.freps = tf.placeholder(tf.float32, frepshape, "feature_reps")
        self.grids = tf.placeholder(
            tf.bool, [None, self.rows, self.cols, 2 * self.n_channels], "grid")
        self.value_target = tf.placeholder(tf.float32, [None, 1], "value_target")
        self.weights = tf.placeholder(tf.float32, [None, 1], "weight")

        freps = scale_freps_big(self.freps) if self.pp['scale_freps'] else self.freps
        net_inp = tf.concat([tf.cast(self.grids, tf.float32), freps], axis=3)
        self.value, online_vars = self._build_net(net_inp, "online")

        self.err = self.value_target - self.value
        if self.pp['huber_loss'] is not None:
            # Linear when loss is above delta and squared difference below
            self.loss = tf.losses.huber_loss(
                labels=self.value_target,
                predictions=self.value,
                delta=self.pp['huber_loss'])
        else:
            self.loss = tf.losses.mean_squared_error(
                labels=self.value_target, predictions=self.value, weights=self.weights)
        return self.loss, online_vars

    def forward(self, grids, freps):
        values = self.sess.run(
            self.value,
            feed_dict={
                self.freps: freps,
                self.grids: prep_data_grids(grids, self.grid_split)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward_supervised(self,
                            grids,
                            freps,
                            value_target,
                            weights=[1],
                            *args,
                            **kwargs):
        weights = np.expand_dims(weights, axis=1)
        data = {
            self.freps: freps,
            self.grids: prep_data_grids(grids, self.grid_split),
            self.value_target: value_target,
            self.weights: weights
        }
        _, loss, lr, err = self.sess.run(
            [self.do_train, self.loss, self.lr, self.err],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if self.pp['avg_reward']:
            self.avg_reward += self.weight_beta * err[0][0]
            # self.weight_beta *= self.weight_beta_decay
            # print(self.avg_reward)
        return loss, lr, err

    def backward(self,
                 grids,
                 freps,
                 rewards,
                 next_grids,
                 next_freps,
                 discount=None,
                 weights=[1]):
        next_value = self.sess.run(
            self.value,
            feed_dict={
                self.freps: next_freps,
                self.grids: prep_data_grids(next_grids, self.grid_split),
            })
        # print(next_value, next_value.shape)
        if self.pp['avg_reward']:
            value_target = rewards + next_value - self.avg_reward
        else:
            rewards = np.expand_dims(rewards, axis=1)
            value_target = rewards + discount * next_value
        # print(value_target, value_target.shape, rewards)
        return self.backward_supervised(grids, freps, value_target, weights)
