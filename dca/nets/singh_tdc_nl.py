import numpy as np
import tensorflow as tf

# yapf: disable
from nets.convlayers import (DepthwiseConv2D, InPlaneSplit,  # noqa
                             InPlaneSplitLocallyConnected2D, SeparableSplit)
# yapf: enable
from nets.net import Net
from nets.utils import build_default_trainer, get_trainable_vars
from utils import prod


class TDCNLSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction for non-linear function approximation
        See 'Convergent Temporal-Difference Learning with Arbitrary
        Smooth Function Approximation' Hamid R. Maei 2009
        http://papers.nips.cc/paper/3809-convergent-temporal-difference-learning-with-arbitrary-smooth-function-approximation.pdf
        """
        self.name = "TDC_NL"
        self.grad_beta = pp['grad_beta']
        self.grad_beta_decay = 1 - pp['grad_beta_decay']
        self.grid_inp = pp['singh_grid']
        self.frepshape = [None, *frepshape]
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _build_net(self, top_inps):
        with tf.variable_scope('model/' + self.name) as scope:
            # conv = DepthwiseConv2D(self.depth, self.pp['conv_kernel_sizes'][0])
            conv1 = tf.layers.Conv2D(
                filters=self.pp['conv_nfilters'][0],
                kernel_size=self.pp['conv_kernel_sizes'][0],
                padding='SAME',
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.conv_regularizer,
                use_bias=False,
                bias_initializer=tf.constant_initializer(0.1),
                activation=self.act_fn,
                name="vconv",
                _reuse=False)
            conv2 = tf.layers.Conv2D(
                filters=self.pp['conv_nfilters'][0],
                kernel_size=self.pp['conv_kernel_sizes'][0],
                padding='SAME',
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.conv_regularizer,
                use_bias=False,
                bias_initializer=tf.constant_initializer(0.1),
                activation=self.act_fn,
                name="vconv",
                _reuse=True)
            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=self.kern_init_dense(),
                use_bias=False,
                activation=None)

            # def napply(inp):
            #     return value_layer.apply(conv.apply(inp))

            # vals = list(map(napply, top_inps))
            vals = [
                value_layer.apply(tf.layers.flatten(conv1.apply(top_inps[0]))),
                value_layer.apply(tf.layers.flatten(conv2.apply(top_inps[1])))
            ]
            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            # self.weight_vars.append(conv.filters)
            # self.weight_names.append(conv.name)
            trainable_vars_by_name = get_trainable_vars(scope)
        return vals, trainable_vars_by_name

    def _build_inputs(self):
        self.frep = tf.placeholder(tf.int32, self.frepshape, "feature_rep")
        self.next_frep = tf.placeholder(tf.int32, self.frepshape, "next_feature_rep")
        self.reward = tf.placeholder(tf.float32, [None], "reward")
        self.avg_reward = tf.placeholder(tf.float32, [], "avg_reward")
        self.discount = tf.placeholder(tf.float32, [None], "discount")
        self.ph_grad_beta = tf.placeholder(tf.float32, [], "grad_beta")

        # frep = tf.layers.flatten(tf.cast(self.frep, tf.float32))
        # next_frep = tf.layers.flatten(tf.cast(self.next_frep, tf.float32))
        frep = tf.cast(self.frep, tf.float32)
        next_frep = tf.cast(self.next_frep, tf.float32)
        if self.grid_inp:
            gridshape = [None, self.rows, self.cols, 2 * self.n_channels]
            self.depth = self.frepshape[-1] + gridshape[-1]
            self.grid = tf.placeholder(tf.bool, gridshape, "grid")
            self.next_grid = tf.placeholder(tf.bool, gridshape, "next_grid")
            grid = tf.cast(self.grid, tf.float32)
            next_grid = tf.cast(self.next_grid, tf.float32)
            net_inp = (tf.concat([grid, frep], axis=3),
                       tf.concat([next_grid, next_frep], axis=3))
        else:
            net_inp = (frep, next_frep)
            self.depth = self.frepshape[-1]
        return net_inp

    def build(self):
        net_inp = self._build_inputs()
        v, trainable_vars_by_name = self._build_net(net_inp)
        self.value, next_value = v
        print("val shapes:", self.value.shape, next_value.shape)
        trainer, self.lr, global_step = build_default_trainer(**self.pp)

        # print(trainable_vars_by_name)
        # print(shapes, fshapes, wdim)
        v_grads, trainable_vars = zip(
            *trainer.compute_gradients(self.value, var_list=trainable_vars_by_name))
        nv_grads, _ = zip(
            *trainer.compute_gradients(next_value, var_list=trainable_vars_by_name))
        shapes = [tuple(map(int, v.shape)) for v in trainable_vars]
        wdim = sum([prod(s) for s in shapes])  # Total number of parameters in net
        weights = tf.Variable(tf.zeros(shape=(wdim, 1)))  # w_t

        def flatten(li, batch_size=1):
            return tf.concat([tf.reshape(e, [-1, batch_size]) for e in li], axis=0)

        v_gradsf = flatten(v_grads)
        nv_gradsf = flatten(nv_grads)

        # print("vgrad", v_grads)
        # print("zip", list(zip(v_grads, weights)))
        self.td_err = self.reward + self.discount * next_value - self.value
        dot = tf.matmul(tf.transpose(v_gradsf), weights)
        nl_hess, _ = zip(*trainer.compute_gradients(dot, var_list=trainable_vars_by_name))
        nl_hessp = []
        for i, v in enumerate(nl_hess):
            if v is None:
                self.logger.error(f"None-gradient in hessian at var {i}")
                nl_hessp.append(0.0)
            else:
                nl_hessp.append(v)
        nl_hessf = flatten(nl_hessp)
        # print(shapes, trainable_vars1a, trainable_vars1b, trainable_vars2)
        h = (self.td_err - dot) * nl_hessf
        # Multiply by 2 to get equivalent magnitude to MSE
        # Multiply by -1 because SGD-variants invert grads
        gradsf = -2 * (self.td_err * v_gradsf - (self.discount * dot) * nv_gradsf - h)

        # Revert back from a single flat weight to one non-flat weight for each layer
        grads = []
        used = 0
        for s in shapes:
            sf = prod(s)
            grads.append(tf.reshape(gradsf[used:used + sf], s))
            used += sf
        # print([grad.shape for grad in grads])

        grads_and_vars = zip(grads, trainable_vars)
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([self.do_train]):
            diff = (self.ph_grad_beta * (self.td_err - dot)) * v_gradsf
            self.update_weights = weights.assign_add(diff)

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
            [self.lr, self.td_err, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.grad_beta *= self.grad_beta_decay
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
