import numpy as np
import tensorflow as tf

from nets.convlayers import DepthwiseConv2D
from nets.net import Net
from nets.utils import build_default_trainer, get_trainable_vars
from utils import prod


class TDCNLSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction for Non-linear func approx
        """
        self.name = "SinghNet"
        self.grad_beta = pp['grad_beta']
        self.grid_inp = pp['singh_grid']
        self.frepshape = [None, *frepshape]
        super().__init__(name=self.name, pp=pp, logger=logger)

    def _build_net(self, top_inps):
        with tf.variable_scope('model/' + self.name) as scope:
            shape = (self.pp['conv_kernel_sizes'][0], self.pp['conv_kernel_sizes'][0],
                     self.depth, 1)
            filters = tf.Variable(self.kern_init_conv()(shape))
            conv1 = tf.nn.depthwise_conv2d(
                top_inps[0], filters, strides=[1, 1, 1, 1], padding="SAME")
            conv2 = tf.nn.depthwise_conv2d(
                top_inps[1], filters, strides=[1, 1, 1, 1], padding="SAME")

            value_layer = tf.layers.Dense(
                units=1,
                kernel_initializer=self.kern_init_dense(),
                use_bias=False,
                activation=None)

            # def napply(inp):
            #     return value_layer.apply(conv.apply(inp))

            print(top_inps)
            # val = value_layer.apply(conv1)
            next_val = value_layer.apply(conv2)
            val = None
            # # val = napply(top_inps[0])
            # next_val = napply(top_inps[1])

            self.weight_vars.append(value_layer.kernel)
            self.weight_names.append(value_layer.name)
            # self.weight_vars.append(conv.filters)
            # self.weight_names.append(conv.name)
            trainable_vars_by_name = get_trainable_vars(scope)
        return (val, next_val), trainable_vars_by_name

    def _build_inputs(self):
        self.frep = tf.placeholder(tf.int32, self.frepshape, "feature_rep")
        self.next_frep = tf.placeholder(tf.int32, self.frepshape, "next_feature_rep")
        self.reward = tf.placeholder(tf.float32, [None], "reward")
        self.discount = tf.placeholder(tf.float32, [None], "discount")

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
        trainer, self.lr, global_step = build_default_trainer(**self.pp)

        # print(trainable_vars_by_name)
        shapes = [tuple(map(int, v.shape)) for v in trainable_vars_by_name.values()]
        fshapes = [prod(s) for s in shapes]
        wdim = sum(fshapes)
        # print(shapes, fshapes, wdim)
        weights = tf.Variable(tf.zeros(shape=(wdim, 1)))  # w_t
        v_grads, _ = zip(
            *trainer.compute_gradients(self.value, var_list=trainable_vars_by_name))
        nv_grads, _ = zip(
            *trainer.compute_gradients(next_value, var_list=trainable_vars_by_name))

        def flatten(li, batch_size=1):
            return tf.concat([tf.reshape(e, [-1, batch_size]) for e in li], axis=0)

        v_gradsf = flatten(v_grads)
        nv_gradsf = flatten(nv_grads)

        # print("vgrad", v_grads)
        # print("zip", list(zip(v_grads, weights)))
        self.td_err = self.reward + self.discount * next_value - self.value
        dot = tf.matmul(tf.transpose(v_gradsf), weights)
        nl_hess, trainable_vars = zip(
            *trainer.compute_gradients(dot, var_list=trainable_vars_by_name))
        nl_hessp = []
        for i, v in enumerate(nl_hess):
            if v is None:
                self.logger.error(f"None-gradient in hessian at var {i}")
                nl_hessp.append(0.0)
            else:
                nl_hessp.append(v)
        nl_hessf = flatten(nl_hessp)
        # print(trainable_vars)
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
            diff = (tf.constant(self.grad_beta) * (self.td_err - dot)) * v_gradsf
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

    def backward(self, *, freps, rewards, next_freps, discount, weights, **kwargs):
        assert len(freps) == 1  # Hard coded for one-step
        assert discount is not None
        data = {
            self.frep: freps,
            self.next_frep: next_freps,
            self.reward: rewards,
            self.discount: [discount]
        }
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.td_err, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
