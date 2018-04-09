import numpy as np
import tensorflow as tf

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
        self.frepshape = [None, *frepshape]
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.freps = tf.placeholder(tf.int32, self.frepshape, "feature_reps")
        self.next_freps = tf.placeholder(tf.int32, self.frepshape, "next_feature_reps")
        self.rewards = tf.placeholder(tf.float32, [None], "rewards")
        self.discount = tf.placeholder(tf.float32, [None], "discount")

        freps_rowvec = tf.layers.flatten(tf.cast(self.freps, tf.float32))
        next_freps_rowvec = tf.layers.flatten(tf.cast(self.next_freps, tf.float32))
        trainer, self.lr, global_step = build_default_trainer(**self.pp)
        # NOTE Code fails if there's only 1 layer

        with tf.variable_scope('model/' + self.name) as scope:
            h1 = tf.layers.Dense(
                units=30,
                kernel_initializer=tf.glorot_uniform_initializer(),
                use_bias=False,
                activation=tf.nn.relu)
            h2 = tf.layers.Dense(
                units=1,
                kernel_initializer=tf.glorot_uniform_initializer(),
                use_bias=False,
                activation=None)

            def net_apply(inp):
                return h2.apply(h1.apply(inp))
                # return h2.apply(inp)

            self.value = net_apply(freps_rowvec)
            next_value = net_apply(next_freps_rowvec)
            trainable_vars_by_name = get_trainable_vars(scope)
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
        self.td_err = self.rewards + self.discount * next_value - self.value
        dot = tf.matmul(tf.transpose(v_gradsf), weights)
        nl_hess, trainable_vars = zip(
            *trainer.compute_gradients(dot, var_list=trainable_vars_by_name))
        nl_hessf = flatten(nl_hess)
        print(trainable_vars)
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
        print([grad.shape for grad in grads])

        grads_and_vars = zip(grads, trainable_vars)
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([self.do_train]):
            diff = (tf.constant(self.grad_beta) * (self.td_err - dot)) * v_gradsf
            self.update_weights = weights.assign_add(diff)

        return None, None

    def forward(self, freps, grids):
        values = self.sess.run(
            self.value,
            feed_dict={self.freps: freps},
            options=self.options,
            run_metadata=self.run_metadata)
        vals = np.reshape(values, [-1])
        return vals

    def backward(self, *, freps, rewards, next_freps, discount, weights, **kwargs):
        assert len(freps) == 1  # Hard coded for one-step
        assert discount is not None
        data = {
            self.freps: freps,
            self.next_freps: next_freps,
            self.rewards: rewards,
            self.discount: [discount]
        }
        lr, td_err, _, _ = self.sess.run(
            [self.lr, self.td_err, self.do_train, self.update_weights],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        td_err = td_err[0, 0]
        return td_err**2, lr, td_err
