import random
from functools import partial, reduce
from operator import mul

import numpy as np
import tensorflow as tf
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import (activations, constraints,
                                                 initializers, regularizers)
from tensorflow.python.keras._impl.keras.engine import InputSpec, Layer
from tensorflow.python.keras._impl.keras.utils import conv_utils

from gridfuncs import GF


def scale_and_centre_freps(freps):
    """Scale feature reps to range [-1, 1]"""
    freps_f = tf.cast(freps, tf.float32)
    mult1 = np.ones(freps.shape[1:], np.float32)
    mult1[:, :, :-1] /= 3.5
    mult1[:, :, -1] /= 35
    freps_f = freps_f * tf.constant(mult1) - 1
    return freps_f


def scale_freps_big(freps):
    """Scale feature reps [0, 1]"""
    freps_f = tf.cast(freps, tf.float32)
    mult1 = np.ones(freps.shape[1:], np.float32)
    mult1[:, :, :70] /= 7
    mult1[:, :, 70:140] /= 7
    mult1[:, :, -1] /= 70
    freps_f = freps_f * tf.constant(mult1)
    return freps_f


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x. If axis is not given,
    softmax over the last dimension."""
    if axis is None:
        axis = np.ndim(x) - 1
    # Subtract maximum value for numerical stability; result will be the same
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def discount(rewards, gamma):
    discounted = []
    r = 0
    for reward in rewards[::-1]:
        r = reward + gamma * r
        discounted.append(r)
    return discounted[::-1]


def set_global_seeds(i):
    "Reproducible results"
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_trainable_vars(scope):
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return trainable_vars_by_name


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def get_init_by_name(name, pp):
    inits = {
        "zeros": tf.zeros_initializer,
        "glorot_unif":  # The default for dense, perhaps for conv2d also. AKA Xavier.
        tf.glorot_uniform_initializer,
        "glorot_norm": tf.glorot_normal_initializer,
        "norm_cols": normalized_columns_initializer,
        "norm_pos": tf.random_normal_initializer(0., 0.2),  # Try for dense kernel
        "const_pos": tf.constant_initializer(0.1),  # Try for dense bias
        "nominal": partial(NominalInitializer, pp['qnom_lo'], pp['qnom_hi'])
    }  # yapf: disable
    return inits[name]


def get_act_fn_by_name(name):
    act_fns = {"relu": tf.nn.relu, "elu": tf.nn.elu, "leaky_relu": tf.nn.leaky_relu}
    return act_fns[name]


def build_default_trainer(net_lr, net_lr_decay, optimizer, **kwargs):
    if net_lr_decay < 1:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(net_lr, global_step, 10000,
                                                   net_lr_decay)
    else:
        global_step = None
        learning_rate = tf.constant(net_lr)
    trainer = get_optimizer_by_name(optimizer, learning_rate)
    return trainer, learning_rate, global_step


def build_default_minimizer(net_lr,
                            net_lr_decay,
                            optimizer,
                            max_grad_norm,
                            loss,
                            var_list=None,
                            **kwargs):
    """
    Build a trainer to minimize loss through adjusting vars in var_list.
    Optionally decay learning rate and clip gradients. Return training op
    and learning rate var.

    If var_list is not specified, defaults to GraphKeys.TRAINABLE_VARIABLES,
    i.e. all trainable variables
    """
    trainer, learning_rate, global_step = build_default_trainer(
        net_lr, net_lr_decay, optimizer)
    # For batch norm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # For L2 reg:
    tot_loss = loss + tf.losses.get_regularization_loss()
    with tf.control_dependencies(update_ops):
        if max_grad_norm is not None:
            gradients, trainable_vars = zip(
                *trainer.compute_gradients(tot_loss, var_list=var_list))
            clipped_grads, grad_norms = tf.clip_by_global_norm(gradients, max_grad_norm)
            do_train = trainer.apply_gradients(
                zip(clipped_grads, trainable_vars), global_step=global_step)
        else:
            do_train = trainer.minimize(
                tot_loss, var_list=var_list, global_step=global_step)
    return do_train, learning_rate


def get_optimizer_by_name(name, lr):
    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(lr),
        "sgd-m": tf.train.MomentumOptimizer(lr, momentum=0.95),
        "adam": tf.train.AdamOptimizer(lr),
        "rmsprop": tf.train.RMSPropOptimizer(lr)
    }
    return optimizers[name]


def copy_net_op(online_vars, target_vars, tau):
    """Move target variables 'tau' towards online variables"""
    copy_ops = []
    for var_name, target_var in target_vars.items():
        online_val = online_vars[var_name].value()
        target_val = target_var.value()
        val = online_val * tau + (1 - tau) * target_val
        op = target_var.assign(val)
        copy_ops.append(op)
    return tf.group(*copy_ops)


def prep_data_grids(grids, split=True):
    """
    split: Copy alloc map and invert second copy (empty as 1; inuse as 0).
    Leaves freps as is.
    """
    assert type(grids) == np.ndarray
    if grids.ndim == 3:
        grids = np.expand_dims(grids, axis=0)
    assert grids.shape[1:] == (7, 7, 70)
    np.concatenate
    if split:
        sgrids = np.zeros((len(grids), 7, 7, 140), dtype=np.bool)
        sgrids[:, :, :, :70] = grids
        sgrids[:, :, :, 70:] = np.invert(grids)
        grids = sgrids
    return grids


def prep_data_cells(cells):
    """One-hot cell encoding"""
    if type(cells) == tuple:
        cells = [cells]
    if type(cells[0]) != tuple:
        raise Exception("WHOAH WHOAH using np arrays for indexing works differently")
        # For python atleast. perhaps admissible in TF
    oh_cells = np.zeros((len(cells), 7, 7, 1), dtype=np.bool)
    for i, cell in enumerate(cells):
        oh_cells[i][cell][0] = 1
    return oh_cells


def prep_data(grids, cells, actions, rewards, next_grids=None, next_cells=None):
    assert type(actions) == np.ndarray
    assert type(rewards) == np.ndarray
    actions = actions.astype(np.int32)
    # Needs to be 32-bit, else will overflow
    rewards = rewards.astype(np.float32)
    # Cells are used as indecies and must be tuples
    if type(cells) == np.ndarray:
        cells = list(map(tuple, cells))
        if next_cells is not None:
            next_cells = list(map(tuple, next_cells))

    grids = prep_data_grids(grids)
    if next_grids is not None:
        next_grids = prep_data_grids(next_grids)
    oh_cells = prep_data_cells(cells)
    if next_cells is not None:
        next_oh_cells = prep_data_cells(next_cells)
        return grids, oh_cells, actions, rewards, next_grids, next_oh_cells
    return grids, oh_cells, actions, rewards


def xavier(n_in, n_out):
    if type(n_in) == list or type(n_in) == tuple:
        return np.random.randn(*n_in, n_out) / np.sqrt(reduce(mul, n_in))
    else:
        return np.random.randn(n_in, n_out) / np.sqrt(n_in)


class NominalInitializer(tf.keras.initializers.Initializer):
    """Initializer that prefers nominal channels"""

    def __init__(self, low, high, dtype=tf.float32):
        # self.low, self.high = low, high
        self.qrange = np.arange(high, low, (low - high) / 10)
        # qrange = np.expand_dims(qrange, 0)
        # self.qrange = np.repeat(qrange, 70, axis=0)

    def __call__(self, shape, dtype=None, partition_info=None):
        # shape: [w_out, h_out, depth_in, nfilters]
        # conv out shape: [w_out, h_out, nfilters]
        # assert shape == [7, 7, 70, 70], shape
        if shape == [49, 70, 70]:
            initvals = xavier((7, 7, 70), 70)
            # initvals = tf.glorot_uniform_initializer()((7, 7, 70, 70))
            # np.zeros((7, 7, 70, 70)[r][c][:][GF.nom_chs[r, c]].shape = [10, 70]
            for r in range(7):
                for c in range(7):
                    for k in range(70):
                        initvals[r][c][k][GF.nom_chs[r, c]] = self.qrange
            initvals = np.reshape(initvals, [49, 70, 70])
        elif shape == [3479, 70]:
            # initvals = tf.glorot_uniform_initializer()((7, 7, 71, 70))
            initvals = xavier((7, 7, 71), 70)
            # initvals = np.zeros((7, 7, 71, 70), np.float32)
            for r in range(7):
                for c in range(7):
                    for k in range(71):
                        initvals[r][c][k][GF.nom_chs[r, c]] = self.qrange
            initvals = np.reshape(initvals, [3479, 70])
            # out = np.random.randn(*shape).astype(np.float32)
            # out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        else:
            raise NotImplementedError(shape)
        return tf.constant(initvals.astype(np.float32))

    def get_config(self):
        return {"dtype": self.dtype.name}


class InPlaneLocallyConnected2D(Layer):
    """In-plane Locally-connected layer for 2D inputs.
  The `InPlaneLocallyConnected2D` layer works similarly
  to the `Conv2D_in_plane` layer, except that weights are unshared,
  that is, a different set of filters is applied at each
  different patch of the input.
  Input shape:
      4D tensor with shape:
      `(samples, rows, cols, channels)`
  Output shape:
      4D tensor with shape:
      `(samples, new_rows, new_cols, channels)`
      `rows` and `cols` values might have changed due to padding.
  """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(InPlaneLocallyConnected2D, self).__init__(**kwargs)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected2D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = conv_utils.normalize_data_format(None)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)

    # @shape_type_conversion
    def build(self, input_shape):
        input_row, input_col = input_shape[1:-1]
        input_filter = input_shape[3]
        # print("Build", input_shape)
        self.filters = input_filter
        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected2D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        self.output_row = output_row
        self.output_col = output_col
        self.kernel_shape = (output_row * output_col,
                             self.kernel_size[0] * self.kernel_size[1], 1)
        # self.kernel_size[0] * self.kernel_size[1] * int(input_filter), 1)
        # print("Kernel shape", self.kernel_shape)
        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(output_row, output_col, input_filter),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
        self.built = True

    # @shape_type_conversion
    def compute_output_shape(self, input_shape):
        print("compute_output_shape CALLED")
        rows = input_shape[1]
        cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1], self.padding,
                                             self.strides[1])

        return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs):
        individual_channels = tf.split(inputs, inputs.shape[3], -1)
        # output = K.local_conv2d(inputs, self.kernel, self.kernel_size, self.strides,
        #                         (self.output_row, self.output_col), self.data_format)
        convs = []
        for channel in individual_channels:
            conv = K.local_conv2d(channel, self.kernel, self.kernel_size, self.strides,
                                  (self.output_row, self.output_col), self.data_format)
            convs.append(conv)
        outputs = tf.concat(convs, -1)
        if self.use_bias:
            raise NotImplementedError
            # output = K.bias_add(outputs, self.bias, data_format=self.data_format)
        outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        print("get_config CALLED")
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(InPlaneLocallyConnected2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
