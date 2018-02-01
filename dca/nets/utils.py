import random

import numpy as np
import tensorflow as tf


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


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def get_init_by_name(name):
    inits = {
        "zeros":
        tf.zeros_initializer,
        "glorot_unif":
        # The default for dense, perhaps for conv2d also. AKA Xavier.
        tf.glorot_uniform_initializer,
        "glorot_norm":
        tf.glorot_normal_initializer,
        "norm_cols":
        normalized_columns_initializer,
        "norm_pos":
        tf.random_normal_initializer(0., 0.2),  # Try for dense kernel
        "const_pos":
        tf.constant_initializer(0.1)  # Try for dense bias
    }
    return inits[name]


def get_act_fn_by_name(name):
    act_fns = {
        "relu": tf.nn.relu,
        "elu": tf.nn.elu,
        "leaky_relu": tf.nn.leaky_relu
    }
    return act_fns[name]


def get_optimizer_by_name(name, l_rate):
    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=l_rate),
        "sgd-m": tf.train.MomentumOptimizer(
            learning_rate=l_rate, momentum=0.95),
        "adam": tf.train.AdamOptimizer(learning_rate=l_rate),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=l_rate)
    }
    return optimizers[name]


def copy_net_op(online_vars, target_vars, tau):
    copy_ops = []
    for var_name, target_var in target_vars.items():
        online_val = online_vars[var_name].value()
        target_val = target_var.value()
        val = online_val * tau + (1 - tau) * target_val
        op = target_var.assign(val)
        copy_ops.append(op)
    return tf.group(*copy_ops)


def prep_data_grids(grids, neg=False, split=True):
    """
    neg: Represent empty channels as -1 instead of 0
    split: Double the depth and represent empty channels as 1 on separate layer
    """
    assert type(grids) == np.ndarray
    assert not (neg and split), "Can't have both options"
    if grids.ndim == 3:
        grids = np.expand_dims(grids, axis=0)
    assert grids.shape[1:] == (7, 7, 70)
    if neg:
        grids = grids.astype(np.int8)
        # Make empty cells -1 instead of 0.
        # Temporarily convert to int8 to save memory
        grids = grids * 2 - 1
    elif split:
        sgrids = np.zeros((len(grids), 7, 7, 140), dtype=np.bool)
        sgrids[:, :, :, :70] = grids
        sgrids[:, :, :, 70:] = np.invert(grids)
        grids = sgrids
    grids = grids.astype(np.float16)
    return grids


def prep_data_cells(cells):
    if type(cells) == tuple:
        cells = [cells]
    if type(cells[0]) != tuple:
        raise Exception(
            "WHOAH WHOAH using np arrays for indexing works differently")
    oh_cells = np.zeros((len(cells), 7, 7), dtype=np.float16)
    # One-hot grid encoding
    for i, cell in enumerate(cells):
        oh_cells[i][cell] = 1
    oh_cells.shape = (-1, 7, 7, 1)
    # Should not be used when predicting, but could save mem when training
    # del cells
    return oh_cells


def prep_data(grids, cells, actions, rewards, next_grids=None,
              next_cells=None):
    assert type(actions) == np.ndarray
    assert type(rewards) == np.ndarray
    actions = actions.astype(np.int32)
    # Needs to be 32-bit, else will overflow
    rewards = rewards.astype(np.float32)
    # Cells are used as indexes and must be tuples
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
