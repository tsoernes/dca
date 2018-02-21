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
    act_fns = {"relu": tf.nn.relu, "elu": tf.nn.elu, "leaky_relu": tf.nn.leaky_relu}
    return act_fns[name]


def build_default_trainer(pp, loss, var_list=None):
    """
    Build a trainer to minimize loss through adjusting vars in var_list.
    Optionally decay learning rate and clip gradients. Return training op
    and learning rate var.

    If var_list is not specified, defaults to GraphKeys.TRAINABLE_VARIABLES,
    i.e. all trainable variables
    """
    if pp['net_lr_decay'] < 1:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(pp['net_lr'], global_step, 10000,
                                                   pp['net_lr_decay'])
    else:
        global_step = None
        learning_rate = tf.constant(pp['net_lr'])
    trainer = get_optimizer_by_name(pp['optimizer'], learning_rate)
    if pp['max_grad_norm'] is not None:
        gradients, trainable_vars = zip(
            *trainer.compute_gradients(loss, var_list=var_list))
        clipped_grads, grad_norms = tf.clip_by_global_norm(gradients, pp['max_grad_norm'])
        do_train = trainer.apply_gradients(
            zip(clipped_grads, trainable_vars), global_step=global_step)
    else:
        do_train = trainer.minimize(loss, var_list=var_list, global_step=global_step)
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
    assert grids.shape[1:-1] == (7, 7)
    if split:
        depth = grids.shape[-1]
        sgrids = np.zeros((len(grids), 7, 7, depth + 70), dtype=np.bool)
        sgrids[:, :, :, :70] = grids[:70]
        sgrids[:, :, :, 70:140] = np.invert(grids[:70])
        if depth > 70:
            # Copy freps as is
            sgrids[:, :, :, -71:] = grids[70:]
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
