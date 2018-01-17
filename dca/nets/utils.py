import random

import numpy as np
import tensorflow as tf


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # Subtract maximum value for numerical stability; result will be the same
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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


def prep_data_grids(grids, empty_neg=True):
    """
    empty_neg: Represent empty channels as -1 instead of 0
    """
    assert type(grids) == np.ndarray
    if grids.ndim == 3:
        grids = np.expand_dims(grids, axis=0)
    grids.shape = (-1, 7, 7, 70)
    if empty_neg:
        grids = grids.astype(np.int8)
        # Make empty cells -1 instead of 0.
        # Temporarily convert to int8 to save memory
        grids = grids * 2 - 1
    grids = grids.astype(np.float16)
    return grids


def prep_data_cells(cells):
    if type(cells) == tuple:
        cells = [cells]
    oh_cells = np.zeros((len(cells), 7, 7), dtype=np.float16)
    # One-hot grid encoding
    for i, cell in enumerate(cells):
        oh_cells[i][cell] = 1
    oh_cells.shape = (-1, 7, 7, 1)
    # Should not be used when predicting, but could save mem when training
    # del cells

    return oh_cells


def prep_data(grids, cells, actions, rewards, next_grids, next_cells):
    assert type(actions) == np.ndarray
    assert type(rewards) == np.ndarray
    actions = actions.astype(np.int32)
    rewards = rewards.astype(
        np.float32)  # Needs to be 32-bit, else will overflow

    grids = prep_data_grids(grids)
    next_grids = prep_data_grids(next_grids)
    oh_cells = prep_data_cells(cells)
    next_oh_cells = prep_data_cells(next_cells)
    return grids, oh_cells, actions, rewards, next_grids, next_oh_cells
