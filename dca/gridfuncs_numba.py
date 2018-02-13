import functools

import numba as nb
import numpy as np
from numba import jit

from eventgen import CEvent


@jit(nb.boolean[:](
    nb.boolean[:], nb.types.UniTuple(nb.int32, 2)),
    nopython=True)
def get_eligible_chs_bitmap(grid, cell):
    """Find eligible chs by bitwise ORing the allocation maps of neighbors"""
    r, c = cell
    neighs = neighbors_sep(2, r, c, include_self=True)
    alloc_map = np.bitwise_or.reduce(grid[neighs])
    return alloc_map


@jit(nb.int32[:](
    nb.boolean[:], nb.types.UniTuple(nb.int32, 2)),
    nopython=True)
def get_eligible_chs(grid, cell):
    """
    Find the channels that are free in 'cell' and all of
    its neighbors with a distance of 2 or less.
    These are the eligible channels, i.e. those that can be assigned
    without violating the reuse constraint.
    """
    alloc_map = get_eligible_chs_bitmap(grid, cell)
    eligible = np.nonzero(np.logical_not(alloc_map))[0]
    return eligible


@jit(nb.int32(
    nb.boolean[:], nb.types.UniTuple(nb.int32, 2)),
    nopython=True)
def get_n_eligible_chs(grid, cell):
    """Return the number of eligible channels"""
    alloc_map = get_eligible_chs_bitmap(grid, cell)
    n_eligible = np.count_nonzero(np.invert(alloc_map))
    return n_eligible


@jit(nb.boolean[:](
    nb.boolean[:], nb.types.UniTuple(nb.int32, 2), nb.int32,
    list(nb.int32), nb.int32, nb.int32, nb.int32),
    nopython=True)
def afterstates(grid, cell, ce_type, chs, rows=7, cols=7, n_channels=70):
    """Make an afterstate (resulting grid) for each possible,
    # eligible action in 'chs'"""
    if ce_type == CEvent.END:
        targ_val = 0
    else:
        targ_val = 1
    grids = np.repeat(np.expand_dims(np.copy(grid), axis=0), len(chs), axis=0)
    for i, ch in enumerate(chs):
        # assert grids[i][cell][ch] != targ_val
        grids[i][cell][ch] = targ_val
    assert grids.shape == (len(chs), rows, cols, n_channels)
    return grids


@jit(nb.int32(
    nb.types.UniTuple(nb.int32, 2), nb.types.UniTuple(nb.int32, 2)),
    nopython=True)
def hex_distance(cell_a, cell_b):
    r1, c1 = cell_a
    r2, c2 = cell_b
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


# @functools.lru_cache(maxsize=None)
@jit(nb.types.UniTuple(list(nb.int32), 2)(
    nb.int32, nb.int32, nb.int32, nb.boolean, nb.int32, nb.int32),
    nopython=True)
def neighbors_sep(dist, row, col, include_self=False, rows=7, cols=7):
    rs = []
    cs = []
    for r2 in range(rows):
        for c2 in range(cols):
            if (include_self or (row, col) != (r2, c2)) \
               and hex_distance((row, col), (r2, c2)) <= dist:
                rs.append(r2)
                cs.append(c2)
    return (rs, cs)


# @functools.lru_cache(maxsize=None)
@jit(list(nb.int32)(nb.int32, nb.int32, nb.int32, nb.boolean, nb.int32, nb.int32), nopython=True)
def neighbors(dist, row, col, include_self=False, rows=7, cols=7):
    idxs = []
    for r2 in range(rows):
        for c2 in range(cols):
            if (include_self or (row, col) != (r2, c2)) \
               and hex_distance((row, col), (r2, c2)) <= dist:
                idxs.append((r2, c2))
    return idxs
