import numpy as np
from numba import njit
from numba.types import Array, List, UniTuple, boolean, int32, intp, optional

from eventgen import CEvent

# Whether or not the part of the feature representation which
# count the number of channels in use at neighbors with
# a distance of 4 or less should include the cell itself in the count.
countself = False
rows, cols, n_channels = 7, 7, 70
neighs1 = np.zeros((rows, cols, 7, 2), dtype=np.int32)
neighs2 = np.zeros((rows, cols, 19, 2), dtype=np.int32)
neighs4 = np.zeros((rows, cols, 43, 2), dtype=np.int32)
n_neighs1 = np.zeros((rows, cols), dtype=np.int32)
n_neighs2 = np.zeros((rows, cols), dtype=np.int32)
n_neighs4 = np.zeros((rows, cols), dtype=np.int32)


def _hex_distance(r1, c1, r2, c2):
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


def _generate_neighbors():
    for r1 in range(rows):
        for c1 in range(cols):
            neighs1[r1, c1, 0] = (r1, c1)
            neighs2[r1, c1, 0] = (r1, c1)
            neighs4[r1, c1, 0] = (r1, c1)
            n_neighs1[r1, c1] += 1
            n_neighs2[r1, c1] += 1
            n_neighs4[r1, c1] += 1
            for r2 in range(rows):
                for c2 in range(cols):
                    dist = _hex_distance(r1, c1, r2, c2)
                    if (r1, c1) != (r2, c2) and dist <= 4:
                        neighs4[r1, c1, n_neighs4[r1, c1]] = (r2, c2)
                        n_neighs4[r1, c1] += 1
                        if dist <= 2:
                            neighs2[r1, c1, n_neighs2[r1, c1]] = (r2, c2)
                            n_neighs2[r1, c1] += 1
                            if dist <= 1:
                                neighs1[r1, c1, n_neighs1[r1, c1]] = (r2, c2)
                                n_neighs1[r1, c1] += 1
    print("Generated neighbors")


_generate_neighbors()
neighs1.setflags(write=False)
neighs2.setflags(write=False)
neighs4.setflags(write=False)
n_neighs1.setflags(write=False)
n_neighs2.setflags(write=False)
n_neighs4.setflags(write=False)


@njit(Array(int32, 2, 'C', readonly=True)
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_np(dist, row, col, include_self=False):
    """np array of 2-dim np arrays"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = neighs1
        n_neighs = n_neighs1
    elif dist == 2:
        neighs = neighs2
        n_neighs = n_neighs2
    elif dist == 4:
        neighs = neighs4
        n_neighs = n_neighs4
    else:
        raise NotImplementedError
    return neighs[row, col, start:n_neighs[row, col]]


@njit(List(UniTuple(int32, 2))
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_tups(dist, row, col, include_self=False):
    """list of tuples"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = neighs1
        n_neighs = n_neighs1
    elif dist == 2:
        neighs = neighs2
        n_neighs = n_neighs2
    elif dist == 4:
        neighs = neighs4
        n_neighs = n_neighs4
    else:
        raise NotImplementedError
    return [(neighs[row, col, i, 0], neighs[row, col, i, 1])
            for i in range(start, n_neighs[row, col])]


@njit(UniTuple(Array(int32, 1, 'A', readonly=True), 2)
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_sep(dist, row, col, include_self=False):
    """2-Tuple of np arrays"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = neighs1
        n_neighs = n_neighs1
    elif dist == 2:
        neighs = neighs2
        n_neighs = n_neighs2
    elif dist == 4:
        neighs = neighs4
        n_neighs = n_neighs4
    else:
        raise NotImplementedError
    return (neighs[row, col, start:n_neighs[row, col], 0],
            neighs[row, col, start:n_neighs[row, col], 1])


def neighbors(dist, row, col, separate=False, include_self=False):
    if separate:
        return neighbors_sep(dist, row, col, include_self)
    else:
        return neighbors_tups(dist, row, col, include_self)


@njit(cache=True)
def _inuse_neighs(grid, r, c):
    # Channels in use at neighbors
    neighs = neighbors_np(2, r, c, False)
    alloc_map = grid[neighs[0, 0], neighs[0, 1]]
    for i in range(1, len(neighs)):
        alloc_map = np.bitwise_or(alloc_map, grid[neighs[i, 0], neighs[i, 1]])
    return alloc_map


@njit(cache=True)
def get_eligible_chs(grid, cell):
    alloc_map = _inuse_neighs(grid, *cell)
    alloc_map = np.bitwise_or(alloc_map, grid[cell])
    eligible = np.nonzero(np.invert(alloc_map))[0]
    return eligible


@njit(cache=True)
def get_n_eligible_chs(grid, cell):
    alloc_map = _inuse_neighs(grid, *cell)
    alloc_map = np.bitwise_or(alloc_map, grid[cell])
    n_eligible = np.sum(np.invert(alloc_map))
    return n_eligible


@njit(cache=True)
def validate_reuse_constr(grid):
    for r in range(rows):
        for c in range(cols):
            # Channels in use at neighbors
            inuse = _inuse_neighs(grid, r, c)
            # Channels in use at a neigh and cell
            inuse_both = np.bitwise_and(grid[r, c], inuse)
            viols = np.where(inuse_both == 1)[0]
            if len(viols) > 0:
                print("Channel Reuse constraint violated")
                return False
    return True


@njit(cache=True)
def afterstates(grid, cell, ce_type, chs):
    if ce_type == CEvent.END:
        targ_val = False
    else:
        targ_val = True
    grids = np.zeros((len(chs), rows, cols, n_channels), dtype=boolean)
    for i, ch in enumerate(chs):
        grids[i] = grid
        grids[i][cell][ch] = targ_val
    return grids


@njit(cache=True)
def feature_rep(grid):
    assert grid.ndim == 3
    frep = np.zeros((intp(rows), intp(cols), n_channels + 1), dtype=int32)
    for r in range(rows):
        for c in range(cols):
            neighs = neighbors_np(4, r, c, countself)
            n_used = np.zeros(n_channels, dtype=int32)
            for i in range(len(neighs)):
                n_used += grid[neighs[i, 0], neighs[i, 1]]
            frep[r, c, :-1] = n_used
            frep[r, c, -1] = get_n_eligible_chs(grid, (r, c))
    return frep


@njit(cache=True)
def feature_reps(grids):
    assert grids.ndim == 4
    n = len(grids)
    freps = np.zeros((n, rows, cols, n_channels + 1), dtype=np.int16)
    for r in range(rows):
        for c in range(cols):
            neighs = neighbors_np(4, r, c, countself)
            for i in range(n):
                n_used = np.zeros(n_channels, dtype=int32)
                for j in range(len(neighs)):
                    n_used += grids[i, neighs[j, 0], neighs[j, 1]]
                freps[i, r, c, :-1] = n_used
                freps[i, r, c, -1] = get_n_eligible_chs(grids[i], (r, c))
    return freps


@njit(cache=True)
def afterstate_freps(grid, cell, ce_type, chs):
    """Feature representation for each of the afterstates of grid"""
    frep = feature_rep(grid)
    return incremental_freps(grid, frep, cell, ce_type, chs)


@njit(cache=True)
def successive_freps(grid, cell, ce_type, chs):
    """Frep for grid and its afterstates"""
    frep = feature_rep(grid)
    next_frep = incremental_freps(grid, frep, cell, ce_type, chs)
    return (frep, next_frep)


@njit(cache=True)
def incremental_freps(grid, frep, cell, ce_type, chs):
    """
    Given a grid, its feature representation frep,
    and a set of actions specified by cell, event type and a list of channels,
    derive feature representations for the afterstates of grid
    """
    r, c = cell
    neighs4 = neighbors_np(4, r, c, countself)
    neighs2 = neighbors_np(2, r, c, True)
    freps = np.zeros((len(chs), intp(rows), intp(cols), n_channels + 1), dtype=int32)
    freps[:] = frep
    if ce_type == CEvent.END:
        n_used_neighs_diff = -1
        n_elig_self_diff = 1
        grid[cell][chs] = 0
    else:
        n_used_neighs_diff = 1
        n_elig_self_diff = -1
    for i in range(len(chs)):
        ch = chs[i]
        for j in range(len(neighs4)):
            freps[i, neighs4[j, 0], neighs4[j, 1], ch] += n_used_neighs_diff
        for j in range(len(neighs2)):
            r2, c2 = neighs2[j, 0], neighs2[j, 1]
            neighs = neighbors_np(2, r2, c2, False)
            not_eligible = grid[r2, c2, ch]
            for k in range(len(neighs)):
                not_eligible = not_eligible or grid[neighs[k, 0], neighs[k, 1], ch]
            if not not_eligible:
                # For END: ch is in use at 'cell', but will become eligible
                # For NEW: ch is eligible at given neighs2, but will be taken in use
                freps[i, neighs2[j, 0], neighs2[j, 1], -1] += n_elig_self_diff
    if ce_type == CEvent.END:
        grid[cell][chs] = 1
    return freps


@njit(cache=True)
def scale_freps(freps):
    # Scale freps in range [0, 1]
    # TODO Try Scale freps in range [-1, 1]
    freps = freps.astype(np.float16)
    if freps.ndim == 4:
        freps[:, :, :, :-1] *= 1 / 43.0
        freps[:, :, :, -1] *= 1 / float(n_channels)
    elif freps.ndim == 3:
        # Max possible neighs within dist 4 including self
        freps[:, :, :-1] *= 1 / 43.0
        freps[:, :, -1] *= 1 / float(n_channels)
    else:
        raise NotImplementedError
    return freps
