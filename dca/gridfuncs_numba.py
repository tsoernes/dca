import numpy as np
from numba import njit
from numba.types import Array, List, UniTuple, boolean, int32, intp, optional

from eventgen import CEvent

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
    fgrid = np.zeros((intp(rows), intp(cols), n_channels + 1), dtype=int32)
    for r in range(rows):
        for c in range(cols):
            neighs = neighbors_np(4, r, c, True)
            n_used = np.zeros(n_channels, dtype=int32)
            for i in range(len(neighs)):
                n_used += grid[neighs[i, 0], neighs[i, 1]]
            fgrid[r, c, :-1] = n_used
            fgrid[r, c, -1] = get_n_eligible_chs(grid, (r, c))
    return fgrid


@njit(cache=True)
def afterstate_freps(grid, cell, ce_type, chs):
    fgrid = feature_rep(grid)
    r, c = cell
    neighs4 = neighbors_np(4, r, c, True)
    # Is this right, in combination with 'neighs' include_self=False?
    neighs2 = neighbors_np(2, r, c, True)
    fgrids = np.zeros((len(chs), intp(rows), intp(cols), n_channels + 1), dtype=int32)
    fgrids[:] = fgrid
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
            fgrids[i, neighs4[j, 0], neighs4[j, 1], ch] += n_used_neighs_diff
        for j in range(len(neighs2)):
            r2, c2 = neighs2[0, j], neighs2[1, j]
            neighs = neighbors_np(2, r2, c2, False)
            inuse = grid[r2, c2, ch]
            for k in range(len(neighs)):
                inuse = inuse or grid[neighs[k, 0], neighs[k, 1], ch]
            if not inuse:
                # For END: ch was in use at (r, c), but is now eligible
                # For NEW: ch was eligible at given neighs2, but is now in use
                fgrids[i, neighs2[j, 0], neighs2[j, 1], -1] += n_elig_self_diff
    if ce_type == CEvent.END:
        grid[cell][chs] = 1
    return fgrids
