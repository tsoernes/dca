import numpy as np
from numba import njit
from numba.types import Array, List, UniTuple, boolean, int32, intp, optional

from eventgen import CEvent

# Whether or not the part of the feature representation which
# count the number of channels in use at neighbors with
# a distance of 4 or less should include the cell itself in the count.
countself = False
rows, cols, n_channels = np.intp(7), np.intp(7), np.intp(70)
# Neighs at dist less than or equal to
_neighs1 = np.zeros((rows, cols, 7, 2), dtype=np.int32)
_neighs2 = np.zeros((rows, cols, 19, 2), dtype=np.int32)
_neighs3 = np.zeros((rows, cols, 37, 2), dtype=np.int32)
_neighs4 = np.zeros((rows, cols, 43, 2), dtype=np.int32)
_n_neighs = np.zeros((4, rows, cols), dtype=np.int32)
# Neighs at exactly dist
_neighs2o = np.zeros((rows, cols, 12, 2), dtype=np.int32)
_neighs3o = np.zeros((rows, cols, 18, 2), dtype=np.int32)
_neighs4o = np.zeros((rows, cols, 15, 2), dtype=np.int32)
_n_neighs_o = np.zeros((3, rows, cols), dtype=np.int32)


def _hex_distance(r1, c1, r2, c2):
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


def _generate_neighbors():
    for r1 in range(rows):
        for c1 in range(cols):
            _neighs1[r1, c1, 0] = (r1, c1)
            _neighs2[r1, c1, 0] = (r1, c1)
            _neighs3[r1, c1, 0] = (r1, c1)
            _neighs4[r1, c1, 0] = (r1, c1)
            _n_neighs[:, r1, c1] += 1
            for r2 in range(rows):
                for c2 in range(cols):
                    dist = _hex_distance(r1, c1, r2, c2)
                    if (r1, c1) != (r2, c2) and dist <= 4:
                        _neighs4[r1, c1, _n_neighs[3, r1, c1]] = (r2, c2)
                        _n_neighs[3, r1, c1] += 1
                        if dist == 4:
                            _neighs4o[r1, c1, _n_neighs_o[2, r1, c1]] = (r2, c2)
                            _n_neighs_o[2, r1, c1] += 1
                        if dist <= 3:
                            _neighs3[r1, c1, _n_neighs[2, r1, c1]] = (r2, c2)
                            _n_neighs[2, r1, c1] += 1
                            if dist == 3:
                                _neighs3o[r1, c1, _n_neighs_o[1, r1, c1]] = (r2, c2)
                                _n_neighs_o[1, r1, c1] += 1
                            if dist <= 2:
                                _neighs2[r1, c1, _n_neighs[1, r1, c1]] = (r2, c2)
                                _n_neighs[1, r1, c1] += 1
                                if dist == 2:
                                    _neighs2o[r1, c1, _n_neighs_o[0, r1, c1]] = (r2, c2)
                                    _n_neighs_o[0, r1, c1] += 1
                                if dist <= 1:
                                    _neighs1[r1, c1, _n_neighs[0, r1, c1]] = (r2, c2)
                                    _n_neighs[0, r1, c1] += 1
    print("Generated neighbors")


_generate_neighbors()
_neighs1.setflags(write=False)
_neighs2.setflags(write=False)
_neighs3.setflags(write=False)
_neighs4.setflags(write=False)
_n_neighs.setflags(write=False)


@njit(Array(int32, 2, 'C', readonly=True)
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_np(dist, row, col, include_self=False):
    """np array of 2-dim np arrays"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = _neighs1
    elif dist == 2:
        neighs = _neighs2
    elif dist == 3:
        neighs = _neighs3
    elif dist == 4:
        neighs = _neighs4
    else:
        raise NotImplementedError
    return neighs[row, col, start:_n_neighs[dist - 1, row, col]]


@njit(List(UniTuple(int32, 2))
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_tups(dist, row, col, include_self=False):
    """List of cells as tuples"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = _neighs1
    elif dist == 2:
        neighs = _neighs2
    elif dist == 3:
        neighs = _neighs3
    elif dist == 4:
        neighs = _neighs4
    else:
        raise NotImplementedError
    return [(neighs[row, col, i, 0], neighs[row, col, i, 1])
            for i in range(start, _n_neighs[dist - 1, row, col])]


@njit(UniTuple(Array(int32, 1, 'A', readonly=True), 2)
      (int32, int32, int32, optional(boolean)), cache=True)  # yapf: disable
def neighbors_sep(dist, row, col, include_self=False):
    """2-Tuple ([row1, row2,..], [col1, col2, ..]) of np arrays"""
    start = 0 if include_self else 1
    if dist == 1:
        neighs = _neighs1
    elif dist == 2:
        neighs = _neighs2
    elif dist == 3:
        neighs = _neighs3
    elif dist == 4:
        neighs = _neighs4
    else:
        raise NotImplementedError
    return (neighs[row, col, start:_n_neighs[dist - 1, row, col], 0],
            neighs[row, col, start:_n_neighs[dist - 1, row, col], 1])


def neighbors(dist, row, col, separate=False, include_self=False):
    if separate:
        return neighbors_sep(dist, row, col, include_self)
    else:
        return neighbors_tups(dist, row, col, include_self)


@njit(cache=True)
def _inuse_neighs(grid, r, c):
    """Channels in use at cell neighbors with distance of 2 or less"""
    neighs = neighbors_np(2, r, c, False)
    alloc_map = grid[neighs[0, 0], neighs[0, 1]]
    for i in range(1, len(neighs)):
        alloc_map = np.bitwise_or(alloc_map, grid[neighs[i, 0], neighs[i, 1]])
    return alloc_map


@njit(cache=True)
def _eligible_map(grid, r, c):
    """Channels that are not in use at cell or its neighbors with distance of 2 or less"""
    inuse = _inuse_neighs(grid, r, c)
    inuse = np.bitwise_or(inuse, grid[(r, c)])
    eligible_map = np.invert(inuse)
    return eligible_map


@njit(cache=True)
def eligible_map_all(grid):
    """For each cell"""
    alloc_map_all = np.zeros((rows, cols, n_channels), dtype=boolean)
    for r in range(rows):
        for c in range(cols):
            neighs = neighbors_np(2, r, c, False)
            alloc_map = grid[neighs[0, 0], neighs[0, 1]]
            for i in range(1, len(neighs)):
                alloc_map = np.bitwise_or(alloc_map, grid[neighs[i, 0], neighs[i, 1]])
            alloc_map_all[r, c] = alloc_map
    return np.invert(alloc_map_all)


@njit(cache=True)
def get_eligible_chs(grid, cell):
    eligible = np.nonzero(_eligible_map(grid, *cell))[0]
    return eligible


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
    """
    As in Singh(96)
    For each cell:
      - The number of eligible channels
    For each cell-channel pair:
      - The number of times the ch is used by neighbors with dist 4 or less
    """
    assert grid.ndim == 3
    frep = np.zeros((intp(rows), intp(cols), n_channels + 1), dtype=int32)
    for r in range(rows):
        for c in range(cols):
            neighs = neighbors_np(4, r, c, countself)
            n_used = np.zeros(n_channels, dtype=int32)
            for i in range(len(neighs)):
                n_used += grid[neighs[i, 0], neighs[i, 1]]
            frep[r, c, :-1] = n_used
            frep[r, c, -1] = np.sum(_eligible_map(grid, r, c))
    return frep


@njit(cache=True)
def feature_rep_big(grid):
    """
    For each cell:
      - The number of eligible channels
    For each cell-channel pair:
      - The number of times the ch is used by neighbors with dist 4 or less
      - The number of times the ch is used by neighbors with dist 3 or less
      - If the ch is eligible or not
    """
    assert grid.ndim == 3
    frep = np.zeros((intp(rows), intp(cols), n_channels * 3 + 1), dtype=int32)
    for r in range(rows):
        for c in range(cols):
            neighs3 = neighbors_np(3, r, c, countself)
            neighs4 = neighbors_np(4, r, c, countself)
            n_used3 = np.zeros(n_channels, dtype=int32)
            n_used4 = np.zeros(n_channels, dtype=int32)
            for i in range(len(neighs3)):
                n_used3 += grid[neighs3[i, 0], neighs3[i, 1]]
            for i in range(len(neighs4)):
                n_used4 += grid[neighs4[i, 0], neighs4[i, 1]]
            frep[r, c, :n_channels] = n_used3
            frep[r, c, n_channels:n_channels * 2] = n_used4
            elig = _eligible_map(grid, r, c)
            frep[r, c, n_channels * 2:n_channels * 3] = elig
            frep[r, c, -1] = np.sum(elig)
    return frep


@njit(cache=True)
def feature_reps_big2(grids):
    assert grids.ndim == 4
    freps = np.zeros((len(grids), intp(rows), intp(cols), n_channels * 5 + 1), dtype=int32)
    for i in range(len(grids)):
        freps[i] = feature_rep_big2(grids[i])
    return freps


@njit(cache=True)
def feature_rep_big2(grid):
    """
    For each cell:
      - The number of eligible channels
    For each cell-channel pair:
      - The number of times the ch is used by neighbors with dist 4
      - The number of times the ch is used by neighbors with dist 3
      - The number of times the ch is used by neighbors with dist 2
      - The number of times the ch is used by neighbors with dist 1
      - If the ch is eligible or not
    """
    assert grid.ndim == 3
    frep = np.zeros((intp(rows), intp(cols), n_channels * 5 + 1), dtype=int32)
    for r in range(rows):
        for c in range(cols):
            neighs1o = neighbors_np(1, r, c, False)
            neighs2o = _neighs2o[r, c, :_n_neighs_o[0, r, c]]
            neighs3o = _neighs3o[r, c, :_n_neighs_o[1, r, c]]
            neighs4o = _neighs4o[r, c, :_n_neighs_o[2, r, c]]
            n_used1 = np.zeros(n_channels, dtype=int32)
            n_used2 = np.zeros(n_channels, dtype=int32)
            n_used3 = np.zeros(n_channels, dtype=int32)
            n_used4 = np.zeros(n_channels, dtype=int32)
            for i in range(len(neighs1o)):
                n_used1 += grid[neighs1o[i, 0], neighs1o[i, 1]]
            for i in range(len(neighs2o)):
                n_used2 += grid[neighs2o[i, 0], neighs2o[i, 1]]
            for i in range(len(neighs3o)):
                n_used3 += grid[neighs3o[i, 0], neighs3o[i, 1]]
            for i in range(len(neighs4o)):
                n_used4 += grid[neighs4o[i, 0], neighs4o[i, 1]]
            frep[r, c, :n_channels] = n_used1
            frep[r, c, n_channels:n_channels * 2] = n_used2
            frep[r, c, n_channels * 2:n_channels * 3] = n_used3
            frep[r, c, n_channels * 3:n_channels * 4] = n_used4
            elig = _eligible_map(grid, r, c)
            frep[r, c, n_channels * 4:n_channels * 5] = elig
            frep[r, c, -1] = np.sum(elig)
    return frep


@njit(cache=True)
def afterstate_freps(grid, cell, ce_type, chs):
    """Feature representation for each of the afterstates of grid"""
    frep = feature_rep(grid)
    return incremental_freps(grid, frep, cell, ce_type, chs)


@njit(cache=True)
def afterstate_freps_big(grid, cell, ce_type, chs):
    """Feature representation for each of the afterstates of grid"""
    frep = feature_rep_big(grid)
    return incremental_freps_big(grid, frep, cell, ce_type, chs)


@njit(cache=True)
def successive_freps(grid, cell, ce_type, chs):
    """Frep for grid and its afterstates"""
    frep = feature_rep(grid)
    next_frep = incremental_freps(grid, frep, cell, ce_type, chs)
    return (frep, next_frep)


@njit(cache=True)
def successive_freps_big(grid, cell, ce_type, chs):
    """Frep for grid and its afterstates"""
    frep = feature_rep_big(grid)
    next_frep = incremental_freps_big(grid, frep, cell, ce_type, chs)
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
def incremental_freps_big(grid, frep, cell, ce_type, chs):
    """
    Given a grid, its feature representation frep,
    and a set of actions specified by cell, event type and a list of channels,
    derive feature representations for the afterstates of grid
    """
    r, c = cell
    neighs3 = neighbors_np(3, r, c, countself)
    neighs4 = neighbors_np(4, r, c, countself)
    neighs2 = neighbors_np(2, r, c, True)
    freps = np.zeros((len(chs), intp(rows), intp(cols), n_channels * 3 + 1), dtype=int32)
    freps[:] = frep
    if ce_type == CEvent.END:
        n_used_neighs_diff = -1
        n_elig_self_diff = 1
        bflip = 1
        grid[cell][chs] = 0
    else:
        n_used_neighs_diff = 1
        n_elig_self_diff = -1
        bflip = 0
    for i in range(len(chs)):
        ch = chs[i]
        for j in range(len(neighs3)):
            freps[i, neighs3[j, 0], neighs3[j, 1], ch] += n_used_neighs_diff
        for j in range(len(neighs4)):
            freps[i, neighs4[j, 0], neighs4[j, 1], n_channels + ch] += n_used_neighs_diff
        for j in range(len(neighs2)):
            r2, c2 = neighs2[j, 0], neighs2[j, 1]
            neighs = neighbors_np(2, r2, c2, False)
            not_eligible = grid[r2, c2, ch]
            for k in range(len(neighs)):
                not_eligible = not_eligible or grid[neighs[k, 0], neighs[k, 1], ch]
            if not not_eligible:
                # For END: ch is in use at 'cell', but will become eligible
                # For NEW: ch is eligible at given neighs2, but will be taken in use
                freps[i, neighs2[j, 0], neighs2[j, 1], n_channels * 2 + ch] = bflip
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
