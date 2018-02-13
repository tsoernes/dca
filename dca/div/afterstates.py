import functools
from enum import Enum
from timeit import timeit

import numpy as np


class Action(Enum):
    END = 0  # Actions of this type switches a bit in 'grid' from 1 to 0
    # Actions of this type switches a bit in 'grid' from 0 to 1,
    # provided that no nearby cells
    # (nearby as defined by the function 'neighbors(dist=2, *cell)')
    # already have this bit 'on' at the given depth
    NEW = 1


n_rows = 7
n_cols = 7
n_depths = 70


def hex_distance(cell_a, cell_b):
    r1, c1 = cell_a
    r2, c2 = cell_b
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


@functools.lru_cache(maxsize=None)
def neighbors(dist, row, col, separate, include_self, rows=7, cols=7):
    """
    Returns a list with indices of neighbors with a distance of 'dist' or less
    from the cell at (row, col)

    If 'include_self' is True, include the given (row, col)

    If 'separate' is True,
    return ([r1, r2, ...], [c1, c2, ...]) else
    return [(r1, c1), (r2, c2), ...]
    """
    if separate:
        rs = []
        cs = []
    else:
        idxs = []
    for r2 in range(rows):
        for c2 in range(cols):
            if (include_self or (row, col) != (r2, c2)) \
               and hex_distance((row, col), (r2, c2)) <= dist:
                if separate:
                    rs.append(r2)
                    cs.append(c2)
                else:
                    idxs.append((r2, c2))
    if separate:
        return (rs, cs)
    return idxs


def get_eligible_depths_bitmap(grid, cell):
    """Find eligible chs by bitwise ORing the allocation maps of neighbors"""
    neighs = neighbors(2, *cell, separate=True, include_self=True)
    alloc_map = np.bitwise_or.reduce(grid[neighs])
    return alloc_map


def get_eligible_depths(grid, cell):
    """
    Find the depths that are free (i.e. 0) in 'cell' and all of
    its neighbors within a distance of 2 or less.
    These are the eligible depths, i.e. those that can be assigned
    (taken in use) without violating the reuse constraint.
    """
    alloc_map = get_eligible_depths_bitmap(grid, cell)
    eligible = np.nonzero(np.logical_not(alloc_map))[0]
    return eligible


def get_n_eligible_depths(grid, cell):
    """Return the number of eligible depths"""
    alloc_map = get_eligible_depths_bitmap(grid, cell)
    n_eligible = np.count_nonzero(np.invert(alloc_map))
    return n_eligible


def afterstates(grid, cell, atype, depths):
    """Make an afterstate (resulting grid) for executing an action of type
    'atype' at position 'cell' on 'grid' for each depth listed 'depths'"""
    if atype == Action.END:
        targ_val = 0
    else:
        targ_val = 1
    grids = np.repeat(np.expand_dims(np.copy(grid), axis=0), len(depths), axis=0)
    for i, depth in enumerate(depths):
        # assert grids[i][cell][depth] != targ_val
        grids[i][cell][depth] = targ_val
    # assert grids.shape == (len(depths), rows, cols, depth)
    return grids


def feature_reps(grids):
    """
    Takes a grid or an array of grids and return the feature representations.

    For each cell, the number of eligible depths in that cell.
    For each cell-depth pair, the number of times that depth is
    occupied by neighbors (or self) with a distance of 4 or less.
    """
    # assert type(grids) == np.ndarray
    if grids.ndim == 3:
        grids = np.expand_dims(grids, axis=0)
    fgrids = np.zeros((len(grids), n_rows, n_cols, n_depths + 1), dtype=np.int16)
    for r in range(n_rows):
        for c in range(n_cols):
            neighs = neighbors(4, r, c, separate=True, include_self=True)
            n_used = np.count_nonzero(grids[:, neighs[0], neighs[1]], axis=1)
            fgrids[:, r, c, :-1] = n_used
            for i in range(len(grids)):
                n_eligible_depths = get_n_eligible_depths(grids[i], (r, c))
                fgrids[i, r, c, -1] = n_eligible_depths
    return fgrids


def afterstate_freps_naive(grid, cell, atype, depths):
    """
    Get the feature representation for each afterstate resulting from
    executing action of type 'atype' at 'depths'
    """
    astates = afterstates(grid, cell, atype, depths)
    freps = feature_reps(astates)
    return freps


def afterstate_freps_incremental(grid, cell, atype, depths):
    """
    Get the feature representation (as described in 'feature_reps')
    of the current grid, and from it derive the f.rep for each possible afterstate.
    """
    fgrid = feature_reps(grid)[0]
    r, c = cell
    neighs4 = neighbors(dist=4, row=r, col=c, separate=True, include_self=True)
    neighs2 = neighbors(dist=2, row=r, col=c, separate=False, include_self=True)
    fgrids = np.repeat(np.expand_dims(fgrid, axis=0), len(depths), axis=0)
    if atype == Action.END:
        # One less depth will be in use by the cell
        n_used_neighs_diff = -1
        # One more depth MIGHT become eligible.
        # Temporarily modify grid and check if that's the case
        n_elig_self_diff = 1
        grid[cell][depths] = 0
    else:
        # One more depth will be occupied
        n_used_neighs_diff = 1
        # One less depth will be eligible
        n_elig_self_diff = -1
    eligible_depths = [get_eligible_depths(grid, neigh2) for neigh2 in neighs2]
    for i, depth in enumerate(depths):
        fgrids[i, neighs4[0], neighs4[1], depth] += n_used_neighs_diff
        for j, neigh2 in enumerate(neighs2):
            if depth in eligible_depths[j]:
                fgrids[i, neigh2[0], neigh2[1], -1] += n_elig_self_diff
    if atype == Action.END:
        grid[cell][depths] = 1
    return fgrids


grid = np.zeros((7, 7, 70), dtype=np.bool)
where = ([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
], [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
    2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
], [
    18, 25, 38, 39, 54, 55, 61, 63, 65, 68, 5, 8, 16, 19, 20, 31, 35, 48, 51, 52, 56, 62,
    4, 7, 21, 23, 30, 32, 36, 37, 53, 58, 69, 1, 6, 14, 15, 24, 29, 34, 46, 47, 50, 57,
    60, 66, 67, 68, 3, 11, 22, 25, 26, 27, 35, 38, 42, 44, 49, 54, 59, 62, 64, 0, 10, 40,
    41, 55, 63, 16, 24, 31, 48, 50, 53, 56, 0, 1, 10, 14, 29, 33, 34, 46, 67, 9, 11, 12,
    22, 26, 42, 43, 44, 49, 17, 18, 39, 40, 41, 55, 65, 5, 19, 28, 48, 52, 56, 7, 13, 20,
    21, 32, 36, 37, 58, 61, 1, 6, 8, 14, 23, 29, 45, 47, 60, 67, 2, 3, 12, 26, 30, 33, 42,
    44, 59, 62, 69, 6, 15, 24, 27, 28, 50, 57, 66, 3, 13, 25, 38, 54, 59, 61, 0, 8, 10,
    16, 45, 51, 4, 9, 12, 30, 31, 43, 53, 69, 18, 34, 46, 68, 11, 15, 19, 22, 27, 35, 38,
    49, 64, 4, 7, 10, 17, 40, 51, 55, 63, 19, 48, 52, 53, 56, 58, 62, 1, 7, 14, 21, 23,
    29, 32, 34, 35, 46, 67, 6, 22, 27, 42, 44, 47, 49, 57, 63, 64, 66, 3, 17, 26, 39, 40,
    54, 0, 5, 24, 25, 41, 50, 56, 9, 28, 31, 48, 53, 58, 61, 65, 1, 6, 8, 14, 23, 32, 37,
    43, 46, 60, 66, 68, 9, 12, 37, 39, 40, 43, 2, 15, 18, 50, 68, 19, 28, 38, 48, 59, 61,
    1, 8, 10, 16, 21, 23, 32, 51, 52, 60, 12, 29, 30, 33, 44, 57, 69, 2, 3, 20, 26, 34,
    42, 47, 62, 0, 22, 25, 38, 45, 49, 64, 3, 8, 10, 13, 16, 26, 0, 25, 30, 31, 33, 41,
    45, 53, 56, 58, 9, 11, 34, 35, 43, 46, 65, 6, 13, 14, 15, 22, 37, 49, 55, 64, 66, 67,
    68, 19, 27, 39, 40, 54, 10, 11, 16, 17, 21, 36, 51, 52, 63, 4, 5, 24, 28, 31, 33, 41,
    48, 53, 57, 58, 1, 4, 5, 7, 14, 21, 22, 23, 49, 52, 20, 27, 39, 40, 57, 62, 2, 3, 17,
    18, 24, 36, 42, 50, 63, 4, 25, 31, 38, 41, 58, 59, 1, 8, 9, 23, 35, 43, 46, 60, 61,
    65, 12, 13, 14, 32, 37, 44, 50, 66, 68, 2, 3, 7, 18, 19, 20, 26, 27, 30, 34, 39, 42,
    47, 54, 62, 69
])
grid[where] = 1
cell_end = (0, 1)
cell_new = (3, 1)
depths_for_end_action = np.nonzero(grid[cell_end])[0]
depths_for_new_action = get_eligible_depths(grid, cell_new)


def f1():
    e1 = afterstate_freps_naive(grid, cell_end, Action.END, depths_for_end_action)
    n1 = afterstate_freps_naive(grid, cell_new, Action.NEW, depths_for_new_action)
    return e1, n1


def f2():
    e2 = afterstate_freps_incremental(grid, cell_end, Action.END, depths_for_end_action)
    n2 = afterstate_freps_incremental(grid, cell_new, Action.NEW, depths_for_new_action)
    return e2, n2


f2()  # Prime the neighbors function
print(timeit(f1, number=1000))
print(timeit(f2, number=1000))
# OK:
# r1, r2 = f1(), f2()
# assert (r1[0] == r2[0]).all()
# assert (r1[1] == r2[1]).all()
