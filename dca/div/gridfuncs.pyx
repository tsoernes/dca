cimport cython
cimport numpy as np
# from libcpp cimport bool as bool_t
import numpy as np
import functools

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] get_eligible_chs_bitmap(
       np.ndarray[np.uint8_t, ndim=3, cast=True] grid, tuple cell):
    """Find eligible chs by bitwise ORing the allocation maps of neighbors"""
    cdef int r, c
    r, c = cell
    cdef tuple neighs = neighbors(2, r, c, separate=True, include_self=True)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] alloc_map = np.bitwise_or.reduce(grid[neighs])
    return alloc_map


def hex_distance(tuple cell_a, tuple cell_b):
    cdef int r1, c1, r2, c2
    r1, c1 = cell_a
    r2, c2 = cell_b
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


@functools.lru_cache(maxsize=None)
def neighbors(dist, row, col, separate=False, include_self=False, rows=7, cols=7):
    """
    Returns a list with indices of neighbors with a distance of 'dist' or less
    from the cell at (row, col)

    If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...]),
    else return [(r1, c1), (r2, c2), ...]
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

# def get_eligible_chs(grid, cell):
#     """
#     Find the channels that are free in 'cell' and all of
#     its neighbors with a distance of 2 or less.
#     These are the eligible channels, i.e. those that can be assigned
#     without violating the reuse constraint.
#     """
#     alloc_map = _get_eligible_chs_bitmap(grid, cell)
#     eligible = np.nonzero(np.logical_not(alloc_map))[0]
#     return eligible
# 
# 
# def get_n_eligible_chs(grid, cell):
#     """Return the number of eligible channels"""
#     alloc_map = _get_eligible_chs_bitmap(grid, cell)
#     n_eligible = np.count_nonzero(np.invert(alloc_map))
#     return n_eligible
# 
# 
# def afterstates(grid, cell, ce_type, chs, rows=7, cols=7, n_channels=70):
#     """Make an afterstate (resulting grid) for each possible,
#     # eligible action in 'chs'"""
#     if ce_type == CEvent.END:
#         targ_val = 0
#     else:
#         targ_val = 1
#     grids = np.repeat(np.expand_dims(np.copy(grid), axis=0), len(chs), axis=0)
#     for i, ch in enumerate(chs):
#         # assert grids[i][cell][ch] != targ_val
#         grids[i][cell][ch] = targ_val
#     assert grids.shape == (len(chs), rows, cols, n_channels)
#     return grids
# 
#     

