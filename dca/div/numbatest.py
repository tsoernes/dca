import numba as nb
import numpy as np
from numba import jit


@jit(nb.int32(
    nb.types.UniTuple(nb.int32, 2), nb.types.UniTuple(nb.int32, 2)),
    nopython=True)
def hex_distance(cell_a, cell_b):
    r1, c1 = cell_a
    r2, c2 = cell_b
    return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2


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
