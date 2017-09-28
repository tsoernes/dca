import numpy as np
import timeit
from bitstring import BitArray

rows, cols = 7, 7
depth = 70


def neighbors2(row, col, separate=False):
    """
    If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...])
    if not, return [(r1, c1), (r2, c2), ...]

    Returns a list of indices of neighbors (in an hexogonal grid)
    within a radius of 2, not including self.
    """
    if separate:
        rs = []
        cs = []
    else:
        idxs = []

    r_low = max(0, row-2)
    r_hi = min(rows-1, row+2)
    c_low = max(0, col-2)
    c_hi = min(cols-1, col+2)
    if col % 2 == 0:
        cross1 = row-2
        cross2 = row+2
    else:
        cross1 = row+2
        cross2 = row-2
    for r in range(r_low, r_hi+1):
        for c in range(c_low, c_hi+1):
            if not ((r, c) == (row, col) or
                    (r, c) == (cross1, col-2) or
                    (r, c) == (cross1, col-1) or
                    (r, c) == (cross1, col+1) or
                    (r, c) == (cross1, col+2) or
                    (r, c) == (cross2, col-2) or
                    (r, c) == (cross2, col+2)):
                if separate:
                    rs.append(r)
                    cs.append(c)
                else:
                    idxs.append((r, c))
    if separate:
        return (rs, cs)
    else:
        return idxs


def get_free(cell):
    """
    Return the indices of a a cell that are 0 and
    where all its neighbors are 0 for the same depth
    """
    candidates = np.where(state[cell] == 0)[0]
    neighs = neighbors2(*cell, False)
    free = []
    # Exclude elements that have non-zero value in neighboring cells
    for candidate in candidates:
        non_zero = False
        for neigh in neighs:
            if state[neigh][candidate]:
                non_zero = True
                break
        if not non_zero:
            free.append(candidate)
    return free


def get_free2(cell):
    """
    Return the indices of a a cell that are 0 and
    where all its neighbors are 0 for the same depth
    """
    neighs = neighbors2(*cell, False)
    free = []
    # Exclude elements that have non-zero value in neighboring cells
    f = np.bitwise_or(state[cell], state[neighs[0]])
    for n in neighs[1:]:
        f = np.bitwise_or(f, state[n])
    free = np.where(f == 0)[0]
    return free


def get_free3(cell):
    """
    Return the indices of a a cell that are 0 and
    where all its neighbors are 0 for the same depth
    """
    neighs = neighbors2(*cell, False)
    free = []
    # Exclude elements that have non-zero value in neighboring cells
    f = binstate[cell[0]][cell[1]] or binstate[neighs[0][0]][neighs[0][1]]
    for n in neighs[1:]:
        f |= binstate[n[0]][n[1]]
    free = where(f, False)
    return free


def where(bits, val):
    idxs = []
    for i, bit in enumerate(bits):
        if bit == val:
            idxs.append(i)
    return idxs


state = np.random.choice([0, 1], size=(rows, cols, depth)).astype(bool)
cell = (4, 4)
state[cell][3] = 0
for n in neighbors2(*cell):
    state[n][3] = 0
binstate = []
for r in range(rows):
    row = []
    for c in range(cols):
        row.append(BitArray(state[r][c]))
    binstate.append(row)


print(get_free(cell))
print(get_free2(cell))
print(get_free3(cell))
print(timeit.timeit("get_free((4, 4))", number=10000,
      setup="from __main__ import get_free"))
print(timeit.timeit("get_free2((4, 4))", number=10000,
      setup="from __main__ import get_free2"))
print(timeit.timeit("get_free3((4, 4))", number=10000,
      setup="from __main__ import get_free3"))
