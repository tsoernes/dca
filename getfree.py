import numpy as np
import timeit

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


def get_neighs():
    # Neighborhood pre-computation for get_free5. This is a one-time thing -
    # the
    # neighborhoods will never change as long as the rows,cols stay the same.
    # NOTE however that these arrays are TRANSPOSED, because that's what free5
    # does.

    neighborhood = []
    for r in range(rows):
        neighborhood.append([])
        for c in range(cols):
            hood = np.zeros((rows, cols), dtype=np.bool_)
            hood[r, c] = 1
            for nb in neighbors2(r, c):
                hood[nb] = 1
            hood = hood.T  # Transpose!
            # Now hood is all 1's for neighbors. Do this same computation
            # in the same fashion that the state-updater does.
            bytestr = np.packbits(hood).tobytes() + b'\0'
            uint = np.frombuffer(bytestr, dtype=np.uint64)
            neighborhood[r].append(uint[0])

    # Neighborhood[r,c] is a bitmask with all neighbors + (r,c) set to 1
    # NOTE: +(r,c) - the neighborhood explicitly includes the center point,
    # which is not true for the neighbors function. I am using this to test for
    # the center ALSO being zero at the same time - just one pass through the
    # state array.
    neighborhood = np.array(neighborhood, dtype=np.uint64)
    return neighborhood


def get_free5(cell, state_asbits, etats, neighboorhood, *, verbose=False):
    """
    Return the indices of a a cell that are 0 and
    where all its neighbors are 0 for the same depth
    """
    result = list(np.nonzero((state_asbits & neighborhood[cell]) == 0)[0])

    # This if statement is used in my self-test code. Delete all this code
    # for production use.
    if verbose:
        orig = get_free(cell)
        missing = [n for n in orig if n not in result]
        extra = [n for n in result if n not in orig]
        print("Missing from result, found in orig:")
        for n in missing:
            print("orig: (transposed)", n)
            print(etats[n])
            print("bits: {:64b}".format(state_asbits[n]))
            print("mask: {:64b}".format(neighborhood[cell]))

        print("Found in result, not found in orig:")
        for n in extra:
            print("orig: (transposed)", n)
            print(etats[n])
            print("bits: {:64b}".format(state_asbits[n]))
            print("mask: {:64b}".format(neighborhood[cell]))

    return result


def encode(state):
    # Setup for get_free5(). This code has to be run whenever the state[]
    # array changes or is computed. It's an alternate encoding of the state
    # array
    # so if you don't need state[] for anything except this check, you can get
    # rid of it. Otherwise, you'll have to call this before you start calling
    # get_free5()

    # Transpose, to get all the values at the same depth in a contiguous area.
    etats = state.T
    bitplane = []
    for d in range(state.shape[-1]):
        # bytestr = np.packbits(etats[d]).tobytes()
        # if len(bytestr) % 8 != 0:
        #     bytestr += b'\00' * (8 - len(bytestr) % 8)

        # Hard-coding for 7x7 - append one extra byte
        bytestr = np.packbits(etats[d]).tobytes() + b'\0'
        uint = np.frombuffer(bytestr, dtype=np.uint64)
        bitplane.append(uint[0])

    # state_asbits[n] is an array of 64-bit numbers, holding bits for each of
    # the
    # 49 cells at depth=n. If the value in state[r,c,n] is 1, then the bit at
    # state_asbits[n].get_bit(r,c) == 1. These values can be trivially checked
    # using bitwise AND operations, with the mask values in the neighbors
    # array.
    state_asbits = np.array(bitplane)
    return etats, state_asbits


state = np.random.choice([0, 1], size=(rows, cols, depth)).astype(bool)
cell = (4, 4)
state[cell][3] = 0
for n in neighbors2(*cell):
    state[n][3] = 0
neighborhood = get_neighs()
etats, state_asbits = encode(state)


def get_free5_pre():
    get_free5(cell, state_asbits, etats, neighborhood)
# state[neighbors2(*cll, True)][3] = 0
# assert (get_free((4, 4)) == get_free2((4, 4)))


print(get_free(cell))
print(get_free2(cell))
print(timeit.timeit("get_free((4, 4))", number=100000,
      setup="from __main__ import get_free"))
print(timeit.timeit("get_free2((4, 4))", number=100000,
      setup="from __main__ import get_free2"))
print(timeit.timeit("get_free5_pre()", number=100000,
      setup="from __main__ import get_free5_pre"))
