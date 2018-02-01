import numpy as np
import timeit

rows = 7
cols = 7


def neighbors(row, col):
    idxs = []

    r_low = max(0, row - 2)
    r_hi = min(rows - 1, row + 2)
    c_low = max(0, col - 2)
    c_hi = min(cols - 1, col + 2)
    if col % 2 == 0:
        cross1 = row - 2
        cross2 = row + 2
    else:
        cross1 = row + 2
        cross2 = row - 2
    for r in range(r_low, r_hi + 1):
        for c in range(c_low, c_hi + 1):
            if not ((r, c) == (row, col) or (r, c) == (cross1, col - 2) or
                    (r, c) == (cross1, col - 1) or (r, c) == (cross1, col + 1) or
                    (r, c) == (cross1, col + 2) or (r, c) == (cross2, col - 2) or
                    (r, c) == (cross2, col + 2)):
                idxs.append((r, c))

    return idxs


def neighbors2all():
    # NOTE the results, i.e. np.arrays, do NOT
    # index the same way as tuples
    neighs2 = np.zeros((7, 7, 18, 2), dtype=np.int32)
    mask = np.zeros((7, 7, 18), dtype=np.bool)
    for row in range(7):
        for col in range(7):
            r_low = max(0, row - 2)
            r_hi = min(rows - 1, row + 2)
            c_low = max(0, col - 2)
            c_hi = min(cols - 1, col + 2)
            if col % 2 == 0:
                cross1 = row - 2
                cross2 = row + 2
            else:
                cross1 = row + 2
                cross2 = row - 2
            oh_idxs = np.zeros((rows + 2, cols + 2), dtype=np.bool)
            oh_idxs[r_low:r_hi + 1, c_low:c_hi + 1] = True

            oh_idxs[row, col] = False
            oh_idxs[cross1, col - 2] = False
            oh_idxs[cross1, col - 1] = False
            oh_idxs[cross1, col + 1] = False
            oh_idxs[cross1, col + 2] = False
            oh_idxs[cross2, col - 2] = False
            oh_idxs[cross2, col + 2] = False
            idxs = np.transpose(np.where(oh_idxs))
            neighs2[row][col][:idxs.shape[0], :idxs.shape[1]] = idxs
            mask[row][col][:idxs.shape[0]] = True
    return (neighs2, mask)


def neighbors2(row, col):
    # NOTE the results, i.e. np.arrays, do NOT
    # index the same way as tuples
    r_low = max(0, row - 2)
    r_hi = min(rows - 1, row + 2)
    c_low = max(0, col - 2)
    c_hi = min(cols - 1, col + 2)
    if col % 2 == 0:
        cross1 = row - 2
        cross2 = row + 2
    else:
        cross1 = row + 2
        cross2 = row - 2
    oh_idxs = np.zeros((rows + 2, cols + 2), dtype=np.bool)
    oh_idxs[r_low:r_hi + 1, c_low:c_hi + 1] = True

    oh_idxs[row, col] = False
    oh_idxs[cross1, col - 2] = False
    oh_idxs[cross1, col - 1] = False
    oh_idxs[cross1, col + 1] = False
    oh_idxs[cross1, col + 2] = False
    oh_idxs[cross2, col - 2] = False
    oh_idxs[cross2, col + 2] = False
    idxs = np.transpose(np.where(oh_idxs))
    return idxs


def test():
    for r in range(0, 7):
        for c in range(0, 7):
            assert ((neighbors(r, c) == neighbors2(r, c)).all())
    print("OK")

    n1 = lambda: neighbors(3, 3)
    n2 = lambda: neighbors2(3, 3)
    print(timeit.timeit(n2, number=200000))
    print(timeit.timeit(n1, number=200000))


if __name__ == "__main__":
    test()
