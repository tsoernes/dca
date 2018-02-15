import numpy as np
from numba import jitclass
from numba.types import boolean, int64

from eventgen import CEvent

spec = [
    ('rows', int64),
    ('cols', int64),
    ('n_channels', int64),
    ('neighs1', int64[:, :, :, :]),
    ('neighs2', int64[:, :, :, :]),
    ('neighs4', int64[:, :, :, :]),
    ('n_neighs1', int64[:, :]),
    ('n_neighs2', int64[:, :]),
    ('n_neighs4', int64[:, :]),
]  # yapf: disable


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


# boolean = np.bool
# int64 = np.int64


@singleton
@jitclass(spec)
class GridFuncs:
    """
    All methods are static in practice (numba does not allow @staticmethod),
    take a grid as input, and read only grid dimensions and neighbors
    from instance-variables.
    """

    def __init__(self, rows, cols, n_channels):
        self.rows, self.cols, self.n_channels = rows, cols, n_channels
        self.neighs1 = np.zeros((rows, cols, 7, 2), dtype=int64)
        self.neighs2 = np.zeros((rows, cols, 19, 2), dtype=int64)
        self.neighs4 = np.zeros((rows, cols, 43, 2), dtype=int64)
        self.n_neighs1 = np.zeros((rows, cols), dtype=int64)
        self.n_neighs2 = np.zeros((rows, cols), dtype=int64)
        self.n_neighs4 = np.zeros((rows, cols), dtype=int64)
        self._generate_neighbors()

    def _hex_distance(self, r1, c1, r2, c2):
        return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2

    def _generate_neighbors(self):
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                self.neighs1[r1, c1, 0] = (r1, c1)
                self.n_neighs1[r1, c1] += 1
                self.neighs2[r1, c1, 0] = (r1, c1)
                self.n_neighs2[r1, c1] += 1
                self.neighs4[r1, c1, 0] = (r1, c1)
                self.n_neighs4[r1, c1] += 1
                for r2 in range(self.rows):
                    for c2 in range(self.cols):
                        dist = self._hex_distance(r1, c1, r2, c2)
                        if (r1, c1) != (r2, c2) and dist <= 4:
                            self.neighs4[r1, c1, self.n_neighs4[r1, c1]] = (r2, c2)
                            self.n_neighs4[r1, c1] += 1
                            if dist <= 2:
                                self.neighs2[r1, c1, self.n_neighs2[r1, c1]] = (r2, c2)
                                self.n_neighs2[r1, c1] += 1
                                if dist <= 1:
                                    self.neighs1[r1, c1, self.n_neighs1[r1, c1]] = (r2,
                                                                                    c2)
                                    self.n_neighs1[r1, c1] += 1
        print("Generated neighbors")

    def neighbors_np(self, dist, row, col, include_self=False):
        """np array of 2-dim np arrays"""
        start = 0 if include_self else 1
        if dist == 1:
            neighs = self.neighs1
            n_neighs = self.n_neighs1
        elif dist == 2:
            neighs = self.neighs2
            n_neighs = self.n_neighs2
        elif dist == 4:
            neighs = self.neighs4
            n_neighs = self.n_neighs4
        else:
            raise NotImplementedError
        return neighs[row, col, start:n_neighs[row, col]]

    def neighbors(self, dist, row, col, include_self=False):
        """list of tuples"""
        start = 0 if include_self else 1
        if dist == 1:
            neighs = self.neighs1
            n_neighs = self.n_neighs1
        elif dist == 2:
            neighs = self.neighs2
            n_neighs = self.n_neighs2
        elif dist == 4:
            neighs = self.neighs4
            n_neighs = self.n_neighs4
        else:
            raise NotImplementedError
        n = []
        for i in range(start, n_neighs[row, col]):
            n.append((neighs[row, col, i, 0], neighs[row, col, i, 1]))
        return n

    def neighbors_sep(self, dist, row, col, include_self=False):
        """2-Tuple of np arrays"""
        start = 0 if include_self else 1
        if dist == 1:
            neighs = self.neighs1
            n_neighs = self.n_neighs1
        elif dist == 2:
            neighs = self.neighs2
            n_neighs = self.n_neighs2
        elif dist == 4:
            neighs = self.neighs4
            n_neighs = self.n_neighs4
        else:
            raise NotImplementedError
        return (neighs[row, col, start:n_neighs[row, col], 0],
                neighs[row, col, start:n_neighs[row, col], 1])

    def _inuse_neighs(self, grid, r, c):
        # Channels in use at neighbors
        neighs = self.neighbors_np(2, r, c, False)
        alloc_map = grid[neighs[0, 0], neighs[0, 1]]
        for i in range(1, len(neighs)):
            nrow, ncol = neighs[i]
            alloc_map = np.bitwise_or(alloc_map, grid[nrow, ncol])
        return alloc_map

    def get_eligible_chs(self, grid, cell):
        alloc_map = self._inuse_neighs(grid, *cell)
        alloc_map = np.bitwise_or(alloc_map, grid[cell])
        eligible = np.nonzero(np.invert(alloc_map))[0]
        return eligible

    def get_n_eligible_chs(self, grid, cell):
        alloc_map = self._inuse_neighs(grid, *cell)
        alloc_map = np.bitwise_or(alloc_map, grid[cell])
        n_eligible = np.sum(np.invert(alloc_map))
        return n_eligible

    def afterstates(self, grid, cell, ce_type, chs):
        if ce_type == CEvent.END:
            targ_val = False
        else:
            targ_val = True
        grids = np.zeros((len(chs), self.rows, self.cols, self.n_channels), dtype=boolean)
        for i, ch in enumerate(chs):
            grids[i] = grid
            grids[i][cell][ch] = targ_val
        return grids

    def feature_rep(self, grid):
        fgrid = np.zeros((self.rows, self.cols, self.n_channels + 1), dtype=np.int16)
        for r in range(self.rows):
            for c in range(self.cols):
                neighs = self.neighbors_sep(4, r, c, True)
                n_used = np.zeros(self.n_channels)
                for i in range(len(neighs[0])):
                    n_used += grid[neighs[0][i], neighs[1][i]]
                fgrid[r, c, :-1] = n_used
                fgrid[r, c, -1] = self.get_n_eligible_chs(grid, (r, c))
        return fgrid

    def afterstate_freps(self, grid, cell, ce_type, chs):
        fgrid = self.feature_rep(grid)
        r, c = cell
        neighs4 = self.neighbors_sep(4, r, c, True)
        fgrids = np.zeros(
            (len(chs), self.rows, self.cols, self.n_channels + 1), dtype=int64)
        fgrids[:] = fgrid
        if ce_type == CEvent.END:
            n_used_neighs_diff = -1
            n_elig_self_diff = 1
            grid[cell][chs] = 0
        else:
            n_used_neighs_diff = 1
            n_elig_self_diff = -1

        neighs2 = self.neighbors_sep(2, r, c, True)
        for i, ch in enumerate(chs):
            for j in range(len(neighs4[0])):
                fgrids[i, neighs4[0][j], neighs4[1][j], ch] += n_used_neighs_diff
            for j in range(len(neighs2[0])):
                r2, c2 = neighs2[0][j], neighs2[1][j]
                neighs = self.neighbors_np(2, r2, c2, False)
                inuse = grid[r2, c2, ch]
                for k in range(len(neighs)):
                    inuse = inuse or grid[neighs[k, 0], neighs[k, 1], ch]
                if not inuse:
                    fgrids[i, neighs2[0][j], neighs2[1][j], -1] += n_elig_self_diff
        if ce_type == CEvent.END:
            grid[cell][chs] = 1
        return fgrids


GF = GridFuncs(7, 7, 70)
