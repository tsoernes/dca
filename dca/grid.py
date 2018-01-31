import functools

import numpy as np

from eventgen import CEvent


class Grid:
    def __init__(self, rows, cols, n_channels, logger, *args, **kwargs):
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels
        self.logger = logger

        self.state = np.zeros(
            (self.rows, self.cols, self.n_channels), dtype=bool)
        self.labels = np.zeros((self.rows, self.cols), dtype=int)
        self._partition_cells()

    def print_cell(self, r, c):
        print(f"Cell ({r}, {c}): {np.where(self.state[r][c] == 1)}")

    def print_neighs(self, row, col):
        print(f"Cell ({row}, {col})"
              f"\nNeighs1: {self.neighbors1(row, col)}"
              f"\nNeighs1sparse: {self.neighbors1sparse(row, col)}"
              f"\nNeighs2: {self.neighbors2(row, col)}")

    def print_neighs2(self, row, col):
        """
        Show all the channels for the given cell and its neighbors
        """
        self.print_cell(row, col)
        for neigh in self.neighbors2(row, col):
            print(f"\n{neigh}: {np.where(self.state[neigh]==1)}")

    def __str__(self):
        strstate = ""
        for r in range(self.rows):
            for c in range(self.cols):
                inuse = np.where(self.state[r][c] == 1)
                strstate += f"\n({r},{c}): {len(inuse)} used - {inuse}"
        return strstate

    def validate_reuse_constr(self):
        """
        Verify that the channel reuse constraint of 3 is not violated,
        e.g. that a channel in use in a cell is not in use in its neighbors.
        Returns True if valid not violated, False otherwise
        """
        # TODO: It might be possible to do this more efficiently.
        # If no neighbors of a cell violate the channel reuse constraint,
        # then the cell itself does not either, so it should be possible
        # to skip checking some cells.
        for r in range(self.rows):
            for c in range(self.cols):
                neighs = self.neighbors2(r, c, True)
                # Channels in use at neighbors
                inuse = np.bitwise_or.reduce(self.state[neighs])
                # Channels in use at a neigh and cell
                inuse_both = np.bitwise_and(self.state[r][c], inuse)
                viols = np.where(inuse_both == 1)[0]
                if len(viols) > 0:
                    self.logger.error("Channel Reuse constraint violated"
                                      f" in Cell {(r, c) }"
                                      f" at channels {viols}")
                    return False
        return True

    def get_eligible_chs(self, cell):
        return self.get_eligible_chs_stat(self.state, cell)

    @staticmethod
    def get_eligible_chs_stat(grid, cell):
        """
        Find the channels that are free in 'cell' and all of
        its neighbors by bitwise ORing their allocation maps.
        These are the eligible channels, i.e. those that can be assigned
        without violating the reuse constraint.
        """
        neighs = RhombusAxialGrid.neighbors2(*cell)
        alloc_map = np.bitwise_or(grid[cell], grid[neighs[0]])
        for n in neighs[1:]:
            alloc_map = np.bitwise_or(alloc_map, grid[n])
        eligible = np.where(alloc_map == 0)[0]
        return eligible

    def afterstates(self, cell, ce_type, chs):
        """Make an afterstate (resulting grid) for each possible,
        # eligible action in 'chs'"""
        if ce_type == CEvent.END:
            targ_val = 0
        else:
            targ_val = 1
        grids = np.repeat(
            np.expand_dims(np.copy(self.state), axis=0), len(chs), axis=0)
        for i, ch in enumerate(chs):
            grids[i][cell][ch] = targ_val
        assert grids.shape == (len(chs), self.rows, self.cols, self.n_channels)
        return grids


class RectOffsetGrid(Grid):
    "Rectangular grid with offset coordinates"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def neighbors1sparse(row, col):
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self. The indexes may not be within grid.
        """
        if col % 2 == 0:
            return [(row - 1, col), (row, col + 1), (row + 1, col + 1),
                    (row + 1, col), (row + 1, col - 1), (row, col - 1)]
        else:
            return [(row - 1, col), (row - 1, col + 1), (row, col + 1),
                    (row + 1, col), (row, col - 1), (row - 1, col - 1)]

    @functools.lru_cache(maxsize=None)
    def neighbors1(self, row, col):
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self
        """
        idxs = []
        r_low = max(0, row - 1)
        r_hi = min(self.rows - 1, row + 1)
        c_low = max(0, col - 1)
        c_hi = min(self.cols - 1, col + 1)
        if col % 2 == 0:
            cross = row - 1
        else:
            cross = row + 1
        for r in range(r_low, r_hi + 1):
            for c in range(c_low, c_hi + 1):
                if not ((r, c) == (cross, col - 1) or
                        (r, c) == (cross, col + 1) or (r, c) == (row, col)):
                    idxs.append((r, c))
        return idxs

    @functools.lru_cache(maxsize=None)
    def neighbors2(self, row, col, separate=False):
        """
        If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...]),
        else return [(r1, c1), (r2, c2), ...]

        Returns a list with indices of neighbors within a radius of 2,
        not including self
        """
        if separate:
            rs = []
            cs = []
        else:
            idxs = []

        r_low = max(0, row - 2)
        r_hi = min(self.rows - 1, row + 2)
        c_low = max(0, col - 2)
        c_hi = min(self.cols - 1, col + 2)
        if col % 2 == 0:
            cross1 = row - 2
            cross2 = row + 2
        else:
            cross1 = row + 2
            cross2 = row - 2
        for r in range(r_low, r_hi + 1):
            for c in range(c_low, c_hi + 1):
                if not ((r, c) == (row, col) or (r, c) == (cross1, col - 2) or
                        (r, c) == (cross1, col - 1) or
                        (r, c) == (cross1, col + 1) or
                        (r, c) == (cross1, col + 2) or
                        (r, c) == (cross2, col - 2) or
                        (r, c) == (cross2, col + 2)):
                    if separate:
                        rs.append(r)
                        cs.append(c)
                    else:
                        idxs.append((r, c))

        # k = self.rows
        # idxs2all = np.array(
        #     np.meshgrid(np.arange(k), np.arange(k), indexing="ij")) \
        #     .transpose([1, 2, 0]) \
        #     .reshape(1, -2, 2)[0}
        if separate:
            return (rs, cs)
        else:
            return idxs

    def neighbors2all(self):
        # NOTE the results, i.e. np.arrays, do NOT
        # index the same way as tuples
        neighs2 = np.zeros((self.rows, self.cols, 18, 2), dtype=np.int32)
        mask = np.zeros((self.rows, self.cols, 18), dtype=np.bool)
        for row in range(self.rows):
            for col in range(self.cols):
                r_low = max(0, row - 2)
                r_hi = min(self.rows - 1, row + 2)
                c_low = max(0, col - 2)
                c_hi = min(self.cols - 1, col + 2)
                if col % 2 == 0:
                    cross1 = row - 2
                    cross2 = row + 2
                else:
                    cross1 = row + 2
                    cross2 = row - 2
                oh_idxs = np.zeros(
                    (self.rows + 2, self.cols + 2), dtype=np.bool)
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

    def _partition_cells(self):
        """
        Partition cells into 7 lots such that the minimum distance
        between cells with the same label ([0..6]) is at least 2
        (which corresponds to a minimum reuse distance of 3).

        Returns an n*m array with the label for each cell.
        """

        def right_up(x, y):
            x_new = x + 3
            y_new = y
            if x % 2 != 0:
                # Odd column
                y_new = y - 1
            return (x_new, y_new)

        def down_left(x, y):
            x_new = x - 1
            if x % 2 == 0:
                # Even column
                y_new = y + 3
            else:
                # Odd Column
                y_new = y + 2
            return (x_new, y_new)

        def label(l, x, y):
            # A center and some part of its subgrid may be out of bounds.
            if (x >= 0 and x < self.cols and y >= 0 and y < self.rows):
                self.labels[y][x] = l

        # Center of a 'circular' 7-cell subgrid in which
        # each cell has a unique label
        center = (0, 0)
        # First center in current row which has neighbors inside grid
        first_row_center = (0, 0)
        # Move center down-left until subgrid goes out of bounds
        while (center[0] >= -1) and (center[1] <= self.rows):
            # Move center right-up until subgrid goes out of bounds
            while (center[0] <= self.cols) and (center[1] >= -1):
                # Label cells 0..6 with given center as 0
                label(0, *center)
                for i, neigh in enumerate(
                        self.neighbors1sparse(center[1], center[0])):
                    label(i + 1, neigh[1], neigh[0])
                center = right_up(*center)
            center = down_left(*first_row_center)
            # Move right until x >= -1
            while center[0] < -1:
                center = right_up(*center)
            first_row_center = center


class RhombusAxialGrid(Grid):
    "Rhombus grid with axial coordinates"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def neighbors1sparse(row, col):
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self. The indexes may not be within grid.
        """
        idxs = []
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if not ((r, c) == (row - 1, col - 1) or
                        (r, c) == (row + 1, col + 1) or (r, c) == (row, col)):
                    idxs.append((r, c))
        return idxs

    @staticmethod
    def neighbors1(row, col, rows=7, cols=7):
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self
        """
        return RhombusAxialGrid.neighbors(1, row, col)

    @staticmethod
    def neighbors2(row, col, separate=False):
        """
        If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...]),
        else return [(r1, c1), (r2, c2), ...]

        Returns a list with indices of neighbors within a radius of 2,
        not including self
        """
        return RhombusAxialGrid.neighbors(
            2, row, col, separate=separate, include_self=False)

    @staticmethod
    def hex_distance(a, b):
        r1, c1 = a
        r2, c2 = b
        return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def neighbors(dist,
                  row,
                  col,
                  separate=False,
                  include_self=False,
                  rows=7,
                  cols=7):
        if separate is True:
            rs = []
            cs = []
        else:
            idxs = []
        for r2 in range(rows):
            for c2 in range(cols):
                if (include_self or (row, col) != (r2, c2)) \
                   and RhombusAxialGrid.hex_distance((row, col), (r2, c2)) <= dist:
                    if separate is True:
                        rs.append(r2)
                        cs.append(c2)
                    else:
                        idxs.append((r2, c2))
        if separate is True:
            return (rs, cs)
        return idxs

    def _partition_cells(self):
        """
        Partition cells into 7 lots such that the minimum distance
        between cells with the same label ([0..6]) is at least 2
        (which corresponds to a minimum reuse distance of 3).

        Create an n*m array with the label for each cell.
        """

        def label(l, y, x):
            # A center and some part of its subgrid may be out of bounds.
            if (x >= 0 and x < self.cols and y >= 0 and y < self.rows):
                self.labels[y][x] = l

        centers = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (3, -1), (4, 1),
                   (5, 3), (6, 5), (7, 7), (-1, 5), (7, 0), (0, 7)]
        for center in centers:
            label(0, *center)
            for i, neigh in enumerate(self.neighbors1sparse(*center)):
                label(i + 1, *neigh)
