import functools

import numpy as np

from eventgen import CEvent
# from gridfuncs import get_eligible_chs_bitmap, hex_distance, neighbors
from gridfuncs_numba import (afterstates, get_eligible_chs,
                             get_eligible_chs_bitmap, get_n_eligible_chs,
                             hex_distance, neighbors, neighbors_sep)


class Grid:
    "Rhombus grid with axial coordinates"

    def __init__(self, rows, cols, n_channels, logger, *args, **kwargs):
        self.rows, self.cols, self.n_channels = rows, cols, n_channels
        self.logger = logger

        self.state = np.zeros((rows, cols, n_channels), dtype=np.bool)
        self.labels = np.zeros((rows, cols), dtype=int)
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

    @staticmethod
    def get_eligible_chs_bitmap(grid, cell):
        get_eligible_chs_bitmap(grid, cell)
    # @staticmethod
    # def get_eligible_chs_bitmap(grid, cell):
    #     """Find eligible chs by bitwise ORing the allocation maps of neighbors"""
    #     r, c = cell
    #     neighs = Grid.neighbors(2, r, c, separate=True, include_self=True)
    #     alloc_map = np.bitwise_or.reduce(grid[neighs])
    #     return alloc_map

    @staticmethod
    def get_eligible_chs(grid, cell):
        get_eligible_chs(grid, cell)
    # @staticmethod
    # def get_eligible_chs(grid, cell):
    #     """
    #     Find the channels that are free in 'cell' and all of
    #     its neighbors with a distance of 2 or less.
    #     These are the eligible channels, i.e. those that can be assigned
    #     without violating the reuse constraint.
    #     """
    #     alloc_map = Grid.get_eligible_chs_bitmap(grid, cell)
    #     eligible = np.nonzero(np.logical_not(alloc_map))[0]
    #     return eligible

    @staticmethod
    def get_n_eligible_chs(grid, cell):
        get_n_eligible_chs(grid, cell)
    # @staticmethod
    # def get_n_eligible_chs(grid, cell):
    #     """Return the number of eligible channels"""
    #     alloc_map = Grid.get_eligible_chs_bitmap(grid, cell)
    #     n_eligible = np.count_nonzero(np.invert(alloc_map))
    #     return n_eligible

    @staticmethod
    def afterstates(grid, cell, ce_type, chs, rows=7, cols=7, n_channels=70):
        return afterstates(grid, cell, ce_type, chs, rows=7, cols=7, n_channels=70)
    # @staticmethod
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

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def neighbors1sparse(row, col):
        """
        Returns a list with the indecies of neighbors within a radius of 1,
        not including self. The indecies may not be within grid boundaries.
        """
        idxs = []
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if not ((r, c) == (row - 1, col - 1) or (r, c) == (row + 1, col + 1) or
                        (r, c) == (row, col)):
                    idxs.append((r, c))
        return idxs

    @staticmethod
    def neighbors1(row, col):
        return neighbors(1, row, col)

    @staticmethod
    def neighbors2(row, col, separate):
        return neighbors(2, row, col, separate=separate, include_self=False)

    @staticmethod
    def hex_distance(cell_a, cell_b):
        return hex_distance(cell_a, cell_b)
    # @staticmethod
    # def hex_distance(cell_a, cell_b):
    #     r1, c1 = cell_a
    #     r2, c2 = cell_b
    #     return (abs(r1 - r2) + abs(r1 + c1 - r2 - c2) + abs(c1 - c2)) / 2

    @staticmethod
    def neighbors(dist, row, col, separate=False, include_self=False, rows=7, cols=7):
        if separate:
            return neighbors_sep(dist, row, col, include_self, rows, cols)
        else:
            return neighbors(dist, row, col, include_self, rows, cols)
    # @staticmethod
    # @functools.lru_cache(maxsize=None)
    # def neighbors(dist, row, col, separate=False, include_self=False, rows=7, cols=7):
    #     """
    #     Returns a list with indices of neighbors with a distance of 'dist' or less
    #     from the cell at (row, col)

    #     If 'separate' is True, return ([r1, r2, ...], [c1, c2, ...]),
    #     else return [(r1, c1), (r2, c2), ...]
    #     """
    #     if separate:
    #         rs = []
    #         cs = []
    #     else:
    #         idxs = []
    #     for r2 in range(rows):
    #         for c2 in range(cols):
    #             if (include_self or (row, col) != (r2, c2)) \
    #                and Grid.hex_distance((row, col), (r2, c2)) <= dist:
    #                 if separate:
    #                     rs.append(r2)
    #                     cs.append(c2)
    #                 else:
    #                     idxs.append((r2, c2))
    #     if separate:
    #         return (rs, cs)
    #     return idxs

    @staticmethod
    def neighbors_all_oh(dist=2, include_self=True, rows=7, cols=7):
        """
        Returns an array where each and every cell has a onehot representation of
        their neigbors
        """
        idxs = np.zeros((rows, cols, rows, cols), dtype=np.bool)
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(rows):
                    for c2 in range(cols):
                        if (include_self or (r1, c1) != (r2, c2)) \
                           and Grid.hex_distance((r1, c1), (r2, c2)) <= dist:
                            idxs[r1, c1, r2, c2] = True
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

        centers = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (3, -1), (4, 1), (5, 3),
                   (6, 5), (7, 7), (-1, 5), (7, 0), (0, 7)]
        for center in centers:
            label(0, *center)
            for i, neigh in enumerate(self.neighbors1sparse(*center)):
                label(i + 1, *neigh)
