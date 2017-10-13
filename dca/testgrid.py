from grid import Grid, BDCLGrid, FixedGrid

import unittest
import logging

import numpy as np


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(rows=7, cols=7, n_channels=70,
                         logger=logging.getLogger(""))

    def test_neighbors1(self):
        neighs = self.grid.neighbors1
        self._li_set_eq(
                [(0, 1), (1, 1), (1, 0)],
                neighs(0, 0))
        self._li_set_eq(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2)],
                neighs(2, 2))
        self._li_set_eq(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3)],
                neighs(2, 3))
        self._li_set_eq(
                [(6, 5), (5, 6)],
                neighs(6, 6))
        self._li_set_eq(
                [(5, 6), (6, 6), (6, 4), (5, 4), (5, 5)],
                neighs(6, 5))
        self._li_set_eq(
                [(6, 6), (6, 5), (5, 5), (4, 6)],
                neighs(5, 6))

    def test_neighbors1sparse(self):
        neighs = self.grid.neighbors1sparse
        self._li_set_eq(
                [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 0)],
                neighs(0, 0))
        self._li_set_eq(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2)],
                neighs(2, 2))
        self._li_set_eq(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3)],
                neighs(2, 3))
        self._li_set_eq(
                [(6, 7), (7, 7), (7, 6), (7, 5), (6, 5), (5, 6)],
                neighs(6, 6))

    def test_neighbors2(self):
        neighs = self.grid.neighbors2
        self._li_set_eq(
                [(0, 1), (1, 1), (1, 0), (0, 2), (1, 2), (2, 1), (2, 0)],
                neighs(0, 0))
        self._li_set_eq(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2),
                    (1, 3), (1, 4), (2, 4), (3, 4), (4, 3), (4, 2),
                    (4, 1), (3, 0), (2, 0), (1, 0), (1, 1), (0, 2)],
                neighs(2, 2))
        self._li_set_eq(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3),
                    (0, 4), (1, 5), (2, 5), (3, 5), (3, 4), (4, 3),
                    (3, 2), (3, 1), (2, 1), (1, 1), (0, 2), (0, 3)],
                neighs(2, 3))
        self._li_set_eq(
                [(6, 5), (5, 6), (6, 4), (5, 4), (5, 5), (4, 6)],
                neighs(6, 6))

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_partition_cells(self):
        self.grid._partition_cells()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                l = self.grid.labels[r][c]
                neighs = self.grid.neighbors2(r, c)
                for neigh in neighs:
                    self.assertNotEqual(l, self.grid.labels[neigh])

    def test_validate_reuse_constr(self):
        self.assertTrue(self.grid.validate_reuse_constr())
        cell = (2, 3)
        ch = 4

        self.grid.state[cell][ch] = 1
        self.assertTrue(self.grid.validate_reuse_constr())

        self.grid.state[(0, 1)][ch] = 1
        self.assertTrue(self.grid.validate_reuse_constr())

        self.grid.state[(1, 4)][ch+1] = 1
        self.grid.state[(1, 4)][ch-1] = 1
        self.assertTrue(self.grid.validate_reuse_constr())

        self.grid.state[(1, 4)][ch] = 1
        self.assertFalse(self.grid.validate_reuse_constr())

    def test_get_free_chs(self):
        self.grid.state = np.ones((7, 7, 70)).astype(bool)
        chs = [0, 4]
        cell = (3, 4)
        self.grid.state[cell][chs] = 0
        for n in self.grid.neighbors2(*cell):
            self.grid.state[n][chs] = 0
        free = self.grid.get_free_chs(cell)
        self._li_set_eq(free, chs)


class TestFixedGrid(unittest.TestCase):
    def setUp(self):
        self.grid = FixedGrid(rows=7, cols=7, n_channels=70,
                              logger=logging.getLogger(""))

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_assign_chs(self):
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.assertGreater(np.sum(self.grid.nom_chs[r][c]), 0)
                neighs = self.grid.neighbors2(r, c)
                for ch, isNom in enumerate(self.grid.nom_chs[r][c]):
                    if isNom:
                        for neigh in neighs:
                            self.assertFalse(self.grid.nom_chs[neigh][ch])

    def _test_max_utilization(self):
        """
        Verify that when all the nominal channels for each cell
        are in use, no channel can be assigned in any cell without
        breaking the reuse constraint.
        THIS WILL NOT BE THE CASE FOR A SQUARE GRID. CORNER CELLS
        DOES NOT HAVE 6 UNIQUELY LABELED NEIGHBORS WITHIN A RADIUS
        OF 2, BUT 5.
        """
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                for ch, isNom in enumerate(self.grid.nom_chs[r][c]):
                    if isNom:
                        self.grid.state[r][c][ch] = 1
        self.assertTrue(self.grid.validate_reuse_constr())
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                neighs = self.grid.neighbors2(r, c)
                f = np.bitwise_or(
                        self.grid.state[r][c], self.grid.state[neighs[0]])
                for n in neighs[1:]:
                    f = np.bitwise_or(f, self.grid.state[n])
                # There should not be any channels that are free
                # in the cell (r, c) and all of its neighbors
                if np.any(f == 0):
                    self.grid.print_neighs2(r, c)
                    self.assertTrue(
                            False,
                            "Grid not maximum utilized under FA")


# class TestBDCLGrid(unittest.TestCase):
class TestBDCLGrid():
    def setUp(self):
        self.grid = BDCLGrid(rows=7, cols=7, n_channels=70,
                             logger=logging.getLogger(""))

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_cochannel_cells(self):
        raise NotImplementedError  # not implemented correctly
        self.grid._partition_cells()
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (3, 3)),
                [(1, 3), (3, 5)])
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (4, 3)),
                [(3, 5), (5, 4)])
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (4, 2)),
                [(5, 4), (6, 1)])
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (4, 1)),
                [(6, 1)])
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (3, 1)),
                [(1, 0)])
        self._li_set_eq(
                self.grid.cochannel_cells((3, 2), (2, 2)),
                [(1, 0), (1, 3)])


if __name__ == '__main__':
    unittest.main()
