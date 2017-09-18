from grid import Grid

import unittest

import numpy as np


class TestNeighbors(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(rows=7, cols=7, n_channels=70)

    def test_neighbors1(self):
        neighs = self.grid.neighbors1
        self._test_neighbors(
                [(0, 1), (1, 1), (1, 0)],
                neighs(0, 0))
        self._test_neighbors(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2)],
                neighs(2, 2))
        self._test_neighbors(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3)],
                neighs(2, 3))
        self._test_neighbors(
                [(6, 5), (5, 6)],
                neighs(6, 6))
        self._test_neighbors(
                [(5, 6), (6, 6), (6, 4), (5, 4), (5, 5)],
                neighs(6, 5))
        self._test_neighbors(
                [(6, 6), (6, 5), (5, 5), (4, 6)],
                neighs(5, 6))

    def test_neighbors1sparse(self):
        neighs = self.grid.neighbors1sparse
        self._test_neighbors(
                [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 0)],
                neighs(0, 0))
        self._test_neighbors(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2)],
                neighs(2, 2))
        self._test_neighbors(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3)],
                neighs(2, 3))
        self._test_neighbors(
                [(6, 7), (7, 7), (7, 6), (7, 5), (6, 5), (5, 6)],
                neighs(6, 6))

    def test_neighbors2(self):
        neighs = self.grid.neighbors2
        self._test_neighbors(
                [(0, 1), (1, 1), (1, 0), (0, 2), (1, 2), (2, 1), (2, 0)],
                neighs(0, 0))
        self._test_neighbors(
                [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 2),
                    (1, 3), (1, 4), (2, 4), (3, 4), (4, 3), (4, 2),
                    (4, 1), (3, 0), (2, 0), (1, 0), (1, 1), (0, 2)],
                neighs(2, 2))
        self._test_neighbors(
                [(1, 4), (2, 4), (3, 3), (2, 2), (1, 2), (1, 3),
                    (0, 4), (1, 5), (2, 5), (3, 5), (3, 4), (4, 3),
                    (3, 2), (3, 1), (2, 1), (1, 1), (0, 2), (0, 3)],
                neighs(2, 3))
        self._test_neighbors(
                [(6, 5), (5, 6), (6, 4), (5, 4), (5, 5), (4, 6)],
                neighs(6, 6))

    def test_partition_cells(self):
        labels = self.grid.partition_cells()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                l = labels[r][c]
                neighs = self.grid.neighbors2(r, c)
                for neigh in neighs:
                    self.assertNotEqual(l, labels[neigh])

    def test_assign_chs(self):
        nom_chs = self.grid.assign_chs()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.assertGreater(np.sum(nom_chs[r][c]), 0)
                neighs = self.grid.neighbors2(r, c)
                for ch, isNom in enumerate(nom_chs[r][c]):
                    if isNom:
                        for neigh in neighs:
                            self.assertFalse(nom_chs[neigh][ch])

    def _test_neighbors(self, targ_neighs, act_neighs):
        self.assertSetEqual(set(targ_neighs), set(act_neighs))


if __name__ == '__main__':
    unittest.main()
