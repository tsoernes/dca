# run 'python3 -m unittest sometest.py'
import logging
import unittest

import numpy as np

from grid import RhombusAxialGrid
from params import get_pparams
from strats import SinghStrat


class TestSinghStrat(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger()
        self.pp, _ = get_pparams(defaults=True)
        self.rows = self.pp['rows']
        self.cols = self.pp['cols']
        self.n_channels = self.pp['n_channels']
        self.logger.error(self.pp)
        self.strat = SinghStrat(self.pp, self.logger)

    def test_feature_rep(self):
        def check_n_free_self(grid, n_free):
            self.assertTrue((grid[:, :, -1] == n_free).all())

        def check_n_used_neighs(grid, n_used):
            self.assertTrue((grid[:, :, :-1] == n_used).all())

        # Three grids in one array. They should not affect each other's
        # feature representation
        grids = np.zeros((3, self.pp['rows'], self.pp['cols'], self.pp['n_channels']))
        grids[1, :, :, 0] = 1
        grids[2, 1, 2, 9] = 1
        fgrids = self.strat.feature_rep(grids)

        # Verify Grid #1
        # No cell has any channels in use, i.e. all are free
        check_n_free_self(fgrids[0], np.ones((self.rows, self.cols)) * self.n_channels)
        # No cell has a channel in use by any of its neighbors
        check_n_used_neighs(fgrids[0], np.zeros((self.rows, self.cols, self.n_channels)))

        # Verify Grid #2
        # All cells have one channel in use
        check_n_free_self(fgrids[1],
                          np.ones((self.rows, self.cols)) * (self.n_channels - 1))
        # Every cell has 'n_neighs(cell)' neighbors4 who uses channel 0
        # ('n_neighs(cell)' depends on cell coordinates)
        n_used2 = np.zeros((self.rows, self.cols, self.n_channels))
        for r in range(self.rows):
            for c in range(self.cols):
                n_neighs = len(RhombusAxialGrid.neighbors(4, r, c))
                n_used2[r][c][0] = n_neighs
        check_n_used_neighs(fgrids[1], n_used2)

        # Verify Grid #3
        # Only cell (row, col) = (1, 2) has a channel in use (ch9)
        n_free3 = np.ones((self.rows, self.cols)) * self.n_channels
        n_free3[1][2] = self.n_channels - 1
        check_n_free_self(fgrids[2], n_free3)
        # Cell (1, 2) has no neighs that use ch9. The neighs of (1, 2)
        # has 1 cell that use ch9.
        n_used3 = np.zeros((self.rows, self.cols, self.n_channels))
        neighs3 = RhombusAxialGrid.neighbors(4, 1, 2, separate=True)
        n_used3[(*neighs3, np.repeat(9, len(neighs3[0])))] = 1
        check_n_used_neighs(fgrids[2], n_used3)
