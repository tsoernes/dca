# run 'python3 -m unittest sometest.py'
import logging
import unittest

import numpy as np

from grid import RhombusAxialGrid
from params import get_pparams
from strats import SinghStrat


class TestSinghStrat(unittest.TestCase):
    def setUp(self):
        # self.assertNotEqual(l, self.grid.labels[neigh])
        # self.assertTrue(self.grid.validate_reuse_constr())
        # self.assertTrue(False, "Grid not maximum utilized under FA")
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

        grid1 = np.zeros(self.pp['dims'])
        fgrids1 = self.strat.feature_rep(grid1)
        # No cell has any channels in use, i.e. all are free
        check_n_free_self(fgrids1[0],
                          np.ones((self.rows, self.cols)) * self.n_channels)
        # No cell has a channel in use by any of its neighbors
        check_n_used_neighs(fgrids1[0],
                            np.zeros((self.rows, self.cols, self.n_channels)))

        grid2 = np.zeros(self.pp['dims'])
        grid2[:, :, 0] = 1
        fgrids2 = self.strat.feature_rep(grid2)
        # All cells have one channel in use
        check_n_free_self(fgrids2[0],
                          np.ones((self.rows,
                                   self.cols)) * (self.n_channels - 1))
        # Every cell has 'n_neighs(cell)' neighbors4 who uses channel 0
        # ('n_neighs(cell)' depends on cell coordinates)
        n_used = np.zeros((self.rows, self.cols, self.n_channels))
        for r in range(self.rows):
            for c in range(self.cols):
                n_neighs = len(RhombusAxialGrid.neighbors(4, r, c))
                n_used[r][c][0] = n_neighs
        check_n_used_neighs(fgrids2[0], n_used)
