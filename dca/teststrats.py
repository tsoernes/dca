# run:
# python3 -m unittest teststrats.py
import logging
import unittest

import numpy as np

from eventgen import CEvent
from grid import RhombusAxialGrid
from params import get_pparams
from strats.net_rl import SinghNetStrat


class TestSinghStrat(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger()
        self.pp, _ = get_pparams(defaults=True)
        self.rows = self.pp['rows']
        self.cols = self.pp['cols']
        self.n_channels = self.pp['n_channels']
        self.logger.error(self.pp)
        self.strat = SinghNetStrat(self.pp, self.logger)

    def test_afterstate_freps(self):
        """
        Test incremental vs naive afterstate feature rep. derivation approach
        """

        def check_freps(freps_inc, freps):
            self.assertTrue(freps_inc.shape == freps.shape,
                            (freps_inc.shape, freps.shape))
            eq_n_free = freps_inc[:, :, :, :-1] == freps[:, :, :, :-1]
            diff_n_free = np.where(np.invert(eq_n_free))
            self.assertTrue(eq_n_free.all(),
                            (diff_n_free, freps_inc[diff_n_free], freps[diff_n_free]))
            eq_n_used = freps_inc[:, :, :, -1] == freps[:, :, :, -1]
            diff_n_used = np.where(np.invert(eq_n_used))
            self.assertTrue(
                eq_n_used.all(),
                ("\n", diff_n_used, freps_inc[:, :, :, -1], freps[:, :, :, -1]))

        grid1 = np.zeros(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        cell1 = (2, 3)
        ce_type1 = CEvent.NEW
        chs1 = RhombusAxialGrid.get_eligible_chs_stat(grid1, cell1)
        freps_inc1 = self.strat.afterstate_freps(grid1, cell1, ce_type1, chs1)
        astates1 = RhombusAxialGrid.afterstates_stat(grid1, cell1, ce_type1, chs1)
        freps1 = self.strat.feature_reps(astates1)
        check_freps(freps_inc1, freps1)

        grid2 = np.ones(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        cell2 = (2, 3)
        grid2[cell2][4] = 0
        grid2[(2, 4)][:] = 0
        ce_type2 = CEvent.END
        chs2 = np.nonzero(grid2[cell2])[0]
        freps_inc2 = self.strat.afterstate_freps(grid2, cell2, ce_type2, chs2)
        astates2 = RhombusAxialGrid.afterstates_stat(grid2, cell2, ce_type2, chs2)
        freps2 = self.strat.feature_reps(astates2)
        check_freps(freps_inc2, freps2)

        grid3 = np.zeros(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        cell3 = (4, 1)
        grid3[cell3][4] = 1
        ce_type3 = CEvent.END
        chs3 = np.nonzero(grid3[cell3])[0]
        freps_inc3 = self.strat.afterstate_freps(grid3, cell3, ce_type3, chs3)
        astates3 = RhombusAxialGrid.afterstates_stat(grid3, cell3, ce_type3, chs3)
        freps3 = self.strat.feature_reps(astates3)
        check_freps(freps_inc3, freps3)

    def test_feature_rep(self):
        def check_n_free_self(grid, n_free):
            self.assertTrue((grid[:, :, -1] == n_free).all(), (grid[:, :, -1], n_free))

        def check_n_used_neighs(grid, n_used):
            self.assertTrue((grid[:, :, :-1] == n_used).all())

        # Three grids in one array. They should not affect each other's
        # feature representation
        grid1 = np.zeros(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        grid2 = np.zeros(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        grid3 = np.zeros(
            (self.pp['rows'], self.pp['cols'], self.pp['n_channels']), dtype=bool)
        grid2[:, :, 0] = 1
        grid3[1, 2, 9] = 1
        fgrid1 = self.strat.feature_reps(grid1)
        fgrid2 = self.strat.feature_reps(grid2)
        fgrid3 = self.strat.feature_reps(grid3)

        # Verify that single- and multi-version works the same
        grids = np.array([grid1, grid2, grid3])
        fgrids = self.strat.feature_reps(grids)
        self.assertTrue((fgrids[0] == fgrid1).all())
        self.assertTrue((fgrids[1] == fgrid2).all())
        self.assertTrue((fgrids[2] == fgrid3).all())

        # TODO NOTE TODO
        # Check if tests below are valid if frep is for ELIGIBLE not
        # FREE chs. Then, if easy to do, mod feature_reps func to do
        # the latter.

        # Verify Grid #1
        # No cell has any channels in use, i.e. all are free
        check_n_free_self(fgrid1, np.ones((self.rows, self.cols)) * self.n_channels)
        # No cell has a channel in use by any of its neighbors
        check_n_used_neighs(fgrid1, np.zeros((self.rows, self.cols, self.n_channels)))

        # Verify Grid #2
        # All cells have one channel in use
        check_n_free_self(fgrid2, np.ones((self.rows, self.cols)) * (self.n_channels - 1))
        # Every cell has 'n_neighs(cell)' neighbors4 who uses channel 0
        # ('n_neighs(cell)' depends on cell coordinates)
        n_used2 = np.zeros((self.rows, self.cols, self.n_channels))
        for r in range(self.rows):
            for c in range(self.cols):
                n_neighs = len(RhombusAxialGrid.neighbors(4, r, c))
                n_used2[r][c][0] = n_neighs
        check_n_used_neighs(fgrid2, n_used2)

        # Verify Grid #3
        # Only cell (row, col) = (1, 2) has a channel in use (ch9)
        n_free3 = np.ones((self.rows, self.cols)) * self.n_channels
        n_free3[1][2] = self.n_channels - 1
        # Cell (1, 2) has no neighs that use ch9. The neighs of (1, 2)
        # has 1 cell that use ch9.
        n_used3 = np.zeros((self.rows, self.cols, self.n_channels))
        neighs3 = RhombusAxialGrid.neighbors(4, 1, 2, separate=True)
        n_used3[(*neighs3, np.repeat(9, len(neighs3[0])))] = 1
        check_n_used_neighs(fgrid3, n_used3)
