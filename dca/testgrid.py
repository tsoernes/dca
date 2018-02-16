import unittest

import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF


class TestAfterstates(unittest.TestCase):
    """
    Assumptions:
        n_used_neighs (frep[:-1]) does NOT include self
        n_free_self (frep[-1]) counts ELIGIBLE chs
    """

    def setUp(self):
        pass

    def test_afterstate_freps(self):
        """
        Test incremental vs naive afterstate feature rep. derivation approach
        """

        def check_frep(frep1, frep2):
            self.assertTrue(frep1.shape == frep2.shape, (frep1.shape, frep2.shape))
            self.assertTrue(frep2.shape == (7, 7, 71))
            eq_n_used = frep1[:, :, -1] == frep2[:, :, -1]
            diff_n_used = np.where(np.invert(eq_n_used))
            self.assertTrue(eq_n_used.all(),
                            ("\n", diff_n_used, frep1[:, :, -1], frep2[:, :, -1]))
            eq_n_free = frep1[:, :, :-1] == frep2[:, :, :-1]
            diff_n_free = np.where(np.invert(eq_n_free))
            self.assertTrue(eq_n_free.all(),
                            (diff_n_free, frep1[diff_n_free], frep2[diff_n_free]))

        def check_freps(freps1, freps2):
            self.assertTrue(freps1.shape == freps2.shape, (freps1.shape, freps2.shape))
            self.assertTrue(freps2.shape[1:] == (7, 7, 71))
            eq_n_free = freps1[:, :, :, :-1] == freps2[:, :, :, :-1]
            diff_n_free = np.where(np.invert(eq_n_free))
            self.assertTrue(eq_n_free.all(),
                            (diff_n_free, freps1[diff_n_free], freps2[diff_n_free]))
            eq_n_used = freps1[:, :, :, -1] == freps2[:, :, :, -1]
            diff_n_used = np.where(np.invert(eq_n_used))
            self.assertTrue(eq_n_used.all(),
                            ("\n", diff_n_used, freps1[:, :, :, -1], freps2[:, :, :, -1]))

        rows, cols, n_channels = 7, 7, 70
        grid1 = np.zeros((rows, cols, n_channels), dtype=bool)
        grid1c = np.copy(grid1)
        cell1 = (2, 3)
        ce_type1 = CEvent.NEW
        chs1 = GF.get_eligible_chs(grid1, cell1)
        freps_inc1 = GF.afterstate_freps(grid1, cell1, ce_type1, chs1)
        freps_inc1b = NGF.afterstate_freps(grid1, cell1, ce_type1, chs1)
        astates1 = GF.afterstates(grid1, cell1, ce_type1, chs1)
        freps1 = GF.feature_reps(astates1)
        frep1b = NGF.feature_rep(astates1[0])
        check_frep(freps1[0], frep1b)
        check_freps(freps_inc1, freps1)
        check_freps(freps_inc1b, freps1)
        assert (grid1 == grid1c).all()

        grid2 = np.ones((rows, cols, n_channels), dtype=bool)
        cell2 = (2, 3)
        grid2[cell2][4] = 0
        grid2[(2, 4)][:] = 0
        grid2c = np.copy(grid2)
        ce_type2 = CEvent.END
        chs2 = np.nonzero(grid2[cell2])[0]
        freps_inc2 = GF.afterstate_freps(grid2, cell2, ce_type2, chs2)
        freps_inc2b = NGF.afterstate_freps(grid2, cell2, ce_type2, chs2)
        astates2 = GF.afterstates(grid2, cell2, ce_type2, chs2)
        freps2 = GF.feature_reps(astates2)
        check_freps(freps_inc2, freps2)
        check_freps(freps_inc2b, freps2)
        assert (grid2 == grid2c).all()

        grid3 = np.zeros((rows, cols, n_channels), dtype=bool)
        cell3 = (4, 1)
        grid3[cell3][4] = 1
        grid3c = np.copy(grid3)
        ce_type3 = CEvent.END
        chs3 = np.nonzero(grid3[cell3])[0]
        freps_inc3 = GF.afterstate_freps(grid3, cell3, ce_type3, chs3)
        freps_inc3b = NGF.afterstate_freps(grid3, cell3, ce_type3, chs3)
        astates3 = GF.afterstates(grid3, cell3, ce_type3, chs3)
        freps3 = GF.feature_reps(astates3)
        check_freps(freps_inc3, freps3)
        check_freps(freps_inc3b, freps3)
        assert (grid3 == grid3c).all()

    def test_feature_rep(self):
        def check_n_free_self(grid, n_free):
            self.assertTrue((grid[:, :, -1] == n_free).all(), (grid[:, :, -1], n_free))

        def check_n_used_neighs(grid, n_used):
            self.assertTrue((grid[:, :, :-1] == n_used).all())

        rows, cols, n_channels = 7, 7, 70
        # Three grids in one array. They should not affect each other's
        # feature representation
        grid1 = np.zeros((rows, cols, n_channels), dtype=bool)
        grid2 = np.zeros((rows, cols, n_channels), dtype=bool)
        grid3 = np.zeros((rows, cols, n_channels), dtype=bool)
        grid2[:, :, 0] = 1
        grid3[1, 2, 9] = 1
        grid3c = np.copy(grid3)
        fgrid1 = GF.feature_reps(grid1)[0]
        fgrid2 = GF.feature_reps(grid2)[0]
        fgrid3 = GF.feature_reps(grid3)[0]

        # Verify that single- and multi-version works the same
        grids = np.array([grid1, grid2, grid3])
        fgrids = GF.feature_reps(grids)
        assert (grid3 == grid3c).all()
        self.assertTrue((fgrids[0] == fgrid1).all())
        self.assertTrue((fgrids[1] == fgrid2).all())
        self.assertTrue((fgrids[2] == fgrid3).all())

        # TODO NOTE TODO
        # Check if tests below are valid if frep is for ELIGIBLE not
        # FREE chs. Then, if easy to do, mod feature_reps func to do
        # the latter.

        # Verify Grid #1
        # No cell has any channels in use, i.e. all are free
        check_n_free_self(fgrid1, np.ones((rows, cols)) * n_channels)
        # No cell has a channel in use by any of its neighbors
        check_n_used_neighs(fgrid1, np.zeros((rows, cols, n_channels)))

        # Verify Grid #2
        # All cells have one channel in use
        check_n_free_self(fgrid2, np.ones((rows, cols)) * (n_channels - 1))
        # Every cell has 'n_neighs(cell)' neighbors4 who uses channel 0
        # ('n_neighs(cell)' depends on cell coordinates)
        n_used2 = np.zeros((rows, cols, n_channels))
        for r in range(rows):
            for c in range(cols):
                n_neighs = len(GF.neighbors(4, r, c))
                n_used2[r][c][0] = n_neighs + 1
        check_n_used_neighs(fgrid2, n_used2)

        # Verify Grid #3
        # Only cell (row, col) = (1, 2) has a channel in use (ch9)
        n_free3 = np.ones((rows, cols)) * n_channels
        n_free3[1][2] = n_channels - 1
        # Cell (1, 2) has no neighs that use ch9. The neighs of (1, 2)
        # has 1 cell that use ch9.
        n_used3 = np.zeros((rows, cols, n_channels))
        neighs3 = GF.neighbors(4, 1, 2, separate=True, include_self=True)
        n_used3[(*neighs3, np.repeat(9, len(neighs3[0])))] = 1
        check_n_used_neighs(fgrid3, n_used3)


class TestNumbaGrid(unittest.TestCase):
    def setUp(self):
        pass

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_neigh_indexing(self):
        """If neighs are np arrays, they index differently than tuples"""
        NGF.get_eligible_chs(np.zeros((7, 7, 70), dtype=np.bool), (3, 2))
        somegrid = np.random.uniform(size=(7, 7, 70))
        n1 = somegrid[NGF.neighbors_sep(2, 3, 2, False)]
        n2 = somegrid[GF.neighbors(2, 3, 2, separate=True, include_self=False)]
        assert (n1 == n2).all()
        n1 = somegrid[NGF.neighbors(2, 3, 2, False)[0]]
        n2 = somegrid[GF.neighbors(2, 3, 2, include_self=False)[0]]
        assert (n1 == n2).all()

    def test_neighs(self):
        for r in range(7):
            for c in range(7):
                e1 = NGF.neighbors(2, r, c, False)
                e2 = GF.neighbors(2, r, c)
                assert (e1 == e2)
                for d in [1, 2, 4]:
                    n1 = NGF.neighbors_sep(d, r, c, False)
                    n2 = GF.neighbors(d, r, c, separate=True, include_self=False)
                    assert ((n1[0] == n2[0]).all())
                assert ((n1[1] == n2[1]).all())

    def test_get_free_chs(self):
        grid = np.ones((7, 7, 70)).astype(bool)
        chs = [0, 4]
        cell = (3, 4)
        grid[cell][chs] = 0
        for n in NGF.neighbors(2, *cell, False):
            grid[n][chs] = 0
        free = NGF.get_eligible_chs(grid, cell)
        self._li_set_eq(free, chs)


# class TestRectOffsetGrid(unittest.TestCase):
class TestRectOffsetGrid():
    """I think the neighbor idxs are hard coded for rect offset"""

    def setUp(self):
        self.grid = np.zeros((7, 7, 70), np.bool)

    def test_neighbors1(self):
        neighs = self.grid.neighbors1
        # yapf: disable
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
        # yapf: enable

    def test_neighbors1sparse(self):
        neighs = self.grid.neighbors1sparse
        # yapf: disable
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
        # yapf: enable

    def test_neighbors2(self):
        neighs = self.grid.neighbors2
        # yapf: disable
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
        # yapf: enable

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_partition_cells(self):
        self.grid._partition_cells()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                label = self.grid.labels[r][c]
                neighs = self.grid.neighbors2(r, c)
                for neigh in neighs:
                    self.assertNotEqual(label, self.grid.labels[neigh])

    def test_validate_reuse_constr(self):
        self.assertTrue(self.grid.validate_reuse_constr())
        cell = (2, 3)
        ch = 4

        self.grid.state[cell][ch] = 1
        self.assertTrue(self.grid.validate_reuse_constr())

        self.grid.state[(0, 1)][ch] = 1
        self.assertTrue(self.grid.validate_reuse_constr())

        self.grid.state[(1, 4)][ch + 1] = 1
        self.grid.state[(1, 4)][ch - 1] = 1
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
        free = self.grid.get_eligible_chs(cell)
        self._li_set_eq(free, chs)


# class TestFixedGrid(unittest.TestCase):
class TestFixedGrid():
    def setUp(self):
        # self.grid = FixedGrid(rows=7, cols=7, n_channels=70, logger=logging.getLogger(""))
        pass

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
                f = np.bitwise_or(self.grid.state[r][c], self.grid.state[neighs[0]])
                for n in neighs[1:]:
                    f = np.bitwise_or(f, self.grid.state[n])
                # There should not be any channels that are free
                # in the cell (r, c) and all of its neighbors
                if np.any(f == 0):
                    self.grid.print_neighs2(r, c)
                    self.assertTrue(False, "Grid not maximum utilized under FA")


# class TestBDCLGrid(unittest.TestCase):
class TestBDCLGrid():
    def setUp(self):
        # self.grid = BDCLGrid(rows=7, cols=7, n_channels=70, logger=logging.getLogger(""))
        pass

    def _li_set_eq(self, targ, act):
        self.assertSetEqual(set(targ), set(act))

    def test_cochannel_cells(self):
        raise NotImplementedError  # not implemented correctly
        self.grid._partition_cells()
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (3, 3)), [(1, 3), (3, 5)])
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (4, 3)), [(3, 5), (5, 4)])
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (4, 2)), [(5, 4), (6, 1)])
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (4, 1)), [(6, 1)])
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (3, 1)), [(1, 0)])
        self._li_set_eq(self.grid.cochannel_cells((3, 2), (2, 2)), [(1, 0), (1, 3)])


if __name__ == '__main__':
    unittest.main()
