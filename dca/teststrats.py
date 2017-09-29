from grid import Grid
from strats import FAStrat

import unittest

import numpy as np


class TestStrats(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(rows=7, cols=7, n_channels=70)

    def test_fa_strat(self):
        fa_strat = FAStrat(7, 7, 70, grid=self.grid)
        # Send 10 calls to a cell and check that (only) the last is blocked
        for i in range(10):
            ch = fa_strat.fn_new(0, 0)
            self.assertNotEqual(ch, -1)
            self.grid.state[0][0][ch] = 1
        self.assertEqual(fa_strat.fn_new(0, 0), -1)


if __name__ == '__main__':
    unittest.main()
