from grid import Grid
from main import FAStrat

import unittest

import numpy as np


pp = {
        'rows': 7,
        'cols': 7,
        'n_channels': 70,
        'call_rates': 150/60,  # Avg. call rate, in calls per minute
        'call_duration': 3,  # Avg. call duration in minutes
        'n_episodes': 10000
        }


class TestStrats(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(**pp)

    def test_fa_strat(self):
        fa_strat = FAStrat(**pp, grid=self.grid)
        for i in range(10):
            ch = fa_strat.fn_new(0, 0)
            self.assertNotEqual(ch, -1)
            self.grid.state[0][0][ch] = 1
        self.assertEqual(fa_strat.fn_new(0, 0), -1)


if __name__ == '__main__':
    unittest.main()
