import math

import numpy as np

from eventgen import CEvent
from strats import Strat


class RandomAssign(Strat):
    """
    On call arrival, an eligible channel is picked
    at random.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_init_action = self.get_action

    def get_action(self, next_cevent, *args):
        ce_type, next_cell = next_cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            free = self.env.grid.get_free_chs(next_cell)
            if len(free) == 0:
                return None
            else:
                return np.random.choice(free)
        elif ce_type == CEvent.END:
            # No rearrangement is done when a call terminates.
            return next_cevent[3]


class FixedAssign(Strat):
    """
    Fixed assignment (FA) channel allocation.

    The set of channels is partitioned, and the partitions are permanently
    assigned to cells so that every cell can use all of its channels
    simultanously without interference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_init_action = self.get_action

        # Nominal channels for each cell
        self.nom_chs = np.zeros((self.rows, self.cols, self.n_channels), dtype=bool)
        self.assign_chs()

    def get_action(self, next_cevent, *args):
        ce_type, next_cell = next_cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            # When a call arrives in a cell,
            # if any pre-assigned channel is unused;
            # it is assigned, else the call is blocked.
            for ch, isNom in enumerate(self.nom_chs[next_cell]):
                if isNom and self.grid[next_cell][ch] == 0:
                    return ch
            return None
        elif ce_type == CEvent.END:
            # No rearrangement is done when a call terminates.
            return next_cevent[3]

    def assign_chs(self, n_nom_channels=0):
        """
        Partition the cells and channels up to and including 'n_nom_channels'
        into 7 lots, and assign
        the channels to cells such that they will not interfere with each
        other within a channel reuse constraint of 3.
        The channels assigned to a cell are its nominal channels.

        Returns a (rows*cols*n_channels) array
        where a channel for a cell has value 1 if nominal, 0 otherwise.
        """
        if n_nom_channels == 0:
            n_nom_channels = self.n_channels
        channels_per_subgrid_cell = []
        channels_per_subgrid_cell_accu = [0]
        channels_per_cell = n_nom_channels / 7
        ceil = math.ceil(channels_per_cell)
        floor = math.floor(channels_per_cell)
        tot = 0
        for i in range(7):
            if tot + ceil + (6 - i) * floor > n_nom_channels:
                tot += ceil
                cell_channels = ceil
            else:
                tot += floor
                cell_channels = floor
            channels_per_subgrid_cell.append(cell_channels)
            channels_per_subgrid_cell_accu.append(tot)
        for r in range(self.rows):
            for c in range(self.cols):
                label = self.env.grid.labels[r][c]
                lo = channels_per_subgrid_cell_accu[label]
                hi = channels_per_subgrid_cell_accu[label + 1]
                self.nom_chs[r][c][lo:hi] = 1
