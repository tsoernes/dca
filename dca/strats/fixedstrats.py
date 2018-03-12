import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF
from strats.base import Strat


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
            free = NGF.get_eligible_chs(self.grid, next_cell)
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

    def get_action(self, next_cevent, *args):
        ce_type, next_cell = next_cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            # When a call arrives in a cell,
            # if any pre-assigned channel is unused;
            # it is assigned, else the call is blocked.
            for ch, isNom in enumerate(GF.nom_chs_mask[next_cell]):
                if isNom and self.grid[next_cell][ch] == 0:
                    return ch
            return None
        elif ce_type == CEvent.END:
            # No rearrangement is done when a call terminates.
            return next_cevent[3]
