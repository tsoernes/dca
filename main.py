from eventgen import CEvent, EventGen
from gui import Gui
from grid import FixedGrid

from heapq import heappush, heappop
import operator

import numpy as np


lr = 0.8  # Learning rate
y = 0.95  # Gamma (discount factor)


# class Params:
#     """
#     Problem parameters. Immutable.
#     """
#     rows = attr.ib()
#     cols = attr.ib()
#     n_channels = attr.ib()
#     call_rates = attr.ib()
#     call_duration = attr.ib()
#     n_episodes = attr.ib()


class Strat:
    def __init__(self, pp, eventgen, grid, gui=None,
                 *args, **kwargs):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.n_episodes = pp['n_episodes']
        self.grid = grid
        self.cevents = []  # Call events
        self.eventgen = None
        self.t = 0  # Current time, in minutes

    def fn_new(self, row, col):
        """
        Assign incoming call in cell in row @row@ column @col@ to a channel.
        Return the channel assigned; -1 if unable to assign a channel.
        """
        raise NotImplementedError()

    def fn_end(self):
        """
        Possibly reassign current calls
        """
        raise NotImplementedError()


class FAStrat(Strat):
    """
    Fixed assignment (FA) channel allocation.
    The set of channels is partitioned, and the partitions are permanently
    assigned to cells so that all cells can use all the channels assigned
    to them simultaneously without interference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fn_new(self, row, col):
        # When a call arrives in a cell,
        # if any pre-assigned channel is unused;
        # it is assigned, else the call is blocked.
        ch = -1
        for idx, isNom in enumerate(self.grid.nom_chs[row][col]):
            if isNom and self.grid.state[row][col][idx] == 0:
                ch = idx
                break
        return ch

    def fn_end(self):
        # No rearrangement is done when a call terminates.
        pass


class RLStrat(Strat):
    def __init__(self, alpha, epsilon):
        """
        :param float alpha - learning rate
        :param float epsilon - best action is selected
            with probability (1-epsilon)
        """
        # "qvals[r][c][n][ch] = v"
        # Assigning channel 'c' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols,
                              self.n_channels, self.n_channels))
        self.epsilon = epsilon
        self.alpha = alpha

    def simulate(self):
        n_rejected = 0  # Number of rejected calls
        n_incoming = 0  # Number of incoming (not necessarily accepted) calls
        # Generate initial call events; one for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                heappush(self.cevents, self.eventgen.event_new(0, r, c))
        prev_cevent = heappop(self.cevents)
        prev_cell = prev_cevent[2]
        prev_n_used, prev_ch = self.optimal_ch(prev_cell, prev_cevent[1])
        # Discrete event simulation
        for _ in range(self.n_episodes):
            t = prev_cevent[0]
            prev_qval = self.qvals[prev_cell][prev_n_used][prev_ch]
            # Take action A, observe R, S'
            self.execute_action(prev_cevent, prev_ch)
            reward = self.reward()
            if prev_cevent[1] == CEvent.NEW:
                n_incoming += 1
                # Generate next incoming call
                heappush(self.cevents, self.eventgen.event_new(t, prev_cell))
                if prev_ch == -1:
                    n_rejected += 1
                    print(f"Rejected call to {prev_cell} when \
                          {prev_n_used} \
                          of {self.n_channels} in use")
                    if self.gui:
                        self.gui.hgrid.mark_cell(prev_cell)
                else:
                    # Generate call duration for incoming call and add event
                    heappush(self.cevents,
                             self.eventgen.event_end(
                                self.t, prev_cell, prev_ch))
            cevent = heappop(self.cevents)
            print(cevent)
            t = cevent[0]
            cell = cevent[2]

            if not self.grid.validate_reuse_constr():
                print("Reuse constraint broken: {self.grid}")
                raise Exception
            if self.gui:
                self.gui.step()

            # Choose A' from S'
            n_used, ch = self.optimal_ch(cell, cevent[1])
            qval = self.qvals[cell][n_used][ch]
            dt = cevent[0] - self.t  # Time until next event
            td_err = reward + self.discount(dt) * qval - prev_qval
            self.qvals[prev_cell][prev_n_used][prev_ch] += self.alpha * td_err

        print(f"Rejected {n_rejected} of {n_incoming} calls")
        print(f"Blocking probability: {n_rejected/n_incoming}")
        print(f"{np.sum(self.grid.state)} calls in progress at simulation end")

    def execute_action(self, cevent, ch):
        assert ch != -1
        cell = cevent[2]
        if cevent[1] == CEvent.NEW:
            if self.grid.state[cell][ch]:
                print("Tried assigning new call {cevent} to \
                        channel {ch} which is already in use")
                raise Exception()

            print(f"Assigned {ch} to {cell}")
            # Add incoming call to current state
            self.grid.state[cell][ch] = 1
        else:
            print(f"Reassigned {ch} to {cevent[3]} in {cell}")
            # Reassign 'ch' to the channel of the terminating call
            self.grid.state[cell][cevent[3]] = 1
            self.grid.state[cell][ch] = 0
            if self.gui:
                self.gui.hgrid.unmark_cell(*cell)

    def optimal_ch(self, ce_type, cell):
        """
        Select the channel fitting for assignment or termination
        that has the maximum (op=gt) or minimum (op=lt) value
        in an epsilon-greedy fasion.
        Return (n_used, ch) where n_used is the number of used channels
        in the event cell before any potential action is taken.
        """
        if ce_type == CEvent.NEW:
            # Find the set of channels that's not in use by this cell
            # or any of its neighbors within a radius of 2.
            potential_chs = np.where(self.state.grid[cell] == 0)[0]
            neighs = self.grid.neighbors2(*cell)
            chs = []  # Channels not in use
            for pch in potential_chs:
                in_use = False
                for neigh in neighs:
                    if self.grid.state[neigh][pch] == 1:
                        in_use = True
                        break
                if not in_use:
                    chs.append(pch)
            n_used = self.n_channels - len(potential_chs)
            op = operator.gt
        else:
            chs = np.nonzero(self.state.grid[cell])[0]  # Channels in use
            n_used = len(chs)
            op = operator.lt

        ch = -1
        if np.random.random() < self.epsilon:
            # Choose an available channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            best_val = 0
            for chan in chs:
                val = self.value[cell][n_used][chan]
                if op(val, best_val):
                    best_val = val
                    ch = chan
        return (n_used, ch)

    def reward(self, action, dt):
        """
        Immediate reward
        dt: Time until next event
        """
        return np.sum(self.state.grid)

    def discount(self, dt):
        """
        Discount factor (gamma)
        """
        # TODO: Find examples (in literature) where
        # gamma is a function of time until next event.
        # How should gamma increase as a function of dt?
        # Linearly, exponentially?
        # discount(0) should probably be 0
        return 0.8


pp = {
        'rows': 7,
        'cols': 7,
        'n_channels': 70,
        'call_rates': 150/60,  # Avg. call rate, in calls per minute
        'call_duration': 3,  # Avg. call duration in minutes
        'n_episodes': 10000
        }


def show():
    grid = FixedGrid(**pp)
    gui = Gui(grid)
    gui.test()


def run():
    grid = FixedGrid(**pp)
    grid.assign_chs()
    eventgen = EventGen(**pp)
    gui = Gui(grid)
    fa_strat = FAStrat(pp, grid=grid, gui=gui, eventgen=eventgen)
    fa_strat.simulate(pp, grid, fa_strat, eventgen, gui)


if __name__ == '__main__':
    run()

# TODO: Sanity checks:
# - The number of accepcted new calls minus the number of ended calls
# minus the number of rejected calls should be equal to the number of calls in
# progress.
