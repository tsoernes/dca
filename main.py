from eventgen import CEvent, EventGen
from gui import Gui
from grid import FixedGrid, Grid

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
        self.eventgen = eventgen
        self.gui = gui

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
    def __init__(self, pp, *args, **kwargs):
        """
        :param float alpha - learning rate
        :param float epsilon - best action is selected
            with probability (1-epsilon)
        """
        super().__init__(pp, *args, **kwargs)
        # "qvals[r][c][n][ch] = v"
        # Assigning channel 'c' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols,
                              self.n_channels, self.n_channels))
        self.epsilon = pp['epsilon']
        self.alpha = pp['alpha']

    def simulate(self):
        n_rejected = 0  # Number of rejected calls
        n_incoming = 0  # Number of incoming (not necessarily accepted) calls
        # Number of channels in progress at a cell when call is blocked
        n_inuse_rej = 0
        # Generate initial call events; one for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                heappush(self.cevents, self.eventgen.event_new(0, (r, c)))
        prev_cevent = heappop(self.cevents)
        prev_cell = prev_cevent[2]
        prev_n_used, prev_ch = self.optimal_ch(prev_cevent[1], prev_cell)
        prev_qval = 0
        # Discrete event simulation
        for _ in range(self.n_episodes):
            t = prev_cevent[0]
            # Take action A, observe R, S'
            self.execute_action(prev_cevent, prev_ch)

            if not self.grid.validate_reuse_constr():
                print("Reuse constraint broken: {self.grid}")
                raise Exception
            if self.gui:
                self.gui.step()

            reward = self.reward()
            if prev_cevent[1] == CEvent.NEW:
                n_incoming += 1
                # Generate next incoming call
                heappush(self.cevents, self.eventgen.event_new(t, prev_cell))
                if prev_ch == -1:
                    n_rejected += 1
                    n_inuse_rej += prev_n_used
                    print(f"Rejected call to {prev_cell} when {prev_n_used}"
                          "of {self.n_channels} channels in use")
                    if self.gui:
                        self.gui.hgrid.mark_cell(*prev_cell)
                else:
                    # Generate call duration for incoming call and add event
                    heappush(self.cevents,
                             self.eventgen.event_end(t, prev_cell, prev_ch))
            cevent = heappop(self.cevents)
            print(cevent)
            t = cevent[0]
            cell = cevent[2]

            # Choose A' from S'
            n_used, ch = self.optimal_ch(cevent[1], cell)
            # Update q-values with one-step lookahead
            qval = self.qvals[cell][n_used][ch]
            dt = -1  # how to calculate this?
            td_err = reward + self.discount(dt) * qval - prev_qval
            self.qvals[prev_cell][prev_n_used][prev_ch] += self.alpha * td_err

            prev_cell = cell
            prev_cevent = cevent
            prev_n_used = n_used
            prev_ch = ch
            prev_qval = qval

        print(f"Rejected {n_rejected} of {n_incoming} calls")
        print(f"Blocking probability: {n_rejected/n_incoming}")
        print(f"Average number of calls in progress when blocking"
              "{n_inuse_rej/n_rejected}")
        print(f"{np.sum(self.grid.state)} calls in progress at simulation end")

    def execute_action(self, cevent, ch):
        if ch == -1:
            return
        cell = cevent[2]
        if cevent[1] == CEvent.NEW:
            if self.grid.state[cell][ch]:
                print("Tried assigning new call {cevent} to \
                        channel {ch} which is already in use")
                raise Exception()

            print(f"Assigned {ch} to {cell}")
            # Add incoming call to current state
            self.grid.state[cell][ch] = 1
            # if not self.grid.validate_reuse_constr():
            #     print("Reuse constraint broken just after assigning \
            #             ch {ch} to cell {cell}")
            #     raise Exception
        else:
            # How does this work if
            # a) there's no channels to reassign, i.e. the end event
            # is for the only in-use channel in the cell
            # b) no channel should be reassigned
            print(f"Reassigned ch {cevent[3]} to ch {ch} in cell {cell}")
            # Reassign 'ch' to the channel of the terminating call
            self.grid.state[cell][ch] = 1
            self.grid.state[cell][cevent[3]] = 0
            if self.gui:
                self.gui.hgrid.unmark_cell(*cell)

    def optimal_ch(self, ce_type, cell):
        """
        Select the channel fitting for assignment or termination
        that has the maximum (op=gt) or minimum (op=lt) value
        in an epsilon-greedy fasion.
        Return (n_used, ch) where n_used is the number of used channels
        in the event cell before any potential action is taken.
        'ch' is -1 if no channel is eligeble for (re)assignment

        TODO: Compare value to the value of no reassignment at all
        TODO: It's not the channel selected for reassignment that's
        in violated the reuse constraint, rather the channel reassigned to.
        How did that happen? The channel reassigned to should not violate
        the constraint in the first place.
        """
        if ce_type == CEvent.NEW:
            # Free channels at cell
            potential_chs = np.where(self.grid.state[cell] == 0)[0]
            neighs = self.grid.neighbors2(*cell)
            chs = []  # Channels eligeble for assignment
            # Exclude channels in use in neighboring cells
            for pch in potential_chs:
                in_use = False
                for neigh in neighs:
                    if self.grid.state[neigh][pch]:
                        in_use = True
                        break
                if not in_use:
                    chs.append(pch)
            n_used = self.n_channels - len(potential_chs)
            op = operator.gt
            best_val = float("-inf")
        else:
            # The channel being reassigned does not need to be
            # available in neighboring cells,
            # because it's being reassigned to a channel
            # that's (supposed to) already be available there.
            # Channels in use at cell
            chs = np.nonzero(self.grid.state[cell])[0]
            n_used = len(chs)
            op = operator.lt
            best_val = float("inf")

        if len(chs) == 0:
            # No channels available for assignment,
            # or no channels in use to reassign
            return (n_used, -1)

        if np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            for chan in chs:
                val = self.qvals[cell][n_used][chan]
                if op(val, best_val):
                    best_val = val
                    ch = chan
        # print(f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (n_used, ch)

    def reward(self):
        """
        Immediate reward
        dt: Time until next event
        """
        return np.sum(self.grid.state)

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
        'call_rates': 200/60,  # Avg. call rate, in calls per minute
        'call_duration': 3,  # Avg. call duration in minutes
        'n_episodes': 100000,
        'epsilon': 0.1,
        'alpha': 0.01
        }


def show():
    grid = FixedGrid(**pp)
    gui = Gui(grid)
    gui.test()


def run():
    grid = Grid(**pp)
    eventgen = EventGen(**pp)
    gui = Gui(grid)
    strat = RLStrat(pp, grid=grid, gui=gui, eventgen=eventgen)
    strat.simulate()


def run_fa():
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

# a) END time is dt, not t
# b) which is probably the cause for events cycling new->end->new->..
