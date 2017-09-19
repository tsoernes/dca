from eventgen import CEvent, EventGen
from gui import Gui
from grid import Grid

from heapq import heappush, heappop

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
    def __init__(self, rows, cols, n_channels,
                 grid, *args, **kwargs):
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels
        self.grid = grid

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
        super(FAStrat, self).__init__(*args, **kwargs)
        self.grid.assign_chs()

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


class FCState():
    def __init__(self):
        pass

    # Fully connected network; attempted replica of Singh
    def _state_frepr(self, state):
        """
        Feature representation of a state
        """
        frepr = np.zeros((self.rows, self.cols, self.n_channels+1))
        # Number of available channels for each cell
        frepr[:, :, -1] = self.n_channels - np.sum(self.state, axis=2)
        # The number of times each channel is used within a 1 cell radius,
        # not including self
        for i in range(self.rows):
            for j in range(self.cols):
                for ch in range(self.n_channels):
                    neighs = self.neighbors2(i, j)
                    for neigh in neighs:
                        frepr[i][j][ch] += self.state[neigh[0]][neigh[1]][ch]
        return frepr


class BDCLStrat(Strat):
    # Borrowing with Directional Channel Locking (BDCL) of Zhang & Yum (1989).
    def __init__(self, *args, **kwargs):
        super(BDCLStrat, self).__init__(*args, **kwargs)
        self.grid.assign_chs()

    def fn_new(self, row, col):
        """
        ON NEW CALL:
        """
        ch = -1
        # If a nominal channel is available when a call arrives in a cell,
        # the smallest numbered such channel is assigned to the call.
        for idx, isNom in enumerate(self.grid.nom_chs[row][col]):
            if isNom and self.state[row][col][idx]:
                ch = idx
                break
        if ch != -1:
            return ch

        # If no nominal channel is available, then the largest numbered
        # free channel is borrowed from the neighbour with
        # the most free channels.
        neigh_idxs = self.neighbors1()
        best_neigh = None
        best_n_free = 0
        for n_idx in neigh_idxs:
            n_free = self.n_channels - np.sum(self.state[n_idx])
            if n_free > best_n_free:
                best_n_free = n_free
                best_neigh = n_idx
        # When a channel is borrowed, careful accounting of the directional
        # effect of which cells can no longer use that channel because
        # of interference is done.
        # The call is blocked if there are no free channels at all.

        # Changing state (assigning call to cell and ch) to the
        # incoming call should not be done here, only rearrangement
        # of existing calls
        # TODO: rearrange existing and keep track of borrowed
        return ch

    def fn_end(self):
        """
        ON CALL TERM:
        When a call terminates in a cell and the channel so freed is a nominal
        channel, say numbered i, of that cell, then if there is a call
        in that cell on a borrowed channel, the call on the smallest numbered
        borrowed channel is reassigned to i and the borrowed channel
        is returned to the
        appropriate cell. If there is no call on a borrowed channel,
        then if there is a call on a nominal channel numbered larger than i,
        the call on the highest numbered nominal channel is reassigned to i.
        If the call just terminated was itself on a borrowed channel, the
        call on the smallest numbered borrowed channel is reassigned to it
        and that
        channel is returned to the cell from which it was borrowed.
        Notice that when a
        borrowed channel is returned to its original cell, a nominal
        channel becomes
        free in that cell and triggers a reassignment.
        """
        pass


class RLState(Strat):
    def __init__(self):
        self.value = np.zeros((self.rows, self.cols, self.n_channels+1))

    def fn_new(self):
        pass

    def fn_end(self):
        pass

    def reward(self, state, action, dt):
        """
        Immediate reward
        dt: Time until next event
        """
        pass

    def discount(self, dt):
        """
        Discount factor (gamma)
        """
        pass

    def value(self, avail, pack):
        """
        avail: The number of available channels per cell.
        pack: Packing feature.
              The number of times each channel is used within a 1 cell radius.
        """
        pass


def simulate(pp, grid, strat, eventgen, gui=None):
    """
    pp: Problem Parameters
    """
    t = 0  # Current time, in minutes
    n_rejected = 0  # Number of rejected calls
    n_incoming = 0  # Number of incoming (not necessarily accepted) calls
    cevents = []  # Call events in a min heap, sorted on time
    # Generate initial call events; one for each cell
    for r in range(pp['rows']):
        for c in range(pp['cols']):
            heappush(cevents, eventgen.event_new(0, r, c))
    # Discrete event simulation
    for _ in range(pp['n_episodes']):
        event = heappop(cevents)
        print(event)
        t = event[0]
        row = event[2][0]
        col = event[2][1]
        # Accept incoming call
        if event[1] == CEvent.NEW:
            n_incoming += 1
            # Assign channel to call
            ch = strat.fn_new(row, col)
            # Generate next incoming call
            heappush(cevents, eventgen.event_new(t, row, col))
            if ch == -1:
                n_rejected += 1
                print(f"Rejected call to {row}, {col} when \
                      {np.sum(grid.state[row][col])} \
                      of {pp['n_channels']} in use")
                if gui:
                    gui.hgrid.mark_cell(row, col)
            else:
                print(f"Assigned {ch} to {row}, {col}")
                # Generate call duration for incoming call and add event
                heappush(cevents, eventgen.event_end(t, event[2], ch))
                # Add incoming call to current state
                grid.state[row][col][ch] = 1
        elif event[1] == CEvent.END:
            strat.fn_end()
            # Remove call from current state
            grid.state[row][col][event[3]] = 0
            if gui:
                gui.hgrid.unmark_cell(row, col)
        if not grid.validate_reuse_constr():
            break
        if gui:
            gui.step()
    print(f"Rejected {n_rejected} of {n_incoming} calls")
    print(f"Blocking probability: {n_rejected/n_incoming}")
    print(f"{np.sum(grid.state)} calls in progress at simulation end")


pp = {
        'rows': 7,
        'cols': 7,
        'n_channels': 70,
        'call_rates': 150/60,  # Avg. call rate, in calls per minute
        'call_duration': 3,  # Avg. call duration in minutes
        'n_episodes': 10000
        }


def run():
    grid = Grid(**pp)
    fa_strat = FAStrat(**pp, grid=grid)
    eventgen = EventGen(**pp)
    gui = Gui(grid)
    gui.test()
    # simulate(pp, grid, fa_strat, eventgen, gui)


if __name__ == '__main__':
    run()

# TODO: Sanity checks:
# - The number of accepcted new calls minus the number of ended calls
# minus the number of rejected calls should be equal to the number of calls in
# progress.
