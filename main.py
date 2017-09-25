from eventgen import CEvent, EventGen
from gui import Gui
from grid import FixedGrid, Grid
from pparams import mk_pparams

import cProfile
from heapq import heappush, heappop
import operator
import time
import logging

import numpy as np
import matplotlib as plt


lr = 0.8  # Learning rate
y = 0.95  # Gamma (discount factor)


class Strat:
    def __init__(self, pp, eventgen, grid, logger, gui=None,
                 sanity_check=True,
                 *args, **kwargs):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.n_episodes = pp['n_episodes']
        self.grid = grid
        self.cevents = []  # Call events
        self.eventgen = eventgen
        self.gui = gui
        self.sanity_check = sanity_check
        self.quit_sim = False
        self.logger = logger

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
    def __init__(self, pp, version='full', *args, **kwargs):
        """
        :param float alpha - learning rate
        :param float epsilon - best action is selected
            with probability (1-epsilon)
        """
        super().__init__(pp, *args, **kwargs)
        self.epsilon = pp['epsilon']
        self.epsilon_decay = pp['epsilon_decay']
        self.alpha = pp['alpha']
        self.alpha_decay = pp['alpha_decay']
        self.gamma = pp['gamma']
        # "qvals[r][c][n][ch] = v"
        # Assigning channel 'c' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n' channels are already
        # in use at that cell.
        if version == 'full':
            self.qvals = np.zeros((self.rows, self.cols,
                                  self.n_channels, self.n_channels))
            self.qval = self.qval_full
            self.update_qval = self.update_qval_full
        elif version == 'trimmed':
            self.qvals = np.zeros((self.rows, self.cols,
                                  30, self.n_channels))
            self.qval = self.qval_trimmed
            self.update_qval = self.update_qval_trimmed
        elif version == 'reduced':
            self.qvals = np.zeros((self.rows, self.cols,
                                   self.n_channels))
            self.qval = self.qval_reduced
            self.update_qval = self.update_qval_reduced

    def simulate(self):
        start_time = time.time()
        n_rejected = 0  # Number of rejected calls
        n_incoming = 0  # Number of incoming (not necessarily accepted) calls
        # Number of channels in progress at a cell when call is blocked
        n_inuse_rej = 0
        n_curr_rejected = 0  # Number of rejected calls last 100 episodes
        n_curr_incoming = 0  # Number of incoming calls last 100 episodes

        # Generate initial call events; one for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                heappush(self.cevents, self.eventgen.event_new(0, (r, c)))
        prev_cevent = heappop(self.cevents)
        prev_cell = prev_cevent[2]
        prev_n_used, prev_ch = self.optimal_ch(prev_cevent[1], prev_cell)
        prev_qval = 0
        # Discrete event simulation
        for i in range(self.n_episodes):
            if self.quit_sim:
                break  # Gracefully quit to print stats

            t = prev_cevent[0]
            self.execute_action(prev_cevent, prev_ch)
            reward = self.reward()

            if self.sanity_check and not self.grid.validate_reuse_constr():
                self.logger.error(f"Reuse constraint broken: {self.grid}")
                raise Exception
            if self.gui:
                self.gui.step()

            if prev_cevent[1] == CEvent.NEW:
                n_incoming += 1
                n_curr_incoming += 1
                # Generate next incoming call
                heappush(self.cevents, self.eventgen.event_new(t, prev_cell))
                if prev_ch == -1:
                    n_rejected += 1
                    n_curr_rejected += 1
                    n_inuse_rej += prev_n_used
                    self.logger.debug(
                            f"Rejected call to {prev_cell} when {prev_n_used}"
                            f" of {self.n_channels} channels in use")
                    if self.gui:
                        self.gui.hgrid.mark_cell(*prev_cell)
                else:
                    # Generate call duration for call and add end event
                    heappush(self.cevents,
                             self.eventgen.event_end(t, prev_cell, prev_ch))

            cevent = heappop(self.cevents)
            t, e_type, cell = cevent[0], cevent[1], cevent[2]
            self.logger.debug(f"{t:.2f}: {e_type.name} {cevent[2:]}")

            # Choose A' from S'
            n_used, ch = self.optimal_ch(e_type, cell)
            # Update q-values with one-step lookahead
            qval = self.qval(cell, n_used, ch)
            dt = -1  # how to calculate this?
            td_err = reward + self.discount(dt) * qval - prev_qval
            self.update_qval(prev_cell, prev_n_used, prev_ch, td_err)
            self.alpha *= self.alpha_decay

            prev_cell = cell
            prev_cevent = cevent
            prev_n_used = n_used
            prev_ch = ch
            prev_qval = qval

            if i > 0 and i % 100000 == 0:
                self.logger.info(
                        f"\nt{t:.2f}: Blocking probability last 100000 events:"
                        f" {n_curr_rejected/(n_curr_incoming+1):.4f}")
                self.logger.info(
                        f"n{i}: Epsilon: {self.epsilon:.5f},"
                        f" Alpha: {self.alpha:.5f}")
                n_curr_rejected = 0
                n_curr_incoming = 0

        self.logger.warn(
            f"\nSimulation duration: {t/24:.2f} hours?,"
            f" {self.n_episodes} episodes"
            f" at {self.n_episodes/(time.time()-start_time):.0f}"
            " episodes/second"
            f"\nRejected {n_rejected} of {n_incoming} calls"
            f"\nBlocking probability: {n_rejected/n_incoming:.4f}"
            f"\nAverage number of calls in progress when blocking: "
            f"{n_inuse_rej/(n_rejected+1):.2f}"  # avoid zero division
            f"\n{np.sum(self.grid.state)} calls in progress at simulation end")

    def update_qval_trimmed(self, cell, n_used, ch, td_err):
        self.qvals[cell][max(29, n_used)][ch] += self.alpha * td_err

    def update_qval_full(self, cell, n_used, ch, td_err):
        self.qvals[cell][n_used][ch] += self.alpha * td_err

    def update_qval_reduced(self, cell, n_used, ch, td_err):
        self.qvals[cell][ch] += self.alpha * td_err

    def qval_trimmed(self, cell, n_used, ch):
        return self.qvals[cell][max(29, n_used)][ch]

    def qval_full(self, cell, n_used, ch):
        return self.qvals[cell][n_used][ch]

    def qval_reduced(self, cell, n_used, ch):
        return self.qvals[cell][ch]

    def execute_action(self, cevent, ch):
        """
        Change the grid state according to the given action
        """
        if ch == -1:
            return
        cell = cevent[2]
        if cevent[1] == CEvent.NEW:
            if self.grid.state[cell][ch]:
                self.logger.error(
                    f"Tried assigning new call {cevent} to"
                    f"channel {ch} which is already in use")
                raise Exception()

            self.logger.debug(f"Assigned ch {ch} to cell {cell}")
            # Add incoming call to current state
            self.grid.state[cell][ch] = 1
        else:
            self.logger.debug(
                    f"Reassigned ch {cevent[3]} to ch {ch} in cell {cell}")
            # Reassign 'ch' to the channel of the terminating call
            self.grid.state[cell][ch] = 1
            self.grid.state[cell][cevent[3]] = 0
            if self.gui:
                self.gui.hgrid.unmark_cell(*cell)

    def optimal_ch(self, ce_type, cell):
        """
        Select the channel fitting for assignment or termination
        that has the maximum (new) or minimum (end) value
        in an epsilon-greedy fasion.

        Return (n_used, ch) where 'n_used' is the number of channels in
        use before any potential action is taken.
        'ch' is -1 if no channel is eligeble for assignment
        """
        if ce_type == CEvent.NEW:
            # Free channels at cell
            potential_chs = np.where(self.grid.state[cell] == 0)[0]
            neighs = self.grid.neighbors2(*cell)
            chs = []  # Channels eligible for assignment
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
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = np.nonzero(self.grid.state[cell])[0]
            n_used = len(chs)
            op = operator.lt
            best_val = float("inf")

        if len(chs) == 0:
            # No channels available for assignment,
            # or no channels in use to reassign
            return (n_used, -1)

        # Might do Greedy in the LImit of Exploration (GLIE) here,
        # like Boltzmann Exploration with decaying temperature.

        # TODO Reduce epsilon. When and by how much?
        if np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            for chan in chs:
                # val = self.qvals[cell][n_used][chan]
                val = self.qval(cell, n_used, chan)
                if op(val, best_val):
                    best_val = val
                    ch = chan
        # print(f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        # Epsilon decay
        self.epsilon *= self.epsilon_decay
        return (n_used, ch)

    def reward(self):
        """
        Immediate reward
        dt: Time until next event
        """
        # Number of calls currently in progress
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
        return self.gamma


class Runner:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger('')
        fh = logging.FileHandler('out.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.pp = mk_pparams()
        self.logger.info(f"Starting simulation with params {self.pp}")

    def run(self, show_gui=False):
        grid = Grid(logger=self.logger, **self.pp)
        eventgen = EventGen(**self.pp)
        if show_gui:
            gui = Gui(grid, self.end_sim)
        else:
            gui = None
        self.strat = RLStrat(
                self.pp, grid=grid, gui=gui, eventgen=eventgen,
                version='trimmed',
                sanity_check=False, logger=self.logger)
        self.strat.simulate()

    def end_sim(self, e):
        """
        Handle key events from Tkinter and quit
        simulation gracefully on 'q'-key
        """
        self.strat.quit_sim = True

    def show(self):
        grid = FixedGrid(**self.pp)
        gui = Gui(grid)
        gui.test()

    def run_fa(self):
        grid = FixedGrid(**self.pp)
        grid.assign_chs()
        eventgen = EventGen(**self.pp)
        gui = Gui(grid)
        fa_strat = FAStrat(self.pp, grid=grid, gui=gui, eventgen=eventgen)
        fa_strat.simulate(self.pp, grid, fa_strat, eventgen, gui)


if __name__ == '__main__':
    r = Runner()
    cProfile.run('r.run()')
    # r.run()

# TODO: Sanity checks:
# - The number of accepcted new calls minus the number of ended calls
# minus the number of rejected calls should be equal to the number of calls in
# progress.

# todo: plot block-rate over time to determine
# if if rl system actually improves over time

