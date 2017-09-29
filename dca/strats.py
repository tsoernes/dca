from eventgen import CEvent, ce_str

from heapq import heappush, heappop
import operator
import time

import numpy as np
import matplotlib as plt


lr = 0.8  # Learning rate
y = 0.95  # Gamma (discount factor)


class Strat:
    def __init__(self, pp, eventgen, grid, logger,
                 gui=None,
                 *args, **kwargs):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.n_episodes = pp['n_episodes']
        self.p_handoff = pp['p_handoff']
        self.sanity_check = pp['sanity_check']
        self.log_iter = pp['log_iter']
        self.grid = grid
        self.eventgen = eventgen
        self.gui = gui
        self.logger = logger

        self.cevents = []  # Call events
        self.quit_sim = False

    def simulate(self):
        start_time = time.time()
        n_rejected = 0  # Number of rejected calls
        n_ended = 0  # Number of ended calls
        n_handoffs = 0  # Number of handoff events
        n_handoffs_rejected = 0  # Number of rejected handoff calls
        n_incoming = 0  # Number of incoming (not necessarily accepted) calls
        # Number of channels in progress at a cell when call is blocked
        n_inuse_rej = 0
        n_curr_rejected = 0  # Number of rejected calls last 100 episodes
        n_curr_incoming = 0  # Number of incoming calls last 100 episodes

        # Generate initial call events; one for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                heappush(self.cevents, self.eventgen.event_new(0, (r, c)))

        cevent = heappop(self.cevents)
        ch = self.get_init_action(cevent)

        # Discrete event simulation
        for i in range(self.n_episodes):
            if self.quit_sim:
                break  # Gracefully quit to print stats

            t, ce_type, cell = cevent[0:3]
            self.logger.debug(f"{t:.2f}: {cevent[1].name} {cevent[2:]}")

            if ch:
                self.execute_action(cevent, ch)
            n_used = np.sum(self.grid.state[cell])

            if self.sanity_check and not self.grid.validate_reuse_constr():
                self.logger.error(f"Reuse constraint broken: {self.grid}")
                raise Exception
            if self.gui:
                self.gui.step()

            if ce_type == CEvent.NEW:
                n_incoming += 1
                n_curr_incoming += 1
                # Generate next incoming call
                heappush(self.cevents, self.eventgen.event_new(t, cell))
                if not ch:
                    n_rejected += 1
                    n_curr_rejected += 1
                    n_inuse_rej += n_used
                    self.logger.debug(
                            f"Rejected call to {cell} when {n_used}"
                            f" of {self.n_channels} channels in use")
                    if self.gui:
                        self.gui.hgrid.mark_cell(*cell)
                else:
                    # With some probability, generate a handoff-event
                    # instead of ending the call
                    if np.random.random() < self.p_handoff:
                        # Generate handoff event
                        (end, hoff) = self.eventgen.event_new_handoff(
                                     t, cell, self.grid.neighbors1(*cell), ch)
                        heappush(self.cevents, end)
                        heappush(self.cevents, hoff)
                    else:
                        # Generate call duration for call and add end event
                        heappush(self.cevents,
                                 self.eventgen.event_end(t, cell, ch))
            elif ce_type == CEvent.HOFF:
                n_handoffs += 1
                if not ch:
                    n_handoffs_rejected += 1
                else:
                    # Generate call duration for call and add end event
                    heappush(self.cevents,
                             self.eventgen.event_end_handoff(t, cell, ch))
            elif ce_type == CEvent.END:
                n_ended += 1

            next_cevent = heappop(self.cevents)
            next_ch = self.get_action(next_cevent, cell, ch)
            ch, cevent = next_ch, next_cevent

            if i > 0 and i % self.log_iter == 0:
                self.logger.info(
                        f"\nt{t:.2f}: Blocking probability events"
                        f" {i-self.log_iter}-{self.log_iter}:"
                        f" {n_curr_rejected/(n_curr_incoming+1):.4f}")
                if hasattr(self, 'epsilon'):
                    self.logger.info(
                        f"\nn{i}: Epsilon: {self.epsilon:.5f},"
                        f" Alpha: {self.alpha:.5f}")
                n_curr_rejected = 0
                n_curr_incoming = 0

        n_inprogress = np.sum(self.grid.state)
        if (n_incoming + n_handoffs - n_ended - n_rejected
                - n_handoffs_rejected) != n_inprogress:
            self.logger.error(
                    f"\nSome calls were lost."
                    f" accepted: {n_incoming}, ended: {n_ended}"
                    f" rejected: {n_rejected}, in progress: {n_inprogress}")
        self.logger.warn(
            f"\nSimulation duration: {t/24:.2f} hours?,"
            f" {self.n_episodes} episodes"
            f" at {self.n_episodes/(time.time()-start_time):.0f}"
            " episodes/second"
            f"\nRejected {n_rejected} of {n_incoming} new calls,"
            f" {n_handoffs_rejected} of {n_handoffs} handoffs"
            f"\nBlocking probability: {n_rejected/n_incoming:.4f} new calls,"
            f" {n_handoffs_rejected/n_handoffs:.4f} for handoffs"
            f"\nAverage number of calls in progress when blocking: "
            f"{n_inuse_rej/(n_rejected+1):.2f}"  # avoid zero division
            f"\n{n_inprogress} calls in progress at simulation end\n")

    def get_init_action(self):
        raise NotImplementedError()

    def get_action(self):
        raise NotImplementedError()

    def execute_action(self):
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
        self.get_init_action = self.get_action

    def get_action(self, next_cevent, *args):
        ce_type, next_cell = next_cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            # When a call arrives in a cell,
            # if any pre-assigned channel is unused;
            # it is assigned, else the call is blocked.
            ch = None
            for idx, isNom in enumerate(self.grid.nom_chs[next_cell]):
                if isNom and self.grid.state[next_cell][idx] == 0:
                    ch = idx
                    break
            return ch
        elif ce_type == CEvent.END:
            # No rearrangement is done when a call terminates.
            return next_cevent[3]

    def execute_action(self, cevent, ch):
        ce_type, cell = cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            self.grid.state[cell][ch] = 1
        else:
            self.grid.state[cell][ch] = 0


class RLStrat(Strat):
    def __init__(self, pp, *args, **kwargs):
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

        self.n_used = 0
        self.qval = 0

    def update_qval():
        raise NotImplementedError

    def get_qval():
        raise NotImplementedError

    def get_init_action(self, cevent):
        _, ch = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, cell, ch):
        """
        Return a channel to be (re)assigned for 'next_cevent'.
        'cell' and 'ch' specify the previous channel (re)assignment.
        """
        # Observe reward from previous action
        reward = self.reward()
        next_cell = next_cevent[2]
        # Choose A' from S'
        next_n_used, next_ch = self.optimal_ch(next_cevent[1], next_cell)
        # Update q-values with one-step lookahead
        next_qval = self.get_qval(next_cell, next_n_used, next_ch)
        dt = -1  # how to calculate this?
        td_err = reward + self.discount(dt) * next_qval - self.qval
        self.update_qval(cell, self.n_used, ch, td_err)
        self.alpha *= self.alpha_decay

        self.n_used, self.qval = next_n_used, next_qval
        return next_ch

    def execute_action(self, cevent, ch):
        """
        Change the grid state according to the given action
        """
        ce_type, cell = cevent[1:3]
        if ce_type == CEvent.NEW:
            if self.grid.state[cell][ch]:
                self.logger.error(
                    f"Tried assigning new call {ce_str(cevent)} to"
                    f" channel {ch} which is already in use")
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
        inuse = np.nonzero(self.grid.state[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            neighs = self.grid.neighbors2(*cell)
            inuse_neigh = np.bitwise_or(
                    self.grid.state[cell], self.grid.state[neighs[0]])
            for n in neighs[1:]:
                inuse_neigh = np.bitwise_or(inuse_neigh, self.grid.state[n])
            chs = np.where(inuse_neigh == 0)[0]
            op = operator.gt
            best_val = float("-inf")
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            op = operator.lt
            best_val = float("inf")

        if len(chs) == 0:
            # No channels available for assignment,
            # or no channels in use to reassign
            return (n_used, None)

        # Might do Greedy in the LImit of Exploration (GLIE) here,
        # like Boltzmann Exploration with decaying temperature.
        if np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            for chan in chs:
                val = self.get_qval(cell, n_used, chan)
                if op(val, best_val):
                    best_val = val
                    ch = chan
        self.logger.debug(
                f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return (n_used, ch)

    def reward(self):
        """
        Immediate reward
        dt: Time until next event
        """
        # Number of calls currently in progress
        # TODO try +1 for accepted and -1 for rejected instead
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


class SARSAStrat(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "qvals[r][c][n][ch] = v"
        # Assigning channel 'c' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols,
                              self.n_channels, self.n_channels))

    def get_qval(self, cell, n_used, ch):
        return self.qvals[cell][n_used][ch]

    def update_qval(self, cell, n_used, ch, td_err):
        self.qvals[cell][n_used][ch] += self.alpha * td_err


class TTSARSAStrat(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # consistent low 7%, sometimes 6% block prob
        # Maximum Number of used channels in a cell in the table.
        # If the actual number is higher, it gets 'merged' to k.
        self.k = 30
        self.qvals = np.zeros((self.rows, self.cols, self.k, self.n_channels))

    def get_qval(self, cell, n_used, ch):
        return self.qvals[cell][min(self.k-1, n_used)][ch]

    def update_qval(self, cell, n_used, ch, td_err):
        self.qvals[cell][min(self.k-1, n_used)][ch] += self.alpha * td_err


class RSSARSAStrat(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels))

    def qval_reduced(self, cell, n_used, ch):
        return self.qvals[cell][ch]

    def update_qval_reduced(self, cell, n_used, ch, td_err):
        self.qvals[cell][ch] += self.alpha * td_err

# todo: plot block-rate over time to determine
# if if rl system actually improves over time

# TODO verify the rl sim loop. is it correct?
# can it be simplified, e.g. remove fn_init?
