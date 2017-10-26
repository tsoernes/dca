from eventgen import EventGen, CEvent, ce_str
from stats import Stats

import signal
import sys
import inspect

import numpy as np


class Strat:
    def __init__(self, pp, grid, logger, pid="",
                 *args, **kwargs):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.n_events = pp['n_events']
        self.p_handoff = pp['p_handoff']
        self.verify_grid = pp['verify_grid']
        self.log_iter = pp['log_iter']
        self.pp = pp
        self.grid = grid
        self.logger = logger

        self.gui = None
        self.epsilon = None  # Not applicable for all strats
        self.alpha = None

        self.quit_sim = False
        self.stats = Stats(pp=pp, logger=logger, pid=pid)
        self.eventgen = EventGen(logger=logger, **pp)

    def exit_handler(self, *args):
        """
        Graceful exit allowing printing of stats on ctrl-c exit from
        command line or on 'q' key-event from gui.
        """
        self.logger.warn("\nPremature exit")
        self.quit_sim = True

    def init_sim(self):
        signal.signal(signal.SIGINT, self.exit_handler)
        # Generate initial call events; one for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                self.eventgen.event_new(0, (r, c))
        self._simulate()
        np.save("data-experience", np.array(self.experience))
        return self.stats.block_prob_tot

    def _simulate(self):
        cevent = self.eventgen.pop()
        ch = self.get_init_action(cevent)

        # Discrete event simulation
        for i in range(self.n_events):
            if self.quit_sim:
                break  # Gracefully exit to print stats

            t, ce_type, cell = cevent[0:3]
            self.stats.iter(t, i, cevent)

            n_used = np.count_nonzero(self.grid.state[cell])
            if ch is not None:
                s = np.copy(self.grid.state)
                self.execute_action(cevent, ch)

                if self.verify_grid and not self.grid.validate_reuse_constr():
                    self.logger.error(f"Reuse constraint broken")
                    raise Exception
            if self.gui:
                self.gui.step()

            # TODO Something seems off here. Why is the event checked
            # after it's executed? n_used has changed?
            if ce_type == CEvent.NEW:
                self.stats.new()
                # Generate next incoming call
                self.eventgen.event_new(t, cell)
                if ch is None:
                    self.stats.new_rej(cell, n_used)
                    if self.gui:
                        self.gui.hgrid.mark_cell(*cell)
                else:
                    # With some probability, generate a handoff-event
                    # instead of ending the call
                    if np.random.random() < self.p_handoff:
                        self.eventgen.event_new_handoff(
                            t, cell, ch, self.grid.neighbors1(*cell))
                    else:
                        # Generate call duration for call and add end event
                        self.eventgen.event_end(t, cell, ch)
            elif ce_type == CEvent.HOFF:
                self.stats.hoff_new()
                if ch is None:
                    self.stats.hoff_rej(cell, n_used)
                    if self.gui:
                        self.gui.hgrid.mark_cell(*cell)
                else:
                    # Generate call duration for call and add end event
                    self.eventgen.event_end_handoff(t, cell, ch)
            elif ce_type == CEvent.END:
                self.stats.end()
                if ch is None:
                    self.logger.error("No channel assigned for end event")
                    raise Exception
                if self.gui:
                    self.gui.hgrid.unmark_cell(*cell)

            next_cevent = self.eventgen.pop()
            if ch is not None and ce_type != CEvent.END \
                    and next_cevent[1] != CEvent.END:
                # Only add (s, a, r, s') tuples for which the events in
                # s and s' are not END events
                r = self.reward()
                s_new = np.copy(self.grid.state)
                cell_new = next_cevent[2]
                self.experience.append([s, cell, ch, r, s_new, cell_new])
            next_ch = self.get_action(next_cevent, cell, ch)
            ch, cevent = next_ch, next_cevent

            if i > 0 and i % self.log_iter == 0:
                self.stats.n_iter(self.epsilon, self.alpha)

        self.stats.end_episode(
            np.count_nonzero(self.grid.state), self.epsilon, self.alpha)
        self.fn_after()
        if self.quit_sim and self.pp['hopt']:
            # Don't want to return block prob for incomplete sims when
            # optimizing hyperparams
            sys.exit(0)

    def get_init_action(self):
        raise NotImplementedError()

    def get_action(self):
        raise NotImplementedError()

    def fn_after(self):
        """
        Cleanup
        """
        pass

    def execute_action(self, cevent, ch):
        ce_type, cell = cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            if self.grid.state[cell][ch]:
                self.logger.error(
                    f"Tried assigning new call {ce_str(cevent)} to"
                    f" channel {ch} which is already in use")
                raise Exception()
            self.logger.debug(f"Assigned ch {ch} to cell {cell}")
            self.grid.state[cell][ch] = 1
        else:
            self.grid.state[cell][ch] = 0


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
            free = self.grid.get_free_chs(next_cell)
            if len(free) == 0:
                return None
            else:
                return np.random.choice(free)
                # return np.random.randint(len(free))
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
            for ch, isNom in enumerate(self.grid.nom_chs[next_cell]):
                if isNom and self.grid.state[next_cell][ch] == 0:
                    return ch
            return None
        elif ce_type == CEvent.END:
            # No rearrangement is done when a call terminates.
            return next_cevent[3]


class RLStrat(Strat):
    def __init__(self, pp, *args, **kwargs):
        """
        """
        super().__init__(pp, *args, **kwargs)
        self.epsilon = pp['epsilon']
        self.epsilon_decay = pp['epsilon_decay']
        self.alpha = pp['alpha']
        self.alpha_decay = pp['alpha_decay']
        self.gamma = pp['gamma']

        self.n_used = 0
        self.qval = 0
        self.experience = []
        self.correct_preds = 0
        self.incorrect_preds = 0

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
        next_ce_type, next_cell = next_cevent[1], next_cevent[2]
        # Choose A' from S'
        next_n_used, next_ch = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, don't update q-value at all
        if next_ce_type != CEvent.END and next_ch is not None:
            # Observe reward from previous action
            reward = self.reward()

            # Update q-values with one-step lookahead
            next_qval = self.get_qval(next_cell, next_n_used, next_ch)
            targetq = reward + self.discount() * next_qval
            self.update_qval(cell, self.n_used, ch, targetq)
            # n_used doesn't change if there's no action to take
            self.n_used, self.qval = next_n_used, next_qval
        if next_ce_type == CEvent.END and next_ch is None:
            self.logger.error(
                "'None' channel for end event"
                f" {ce_str(next_cevent)}"
                f" {np.where(self.grid.state[next_cell] == 1)}")
            raise Exception
        return next_ch

    def optimal_ch(self, ce_type, cell):
        # TODO this isn't really the 'optimal' ch since
        # it's chosen in an epsilon-greedy fashion
        """
        Select the channel fitting for assignment that
        that has the maximum q-value in an epsilon-greedy fasion,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (n_used, ch) where 'n_used' is the number of channels in
        use before any potential action is taken.
        'ch' is None if no channel is eligeble for assignment
        """
        inuse = np.nonzero(self.grid.state[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.grid.get_free_chs(cell)
            op = np.argmax
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            op = np.argmin

        if len(chs) == 0:
            # No channels available for assignment,
            # or no channels in use to reassign
            assert ce_type != CEvent.END
            return (n_used, None)

        # Might do Greedy in the LImit of Exploration (GLIE) here,
        # like Boltzmann Exploration with decaying temperature.
        if ce_type != CEvent.END and np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily (either minimum or maximum)
            idx = op(self.get_qval(cell, n_used, chs))
            ch = chs[idx]

        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return (n_used, ch)

    def execute_action(self, cevent, ch):
        """
        Change the grid state according to the given action.
        """
        ce_type, cell = cevent[1:3]
        if ce_type == CEvent.END:
            end_ch = cevent[3]
            self.logger.debug(
                f"Reassigned cell {cell} ch {ch} to ch {end_ch}")
            assert self.grid.state[cell][end_ch] == 1
            assert self.grid.state[cell][ch] == 1
            if end_ch != ch:
                self.eventgen.reassign(cevent[2], ch, end_ch)
            self.grid.state[cell][ch] = 0
        super().execute_action(cevent, ch)

    def reward(self):
        """
        Immediate reward
        dt: Time until next event
        """
        # Number of calls currently in progress
        # TODO try +1 for accepted and -1 for rejected instead
        return np.count_nonzero(self.grid.state)

    def discount(self):
        """
        Discount factor (gamma)
        """
        # TODO: Find examples where
        # gamma is a function of time until next event.
        # How should gamma increase as a function of dt?
        # Linearly, exponentially?
        # discount(0) should probably be 0
        return self.gamma


class SARSA(RLStrat):
    """
    State consists of coordinates + number of used channels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "qvals[r][c][n_used][ch] = v"
        # Assigning channel 'ch' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n_used' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols,
                              self.n_channels, self.n_channels))

    def get_qval(self, cell, n_used, ch):
        return self.qvals[cell][n_used][ch]

    def update_qval(self, cell, n_used, ch, targetq):
        td_err = targetq - self.qval
        self.qvals[cell][n_used][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class TT_SARSA(RLStrat):
    """
    State consists of coordinates + the number of used channels.
    If the number of used channels exceeds 'k', their values are
    aggregated to 'k'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 30
        self.qvals = np.zeros((self.rows, self.cols, self.k, self.n_channels))

    def get_qval(self, cell, n_used, ch):
        return self.qvals[cell][min(self.k - 1, n_used)][ch]

    def update_qval(self, cell, n_used, ch, td_err):
        self.qvals[cell][min(self.k - 1, n_used)][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class RS_SARSA(RLStrat):
    """
    State consists of coordinates only
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels))
        self.fmax = np.argmax
        self.fmin = np.argmin

    def get_qval(self, cell, n_used, ch):
        return self.qvals[cell][ch]

    def update_qval(self, cell, n_used, ch, td_err):
        self.qvals[cell][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class SARSAQNet(RLStrat):
    """
    State consists of coordinates + number of used channels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros((70))

    def get_action(self, next_cevent, cell, ch):
        """
        Return a channel to be (re)assigned for 'next_cevent'.
        'cell' and 'ch' specify the previous channel (re)assignment.
        """
        next_ce_type, next_cell = next_cevent[1], next_cevent[2]
        # Choose A' from S'
        next_n_used, next_ch, next_qvals = self.optimal_ch(
            next_ce_type, next_cell)
        # If there's no action to take, don't update q-value at all
        if next_ce_type != CEvent.END and next_ch is not None:
            # Observe reward from previous action
            reward = self.reward()
            # Update q-values with one-step lookahead
            next_qval = next_qvals[next_ch]
            targetq = reward + self.discount() * next_qval
            self.update_qval(cell, self.n_used, ch, targetq)
            # n_used doesn't change if there's no action to take
            self.n_used, self.qvals = next_n_used, next_qvals
        if next_ce_type == CEvent.END and next_ch is None:
            self.logger.error(
                "'None' channel for end event"
                f" {ce_str(next_cevent)}"
                f" {np.where(self.grid.state[next_cell] == 1)}")
            raise Exception
        return next_ch

    def optimal_ch(self, ce_type, cell):
        # TODO this isn't really the 'optimal' ch since
        # it's chosen in an epsilon-greedy fashion
        # TODO This, and get_action, should be merged with RL Strat
        # with some sensible refactoring
        """
        Select the channel fitting for assignment that
        that has the maximum q-value in an epsilon-greedy fasion,
        or select the channel for termination that has the minimum:
        q-value in a greedy fashion.

        Return (n_used, ch) where 'n_used' is the number of channels in
        use before any potential action is taken.
        'ch' is None if no channel is eligeble for assignment
        """
        inuse = np.nonzero(self.grid.state[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.grid.get_free_chs(cell)
            op = np.argmax
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            op = np.argmin

        if len(chs) == 0:
            # No channels available for assignment,
            # or no channels in use to reassign.
            return (None, None, None)

        qvals = self.get_qvals(cell, n_used)
        # Might do Greedy in the LImit of Exploration (GLIE) here,
        # like Boltzmann Exploration with decaying temperature.
        # TODO Why are END events always greedy??
        if ce_type != CEvent.END and np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily (from eligible channels only)
            idx = op(qvals[chs])
            ch = chs[idx]
            # TODO
            # If qvals blow up, you get a lot of 'NaN's and 'inf's
            # in the qvals and ch becomes none.
            if ch is None:
                self.logger.error(f"{ce_type}\n{chs}\n{qvals}\n\n")
                raise Exception

        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return (n_used, ch, qvals)

    def fn_after(self):
        self.net.sess.close()

    def get_init_action(self, cevent):
        _, ch, _ = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_qvals(self, cell, n_used):
        state = self.encode_state(cell, n_used)
        _, qvals = self.net.forward(state)
        return qvals

    def update_qval(self, cell, n_used, ch, target):
        self.qvals[ch] = target
        state = self.encode_state(cell, n_used)
        self.net.backward(state, self.qvals)

    def encode_state(self, *args):
        raise NotImplementedError


class SARSAQNet_idx_nused(SARSAQNet):
    """
    State consists of: Index of cell, one-hot encoded and
    number of channels in use for that cell (integer)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from net import Net
        self.net = Net(self.logger, 50, 70, self.alpha)

    def encode_state(self, cell, n_used):
        state = np.identity(50)[(cell[0] + 1) * (cell[1] + 1) - 1]
        state[49] = n_used
        state.shape = (1, 50)
        return state


class SARSAQNet_singh(SARSAQNet):
    """
    Nearly Features from Singh paper
    For each cell, the number of available channels.
    For each cell-channel pair, the number of times the
    channel is used in a 4 cell radius.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from net import Net
        self.net = Net(
            self.logger, self.n_channels,
            self.alpha, *self.grid.neighbors2all())

    def encode_state(self, cell, n_used, empty_neg=True):
        pos = np.zeros((self.grid.rows, self.grid.cols))
        pos[cell] = 1
        if empty_neg:
            state = self.grid.state * 2 - 1
        else:
            state = self.grid.state
        features = np.dstack((state, pos))
        features.shape = (1, self.rows, self.cols, self.n_channels + 1)
        return features


# TODO verify the rl sim loop. is it correct?
# can it be simplified, e.g. remove fn_init?
def strat_classes():
    """
    Return a list with (name, class) for all the strats
    """
    def is_class_member(member):
        return inspect.isclass(member) and member.__module__ == __name__
    clsmembers = inspect.getmembers(sys.modules[__name__], is_class_member)
    return clsmembers
