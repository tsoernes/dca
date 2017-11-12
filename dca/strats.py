import signal
from typing import Tuple

import numpy as np

from environment import Env
from eventgen import CEvent
from grid import RhombusAxialGrid
from utils import h5py_save_append

Cell = Tuple[int, int]


class Strat:
    def __init__(self, pp, logger, pid="", gui=None, *args, **kwargs):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.save = pp['save_exp_data']
        self.batch_size = pp['batch_size']
        self.pp = pp
        self.logger = logger

        grid = RhombusAxialGrid(self.rows, self.cols, self.n_channels,
                                self.logger)
        self.env = Env(self.pp, grid, self.logger, pid)
        self.state = self.env.grid.state

        self.quit_sim = False
        signal.signal(signal.SIGINT, self.exit_handler)

        self.experience_store = {
            'grids': [],
            'cells': [],
            'actions': [],
            'rewards': [],
            'next_grids': [],
            'next_cells': [],
            'next_actions': []
        }

    def exit_handler(self, *args):
        """
        Graceful exit on ctrl-c signal from
        command line or on 'q' key-event from gui.
        """
        self.logger.warn("\nPremature exit")
        self.quit_sim = True

    def simulate(self):
        cevent = self.env.init()
        ch = self.get_init_action(cevent)

        # Discrete event simulation
        for i in range(self.pp['n_events']):
            if self.quit_sim:
                break  # Gracefully exit to print stats, clean up etc.

            t, ce_type, cell = cevent[0:3]

            if ch is not None:
                if self.save or self.batch_size > 1:
                    s = np.copy(self.state)  # Copy before state is modified

            reward, next_cevent = self.env.step(ch)
            next_ch = self.get_action(next_cevent, cell, ch, reward)
            if (self.save or self.batch_size > 1) \
                    and ch is not None \
                    and next_ch is not None \
                    and ce_type != CEvent.END \
                    and next_cevent[1] != CEvent.END:
                # Only add (s, a, r, s') tuples for which the events in
                # s and s' are not END events
                s_new = np.copy(self.state)
                cell_new = next_cevent[2]
                self.experience_store['grids'].append(s)
                self.experience_store['cells'].append(cell)
                self.experience_store['actions'].append(ch)
                self.experience_store['rewards'].append(reward)
                self.experience_store['next_grids'].append(s_new)
                self.experience_store['next_cells'].append(cell_new)
                self.experience_store['next_actions'].append(next_ch)

            if i % self.pp['log_iter'] == 0 and i > 0:
                self.fn_report()
            ch, cevent = next_ch, next_cevent
        self.env.stats.end_episode(reward)
        self.fn_after()
        if self.save:
            self.save_experience_to_disk()
        if self.quit_sim and self.pp['hopt']:
            # Don't want to return actual block prob for incomplete sims when
            # optimizing params, because block prob is much lower at sim start
            return 1
        return self.env.stats.block_prob_cum

    def save_experience_to_disk(self):
        raise NotImplementedError
        # NOTE UNTESTED. May be better to append to different data sets
        start = 10000  # Ignore initial period
        h5py_save_append("data-experience",
                         map(lambda li: li[start:],
                             self.experience_store.values()))

    def get_init_action(self, next_cevent):
        raise NotImplementedError

    def get_action(self, next_cevent, cell: Cell, ch: int, reward: int) -> int:
        raise NotImplementedError

    def fn_report(self):
        """
        Report stats for different strategies
        """
        pass

    def fn_after(self):
        """
        Cleanup
        """
        pass


class RLStrat(Strat):
    def __init__(self, pp, *args, **kwargs):
        super().__init__(pp, *args, **kwargs)
        self.epsilon = pp['epsilon']
        self.epsilon_decay = pp['epsilon_decay']
        self.alpha = pp['alpha']
        self.alpha_decay = pp['alpha_decay']
        self.gamma = pp['gamma']

    def fn_report(self):
        self.env.stats.report_rl(self.epsilon, self.alpha)

    def update_qval(self, cell: Cell, ch: np.int64, target_q: np.float64):
        raise NotImplementedError

    def get_qvals(self, cell: Cell, *args):
        """
        Different strats may use additional arguments,
        depending on the features
        """
        raise NotImplementedError

    def get_init_action(self, cevent):
        ch, _ = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, cell: Cell, ch: int, reward) -> int:
        """
        Return a channel to be (re)assigned for 'next_cevent'.
        'cell' and 'ch' specify the previously executed action.
        """
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, next_qval = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if next_ce_type != CEvent.END \
                and ch is not None and next_ch is not None:
            # Observe reward from previous action, and
            # update q-values with one-step lookahead
            target_q = reward + self.gamma * next_qval
            self.update_qval(cell, ch, target_q)
        return next_ch

    def optimal_ch(self, ce_type: CEvent, cell: Cell) -> Tuple[int, float]:
        # NOTE this isn't really the optimal ch since
        # it's chosen in an epsilon-greedy fashion
        """
        Select the channel fitting for assignment that
        that has the maximum q-value in an epsilon-greedy fasion,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (ch, qval) where 'qval' is the q-value for the
        selected channel.
        'ch' is None if no channel is eligible for assignment.
        """
        inuse = np.nonzero(self.state[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_free_chs(cell)
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
            return (None, None)

        qvals = self.get_qvals(cell, n_used)
        # Might do Greedy in the LImit of Exploration (GLIE) here,
        # like Boltzmann Exploration with decaying temperature.
        # Selecting a ch for reassigment is always greedy because no learning
        # is done on the reassignment actions.
        if ce_type != CEvent.END and np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily (either minimum or maximum)
            idx = op(qvals[chs])
            ch = chs[idx]
            # If qvals blow up, you get a lot of 'NaN's and 'inf's
            # in the qvals and ch becomes none.
            if ch is None:
                self.logger.error(f"{ce_type}\n{chs}\n{qvals}\n\n")
                raise Exception

        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return (ch, qvals[ch])


class SARSA(RLStrat):
    """
    State consists of coordinates and the number of used channels in that cell.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "qvals[r][c][n_used][ch] = v"
        # Assigning channel 'ch' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n_used' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels,
                               self.n_channels))

    def get_qvals(self, cell: Cell, n_used):
        return self.qvals[cell][n_used]

    def update_qval(self, cell: Cell, ch: np.int64, target_q: np.float32):
        assert type(ch) == np.int64
        n_used = np.count_nonzero(self.state[cell])
        td_err = target_q - self.get_qvals(cell, n_used)[ch]
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

    def get_qvals(self, cell: Cell, n_used):
        return self.qvals[cell][min(self.k - 1, n_used)]

    def update_qval(self, cell, ch, target_q):
        assert type(ch) == np.int64
        n_used = np.count_nonzero(self.state[cell])
        td_err = target_q - self.get_qvals(cell, n_used)[ch]
        self.qvals[cell][min(self.k - 1, n_used)][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class RS_SARSA(RLStrat):
    """
    State consists of coordinates only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels))

    def get_qvals(self, cell, *args):
        return self.qvals[cell]

    def update_qval(self, cell, ch, target_q):
        assert type(ch) == np.int64
        td_err = target_q - self.get_qvals(cell)[ch]
        self.qvals[cell][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class SARSAQNet(RLStrat):
    """
    State consists of coordinates + number of used channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []
        if self.batch_size > 1:
            self.update_qval = self.update_qval_experience
        else:
            self.update_qval = self.update_qval_single
        from net import Net
        self.net = Net(self.pp, self.logger, restore=False, save=False)

    def fn_report(self):
        self.env.stats.report_net(self.losses)
        super().fn_report()

    def fn_after(self):
        self.net.save_model()
        self.net.save_timeline()
        self.net.sess.close()

    def get_qvals(self, cell, *args):
        qvals, _, _ = self.net.forward(self.state, cell)
        return qvals

    def update_qval_single(self, cell: Cell, ch: int, q_target: float):
        """ Update qval for one experience tuple"""
        loss = self.net.backward(self.state, cell, ch, q_target)
        self.losses.append(loss)
        if np.isinf(loss) or np.isnan(loss):
            self.quit_sim = True

    def update_qval_experience(self, *args):
        """
        Update qval for pp['batch_size'] experience tuples,
        randomly sampled from the experience replay memory.
        """
        n = len(self.experience_store['grids'])
        if n < self.batch_size:
            # Can't backprop before exp store has enough experiences
            return
        idxs = np.random.randint(0, n, self.batch_size)
        grids = np.zeros(
            (self.batch_size, self.rows, self.cols, self.n_channels),
            dtype=np.int8)
        cells = []
        actions = np.zeros(self.batch_size, dtype=np.int32)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        next_grids = np.zeros(
            (self.batch_size, self.rows, self.cols, self.n_channels),
            dtype=np.int8)
        next_cells = []
        next_actions = np.zeros(self.batch_size, dtype=np.int32)
        for i, idx in enumerate(idxs):
            grids[i][:] = self.experience_store['grids'][idx]
            cells.append(self.experience_store['cells'][idx])
            actions[i] = self.experience_store['actions'][idx]
            rewards[i] = self.experience_store['rewards'][idx]
            next_grids[i][:] = self.experience_store['next_grids'][idx]
            next_cells.append(self.experience_store['next_cells'][idx])
            next_actions[i] = self.experience_store['next_actions'][idx]
        loss = self.net.backward_exp_replay(grids, cells, actions, rewards,
                                            next_grids, next_cells,
                                            next_actions)
        self.losses.append(loss)
        if np.isinf(loss) or np.isnan(loss):
            self.quit_sim = True
