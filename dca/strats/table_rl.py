import numpy as np

from strats.base import RLStrat


class QTable(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmbda = self.pp['lambda']
        if self.lmbda is not None:
            self.logger.error("Using lambda returns")

    def load_qvals(self):
        """Load Q-values from file"""
        fname = self.pp['restore_qtable']
        if fname != '':
            self.qvals = np.load(fname)
            self.logger.error(f"Restored qvals from {fname}")

    def fn_after(self):
        self.logger.info(f"Max qval: {np.max(self.qvals)}")
        if self.save:
            fname = "qtable"
            self.logger.error(f"Saved Q-table to {fname}.npy")
            np.save(fname, self.qvals)

    def get_qvals(self, cell, n_used, chs=None, *args, **kwargs):
        rep = self.feature_rep(cell, n_used)
        if chs is None:
            return self.qvals[rep]
        else:
            return self.qvals[rep][chs]

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        assert type(ch) == np.int64
        assert ch is not None
        if self.pp['verify_grid']:
            assert np.sum(grid != self.grid) == 1
        next_n_used = np.count_nonzero(self.grid[next_cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        gamma = bdisc if self.pp['dt_rewards'] else self.gamma
        target_q = reward + gamma * next_qval
        # Counting n_used of self.grid instead of grid yields significantly lower
        # blockprob on (TT-)SARSA for unknown reasons.
        n_used = np.count_nonzero(grid[cell])
        q = self.get_qvals(cell, n_used, ch)
        td_err = target_q - q
        self.losses.append(td_err**2)
        frep = self.feature_rep(cell, n_used)
        if self.lmbda is None:
            self.qvals[frep][ch] += self.alpha * td_err
        else:
            self.el_traces[frep][ch] += 1
            self.qvals += self.alpha * td_err * self.el_traces
            self.el_traces *= gamma * self.lmbda
        if self.alpha > self.pp['min_alpha']:
            self.alpha *= self.alpha_decay
        next_frep = self.feature_rep(next_cell, next_n_used)
        self.logger.debug(
            f"Q[{frep}][{ch}]:{q:.1f} -> {reward:.1f} + Q[{next_frep}][{next_ch}]:{next_qval:.1f}"
        )


class SARSA(QTable):
    """
    State consists of cell coordinates and the number of used channels in that cell.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "qvals[r, c, n_used, ch] = v"
        # Assigning channel 'ch' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n_used' channels are already
        # in use at that cell.
        self.qvals = np.zeros(
            (self.rows, self.cols, self.n_channels, self.n_channels), dtype=np.float32)
        self.load_qvals()
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros(self.dims)

    def feature_rep(self, cell, n_used):
        return (*cell, n_used)


class TT_SARSA(QTable):
    """
    Table-trimmed SARSA.
    State consists of cell coordinates and the number of used channels.
    States where the number of used channels is or exceeds 'k' have their values are
    aggregated to the state where the number of used channels is 'k-1'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 30
        self.qvals = np.zeros((self.rows, self.cols, self.k, self.n_channels))
        self.load_qvals()
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros((self.rows, self.cols, self._k, self.n_channels))

    def feature_rep(self, cell, n_used):
        return (*cell, min(self.k - 1, n_used))


class RS_SARSA(QTable):
    """
    Reduced-state SARSA.
    State consists of cell coordinates only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros(self.dims)
        self.load_qvals()
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros(self.dims)

    def feature_rep(self, cell, n_used):
        return cell
