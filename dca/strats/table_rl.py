import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from strats.base import RLStrat
from utils import prod


class QTable(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.pp['alpha']
        self.alpha_decay = self.pp['alpha_decay']
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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount, p):
        assert type(ch) == np.int64
        assert ch is not None
        if self.pp['verify_grid']:
            assert np.sum(grid != self.grid) == 1
        next_n_used = np.count_nonzero(self.grid[next_cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        target_q = reward + discount * next_qval
        # Counting n_used of self.grid instead of grid yields significantly lower
        # blockprob on (TT-)SARSA for unknown reasons.
        n_used = np.count_nonzero(grid[cell])
        q = self.get_qvals(cell, n_used, ch)
        self.qval_means.append(np.mean(self.get_qvals(cell, n_used)))
        td_err = target_q - q
        self.losses.append(td_err**2)
        frep = self.feature_rep(cell, n_used)
        if self.lmbda is None:
            self.qvals[frep][ch] += self.alpha * td_err
        else:
            self.el_traces[frep][ch] += 1
            self.qvals += self.alpha * td_err * self.el_traces
            self.el_traces *= discount * self.lmbda
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


class HLA_RS_SARSA(QTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros(self.dims)

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_max_ch, p = self.optimal_ch(next_ce_type, next_cell)
        # NOTE TODO this might not be correct for HLA
        if ce_type != CEvent.END and ch is not None and next_ch is not None:
            assert next_max_ch is not None
            self.update_qval(grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                             discount, p)
        return next_ch

    def optimal_ch(self, ce_type, cell):
        inuse = np.nonzero(self.grid[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = NGF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return (None, None, 0)
        else:
            chs = inuse
            assert n_used > 0

        next_event = self.env.eventgen.peek()
        if self.pp['hoff_lookahead'] and next_event[1] == CEvent.HOFF:
            assert ce_type == CEvent.END
            qvals_dense, cur_frep, freps = self.get_hoff_qvals(self.grid, cell, ce_type,
                                                               chs, next_event[2])
        else:
            qvals_dense, cur_frep, freps = self.get_qvals(self.grid, cell, ce_type, chs)

        qvals_dense = self.get_qvals(cell=cell, n_used=n_used, ce_type=ce_type, chs=chs)

        if ce_type == CEvent.END:
            amin_idx = np.argmin(qvals_dense)
            ch = max_ch = chs[amin_idx]
            p = 1
        else:
            ch, idx, p = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            self.epsilon *= self.epsilon_decay
            amax_idx = np.argmax(qvals_dense)
            max_ch = chs[amax_idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
            raise Exception
        self.logger.debug(f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (ch, max_ch, p)

    def get_qvals(self, cell, n_used, chs=None, *args, **kwargs):
        if chs is None:
            return self.qvals[cell]
        else:
            return self.qvals[cell][chs]

    def get_hoff_qvals(self, grid, cell, ce_type, chs, h_cell):
        """ Look ahead for handoffs """
        end_astates = NGF.afterstates(grid, cell, ce_type, chs)
        hoff_astates = []
        n_hoff_astates = []  # For a given end_astate, how many hoff astates?
        for astate in end_astates:
            h_chs = NGF.get_eligible_chs(astate, h_cell)
            if len(h_chs) > 0:
                h_astates = NGF.afterstates(astate, h_cell, CEvent.HOFF, h_chs)
                hoff_astates.extend(h_astates)
                n = len(h_astates)
            else:
                n = 0
            n_hoff_astates.append(n)
        cur_frep, freps = self.feature_rep(grid), self.feature_reps(end_astates)
        if len(hoff_astates) > 0:
            hoff_astates = np.array(hoff_astates)
            hfreps = self.feature_reps(hoff_astates)
            hqvals_dense = self.net.forward(freps=hfreps, grids=hoff_astates)
            assert hqvals_dense.shape == (len(hoff_astates), ), hqvals_dense.shape
            qvals_dense = np.zeros(len(chs))
            t = 0
            for i, n in enumerate(n_hoff_astates):
                qvals_dense[i] = np.max(hqvals_dense[t:t + n]) if n > 0 else 0
                t += n
        else:
            # Not possible to assign HOFF for any reass on END.
            qvals_dense = self.net.forward(freps=freps, grids=end_astates)

        return qvals_dense, cur_frep, freps

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount, p):
        assert (type(ch) == np.int64) and ch is not None
        next_n_used = np.count_nonzero(self.grid[next_cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        target_q = reward + discount * next_qval
        n_used = np.count_nonzero(grid[cell])
        q = self.get_qvals(cell, n_used, ch)
        self.qval_means.append(np.mean(self.get_qvals(cell, n_used)))
        td_err = target_q - q
        self.losses.append(td_err**2)
        self.qvals[cell][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class NT_RS_SARSA(QTable):
    """
    No-target Reduced-state SARSA.
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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount):
        assert type(ch) == np.int64
        assert ch is not None
        self.qval_means.append(np.mean(self.qvals[cell]))
        td_err = (reward - self.qvals[cell][ch])
        self.losses.append(td_err**2)
        self.qvals[cell][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay


class E_RS_SARSA(QTable):
    """
    Expected Reduced-state SARSA.
    State consists of cell coordinates only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros(self.dims)
        self.load_qvals()

    def feature_rep(self, cell, n_used):
        return cell

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount):
        assert type(ch) == np.int64
        assert ch is not None
        next_n_used = np.count_nonzero(self.grid[next_cell])
        next_qvals = self.get_qvals(next_cell, next_n_used)
        scaled = np.exp((next_qvals - np.max(next_qvals)) / self.epsilon)
        probs = scaled / np.sum(scaled)
        expected_next_q = np.sum(probs * next_qvals)
        target_q = reward + discount * expected_next_q

        n_used = np.count_nonzero(grid[cell])
        q = self.get_qvals(cell, n_used, ch)
        td_err = target_q - q
        self.losses.append(td_err**2)

        frep = self.feature_rep(cell, n_used)
        self.qvals[frep][ch] += self.alpha * td_err
        if self.alpha > self.pp['min_alpha']:
            self.alpha *= self.alpha_decay


class ZapQ(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.pp['alpha']
        self.alpha_decay = self.pp['alpha_decay']
        # State-action pair numbers
        self.d = prod(self.dims)
        self.qvals = np.zeros(self.d)
        self.sa_nums = np.arange(self.d).reshape(self.dims)
        self.azap = np.zeros((self.d, self.d))

    def fn_after(self):
        self.logger.info(f"Max qval: {np.max(self.qvals)}")

    def get_qvals(self, cell, n_used=None, chs=None, *args, **kwargs):
        if chs is None or type(chs) is list:
            raise NotImplementedError(chs)
        else:
            return self.qvals[self.sa_nums[cell][chs]]

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount, p):
        assert type(ch) == np.int64
        assert ch is not None
        assert next_max_ch is not None

        sa_num = self.sa_nums[cell][ch]
        next_sa_num = self.sa_nums[next_cell][next_max_ch]
        outer1 = np.zeros((self.d, self.d))
        outer2 = np.zeros((self.d, self.d))
        outer1[sa_num, sa_num] = 1
        outer2[sa_num, next_sa_num] = 1

        azap_gam = np.power((self.i + 1), -0.85)  # stepsize for matrix gain recursion
        self.azap += azap_gam * ((-outer1 + discount * outer2) - self.azap)
        azap_inv = np.linalg.pinv(self.azap)
        a_inv_dot = azap_inv[:, sa_num]

        q = self.get_qvals(cell, chs=ch)
        next_q = self.get_qvals(next_cell, chs=next_max_ch)
        td_err = reward + discount * next_q - q
        self.qvals -= self.alpha * a_inv_dot * td_err

        self.losses.append(td_err**2)
        self.alpha *= self.alpha_decay
