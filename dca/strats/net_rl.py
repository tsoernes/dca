from typing import List, Tuple

import numpy as np

from eventgen import CEvent
from grid import RhombusAxialGrid
from nets.acnet import ACNet
from nets.dqnet import DistQNet
from nets.qnet import QNet
from nets.singh import SinghNet
from nets.utils import softmax
from strats.base import RLStrat


class NetStrat(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_copy_iter = self.pp['net_copy_iter']
        self.losses = [0]

    def fn_report(self):
        self.env.stats.report_net(self.losses)
        self.env.stats.report_rl(self.epsilon)

    def fn_after(self):
        ra = self.net.rand_uniform()
        self.logger.debug(f"TF Rand: {ra}, NP Rand: {np.random.uniform()}")
        if self.pp['save_net']:
            inp = ""
            if self.quit_sim:
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save model? Y/N: ").upper()
            if inp in ["", "Y"]:
                self.net.save_model()
        self.net.save_timeline()
        self.net.sess.close()


class QNetStrat(NetStrat):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = QNet(name, self.pp, self.logger)
        ra = self.net.rand_uniform()
        self.logger.debug(f"TF Rand: {ra}")
        if self.batch_size > 1:
            self.update_qval = self.update_qval_experience
            self.logger.warn("Using experience replay with batch"
                             f" size of {self.batch_size}")

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

    def get_qvals(self, cell, ce_type, chs, *args, **kwargs):
        qvals = self.net.forward(self.grid, cell, ce_type)
        return qvals[chs]

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch):
        """ Update qval for one experience tuple"""
        loss = self.backward(grid, cell, [ch], [reward], self.grid, next_cell, [next_ch],
                             [next_max_ch])
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)

    def update_qval_experience(self, *args, **kwargs):
        """
        Update qval for pp['batch_size'] experience tuples,
        randomly sampled from the experience replay memory.
        """
        if len(self.exp_buffer) < self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            return
        loss = self.net.backward(**self.exp_buffer.sample(self.pp['batch_size']))
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)


class QLearnNetStrat(QNetStrat):
    """Update towards greedy, possibly illegal, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("QLearnNet", *args, **kwargs)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, *args, **kwargs):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell)
        return loss


class QLearnEligibleNetStrat(QNetStrat):
    """Update towards greedy, eligible, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("QlearnEligibleNet", *args, **kwargs)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, next_ch,
                 next_max_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell,
                                 next_max_ch)
        return loss


class SARSANetStrat(QNetStrat):
    """Update towards policy action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("SARSANet", *args, **kwargs)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, next_ch,
                 next_max_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell, next_ch)
        return loss


class DistQNetStrat(QNetStrat):
    """
    TODO:
    - Try SARSA
    - Try Double Q
    - 
    """

    def __init__(self, *args, **kwargs):
        super().__init__("DistQNet", *args, **kwargs)
        self.net = DistQNet(pp=self.pp, logger=self.logger)

    def update_target_net(self):
        pass

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, *args, **kwargs):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell)
        return loss


class ACNetStrat(NetStrat):
    """Actor Critic"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ACNet(pp=self.pp, logger=self.logger)
        self.logger.info(
            "Loss legend (scaled): [ total, policy_grad, value_fn, entropy ]")

    def forward(self, cell, ce_type) -> Tuple[List[float], float]:
        if ce_type == CEvent.END:
            state = np.copy(self.grid)
            state[cell] = np.zeros(self.n_channels)
        else:
            state = self.grid
        a, v = self.net.forward(state, cell)
        return a, v

    def update_target_net(self):
        pass

    def get_init_action(self, cevent):
        ch, self.val = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        # TODO perhaps for n-step returns, everything should be included, or
        # next_ce_type == END should be excluded.
        if ce_type != CEvent.END and ch is not None and next_ch is not None:
            # Observe reward from previous action, and
            # update q-values with one-step lookahead
            self.update_qval(grid, cell, ch, reward, self.grid, next_cell, next_ch)
            self.val = next_val
        return next_ch

    def optimal_ch(self, ce_type, cell) -> Tuple[int, float]:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            # Calls eligible for assignment
            chs = self.env.grid.get_eligible_chs(cell)
        else:
            # Calls in progress
            chs = np.nonzero(self.grid[cell])[0]
        if len(chs) == 0:
            assert ce_type != CEvent.END
            return (None, None)

        a_dist, val = self.forward(cell=cell, ce_type=ce_type)
        greedy = True
        if ce_type == CEvent.END:
            if greedy:
                idx = np.argmin(a_dist[chs])
            else:
                valid_a_dist = softmax(1 - a_dist[chs])
                idx = np.random.choice(np.range(len(valid_a_dist)), p=valid_a_dist)
        else:
            if greedy:
                idx = np.argmax(a_dist[chs])
            else:
                valid_a_dist = softmax(a_dist[chs])
                idx = np.random.choice(np.range(len(valid_a_dist)), p=valid_a_dist)
        ch = chs[idx]
        # print(ce_type, a_dist, ch, a_dist[ch], chs)
        # TODO NOTE verify the above

        # If vals blow up ('NaN's and 'inf's), ch becomes none.
        if np.isinf(val) or np.isnan(val):
            self.logger.error(f"{ce_type}\n{chs}\n{val}\n\n")
            raise Exception

        self.logger.debug(f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (ch, val)

    def update_qval(self, grid, cell, ch, reward, next_grid, next_cell, next_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell)
        if np.isinf(loss[0]) or np.isnan(loss[0]):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)

    def update_qval_n_step(self, grid, cell, ch, reward, next_grid, next_cell, next_ch):
        """
        Update qval for pp['batch_size'] experience tuple.
        """
        if len(self.exp_buffer) < self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            return
        loss = self.net.backward_gae(
            **self.exp_buffer.pop(), next_grid=next_grid, next_cell=next_cell)
        if np.isinf(loss[0]) or np.isnan(loss[0]):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)


class VNetStrat(NetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_target_net(self):
        pass

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, _ = self.optimal_ch(next_ce_type, next_cell)
        if ce_type != CEvent.END and \
           ch is not None and next_ch is not None:
            self.update_qval(grid, cell, ch, reward, self.grid, next_cell, next_ch)
        return next_ch

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch):
        """ Update qval for one experience tuple"""
        # TODO assert that grid and self.grid only differs by ch in cell
        # assert not (grid == self.grid).all()
        loss = self.backward(grid, reward, self.grid)
        assert np.min(self.grid) >= 0
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)


class SinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.val = 0.0

    # def get_init_action(self, cevent) -> int:
    #     return self.optimal_ch(ce_type=cevent[1], cell=cevent[2])

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        if ch is not None:
            loss = self.backward(grid, cell, reward, self.grid)
            if np.isinf(loss) or np.isnan(loss):
                self.logger.error(f"Invalid loss: {loss}")
                self.quit_sim = True
            else:
                self.losses.append(loss)

        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        self.val = next_val
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_eligible_chs(cell)
            if len(chs) == 0:
                return None, 0
        else:
            chs = np.nonzero(self.grid[cell])[0]

        fgrids = self.afterstate_freps(self.grid, cell, ce_type, chs)
        # fgrids2 = self.afterstate_freps2(self.grid, cell, ce_type, chs)
        # assert (fgrids == fgrids2).all()
        qvals_dense = self.net.forward(fgrids)
        assert qvals_dense.shape == (len(chs), )
        amax_idx = np.argmax(qvals_dense)
        ch = chs[amax_idx]
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[amax_idx]

    def backward(self, grid, cell, reward, next_grid):
        value_target = reward + self.gamma * np.array([[self.val]])
        loss = self.net.backward([self.feature_reps(grid, cell)],
                                 [self.feature_reps(next_grid, cell)], value_target)
        return loss

    def afterstate_freps_naive(self, grid, cell, ce_type, chs):
        astates = RhombusAxialGrid.afterstates_stat(grid, cell, ce_type, chs)
        freps = self.feature_reps(astates)
        return freps

    def afterstate_freps(self, grid, cell, ce_type, chs):
        """ Get the feature representation for the current grid,
        and from it derive the f.rep for each possible afterstate.
        Current assumptions:
        n_used_neighs (frep[:-1]) does include self
        n_free_self (frep[-1]) counts ELIGIBLE chs
        """
        fgrid = self.feature_reps(grid)
        r, c = cell
        neighs4 = RhombusAxialGrid.neighbors(
            dist=4, row=r, col=c, separate=True, include_self=False)
        neighs2 = RhombusAxialGrid.neighbors(dist=2, row=r, col=c, include_self=True)
        fgrids = np.repeat(np.expand_dims(fgrid, axis=0), len(chs), axis=0)
        if ce_type == CEvent.END:
            # One less channel will be in use by the cell
            n_used_neighs_diff = -1
            # One more channel MIGHT become eligible
            # Temporarily modify grid and check if that's the case
            n_elig_self_diff = 1
            grid[cell][chs] = 0
        else:
            # One more ch will be in use
            n_used_neighs_diff = 1
            # One less ch will be eligible
            n_elig_self_diff = -1
        eligible_chs = [
            RhombusAxialGrid.get_eligible_chs_stat(grid, neigh2) for neigh2 in neighs2
        ]
        for i, ch in enumerate(chs):
            fgrids[i, neighs4[0], neighs4[1], ch] += n_used_neighs_diff
            for j, neigh2 in enumerate(neighs2):
                if ch in eligible_chs[j]:
                    fgrids[i, neigh2[0], neigh2[1], -1] += n_elig_self_diff
        if ce_type == CEvent.END:
            grid[cell][chs] = 1
        return fgrids

    def feature_reps(self, grids, *args):
        """
        Takes a grid or an array of grids and return the feature representation(s).

        For each cell, the number of ELIGIBLE channels in that cell.
        For each cell-channel pair, the number of times that channel is
        used by neighbors with a distance of 4 or less.
        NOTE The latter does not include whether or not the channel is
        in use by the cell itself, though that may be the better option.
        """
        assert type(grids) == np.ndarray
        single = False  # Only one grid to create frep for
        if grids.ndim == 3:
            grids = np.expand_dims(grids, axis=0)
            single = True
        fgrids = np.zeros(
            (len(grids), self.rows, self.cols, self.n_channels + 1), dtype=np.int32)
        # fgrids[:, :, :, self.n_channels] = self.n_channels \
        #     - np.count_nonzero(grids, axis=3)
        for r in range(self.rows):
            for c in range(self.cols):
                neighs = self.env.grid.neighbors(4, r, c, separate=True)
                n_used = np.count_nonzero(grids[:, neighs[0], neighs[1]], axis=1)
                fgrids[:, r, c, :-1] = n_used
                for i in range(len(grids)):
                    n_eligible_chs = RhombusAxialGrid.get_n_eligible_chs_stat(
                        grids[i], (r, c))
                    fgrids[i, r, c, -1] = n_eligible_chs
        if single:
            return fgrids[0]
        return fgrids
