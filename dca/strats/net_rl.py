from typing import List, Tuple

import numpy as np

from eventgen import CEvent
from grid import RhombusAxialGrid
from nets.acnet import ACNet
from nets.qnet import QNet
from nets.singh import SinghNet
from nets.utils import softmax
from replaybuffer import ExperienceBuffer
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
        # ra = self.net.rand_uniform()
        # self.logger.info(f"TF Rand: {ra}, NP Rand: {np.random.uniform()}")
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
        # ra = self.net.rand_uniform()
        # self.logger.info(f"TF Rand: {ra}")
        if self.batch_size > 1:
            self.update_qval = self.update_qval_experience
            self.logger.warn("Using experience replay with batch"
                             f" size of {self.batch_size}")

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

    def get_qvals(self, cell, ce_type, chs, *args, **kwargs):
        if ce_type == CEvent.END:
            # Zero out channel usage in cell of END event
            grid = np.copy(self.grid)
            grid[cell] = np.zeros(self.n_channels)
        else:
            grid = self.grid
        qvals = self.net.forward(grid, cell)
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
        if len(self.replaybuffer) < self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            return
        loss = self.net.backward(*self.replaybuffer.sample(self.pp['batch_size']))
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


class ACNetStrat(NetStrat):
    """Actor Critic"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ACNet(pp=self.pp, logger=self.logger)
        self.exp_buffer = ExperienceBuffer()
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
        inuse = np.nonzero(self.grid[cell])[0]

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_eligible_chs(cell)
        else:
            chs = inuse
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
        self.exp_buffer.add(grid, cell, self.val, ch, reward)
        if len(self.exp_buffer) < self.pp['n_step']:
            # Can't backprop before exp store has enough experiences
            return
        loss = self.net.backward_gae(*self.exp_buffer.pop(), next_grid, next_cell)
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

    def get_init_action(self, cevent) -> int:
        return self.optimal_ch(ce_type=cevent[1], cell=cevent[2])

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        if ch is not None:
            loss = self.backward(grid, reward, self.grid)
            if np.isinf(loss) or np.isnan(loss):
                self.logger.error(f"Invalid loss: {loss}")
                self.quit_sim = True
            else:
                self.losses.append(loss)

        next_ce_type, next_cell = next_cevent[1:3]
        next_ch = self.optimal_ch(next_ce_type, next_cell)
        return next_ch

    def optimal_ch(self, ce_type, cell) -> Tuple[int, float, int]:
        inuse = np.nonzero(self.grid[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_eligible_chs(cell)
            if len(chs) == 0:
                return None
        else:
            chs = inuse
            # or no channels in use to reassign
            assert n_used > 0

        fgrids = self.afterstate_freps(self.grid, cell, ce_type, chs)
        qvals_sparse = self.net.forward(fgrids)
        assert qvals_sparse.shape == (len(chs), )
        amax_idx = np.argmax(qvals_sparse)
        ch = chs[amax_idx]

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_sparse}\n")
        return ch

    def afterstate_freps(self, grid, cell, ce_type, chs):
        """ Get the feature representation for the current grid,
        and from it derive the f.rep for each possible afterstate.
        Current assumptions:
        n_used_neighs (:-1) does NOT include self
        n_free_self (-1) counts ELIGIBLE chs

        TODO NOTE Should handoffs be handled differently?
        """
        fgrid = self.feature_rep(grid)
        if ce_type == CEvent.END:
            # One more channel might become eligible if the ch is
            # not in use by neighs2
            n_elig_self_diff = 1
            # One less channel will be in use by the cell
            n_used_neighs_diff = -1
        else:
            # One less ch will be eligible
            n_elig_self_diff = -1
            # One more ch will be in use
            n_used_neighs_diff = 1
        r, c = cell
        neighs4 = RhombusAxialGrid.neighbors(dist=4, row=r, col=c, separate=True)
        neighs2 = RhombusAxialGrid.neighbors(dist=2, row=r, col=c, include_self=True)
        fgrids = np.repeat(np.expand_dims(fgrid, axis=0), len(chs), axis=0)
        for i, ch in enumerate(chs):
            fgrids[i, neighs4[0], neighs4[1], ch] += n_used_neighs_diff
            for neigh2 in neighs2:
                eligible_chs = RhombusAxialGrid.get_eligible_chs_stat(grid, neigh2)
                if ch in eligible_chs:
                    fgrids[i, neigh2[0], neigh2[1], -1] += n_elig_self_diff
        return fgrids

    def backward(self, grid, reward, next_grid):
        loss = self.net.backward([self.feature_rep(grid)], reward,
                                 [self.feature_rep(next_grid)])
        return loss

    def feature_rep(self, grid):
        # For each cell, the number of ELIGBLE channels in that cell.
        # For each cell-channel pair, the number of times that channel is
        # used by neighbors with a distance of 4 or less.
        # NOTE Should that include
        # whether or not the channel is in use by the cell itself??
        # Currently, DOES NOT
        assert type(grid) == np.ndarray
        assert grid.shape == self.dims, (grid.shape, self.dims)
        fgrid = np.zeros((self.rows, self.cols, self.n_channels + 1), dtype=np.int32)
        # fgrids[:, :, :, self.n_channels] = self.n_channels \
        #     - np.count_nonzero(grids, axis=3)
        for r in range(self.rows):
            for c in range(self.cols):
                # Used neighs
                neighs = self.env.grid.neighbors(4, r, c, separate=True)
                n_used = np.count_nonzero(grid[neighs], axis=0)
                fgrid[r, c, :-1] = n_used
                # Eligible self
                eligible_chs = RhombusAxialGrid.get_eligible_chs_stat(grid, (r, c))
                fgrid[r, c, -1] = len(eligible_chs)
        return fgrid

    def feature_reps(self, grids):
        # For each cell, the number of ELIGIBLE channels in that cell.
        # For each cell-channel pair, the number of times that channel is
        # used by neighbors with a distance of 4 or less.
        # NOTE Should that include
        # whether or not the channel is in use by the cell itself??
        # Currently, DOES NOT
        assert type(grids) == np.ndarray
        if grids.ndim == 3:
            grids = np.expand_dims(grids, axis=0)
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
                    eligible_chs = RhombusAxialGrid.get_eligible_chs_stat(
                        grids[i], (r, c))
                    fgrids[i, r, c, -1] = len(eligible_chs)
        return fgrids
