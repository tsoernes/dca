from typing import List, Tuple

import numpy as np

from eventgen import CEvent
from gridfuncs_numba import GF
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
        self.last_lr = 1

    def fn_report(self):
        self.env.stats.report_loss(self.losses)
        self.env.stats.report_rl(self.epsilon, self.last_lr)

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

    def backward(self, *args, **kwargs):
        loss, lr = self.net.backward(*args, **kwargs)
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.invalid_loss = True
            self.quit_sim = True
        else:
            self.losses.append(loss)
            self.last_lr = lr


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

    def update_qval_experience(self, *args, **kwargs):
        """
        Update qval for pp['batch_size'] experience tuples,
        randomly sampled from the experience replay memory.
        """
        if len(self.exp_buffer) >= self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            self.backward(**self.exp_buffer.sample(self.pp['batch_size']))


class QLearnNetStrat(QNetStrat):
    """Update towards greedy, possibly illegal, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("QLearnNet", *args, **kwargs)

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, [ch], [reward], self.grid, next_cell)


class QLearnEligibleNetStrat(QNetStrat):
    """Update towards greedy, eligible, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("QlearnEligibleNet", *args, **kwargs)

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, [ch], [reward], self.grid, next_cell, [next_max_ch])


class SARSANetStrat(QNetStrat):
    """Update towards policy action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("SARSANet", *args, **kwargs)

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, [ch], [reward], self.grid, next_cell, [next_ch])


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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, [ch], [reward], self.grid, next_cell)


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
            chs = GF.get_eligible_chs(self.grid, cell)
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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, *args):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, ch, reward, self.grid, next_cell)

    def update_qval_n_step(self, grid, cell, ch, reward, next_grid, next_cell, next_ch,
                           *args):
        """
        Update qval for pp['batch_size'] experience tuple.
        """
        if len(self.exp_buffer) < self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            return
        loss, lr = self.net.backward_gae(
            **self.exp_buffer.pop(), next_grid=next_grid, next_cell=next_cell)
        if np.isinf(loss[0]) or np.isnan(loss[0]):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
            self.invalid_loss = True
        else:
            self.losses.append(loss)
            self.learning_rates.append(lr)


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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, bdisc):
        """ Update qval for one experience tuple"""
        self.backward(grid, reward, self.grid)


class SinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.val = 0.0

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, bdisc) -> int:
        # if self.pp['dt_rewards']:
        #     count = np.count_nonzero(self.grid)
        #     reward = ((1 - bdisc) / self.pp['beta']) * count
        if ch is not None:
            gamma = bdisc if self.pp['dt_rewards'] else self.gamma
            value_target = reward + gamma * np.array([[self.val]])
            self.backward(
                self.scale_freps(GF.feature_reps(grid)),
                self.scale_freps(GF.feature_reps(self.grid)), value_target)

        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        self.val = next_val
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return None, 0
        else:
            chs = np.nonzero(self.grid[cell])[0]

        fgrids = GF.afterstate_freps(self.grid, cell, ce_type, chs)
        fgrids = self.scale_freps(fgrids)
        # fgrids2 = self.afterstate_freps2(self.grid, cell, ce_type, chs)
        # assert (fgrids == fgrids2).all()
        qvals_dense = self.net.forward(fgrids)
        assert qvals_dense.shape == (len(chs), )
        amax_idx = np.argmax(qvals_dense)
        ch = chs[amax_idx]
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[amax_idx]

    def afterstate_freps_naive(self, grid, cell, ce_type, chs):
        astates = GF.afterstates(grid, cell, ce_type, chs)
        freps = GF.feature_reps(astates)
        return freps

    def scale_freps(self, freps):
        assert freps.ndim == 4
        # TODO Try Scale freps in range [-1, 1]
        # Scale freps in range [0, 1]
        freps = freps.astype(np.float16)
        freps[:, :, :, :-1] /= 43.0  # Max possible neighs within dist 4 including self
        freps[:, :, :, -1] /= float(self.n_channels)
        return freps
