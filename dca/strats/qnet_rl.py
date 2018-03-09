from typing import List, Tuple

import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF
from nets.acnet import ACNet
from nets.dqnet import DistQNet
from nets.qnet import QNet
from nets.utils import softmax
from strats.base import NetStrat


class QNetStrat(NetStrat):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pp['rnn']:
            from nets.rnn_qnet import RQNet
            self.net = RQNet(name, self.pp, self.logger)
        elif self.pp['bighead']:
            from nets.bighead import BigHeadQNet
            self.net = BigHeadQNet(name, self.pp, self.logger)
        else:
            self.net = QNet(name, self.pp, self.logger)
        self.logger.info(f"TF Rand: {self.net.rand_uniform()}")
        if self.batch_size > 1:
            self.update_qval = self.update_qval_experience
            self.gamma = self.pp['gamma']
            self.logger.warn("Using experience replay with batch"
                             f" size of {self.batch_size}")

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

    def get_qvals(self, cell, ce_type, chs, *args, **kwargs):
        frep = NGF.feature_rep(self.grid) if self.pp['freps'] else None
        qvals = self.net.forward(self.grid, cell, ce_type, frep)
        self.qval_means.append(np.mean(qvals))
        return qvals[chs]

    def update_qval_experience(self, *args, **kwargs):
        """
        Update qval for pp['batch_size'] experience tuples,
        sampled from the experience replay memory.
        """
        if len(self.exp_buffer) >= self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            data, weights, batch_idxes = self.exp_buffer.sample(
                self.pp['batch_size'], beta=self.pri_beta_schedule.value(self.i))
            if self.pp['freps']:
                freps = GF.feature_reps(data['grids'])
                next_freps = GF.feature_reps(data['next_grids'])
            data.update({
                'freps': freps,
                'next_freps': next_freps,
                'next_chs': None,
            })
            data['weights'] = weights
            td_errs = self.backward(**data, gamma=self.gamma)
            new_priorities = np.abs(td_errs) + self.prioritized_replay_eps
            self.exp_buffer.update_priorities(batch_idxes, new_priorities)

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        # NOTE For greedy q-learning, strictly speaking, the correct thing to do
        # is to update_q-val before finding optimal_ch
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell, next_ch,
                         next_max_ch, discount)
        return next_ch


class QLearnNetStrat(QNetStrat):
    """Update towards greedy, possibly illegal, action selection
    """

    def __init__(self, *args, **kwargs):
        super().__init__("QLearnNet", *args, **kwargs)
        self.exps = []

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_ch,
                    next_max_ch, discount):
        if ce_type != CEvent.END and ch is not None:
            if self.pp['freps']:
                frep, next_freps = NGF.successive_freps(grid, cell, ce_type,
                                                        np.array([ch]))
                freps = [frep]
            else:
                freps, next_freps = None, None
            self.backward(
                grids=grid,
                cells=cell,
                chs=[ch],
                rewards=[reward],
                next_grids=next_grid,
                next_cells=next_cell,
                gamma=discount,
                freps=freps,
                next_freps=next_freps,
                next_chs=None,
                weights=None)


class NQLearnNetStrat(QNetStrat):
    """Every iteration, train on n-step return"""

    def __init__(self, *args, **kwargs):
        super().__init__("NQLearnNet", *args, **kwargs)
        self.n_step = self.pp['n_step']
        self.exps = []

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        self.exps.append((grid, cell, ch, reward, ce_type))
        if len(self.exps) == self.n_step:
            agrid, acell, ach, _, ace_type = self.exps[0]
            if ace_type != CEvent.END and ach is not None:
                rewards = [exp[3] for exp in self.exps]
                self.backward(
                    agrid,
                    acell, [ach],
                    rewards,
                    self.grid,
                    next_cell,
                    gamma=discount,
                    freps=None,
                    next_freps=None,
                    next_chs=None)
            del self.exps[0]

        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        return next_ch


class MNQLearnNetStrat(QNetStrat):
    """
    Gather n experiences, then train on n-step return,
    (n-1)-step return ..., 1-step return"""

    def __init__(self, *args, **kwargs):
        super().__init__("MNQLearnNet", *args, **kwargs)
        self.net.backward = self.net.backward_multi_nstep
        self.empty()

    def empty(self):
        self.grids, self.cells, self.chs, self.rewards, self.ce_types = [], [], [], [], []

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if ch is not None and ce_type != CEvent.END:
            self.grids.append(grid)
            self.cells.append(cell)
            self.chs.append(ch)
            self.rewards.append(reward)
            self.ce_types.append(ce_type)
        if len(self.grids) == self.n_step:
            self.backward(
                np.array(self.grids), self.cells, self.chs, self.rewards, self.grid,
                next_cell, None)
            self.empty()

        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        return next_ch


class GAEQLearnNetStrat(MNQLearnNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DIRTY HACK
        self.net.backward = self.net.backward_gae


class QLearnEligibleNetStrat(QNetStrat):
    """Update towards greedy, eligible, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("QlearnEligibleNet", *args, **kwargs)

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_ch,
                    next_max_ch, discount):
        """ Update qval for one experience tuple"""
        if ce_type != CEvent.END and ch is not None and next_max_ch is not None:
            if self.pp['freps']:
                frep, next_freps = GF.successive_freps(grid, cell, ce_type,
                                                       np.array([ch]))
                freps = [frep]
            else:
                freps, next_freps = None, None
            self.backward(
                grids=grid,
                cells=cell,
                chs=[ch],
                rewards=[reward],
                next_grids=next_grid,
                next_cells=next_cell,
                gamma=discount,
                freps=freps,
                next_freps=next_freps,
                next_chs=[next_max_ch],
                weights=None)


class SARSANetStrat(QNetStrat):
    """Update towards policy action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__("SARSANet", *args, **kwargs)

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_ch,
                    next_max_ch, discount):
        """ Update qval for one experience tuple"""
        if ce_type != CEvent.END and ch is not None and next_ch is not None:
            if self.pp['freps']:
                frep, next_freps = GF.successive_freps(grid, cell, ce_type,
                                                       np.array([ch]))
                freps = [frep]
            else:
                freps, next_freps = None, None
            self.backward(
                grids=grid,
                cells=cell,
                chs=[ch],
                rewards=[reward],
                next_grids=next_grid,
                next_cells=next_cell,
                gamma=discount,
                freps=freps,
                next_freps=next_freps,
                next_chs=[next_ch],
                weights=None)


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

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch,
                    discount):
        """ Update qval for one experience tuple"""
        self.backward(grid, cell, [ch], [reward], self.grid, next_cell, discount)


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
