import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF
from nets.afterstate import AfterstateNet  # noqa
from nets.singh import SinghNet
from nets.singh_ac import ACSinghNet
from nets.singh_lstd import LSTDSinghNet
from nets.singh_man import ManSinghNet
from nets.singh_resid import ResidSinghNet
from nets.singh_tdc import TDCSinghNet
from nets.singh_tdc_tf import TFTDCSinghNet
from nets.singhq import SinghQNet
from nets.utils import softmax
from strats.base import NetStrat


class VNetStrat(NetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = self.pp['beta']
        self.importance_sampl = self.pp['importance_sampling']
        self.next_p = 1
        self.next_max_ch = 0
        self.next_max_val = 0
        self.next_val = np.float32(0)
        self.next_ch = None

    def update_target_net(self):
        pass

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if ch is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell,
                             self.next_val, discount, self.next_max_ch, self.next_max_val,
                             self.next_p)
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        res = self.optimal_ch(next_ce_type, next_cell)
        next_ch, self.next_val, self.next_max_ch, self.next_max_val, self.next_p = res
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return None, 0, None, 0, 1
        else:
            chs = np.nonzero(self.grid[cell])[0]

        qvals_dense = self.get_qvals(self.grid, cell, ce_type, chs)
        self.qval_means.append(np.mean(qvals_dense))
        if ce_type == CEvent.END:
            idx = max_idx = np.argmax(qvals_dense)
            ch = max_ch = chs[idx]
            p = 1
            # print(qvals_dense, idx, np.max(qvals_dense))
            if self.pp['debug']:
                val = qvals_dense[idx]
                freps1 = NGF.afterstate_freps(self.grid, cell, ce_type, np.array([ch]))
                v1 = self.net.forward(freps1)[0]
                # print("\n", qvals_dense, idx)
                frep2 = GF.afterstate_freps_naive(self.grid, cell, ce_type, chs)[idx]
                v2 = self.net.forward([frep2])[0]
                v3 = self.net.forward([freps1[0], freps1[0]])[0]
                v4 = self.net.forward([frep2, frep2])[0]
                # val: multi freps, multi tf
                # v1: single frep, single tf
                # v2: multi freps, single tf
                # v3: single freps, multi tf
                # v4: multi freps, multi tf
                # (val == v3 == v4) != (v1 == v2)
                # CONCLUSION: Running multiple samples at once through TF
                # yields different (higher accuracy) results
                print(val, v1, v2, v3, v4, "\n")
        else:
            ch, idx, p = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            max_idx = np.argmax(qvals_dense)
            max_ch = chs[max_idx]
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")

        return ch, qvals_dense[idx], max_ch, qvals_dense[max_idx], p

    def get_qvals(self, grid, cell, ce_type, chs):
        freps = NGF.afterstate_freps(grid, cell, ce_type, chs)
        # Q-value for each ch in 'chs'
        qvals_dense = self.net.forward(freps)
        assert qvals_dense.shape == (len(chs), ), qvals_dense.shape
        return qvals_dense

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount, next_max_ch, next_max_val, next_p):
        if self.importance_sampl:
            weight = next_p
            ch = next_max_ch
        else:
            weight = 1
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        if self.pp['debug']:
            frep2 = GF.feature_reps(grid)[0]
            assert (frep == frep2).all()
            astates = GF.afterstates(grid, cell, ce_type, np.array([ch]))
            next_freps2 = GF.feature_reps(astates)
            assert (next_freps == next_freps2).all()
        self.backward(
            freps=[frep],
            rewards=[reward],
            next_freps=next_freps,
            weight=weight,
            discount=discount,
            next_val=next_val)


class SinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pre_conv=False, pp=self.pp, logger=self.logger)
        self.backward_fn = self.net.backward_supervised

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount, next_max_ch, next_max_val, next_p):
        if self.importance_sampl:
            weight = next_p
            next_val = next_max_val
        else:
            weight = 1
        frep = NGF.feature_rep(grid)
        value_target = reward + discount * next_val
        self.backward(freps=[frep], value_target=[[value_target]], weight=weight)


class DoubleSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(
            pre_conv=False, double_net=True, pp=self.pp, logger=self.logger)
        self.backward_fn = self.net.backward

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)


class WolfSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward_supervised
        self.val = 0

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        res = self.optimal_ch(next_ce_type, next_cell)
        next_ch, next_val, next_max_ch, next_max_val, next_p = res
        if ch is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell,
                             self.val, discount, next_ch)
        self.val = next_val
        return next_ch

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount, next_ch):
        weight = self.pp['wolf'] if next_ch is None else 1
        frep = NGF.feature_rep(grid)
        value_target = reward + discount * next_val
        self.backward(freps=[frep], value_target=[[value_target]], weight=weight)


class ManSinghNetStrat(VNetStrat):
    """Manual gradient calculation, just for illustration"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ManSinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward


class ResidSinghNetStrat(VNetStrat):
    """Residual gradients"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ResidSinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward


class TDCSinghNetStrat(VNetStrat):
    """TD(0) with Gradient Correction
    Without dt_rewards,
        lr=1e-6, weight_beta=1e-6
    seems like good strating points
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = TDCSinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward


class TFTDCSinghNetStrat(VNetStrat):
    """TensorFlow impl. of TDC"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = TFTDCSinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward


class AvgSinghNetStrat(VNetStrat):
    """Average reward formulation
    Without dt_rewards,
        lr=1e-7, weight_beta=1e-2
        lr=1e-6, weight_beta=1e-2
        lr=1e-5, weight_beta=1e-2
    seems like good strating points
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward
        assert self.pp['avg_reward']


class LSTDSinghNetStrat(VNetStrat):
    """Least Squares
    Good starting point
    --alpha 0.00000001 -epol greedy --beta 2600
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = LSTDSinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward


class VConvNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pre_conv=True, pp=self.pp, logger=self.logger)
        self.backward_fn = self.net.backward


class AlfySinghNetStrat(VNetStrat):
    """
    El-Alfy average semi-markov
    Ballpark
    -opt sgd -lr 1e-7
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        assert not self.pp['dt_rewards']
        self.backward_fn = self.net.backward_supervised
        self.sojourn_times = np.zeros(self.dims, np.float64)
        self.rewards = np.zeros(self.dims, np.float64)
        self.act_count = np.zeros(self.dims, np.int64)
        self.tot_rewards = 0
        self.tot_sotimes = 0
        self.t0 = 0

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        t1, next_ce_type, next_cell = next_cevent[:3]
        if ch is not None:
            dt = t1 - self.t0
            self.act_count[cell][ch] += 1
            immediate_reward = reward * dt
            self.rewards[cell][ch] += (immediate_reward - self.rewards[cell][ch]) \
                / self.act_count[cell][ch]
            immediate_avg_reward = self.rewards[cell][ch]
            self.sojourn_times[cell][ch] += (dt - self.sojourn_times[cell][ch]) \
                / self.act_count[cell][ch]
            avg_sojourn_time = self.sojourn_times[cell][ch]
            self.tot_rewards += immediate_reward
            self.tot_sotimes += dt
            frep = NGF.feature_rep(grid)
            value_target = immediate_avg_reward - self.tot_rewards / self.tot_sotimes \
                * avg_sojourn_time + self.next_val
            self.backward(freps=[frep], value_target=[[value_target]])
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        next_ch, self.next_val, _ = self.optimal_ch(next_ce_type, next_cell)
        self.t0 = t1
        return next_ch


class PSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.backward_fn = self.net.backward_supervised
        assert not self.pp['dt_rewards']

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, _ = self.optimal_ch(next_ce_type, next_cell)
        if ch is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cevent,
                             next_ch, discount)
        return next_ch

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cevent,
                    next_ch, discount):
        """If dt rewards is not used, the next reward is
        deterministic given next_cevent and state"""
        frep = NGF.feature_rep(grid)
        if next_ch is not None:
            next_ce_type, next_cell = next_cevent[1:3]
            reward2 = reward
            if next_cevent == CEvent.END:
                reward2 -= 1
            else:
                reward2 += 1
            bfreps = NGF.afterstate_freps(next_grid, next_cell, next_ce_type,
                                          np.array([next_ch]))
            bootstrap_val = self.net.forward(bfreps)[0]
            value_target = reward + discount * reward2 + (discount**2) * bootstrap_val
        else:
            next_freps = NGF.incremental_freps(grid, frep, cell, ce_type, np.array([ch]))
            bootstrap_val = self.net.forward(next_freps)[0]
            value_target = reward + discount * bootstrap_val
        self.backward(freps=[frep], value_target=[[value_target]])


class SinghQNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghQNet(self.pp, self.logger)
        self.backward_fn = self.net.backward

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        if ch is not None and next_ch is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell,
                             next_ch, discount)
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        return next_ch

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_ch,
                    discount):
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        self.backward(
            freps=[frep],
            cells=cell,
            chs=[ch],
            rewards=reward,
            next_freps=next_freps,
            next_cells=next_cell,
            next_chs=[next_ch],
            discount=discount)

    def get_qvals(self, grid, cell, ce_type, chs):
        frep = NGF.feature_rep(grid)
        # Q-value for each ch in 'chs'
        qvals_sparse = self.net.forward([frep], [cell])
        qvals_dense = qvals_sparse[chs]
        assert qvals_dense.shape == (len(chs), ), qvals_dense.shape
        return qvals_dense

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return None, 0
        else:
            chs = np.nonzero(self.grid[cell])[0]

        qvals_dense = self.get_qvals(self.grid, cell, ce_type, chs)
        self.qval_means.append(np.mean(qvals_dense))
        if ce_type == CEvent.END:
            idx = np.argmin(qvals_dense)
            ch = chs[idx]
        else:
            ch, idx = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[idx]


class RSMART(VNetStrat):
    """-lr 1e-7 --weight_beta 1e-5 --beta 2500"""

    # TODO Try beta gamma
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pp=self.pp, logger=self.logger)
        self.backward_fn = self.net.backward
        self.avg_reward = 0
        self.tot_reward = 0
        self.tot_time = 0
        self.t0 = 0
        self.weight_beta = self.pp['weight_beta']
        self.weight_beta_decay = self.pp['weight_beta_decay']

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        # value_target = reward + self.gamma * np.array([[self.val]])

        if ch is not None:
            frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
            dt = next_cevent[0] - self.t0
            # treward = reward * dt - self.avg_reward * dt
            treward = reward - self.avg_reward * dt
            self.backward(
                freps=[frep], rewards=treward, next_freps=next_freps, discount=discount)
            self.tot_reward += float(reward)
            self.tot_time += dt
            self.avg_reward = (1 - self.weight_beta) * self.avg_reward \
                + self.weight_beta * (self.tot_reward / self.tot_time)
            self.weight_beta *= self.weight_beta_decay

        self.t0 = next_cevent[0]
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, qval, next_max_ch, qval_max, p = self.optimal_ch(next_ce_type, next_cell)
        return next_ch


class ACNSinghStrat(NetStrat):
    """Actor Critic"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ACSinghNet(pp=self.pp, logger=self.logger)
        self.logger.info(
            "Loss legend (scaled): [ total, policy_grad, value_fn, entropy ]")

    def forward(self, cell):
        a, v = self.net.forward(self.grid, cell)
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

    def optimal_ch(self, ce_type, cell):
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
