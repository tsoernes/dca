import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF
from nets.afterstate import AfterstateNet
from nets.lstd import LSTDNet
from nets.singh import SinghNet
from nets.singh_man import ManSinghNet
from nets.singh_resid import ResidSinghNet
from nets.singh_tdc import TDCSinghNet
from nets.singhq import SinghQNet
from strats.base import NetStrat


class VNetStrat(NetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = self.pp['beta']
        self.next_val = 0

    def update_target_net(self):
        pass

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if ch is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell,
                             self.next_val, discount)
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        next_ch, self.next_val = self.optimal_ch(next_ce_type, next_cell)
        return next_ch

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
            idx = np.argmax(qvals_dense)
            ch = chs[idx]
        else:
            ch, idx = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[idx]

    def get_qvals(self, grid, cell, ce_type, chs):
        freps = NGF.afterstate_freps(grid, cell, ce_type, chs)
        # Q-value for each ch in 'chs'
        qvals_dense = self.net.forward(freps)
        assert qvals_dense.shape == (len(chs), ), qvals_dense.shape
        return qvals_dense

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount):
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        self.backward(
            freps=[frep], rewards=[reward], next_freps=next_freps, discount=discount)


class SinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.net.backward = self.net.backward_supervised

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount):
        frep = NGF.feature_rep(grid)
        value_target = reward + discount * next_val
        self.backward(freps=[frep], value_target=[[value_target]])


class ManSinghNetStrat(VNetStrat):
    """Manual gradient calculation, just for example"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ManSinghNet(self.pp, self.logger)


class ResidSinghNetStrat(VNetStrat):
    """Manual gradient calculation, just for example"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ResidSinghNet(self.pp, self.logger)


class TDCSinghNetStrat(VNetStrat):
    """TD(0) with Gradient Correction
    Without dt_rewards,
        lr=1e-6, weight_beta=1e-6
    seems like good strating points

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = TDCSinghNet(self.pp, self.logger)


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
        assert self.pp['avg_reward']

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount):
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        self.backward(freps=[frep], rewards=reward, next_freps=next_freps, gamma=None)


class LSTDSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = LSTDNet(self.pp, self.logger)

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount):
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        self.net.backward(
            frep=frep, reward=reward, next_frep=next_freps[0], gamma=discount)


class PSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.net.backward = self.net.backward_supervised
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
            value_target = reward + self.discount * reward2 + (
                self.discount**2) * bootstrap_val
        else:
            next_freps = NGF.incremental_freps(grid, frep, cell, ce_type, np.array([ch]))
            bootstrap_val = self.net.forward(next_freps)[0]
            value_target = reward + discount * bootstrap_val
        self.backward(freps=[frep], value_target=[[value_target]])


class WSinghNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        """Importance sampling"""
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.w = 1
        self.max_ch = 0

    def get_init_action(self, cevent):
        ch, self.w, self.max_ch = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if self.max_ch is not None:
            freps, next_freps = NGF.successive_freps(grid, cell, ce_type,
                                                     np.array([self.max_ch]))
            self.backward(
                freps=[freps],
                rewards=[reward],
                next_freps=next_freps,
                weight=self.w,
                gamma=discount)

        next_ch, self.w, self.max_ch = self.optimal_ch(next_ce_type, next_cell)
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return None, 0, None
        else:
            chs = np.nonzero(self.grid[cell])[0]

        qvals_dense = self.get_qvals(self.grid, cell, ce_type, chs)
        self.qval_means.append(np.mean(qvals_dense))
        amax_idx = np.argmax(qvals_dense)
        max_ch = chs[amax_idx]
        if ce_type == CEvent.END:
            ch = max_ch
            weight = 1
        else:
            ch = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            scaled = np.exp((qvals_dense - np.max(qvals_dense)) / self.epsilon)
            probs = scaled / np.sum(scaled)
            for i, ch2 in enumerate(chs):
                if ch == ch2:
                    weight = probs[i]
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, weight, max_ch


class SinghQNetStrat(VNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghQNet(self.pp, self.logger)

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell,
                    discount):
        frep, next_freps = NGF.successive_freps(grid, cell, ce_type, np.array([ch]))
        self.backward(
            freps=[frep],
            cells=cell,
            chs=[ch],
            rewards=[reward],
            next_freps=next_freps,
            next_cells=next_cell,
            gamma=discount)

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

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
            amax_idx = np.argmin(qvals_dense)
            ch = chs[amax_idx]
        else:
            ch = self.policy_part_eps_greedy(chs, qvals_dense, cell)

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, None


class VConvNetStrat(SinghNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = AfterstateNet(self.pp, self.logger)

    def get_qvals(self, grid, cell, ce_type, chs):
        # Just contains qvals for 'chs'
        qvals_dense = self.net.forward(NGF.afterstates(self.grid, cell, ce_type, chs))
        assert qvals_dense.shape == (len(chs), )
        return qvals_dense

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, discount):
        # self.logger.error(
        #     f"{gamma}, {reward}, {gamma/(gamma+(1-gamma)/self.pp['beta'])}")
        self.backward(
            freps=[grid], rewards=reward, next_freps=[self.grid], gamma=discount)


class RSMART(SinghNetStrat):
    """-lr 1e-7 --weight_beta 1e-5 --beta 2500"""

    # TODO Try beta gamma
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        self.avg_reward = 0
        self.tot_reward = 0
        self.tot_time = 0
        self.t0 = 0
        self.weight_beta = self.pp['weight_beta']

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        # value_target = reward + self.gamma * np.array([[self.val]])

        if ch is not None:
            freps = np.expand_dims(NGF.feature_rep(grid), axis=0)
            next_freps = NGF.incremental_freps(grid, freps[0], cell, ce_type,
                                               np.array([ch]))

            dt = next_cevent[0] - self.t0
            treward = reward - self.avg_reward * dt
            self.backward(
                freps=freps, rewards=treward, next_freps=next_freps, gamma=discount)
            self.tot_reward += float(reward)
            self.tot_time += dt
            self.avg_reward = (1 - self.weight_beta) * self.avg_reward \
                + self.weight_beta * (self.tot_reward / self.tot_time)
            # self.weight_beta *= self.alpha_decay

        self.t0 = next_cevent[0]
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        return next_ch
