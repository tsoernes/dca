from collections import namedtuple

import numpy as np

import gridfuncs_numba as NGF
from eventgen import CEvent
from gridfuncs import GF
from nets.afterstate import AfterstateNet  # noqa
from nets.singh import SinghNet
from nets.singh_ac import ACSinghNet
from nets.singh_gtd2 import GTD2SinghNet
from nets.singh_lstd import LSTDSinghNet
from nets.singh_man import ManSinghNet
from nets.singh_ppo import PPOSinghNet
from nets.singh_q import SinghQNet
from nets.singh_resid import ResidSinghNet
from nets.singh_tdc import TDCSinghNet
from nets.singh_tdc_tf import TFTDCSinghNet
from nets.utils import softmax
from strats.base import NetStrat
from strats.exp_policies import BoltzmannGumbel


class VNetBase(NetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_sampl = self.pp['importance_sampling']
        self.p = 1
        self.max_ch = 0
        self.next_val = np.float32(0)

        if self.pp['avg_reward']:
            self.weight_beta = self.pp['weight_beta']
            self.weight_beta_decay = self.pp['weight_beta_decay']
            self.avg_reward = 0
            self.update_qval = self.update_qval_avg
        else:
            self.update_qval = self.update_qval_disc

        frepfuncs = NGF.get_frep_funcs(self.pp['frep_type'])
        self.afterstate_freps = frepfuncs['afterstate_freps']
        self.feature_rep = frepfuncs['feature_rep']
        self.feature_reps = frepfuncs['feature_reps']
        self.prep_net(self.pp['prep_net'])

    def prep_net(self, n=200):
        """ Pre-train net on nominal chs """
        r = np.count_nonzero(GF.nom_chs_mask)
        frep = self.feature_rep(GF.nom_chs_mask)
        for _ in range(n):
            self.net.backward(freps=[frep], value_target=[r], grids=GF.nom_chs_mask)

    def update_target_net(self):
        pass

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        if ch is not None:
            self.update_qval(grid, cell, ce_type, ch, self.max_ch, self.p, reward,
                             self.grid, self.next_val, discount)
        # 'next_ch' will be passed as 'ch' next time get_action is called,
        # and self.next_val will be the value of executing then 'ch' on then 'grid'
        # i.e. the value of then 'self.grid'
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, next_val, self.max_ch, next_max_val, p = self.optimal_ch(
            next_ce_type, next_cell)
        # NOTE This looks funny. If imp sampling, is 'p' prob of max ch?
        if self.importance_sampl:
            self.p = p
            self.next_val = next_max_val
        else:
            self.p = 1
            self.next_val = next_val
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return (None, ) * 5
        else:
            chs = np.nonzero(self.grid[cell])[0]

        qvals_dense = self.get_qvals(self.grid, cell, ce_type, chs)
        self.qval_means.append(np.mean(qvals_dense))
        if ce_type == CEvent.END:
            idx = max_idx = np.argmax(qvals_dense)
            ch = max_ch = chs[idx]
            p = 1
            if self.pp['debug']:
                val = qvals_dense[idx]
                _, freps1 = self.afterstate_freps(self.grid, cell, ce_type,
                                                  np.array([ch]))
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
                # ON CPU:
                # (val == v3 == v4) != (v1 == v2)
                # ON GPU: All the same...
                # CONCLUSION: Running multiple samples at once through TF
                # yields different (higher accuracy) results
                print(val, v1, v2, v3, v4, "\n")
        else:
            ch, idx, p = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            max_idx = np.argmax(qvals_dense)
            max_ch = chs[max_idx]
            self.epsilon *= self.epsilon_decay

        assert ch is not None, f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n"
        return ch, qvals_dense[idx], max_ch, qvals_dense[max_idx], p

    def get_qvals(self, grid, cell, ce_type, chs):
        _, freps = self.afterstate_freps(grid, cell, ce_type, chs)
        grids = NGF.afterstates(grid, cell, ce_type, chs)
        # Q-value for each ch in 'chs'
        qvals_dense = self.net.forward(freps, grids)
        assert qvals_dense.shape == (len(chs), ), qvals_dense.shape
        return qvals_dense

    def update_qval_disc(self, grid, cell, ce_type, ch, max_ch, p, reward, next_grid,
                         next_val, discount):
        """
        :param grid: Grid on which action is executed
        :param cell: Cell in which action is executed
        :param ce_type: Action type
        :param ch: Selected action
        :param max_ch: Greedy action
        :param p: Probability of action under policy
        :param reward: Reward for executing action
        :param next_grid: Resulting grid
        :param next_val: Value of next_grid
        :param discount: Discount factor, e.g. gamma
        :returns: None

        """
        frep = self.feature_rep(grid)
        value_target = reward + discount * next_val
        self.backward(freps=[frep], value_targets=[value_target], grids=grid, weights=[p])

    def update_qval_avg(self, grid, cell, ce_type, ch, max_ch, p, reward, next_grid,
                        next_val, discount):
        """ Average reward formulation """
        frep = self.feature_rep(grid)
        value_target = reward + next_val - self.avg_reward
        err = self.backward(
            freps=[frep], value_targets=[value_target], grids=grid, weights=[p])
        if ch == max_ch:
            self.avg_reward += self.weight_beta * np.mean(err)
            # self.weight_beta *= self.weight_beta_decay

    def update_qval_rsmart_mdp(self, grid, cell, ce_type, ch, max_ch, p, reward,
                               next_grid, next_val, discount):
        """ RSMART for MDP """
        frep = self.feature_rep(grid)
        value_target = reward + next_val - self.avg_reward
        self.backward(freps=[frep], value_targets=[value_target], grids=grid, weights=[p])
        if ch == max_ch:
            self.avg_reward = (
                1 - self.weight_beta) * self.avg_reward + self.weight_beta * reward
            self.weight_beta *= self.weight_beta_decay

    def update_qval_rsmart_smdp(self, grid, cell, ce_type, ch, max_ch, p, reward,
                                next_grid, next_val, discount):
        """ RSMART for SMDP """
        frep = self.feature_rep(grid)
        value_target = reward + next_val - self.avg_reward
        self.backward(freps=[frep], value_targets=[value_target], grids=grid, weights=[p])
        if ch == max_ch:
            self.avg_reward = (
                1 - self.weight_beta) * self.avg_reward + self.weight_beta * reward
            self.weight_beta *= self.weight_beta_decay


class SinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pp=self.pp, logger=self.logger)


class ExpSinghNetStrat(VNetBase):
    """ Experience replay """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pp=self.pp, logger=self.logger)
        assert not self.pp['avg_reward']
        assert self.batch_size > 1
        self.bgumbel = BoltzmannGumbel(c=self.pp['exp_policy_param']).select_action

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount, max_ch, next_max_val, p):
        """
        Update qval for pp['batch_size'] experience tuples,
        sampled from the experience replay memory.
        """
        if self.i == 30000:
            self.logger.error("SWITCHED TO BGUMBEL")
            self.exploration_policy = self.bgumbel
        if self.importance_sampl:
            chs = [ch, max_ch]
        else:
            chs = [ch]
        frep, next_freps = self.afterstate_freps(grid, cell, ce_type, np.array(chs))
        gamma = self.gamma_schedule.value(self.i)
        td_err = self.backward(
            grids=grid,
            freps=[frep],
            rewards=[reward],
            next_grids=self.grid,
            next_freps=[next_freps[0]],
            weights=[1],
            discount=gamma).reshape([-1])

        if len(self.exp_buffer) >= 1000:  # self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            data, weights, batch_idxes = self.exp_buffer.sample(
                self.pp['batch_size'], beta=self.pri_beta_schedule.value(self.i))
            data['weights'] = weights
            td_errs = self.backward(**data, discount=gamma).reshape([-1])
            new_priorities = np.abs(td_errs) + self.prioritized_replay_eps
            self.exp_buffer.update_priorities(batch_idxes, new_priorities)

        pri = np.abs(td_err) + self.prioritized_replay_eps
        self.exp_buffer.add_with_pri(
            priority=pri,
            grid=grid,
            frep=frep,
            reward=reward,
            next_grid=self.grid,
            next_frep=next_freps[-1])


class ExpAvgSinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_size > 1
        assert self.pp['avg_reward']
        self.next_elig_freps = None

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if ch is not None and self.next_elig_freps is not None:
            self.update_qval(grid, cell, ce_type, ch, reward, self.grid, next_cell,
                             self.next_val, discount, self.max_ch, self.next_elig_freps)
        res = self.optimal_ch(next_ce_type, next_cell)
        next_ch, self.next_val, self.max_ch, self.next_elig_freps = res
        return next_ch

    def optimal_ch(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return (None, ) * 4
        else:
            chs = np.nonzero(self.grid[cell])[0]

        qvals_dense, freps = self.get_qvals(self.grid, cell, ce_type, chs)
        self.qval_means.append(np.mean(qvals_dense))
        if ce_type == CEvent.END:
            idx = max_idx = np.argmax(qvals_dense)
            ch = max_ch = chs[idx]
        else:
            ch, idx, _ = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            max_idx = np.argmax(qvals_dense)
            max_ch = chs[max_idx]
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")

        return ch, qvals_dense[idx], max_ch, freps

    def update_qval(self, grid, cell, ce_type, ch, reward, next_grid, next_cell, next_val,
                    discount, max_ch, next_elig_freps):
        """
        Update qval for pp['batch_size'] experience tuples,
        sampled from the experience replay memory.
        """
        # frep, next_freps = self.afterstate_freps(grid, cell, ce_type, np.array(chs))
        frep = self.feature_rep_big2(grid)
        value_target = reward + next_val - self.avg_reward
        td_err = self.backward(freps=[frep], value_target=[[value_target]]).reshape([-1])
        if ch == max_ch:
            # TODO can possibly move this to optimal ch and update each iter since reward
            # and next_val is known
            self.avg_reward += self.weight_beta * td_err[0]

        if len(self.exp_buffer) >= 1000:  # self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            data, weights, batch_idxes = self.exp_buffer.sample(
                self.pp['batch_size'], beta=self.pri_beta_schedule.value(self.i))
            data['weights'] = weights
            data['value_target'] = np.zeros(self.pp['batch_size'])
            for i, nef in enumerate(data['next_elig_freps']):
                next_vals = self.net.forward(np.array(nef))
                data['value_target'][
                    i] = data['rewards'][i] - self.avg_reward + np.max(next_vals)
            # data['value_target'] = np.expand_dims(
            #     data['rewards'] - self.avg_reward + next_vals, axis=1)
            data['value_target'] = np.expand_dims(data['value_target'], axis=1)
            td_errs = self.backward(**data).reshape([-1])
            self.avg_reward += self.weight_beta * np.mean(td_errs)
            new_priorities = np.abs(td_errs) + self.prioritized_replay_eps
            self.exp_buffer.update_priorities(batch_idxes, new_priorities)

        pri = np.abs(td_err) + self.prioritized_replay_eps
        self.exp_buffer.add_with_pri(
            priority=pri, frep=frep, reward=reward, next_elig_freps=next_elig_freps)


class ASinghNetStrat(VNetBase):
    """Update on none"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(pp=self.pp, logger=self.logger)

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        # if ce_type != CEvent.END:
        frep = self.feature_rep(grid)
        next_frep = self.feature_rep(self.grid)
        self.backward(
            freps=[frep],
            rewards=[reward],
            next_freps=[next_frep],
            discount=discount,
            weight=1)
        # 'next_ch' will be 'ch' next iteration, thus "self.next_val" the value
        # of 'self.grid' after its execution.
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, self.next_val, *_ = self.optimal_ch(next_ce_type, next_cell)
        return next_ch


class WolfSinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
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
        frep = self.feature_rep(grid)
        value_target = reward + discount * next_val
        self.backward(freps=[frep], value_target=[[value_target]], weight=weight)


class ManSinghNetStrat(VNetBase):
    """Manual gradient calculation, for illustration purposes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ManSinghNet(self.pp, self.logger)


class ResidSinghNetStrat(VNetBase):
    """Residual gradients"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = ResidSinghNet(self.pp, self.logger)


class TDCSinghNetStrat(VNetBase):
    """TD(0) with Gradient Correction
    Without beta/dt_rewards,
        lr=1e-6, weight_beta=1e-6
    seems like good strating points
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = TDCSinghNet(self.pp, self.logger)


class TFTDCSinghNetStrat(VNetBase):
    """TensorFlow impl. of TDC"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = TFTDCSinghNet(self.pp, self.logger)


class GTD2SinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = GTD2SinghNet(self.pp, self.logger)


class LSTDSinghNetStrat(VNetBase):
    """Least Squares
    Good starting point
    --alpha 0.00000001 -epol greedy --beta 2600
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = LSTDSinghNet(self.pp, self.logger)


class AlfySinghNetStrat(VNetBase):
    """
    El-Alfy average semi-markov
    Ballpark
    -opt sgd -lr 1e-7
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
        assert not self.pp['dt_rewards']
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
            frep = self.feature_rep(grid)
            value_target = immediate_avg_reward - self.tot_rewards / self.tot_sotimes \
                * avg_sojourn_time + self.next_val
            self.backward(freps=[frep], value_target=[[value_target]])
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        next_ch, self.next_val, _ = self.optimal_ch(next_ce_type, next_cell)
        self.t0 = t1
        return next_ch


class PSinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghNet(self.pp, self.logger)
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
        frep = self.feature_rep(grid)
        if next_ch is not None:
            next_ce_type, next_cell = next_cevent[1:3]
            reward2 = reward
            if next_cevent == CEvent.END:
                reward2 -= 1
            else:
                reward2 += 1
            _, bfreps = self.afterstate_freps(next_grid, next_cell, next_ce_type,
                                              np.array([next_ch]))
            bootstrap_val = self.net.forward(bfreps)[0]
            value_target = reward + discount * reward2 + (discount**2) * bootstrap_val
        else:
            next_freps = NGF.incremental_freps(grid, frep, cell, ce_type, np.array([ch]))
            bootstrap_val = self.net.forward(next_freps)[0]
            value_target = reward + discount * bootstrap_val
        self.backward(freps=[frep], value_target=[[value_target]])


class SinghQNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = SinghQNet(self.pp, self.logger)

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        if ch is not None:
            frep, next_freps = self.afterstate_freps(grid, cell, ce_type, np.array([ch]))
            elig_map = NGF.eligible_map_all(self.grid)
            self.backward(
                grids=[grid],
                freps=[frep],
                cells=[cell],
                chs=[ch],
                rewards=reward,
                next_elig=[elig_map],
                next_grids=[self.grid],
                next_freps=next_freps,
                next_cells=next_cell,
                discount=discount)
        next_ch, next_val = self.optimal_ch(next_ce_type, next_cell)
        # 'next_ch' will be 'ch' next iteration, thus the value of 'self.grid' after
        # its execution.
        return next_ch

    def get_qvals(self, grid, cell, ce_type, chs):
        frep = self.feature_rep(grid)
        # Q-value for each ch in 'chs'
        qvals_sparse = self.net.forward([grid], [frep], [cell])
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
            ch, idx, p = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[idx]


class RSMARTSMDPNet(SinghNetStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pp['beta']
        self.tot_reward = 0
        self.tot_time = 0
        self.t0 = 0

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        if ch is not None:
            frep, next_freps = self.afterstate_freps(grid, cell, ce_type, np.array([ch]))
            dt = next_cevent[0] - self.t0
            # treward = reward * dt - self.avg_reward * dt
            value_target = reward - self.avg_reward * dt + self.next_val
            self.backward(grids=grid, freps=[frep], value_targets=[value_target])
            self.tot_reward = (
                1 - self.weight_beta) * self.tot_reward + self.weight_beta * float(reward)
            self.tot_time = (
                1 - self.weight_beta) * self.tot_time + self.weight_beta * float(dt)
            self.avg_reward = (1 - self.weight_beta) * self.avg_reward \
                + self.weight_beta * (self.tot_reward / self.tot_time)
            self.weight_beta *= self.weight_beta_decay

        self.t0, next_ce_type, next_cell = next_cevent[:3]
        next_ch, self.next_val, next_max_ch, qval_max, p = self.optimal_ch(
            next_ce_type, next_cell)
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


class PPOSinghNetStrat(VNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = PPOSinghNet(pp=self.pp, logger=self.logger)
        assert self.pp['avg_reward']
        self.rest = None
        self.buf = []
        self.n_step = self.pp['n_step']
        self.step = namedtuple(
            'step',
            ('ch', 'next_val', 'frep', 'next_frep', 'neglogpac', 'reward', 'cell'))
        self.i_pol_train = 5_000
        self.i_pol_act = 20_000
        self.pol_train = False
        self.pol_act = False
        self.optimal_ch = self.optimal_ch_val

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        if self.i == self.i_pol_train:
            self.pol_train = True
            print("POL TRAIN")
        if self.i == self.i_pol_act:
            self.pol_act = True
            self.optimal_ch = self.optimal_ch_pol
            print("POL ACT")
        if ch is not None and self.rest is not None:
            # Train policy network
            step = self.step(ch, *self.rest, reward, cell)
            if self.pol_train:
                self.buf.append(step)
            self.backward(step, self.buf, n_step=self.n_step)
            if self.pol_train:
                if len(self.buf) == self.n_step:
                    self.buf = []
        next_ce_type, next_cell = next_cevent[1:3]
        next_ch, *self.rest = self.optimal_ch(next_ce_type, next_cell)
        return next_ch

    def optimal_ch_pol(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return (None, ) * 5
        else:
            chs = np.nonzero(self.grid[cell])[0]

        frep = self.feature_rep(self.grid)
        ch, neglogpac = self.net.forward_action(frep, cell, ce_type, chs)
        next_frep = NGF.incremental_freps(self.grid, frep, cell, ce_type, np.array([ch]))
        val = self.net.forward_value(next_frep)
        return ch, val, frep, next_frep, neglogpac

    def optimal_ch_val(self, ce_type, cell) -> int:
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = GF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                return (None, ) * 5
        else:
            chs = np.nonzero(self.grid[cell])[0]
        old_frep, freps = self.afterstate_freps(self.grid, cell, ce_type, chs)
        # Q-value for each ch in 'chs'
        qvals_dense = self.net.forward_value(freps).reshape(len(chs))
        if ce_type == CEvent.END:
            idx = np.argmax(qvals_dense)
            ch = chs[idx]
            # max_idx = idx
            # max_ch  = ch
        else:
            ch, idx, _ = self.exploration_policy(self.epsilon, chs, qvals_dense, cell)
            # max_idx = np.argmax(qvals_dense)
            # max_ch = chs[max_idx]
            self.epsilon *= self.epsilon_decay

        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
        return ch, qvals_dense[idx], old_frep, freps[idx], self.net.get_neglogpac(
            old_frep, cell, ch)
