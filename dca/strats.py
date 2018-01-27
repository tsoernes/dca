import signal
from typing import List, Tuple

import numpy as np

from environment import Env
from eventgen import CEvent
from grid import RhombusAxialGrid
from nets.acnet import ACNet
from nets.utils import softmax
from replaybuffer import ExperienceBuffer, ReplayBuffer


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
        self.grid = self.env.grid.state
        self.replaybuffer = ReplayBuffer(pp['buffer_size'], self.rows,
                                         self.cols, self.n_channels)

        self.quit_sim = False
        self.t = 0.1
        signal.signal(signal.SIGINT, self.exit_handler)

    def exit_handler(self, *args):
        """
        Graceful exit on ctrl-c signal from
        command line or on 'q' key-event from gui.
        """
        self.logger.error("\nPremature exit")
        self.quit_sim = True

    def simulate(self) -> Tuple[float, float]:
        """
        Run simulation and return a tuple with cumulative call
        block probability and cumulative handoff block probability
        """
        cevent = self.env.init_calls()
        ch = self.get_init_action(cevent)

        # Discrete event simulation
        i = 0
        while self.check_stop(i):
            if self.quit_sim:
                break  # Gracefully exit to print stats, clean up etc.

            self.t, ce_type, cell = cevent[0:3]

            grid = np.copy(self.grid)  # Copy before state is modified

            reward, next_cevent = self.env.step(ch)
            next_ch = self.get_action(next_cevent, grid, cell, ch, reward,
                                      ce_type)
            if (self.save or self.batch_size > 1) \
                    and ch is not None \
                    and ce_type != CEvent.END:
                # Only add (s, a, r, s', a') tuples for which the events in
                # s is not an END events, and for which there is an
                # available action a.
                # If there is no available action, that is, there are no
                # free channels which to assign, the neural net is not used
                # for selection and so it should not be trained on that data.
                # END events are not trained on either because the network is
                # supposed to predict the q-values for different channel
                # assignments; however the channels available for reassignment
                # are always busy in a grid corresponding to an END event.
                next_grid = np.copy(self.grid)
                next_cell = next_cevent[2]
                self.replaybuffer.add(grid, cell, ch, reward, next_grid,
                                      next_cell)

            if i > 0:
                if i % self.pp['log_iter'] == 0:
                    self.fn_report()
                if self.pp['net'] and \
                        i % self.net_copy_iter == 0:
                    self.update_target_net()
                if self.pp['net_copy_iter_decr'] and \
                   i % self.pp['net_copy_iter_decr'] == 0 and \
                   self.net_copy_iter > 0:
                    self.net_copy_iter -= 1
                    self.logger.info(
                        f"Decreased net copy iter to {self.net_copy_iter}")
            ch, cevent = next_ch, next_cevent
            i += 1
        self.env.stats.end_episode(reward)
        self.fn_after()
        if self.save:
            self.replaybuffer.save_experience_to_disk()
        if self.quit_sim and (self.pp['hopt'] or self.pp['avg_runs']):
            # Don't want to return actual block prob for incomplete sims when
            # optimizing params, because block prob is much lower at sim start
            return (1.0, 1.0)
        return (self.env.stats.block_prob_cum,
                self.env.stats.block_prob_cum_hoff)

    def check_stop(self, i) -> bool:
        if self.pp['n_hours'] is not None:
            return self.t < self.pp['n_hours']
        else:
            return i < self.pp['n_events']

    def get_init_action(self, next_cevent) -> int:
        """Return a channel to be (re)assigned in response to 'next_cevent'."""
        raise NotImplementedError

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        """Return a channel to be (re)assigned in response to 'next_cevent'.

        'cell' and 'ch' specify the action that was previously executed on
        'grid' in response to an event of type 'ce_type', resulting in
        'reward'.
        """
        raise NotImplementedError

    def fn_report(self):
        """
        Report stats for different strategies
        """
        pass

    def fn_after(self):
        """
        Cleanup after simulation
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
        self.logger.info(f"NP Rand: {np.random.uniform()}")

    def fn_report(self):
        self.env.stats.report_rl(self.epsilon, self.alpha)

    def fn_after(self):
        self.logger.info(f"NP Rand: {np.random.uniform()}")

    def get_init_action(self, cevent):
        ch, _, = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if ce_type != CEvent.END and ch is not None and next_ch is not None:
            # Observe reward from previous action, and
            # update q-values with one-step lookahead
            self.update_qval(grid, cell, ch, reward, next_cell, next_ch,
                             next_max_ch)
        return next_ch

    def policy_eps_greedy(self, chs, qvals):
        """Epsilon greedy action selection with expontential decay"""
        if np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            idx = np.argmax(qvals[chs])
            ch = chs[idx]
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return ch

    def policy_boltzmann(self, chs, qvals):
        scaled = np.exp((qvals[chs] - np.max(qvals[chs])) / self.temp)
        probs = scaled / np.sum(scaled)
        ch = np.random.choice(chs, p=probs)
        return ch

    def optimal_ch(self, ce_type, cell) -> Tuple[int, float, int]:
        """Select the channel fitting for assignment that
        that has the maximum q-value according to an exploration policy,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (ch, max_ch) where 'ch' is the selected channel according to
        exploration policy and max_ch' is the greedy (still eligible) channel.
        'ch' (and 'max_ch') is None if no channel is eligible for assignment.
        """
        inuse = np.nonzero(self.grid[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_free_chs(cell)
            if len(chs) == 0:
                # No channels available for assignment,
                return (None, None)
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            # or no channels in use to reassign
            assert n_used > 0

        # TODO If 'max_ch' turns out not to be useful, then don't return it and
        # avoid running a forward pass through the net if a random action is selected.
        qvals = self.get_qvals(cell=cell, n_used=n_used, ce_type=ce_type)
        # Selecting a ch for reassigment is always greedy because no learning
        # is done on the reassignment actions.
        if ce_type == CEvent.END:
            idx = np.argmin(qvals[chs])
            ch = chs[idx]
            max_ch = ch
        else:
            ch = self.policy_eps_greedy(chs, qvals)
            idx = np.argmax(qvals[chs])
            max_ch = chs[idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"{ce_type}\n{chs}\n{qvals}\n")
            raise Exception
        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (ch, max_ch)


class QTable(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmbda = self.pp['lambda']

    def get_qvals(self, cell, n_used, ch=None, *args, **kwargs):
        rep = self.feature_rep(cell, n_used)
        if ch is None:
            return self.qvals[rep]
        else:
            return self.qvals[rep][ch]

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch, *args):
        assert type(ch) == np.int64
        assert ch is not None
        # Counting n_used of self.grid instead of grid yields significantly better
        # performance for unknown reasons.
        next_n_used = np.count_nonzero(self.grid[cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        target_q = reward + self.gamma * next_qval
        n_used = np.count_nonzero(grid[cell])
        td_err = target_q - self.get_qvals(cell, n_used, ch)
        frep = self.feature_rep(cell, n_used)
        if self.lmbda is None:
            self.qvals[frep][ch] += self.alpha * td_err
        else:
            self.el_traces[frep][ch] += 1
            self.qvals += self.alpha * td_err * self.el_traces
            self.el_traces *= self.gamma * self.lmbda
        if self.alpha > self.pp['min_alpha']:
            self.alpha *= self.alpha_decay


class SARSA(QTable):
    """
    State consists of cell coordinates and the number of used channels in that cell.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "qvals[r][c][n_used][ch] = v"
        # Assigning channel 'ch' to the cell at row 'r', col 'c'
        # has q-value 'v' given that 'n_used' channels are already
        # in use at that cell.
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels,
                               self.n_channels))
        self.old_qvals = None  # Compare policy of current qvals to this
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros((self.rows, self.cols, self.n_channels,
                                       self.n_channels))

    def feature_rep(self, cell, n_used):
        return (*cell, n_used)


class TT_SARSA(RLStrat):
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
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros((self.rows, self.cols, self._k,
                                       self.n_channels))

    def feature_rep(self, cell, n_used):
        return (*cell, min(self.k - 1, n_used))


class RS_SARSA(QTable):
    """
    Reduced-state SARSA.
    State consists of cell coordinates only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qvals = np.zeros((self.rows, self.cols, self.n_channels))
        if self.lmbda is not None:
            # Eligibility traces
            self.el_traces = np.zeros((self.rows, self.cols, self.n_channels))

    def feature_rep(self, cell, n_used):
        return cell


class NetStrat(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_copy_iter = self.pp['net_copy_iter']
        self.losses = []

    def fn_report(self):
        self.env.stats.report_net(self.losses)
        self.env.stats.report_rl(self.epsilon)

    def fn_after(self):
        ra = self.net.rand_uniform()
        self.logger.info(f"TF Rand: {ra}, NP Rand: {np.random.uniform()}")
        if self.pp['save_net']:
            inp = ""
            if self.quit_sim:
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save model? Y/N: ")
            if inp in ["", "Y"]:
                self.net.save_model()
        self.net.save_timeline()
        self.net.sess.close()


class QNetStrat(NetStrat):
    def __init__(self, max_next_action, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from nets.qnet import QNet
        self.net = QNet(max_next_action, self.pp, self.logger)
        ra = self.net.rand_uniform()
        self.logger.info(f"TF Rand: {ra}")

    def update_target_net(self):
        self.net.sess.run(self.net.copy_online_to_target)

    def get_qvals(self, cell, ce_type, *args, **kwargs):
        if ce_type == CEvent.END:
            grid = np.copy(self.grid)
            grid[cell] = np.zeros(self.n_channels)
        else:
            grid = self.grid
        qvals = self.net.forward(grid, cell)
        return qvals

    def update_qval(self, grid, cell, ch, reward, next_cell, next_ch,
                    next_max_ch):
        """ Update qval for one experience tuple"""
        loss = self.backward(grid, cell, [ch], [reward], self.grid, next_cell,
                             [next_ch], [next_max_ch])
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)

    def backward(self, *args):
        raise NotImplementedError


class QLearnNetStrat(QNetStrat):
    """Update towards greedy, possibly illegal, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__(True, *args, **kwargs)
        if self.batch_size > 1:
            self.update_qval = self.update_qval_experience
            self.logger.warn("Using experience replay with batch"
                             f" size of {self.batch_size}")

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, *args,
                 **kwargs):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell)
        return loss

    def update_qval_experience(self, *args):
        """
        Update qval for pp['batch_size'] experience tuples,
        randomly sampled from the experience replay memory.
        """
        if len(self.replaybuffer) < self.pp['buffer_size']:
            # Can't backprop before exp store has enough experiences
            print("Not training" + str(len(self.replaybuffer)))
            return
        loss = self.net.backward(*self.replaybuffer.sample(
            self.pp['batch_size']))
        if np.isinf(loss) or np.isnan(loss):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)


class QLearnEligibleNetStrat(QNetStrat):
    """Update towards greedy, eligible, action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, next_ch,
                 next_max_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell,
                                 next_max_ch)
        return loss


class SARSANetStrat(QNetStrat):
    """Update towards policy action selection"""

    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

    def backward(self, grid, cell, ch, reward, next_grid, next_cell, next_ch,
                 next_max_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell,
                                 next_ch)
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
            self.update_qval(grid, cell, ch, reward, self.grid, next_cell,
                             next_ch)
            self.val = next_val
        return next_ch

    def optimal_ch(self, ce_type, cell) -> Tuple[int, float]:
        inuse = np.nonzero(self.grid[cell])[0]

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = self.env.grid.get_free_chs(cell)
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
                idx = np.random.choice(
                    np.range(len(valid_a_dist)), p=valid_a_dist)
        else:
            if greedy:
                idx = np.argmax(a_dist[chs])
            else:
                valid_a_dist = softmax(a_dist[chs])
                idx = np.random.choice(
                    np.range(len(valid_a_dist)), p=valid_a_dist)
        ch = chs[idx]
        # print(ce_type, a_dist, ch, a_dist[ch], chs)
        # TODO NOTE verify the above

        # If vals blow up ('NaN's and 'inf's), ch becomes none.
        if np.isinf(val) or np.isnan(val):
            self.logger.error(f"{ce_type}\n{chs}\n{val}\n\n")
            raise Exception

        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (ch, val)

    def update_qval(self, grid, cell, ch, reward, next_grid, next_cell,
                    next_ch):
        loss = self.net.backward(grid, cell, ch, reward, next_grid, next_cell)
        if np.isinf(loss[0]) or np.isnan(loss[0]):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)

    def update_qval_n_step(self, grid, cell, ch, reward, next_grid, next_cell,
                           next_ch):
        """
        Update qval for pp['batch_size'] experience tuple.
        """
        self.exp_buffer.add(grid, cell, self.val, ch, reward)
        if len(self.exp_buffer) < self.pp['n_step']:
            # Can't backprop before exp store has enough experiences
            return
        loss = self.net.backward_gae(*self.exp_buffer.pop(), next_grid,
                                     next_cell)
        if np.isinf(loss[0]) or np.isnan(loss[0]):
            self.logger.error(f"Invalid loss: {loss}")
            self.quit_sim = True
        else:
            self.losses.append(loss)
