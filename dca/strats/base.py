import signal
from typing import Tuple

import numpy as np

import gridfuncs_numba as GF
from environment import Env
from eventgen import CEvent
from replaybuffer import PrioritizedReplayBuffer
from utils import LinearSchedule


class Strat:
    def __init__(self, pp, logger, pid="", gui=None, *args, **kwargs):
        self.rows, self.cols, self.n_channels = self.dims = pp['dims']
        self.save = pp['save_exp_data']
        self.batch_size = pp['batch_size']
        self.n_hours, self.n_events = pp['n_hours'], pp['n_events']
        self.pp = pp
        self.logger = logger

        self.grid = np.zeros(pp['dims'], np.bool)
        self.env = Env(pp, self.grid, logger, pid)
        if (self.save or self.batch_size > 1):
            self.exp_buffer = PrioritizedReplayBuffer(
                size=pp['buffer_size'],
                rows=self.rows,
                cols=self.cols,
                n_channels=self.n_channels,
                alpha=0.6)
            self.pri_beta_schedule = LinearSchedule(
                self.n_events,
                initial_p=0.4,  # pp['prioritized_replay_beta'],
                final_p=1.0)
            self.prioritized_replay_eps = float(1e-6)

        self.quit_sim, self.invalid_loss, self.exceeded_bthresh = False, False, False
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
        self.i, t = 0, 0  # Iteration, time
        while self.continue_sim(self.i, t):
            t, ce_type, cell = cevent[0:3]
            grid = np.copy(self.grid)  # Copy before state is modified
            reward, beta_disc, next_cevent = self.env.step(ch)
            if self.pp['dt_rewards']:
                self.gamma = beta_disc
            next_ch = self.get_action(next_cevent, grid, cell, ch, reward, ce_type)
            # NOTE Could do per-strat saving here, as they save different stuff
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
                self.exp_buffer.add(
                    grid, cell, ch, reward, next_grid=next_grid, next_cell=next_cell)

            if self.i > 0:
                if self.i % self.pp['log_iter'] == 0:
                    self.fn_report()
                # NOTE Could do per iteration stuff in strats
                if self.pp['net'] and \
                        self.i % self.net_copy_iter == 0:
                    self.update_target_net()
                if self.pp['net_copy_iter_decr'] and \
                   self.i % self.pp['net_copy_iter_decr'] == 0 and \
                   self.net_copy_iter > 1:
                    self.net_copy_iter -= 1
                    self.logger.info(f"Decreased net copy iter to {self.net_copy_iter}")
            if (self.env.stats.block_probs_cum and  # yapf: disable
                    self.env.stats.block_probs_cum[-1] > self.pp['breakout_thresh']):
                self.logger.error("Block prob threshold exceeded; breaking out early")
                self.quit_sim = True
                self.exceeded_bthresh = True
            ch, cevent = next_ch, next_cevent
            self.i += 1
        self.env.stats.end_episode(np.count_nonzero(self.grid))
        self.fn_after()
        if self.save:
            self.exp_buffer.save_experience_to_disk()
        if self.quit_sim and (self.pp['hopt'] or self.pp['avg_runs']):
            # Don't want to return actual block prob for incomplete sims when
            # optimizing params, because block prob is much lower at sim start
            if self.invalid_loss:
                (None, None)
            return (1, 1)
        return (self.env.stats.block_prob_cum, self.env.stats.block_prob_cum_hoff)

    def continue_sim(self, i, t) -> bool:
        if self.quit_sim:
            return False  # Gracefully exit to print stats, clean up etc.
        elif self.n_hours is not None:
            return (t / 60) < self.n_hours
        else:
            return i < self.n_events

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
        if self.pp['dt_rewards'] is False:
            self.gamma = pp['gamma']
        self.logger.info(f"NP seed: {np.random.get_state()[1][0]}")
        self.losses = [0]

    def fn_report(self):
        self.env.stats.report_rl(self.epsilon, self.alpha, self.losses)

    def get_init_action(self, cevent):
        ch, _, = self.optimal_ch(ce_type=cevent[1], cell=cevent[2])
        return ch

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if ce_type != CEvent.END and  \
           ch is not None and next_ch is not None:
            # Observe reward from previous action, and
            # update q-values with one-step lookahead
            self.update_qval(grid, cell, ch, reward, next_cell, next_ch, next_max_ch)
        return next_ch

    def policy_eps_greedy(self, chs, qvals_dense):
        """Epsilon greedy action selection with expontential decay"""
        if np.random.random() < self.epsilon:
            # Choose an eligible channel at random
            ch = np.random.choice(chs)
        else:
            # Choose greedily
            idx = np.argmax(qvals_dense)
            ch = chs[idx]
        self.epsilon *= self.epsilon_decay  # Epsilon decay
        return ch

    def policy_boltzmann(self, chs, qvals_dense):
        scaled = np.exp((qvals_dense - np.max(qvals_dense)) / self.temp)
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
            chs = GF.get_eligible_chs(self.grid, cell)
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
        qvals_dense = self.get_qvals(cell=cell, n_used=n_used, ce_type=ce_type, chs=chs)
        # Selecting a ch for reassigment is always greedy because no learning
        # is done on the reassignment actions.
        if ce_type == CEvent.END:
            amin_idx = np.argmin(qvals_dense)
            ch = chs[amin_idx]
            max_ch = ch
        else:
            # print(qvals_dense.shape, chs.shape)
            ch = self.policy_eps_greedy(chs, qvals_dense)
            amax_idx = np.argmax(qvals_dense)
            max_ch = chs[amax_idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
            raise Exception
        self.logger.debug(f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}")
        return (ch, max_ch)
