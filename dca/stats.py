import time

import numpy as np
from matplotlib import pyplot as plt

from eventgen import ce_str


class Stats:
    """
    For monitoring (and warning about) simulation statistics
    """

    def __init__(self, pp, logger, pid="", *args, **kwargs):
        self.pp = pp
        self.logger = logger
        self.pid_str = "" if pid is "" else f"T{pid} "
        self.start_time = time.time()

        self.n_rejected = 0  # Number of rejected calls
        self.n_ended = 0  # Number of ended calls
        self.n_handoffs = 0  # Number of handoff events
        self.n_handoffs_rejected = 0  # Number of rejected handoff calls
        # Number of incoming (not necessarily accepted) calls
        self.n_incoming = 0
        # Number of channels in use at a cell when call is blocked
        self.n_inuse_rej = 0
        self.n_curr_rejected = 0  # Number of rejected calls last log_iter events
        self.n_curr_incoming = 0  # Number of incoming calls last log_iter events
        self.block_probs = []
        self.block_probs_cum = []  # For each log iter, cumulative new call block prob
        self.block_probs_cum_h = []  # For hoffs
        self.block_probs_cum_t = []  # New + hoff
        self.i = 0  # Current iteration
        self.t = 0  # Current time

        self.alphas = []  # Monitor alpha decay
        self.epsilons = []  # Monitor epsilon decay

    def event_new(self):
        self.n_incoming += 1
        self.n_curr_incoming += 1

    def event_new_reject(self, cell, n_used):
        self.n_rejected += 1
        self.n_curr_rejected += 1
        self.n_inuse_rej += n_used
        self.event_reject(cell, n_used)

    def event_end(self):
        self.n_ended += 1

    def event_hoff_new(self):
        # TODO Because n_incoming/n_rejected does not include
        # handoffs, a significant amount of data, 15 %, is
        # not included when judgning hyperparams.
        # A separate counter for new+hoff should be made.
        self.n_handoffs += 1

    def event_hoff_reject(self, cell, n_used):
        self.n_handoffs_rejected += 1
        self.event_reject(cell, n_used)

    def event_reject(self, cell, n_used):
        if n_used == 0:
            self.logger.debug(f"Rejected call to {cell} when {n_used}"
                              f" of {self.pp['n_channels']} channels in use")

    def iter(self, t, cevent):
        self.t = t
        self.i += 1
        self.logger.debug(ce_str(cevent))

        niter = self.pp['log_iter']
        if self.i > 0 and self.i % niter == 0:
            block_prob_cum = self.n_rejected / (self.n_incoming + 1)
            block_prob_cum_h = self.n_handoffs_rejected / (self.n_handoffs + 1)
            block_prob_cum_t = (self.n_rejected + self.n_handoffs_rejected) / (
                self.n_handoffs + self.n_incoming + 1)
            self.block_probs_cum.append(block_prob_cum)
            self.block_probs_cum_h.append(block_prob_cum_h)
            self.block_probs_cum_t.append(block_prob_cum_t)

            block_prob = self.n_curr_rejected / (self.n_curr_incoming + 1)
            self.block_probs.append(block_prob)
            self.logger.info(f"\n{self.pid_str}Blocking probability events"
                             f" {self.i-niter}-{self.i}:"
                             f" {block_prob:.4f}, cumulative {block_prob_cum:.4f}")
            self.n_curr_rejected = 0
            self.n_curr_incoming = 0

    def report_cac(self, admits=None, denies=None):
        admit_s = f"deny perc: {denies/(denies+admits)}"
        self.logger.error(admit_s)

    def report_rl(
            self,
            epsilon,
            alpha,
            losses,
            qval_means=None,
            avg_reward=None,
    ):
        self.alphas.append(alpha)
        self.epsilons.append(epsilon)

        niter = self.pp['log_iter']
        if type(losses[0]) is tuple:
            avg_losses = np.zeros(len(losses[0]))
            # Policy net has multiple losses
            for losses in losses[-niter:]:
                for i, loss in enumerate(losses):
                    avg_losses[i] += loss
            avg_losses /= niter
            # avg_loss = str.join(", ", [f"{i:.1f}" for i in avg_losses])
            avg_loss = f"Tot:{avg_losses[0]:.2E}, PG:{avg_losses[1]:.2E}" \
                       f" Val:{avg_losses[2]:.2E}, Ent:{avg_losses[3]:.2E}"
        else:
            avg_loss = f"{sum(losses[-niter:]) / niter :.2E}"
            max_loss = f"{max(losses[-niter:]):.2E}"

        avg_reward_s, avg_qval = "", ""
        if qval_means:
            avg_qval = f", qvals: {sum(qval_means[-niter:]) / niter :.2E}"
        if avg_reward:
            avg_reward_s = f", reward: {avg_reward:.2f}"
        self.logger.info(
            f"Epsilon: {epsilon:.5f}, Alpha: {alpha:.3E},"
            f" Avg|max loss last {niter} events: {avg_loss}|{max_loss}{avg_qval}{avg_reward_s}"
        )

    def report_weights(self, weights, names):
        for i, w in enumerate(weights):
            wmin, wmax, wavg = np.min(w), np.max(w), np.mean(w)
            name = names[i].replace("model/", "")
            self.logger.info(f"{name}:\t{wmin:.4f}, {wmax:.4f}, {wavg:.4f}")

    def end_episode(self, n_inprogress):
        delta = self.n_incoming + self.n_handoffs \
            - self.n_rejected - self.n_handoffs_rejected - self.n_ended
        if delta != n_inprogress:
            self.logger.error(f"\nSome calls were lost. Counted in progress {delta}."
                              f" Actual in progress: {n_inprogress}"
                              f"\nIncoming: {self.n_incoming}"
                              f"\nIncoming handoffs: {self.n_handoffs}"
                              f"\nRejected: {self.n_rejected}"
                              f"\nRejected handoffs: {self.n_handoffs_rejected}"
                              f"\nEnded: {self.n_ended}")

        t = time.time() - self.start_time
        m, s = map(int, divmod(t, 60))
        self.logger.warn(
            f"\nSimulation duration: {self.t/60:.2f} sim hours,"
            f" {m}m{s}s real, {self.i} events at {self.i/t:.0f} events/second")
        # Avoid zero divisions by adding 1 do dividers
        self.block_prob_cum = self.n_rejected / (self.n_incoming + 1)
        self.block_prob_cum_hoff = self.n_handoffs_rejected / (self.n_handoffs + 1)
        self.block_prob_cum_tot = (self.n_handoffs_rejected + self.n_rejected) / (
            self.n_handoffs + self.n_incoming + 1)
        if self.n_handoffs_rejected > 0:
            hoff_str = f", {self.block_prob_cum_hoff:.4f} for handoffs" \
                       f", {self.block_prob_cum_tot:.4f} total"
        else:
            hoff_str = ""
        self.logger.error(f"\n{self.pid_str}Blocking probability:"
                          f" {self.block_prob_cum:.4f} for new calls" + hoff_str)
        # self.logger.info(f"\nAverage number of calls in progress when blocking: "
        #                  f"{self.n_inuse_rej/(self.n_rejected+1):.2f}")

        # if self.pp['do_plot']:
        #     self.plot()

    def plot(self, losses=None):
        xlabel_iters = f"Iterations, in {self.pp['log_iter']}s"
        plt.subplot(221)
        plt.plot(self.block_probs_cum)
        plt.ylabel("Blocking probability")
        plt.xlabel(xlabel_iters)
        if self.alphas:
            plt.subplot(223)
            plt.plot(self.alphas)
            plt.ylabel("Alpha (learning rate)")
            plt.xlabel(xlabel_iters)
            plt.subplot(224)
            plt.plot(self.epsilons)
            plt.ylabel("Epsilon")
            plt.xlabel(xlabel_iters)
        if losses:
            plt.subplot(222)
            plt.plot(self.losses)
            plt.ylabel("Loss")
            plt.xlabel(xlabel_iters)
        plt.show()
