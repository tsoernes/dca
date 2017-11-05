from eventgen import ce_str

import time

from matplotlib import pyplot as plt


class Stats:
    """
    For monitoring (and warning about) simulation statistics
    """

    def __init__(self, pp, logger, pid="",
                 *args, **kwargs):
        self.pp = pp
        self.logger = logger
        self.pid = pid

        self.start_time = time.time()
        self.n_rejected = 0  # Number of rejected calls
        self.n_ended = 0  # Number of ended calls
        self.n_handoffs = 0  # Number of handoff events
        self.n_handoffs_rejected = 0  # Number of rejected handoff calls
        # Number of incoming (not necessarily accepted) calls
        self.n_incoming = 0
        # Number of channels in progress at a cell when call is blocked
        self.n_inuse_rej = 0
        self.n_curr_rejected = 0  # Number of rejected calls last 100 episodes
        self.n_curr_incoming = 0  # Number of incoming calls last 100 episodes
        self.block_probs = []
        self.block_probs_cum = []
        self.alphas = []  # Monitor alpha decay
        self.epsilons = []  # Monitor epsilon decay
        self.i = 0  # Current iteration
        self.t = 0  # Current time

    def new(self):
        self.n_incoming += 1
        self.n_curr_incoming += 1

    def new_rej(self, cell, n_used):
        self.n_rejected += 1
        self.n_curr_rejected += 1
        self.n_inuse_rej += n_used
        self.rej(cell, n_used)

    def end(self):
        self.n_ended += 1

    def hoff_new(self):
        # TODO Because n_incoming/n_rejected does not include
        # handoffs, a significant amount of data, 15 %, is
        # not included when judgning hyperparams.
        # A separate counter for new+hoff should be made.
        self.n_handoffs += 1

    def hoff_rej(self, cell, n_used):
        self.n_handoffs_rejected += 1
        self.rej(cell, n_used)

    def rej(self, cell, n_used):
        if n_used == 0:
            self.logger.debug(
                f"Rejected call to {cell} when {n_used}"
                f" of {self.pp['n_channels']} channels in use")

    def iter(self, t, i, cevent):
        self.t = t
        self.i = i
        self.logger.debug(ce_str(cevent))

    def n_iter(self, epsilon, alpha, losses):
        niter = self.pp['log_iter']
        # NOTE excluding handoffs
        block_prob = self.n_curr_rejected / (self.n_curr_incoming + 1)
        self.block_probs.append(block_prob)
        block_prob_cum = self.n_rejected / (self.n_incoming + 1)
        self.block_probs_cum.append(block_prob_cum)
        self.logger.info(
            f"\n{self.t:.2f}-{self.i}: Blocking probability events"
            f" {self.i-niter}-{self.i}:"
            f" {block_prob:.4f}, cumulative {block_prob_cum:.4f}")
        if epsilon:
            self.alphas.append(alpha)
            self.epsilons.append(epsilon)
            self.logger.info(f"Epsilon: {epsilon:.5f}," f" Alpha: {alpha:.5f}")
        if losses:
            avg_loss = sum(losses[-niter:]) / niter
            self.logger.info(f"Avg. loss: {avg_loss:.5f}")
        self.n_curr_rejected = 0
        self.n_curr_incoming = 0

    def end_episode(self, n_inprogress, epsilon, alpha):
        delta = self.n_incoming + self.n_handoffs \
            - self.n_rejected - self.n_handoffs_rejected - self.n_ended
        self.block_prob_cum = self.n_rejected / (self.n_incoming + 1)
        if delta != n_inprogress:
            self.logger.error(
                f"\nSome calls were lost. Counted in progress {delta}. "
                f" Actual in progress: {n_inprogress}"
                f"\nIncoming: {self.n_incoming}"
                f"\nIncoming handoffs: {self.n_handoffs}"
                f"\nRejected: {self.n_rejected}"
                f"\nRejected handoffs: {self.n_handoffs_rejected}"
                f"\nEnded: {self.n_ended}")
        self.logger.warn(
            f"\nSimulation duration: {self.t/24:.2f} sim hours(?),"
            f" {self.i+1} episodes"
            f" at {self.pp['n_events']/(time.time()-self.start_time):.0f}"
            " episodes/second"

            f"\nRejected {self.n_rejected} of {self.n_incoming} new calls,"
            f" {self.n_handoffs_rejected} of {self.n_handoffs} handoffs")
        if self.pp['test_params']:
            self.logger.error(f"\nT{self.pid} Using params:"
                              f" gamma {self.pp['gamma']:.6f},"
                              f" alpha {self.pp['alpha']:.8f},"
                              f" alphadec {self.pp['alpha_decay']:.8f}"
                              f" epsilon {self.pp['epsilon']:.8f},"
                              f" epsilondec {self.pp['alpha_decay']:.8f}")
        # Avoid zero divisions by adding 1 do dividers
        self.logger.error(
            f"\nT{self.pid} Blocking probability:"
            f" {self.block_prob_cum:.4f} for new calls, "
            f"{self.n_handoffs_rejected/(self.n_handoffs+1):.4f} for handoffs")
        self.logger.warn(
            f"\nAverage number of calls in progress when blocking: "
            f"{self.n_inuse_rej/(self.n_rejected+1):.2f}"

            f"\n{n_inprogress} calls in progress at simulation end\n")
        if epsilon:
            self.logger.warn(f"\nEnd epsilon: {epsilon}\nEnd alpha: {alpha}")
        if self.pp['do_plot']:
            self.plot()

    def plot(self):
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
        plt.show()
