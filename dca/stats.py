from eventgen import ce_str

import time

from matplotlib import pyplot as plt


class Stats:
    """
    For monitoring (and warning about) simulation statistics
    """

    def __init__(self, logger, n_channels, log_iter, n_episodes, do_plot,
                 *args, **kwargs):
        self.logger = logger
        self.n_channels = n_channels
        self.log_iter = log_iter
        self.n_episodes = n_episodes
        self.do_plot = do_plot

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
        self.alphas = []
        self.epsilons = []
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
        self.n_handoffs += 1

    def hoff_rej(self, cell, n_used):
        self.n_handoffs_rejected += 1
        self.rej(cell, n_used)

    def rej(self, cell, n_used):
        if n_used == 0:
            lgger = self.logger.info
        else:
            lgger = self.logger.debug
        lgger(f"Rejected call to {cell} when {n_used}"
              f" of {self.n_channels} channels in use")

    def iter(self, t, i, cevent):
        self.t = t
        self.i = i
        self.logger.debug(ce_str(cevent))

    def n_iter(self, epsilon, alpha):
        # NOTE excluding handoffs
        block_prob = self.n_curr_rejected/(self.n_curr_incoming+1)
        self.block_probs.append(block_prob)
        # self.logger.info(
        #         f"\n{self.t:.2f}-{self.i}: Blocking probability events"
        #         f" {self.i-self.log_iter}-{self.log_iter}:"
        #         f" {block_prob:.4f}")
        if epsilon:
            self.alphas.append(alpha)
            self.epsilons.append(epsilon)
            self.logger.info(
                f"Epsilon: {epsilon:.5f},"
                f" Alpha: {alpha:.5f}")
        self.n_curr_rejected = 0
        self.n_curr_incoming = 0

    def endsim(self, n_inprogress):
        delta = self.n_incoming + self.n_handoffs \
                - self.n_rejected - self.n_handoffs_rejected - self.n_ended
        if delta != n_inprogress:
            self.logger.error(
                    f"\nSome calls were lost. Counted in progress {delta}. "
                    f" Actual in progress: {n_inprogress}"
                    f"\nIncoming: {self.n_incoming}"
                    f"\nIncoming handoffs: {self.n_handoffs}"
                    f"\nRejected: {self.n_rejected}"
                    f"\nRejected handoffs: {self.n_handoffs_rejected}"
                    f"\nEnded: {self.n_ended}")
        # Avoid zero divisions by adding 1 do dividers
        self.logger.warn(
            f"\nSimulation duration: {self.t/24:.2f} sim hours(?),"
            f" {self.i+1} episodes"
            f" at {self.n_episodes/(time.time()-self.start_time):.0f}"
            " episodes/second"

            f"\nRejected {self.n_rejected} of {self.n_incoming} new calls,"
            f" {self.n_handoffs_rejected} of {self.n_handoffs} handoffs"

            f"\nBlocking probability:"
            f" {self.n_rejected/(self.n_incoming+1):.4f} for new calls,"
            f" {self.n_handoffs_rejected/(self.n_handoffs+1):.4f} for handoffs"

            f"\nAverage number of calls in progress when blocking: "
            f"{self.n_inuse_rej/(self.n_rejected+1):.2f}"

            f"\n{n_inprogress} calls in progress at simulation end\n")
        if self.do_plot:
            self.plot()

    def plot(self):
        plt.subplot(221)
        plt.plot(self.block_probs)
        plt.ylabel("Blocking probability")
        plt.xlabel(f"Iterations, in {self.log_iter}s")
        if self.alphas:
            plt.subplot(223)
            plt.plot(self.alphas)
            plt.ylabel("Alpha (learning rate)")
            plt.xlabel(f"Iterations, in {self.log_iter}s")
            plt.subplot(224)
            plt.plot(self.epsilons)
            plt.ylabel("Epsilon")
            plt.xlabel(f"Iterations, in {self.log_iter}s")
        plt.show()
