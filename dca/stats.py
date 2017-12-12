import time

from matplotlib import pyplot as plt

from eventgen import ce_str


class Stats:
    """
    For monitoring (and warning about) simulation statistics
    """

    def __init__(self, pp, logger, pid="", *args, **kwargs):
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
        self.i = 0  # Current iteration
        self.t = 0  # Current time

        self.alphas = []  # Monitor alpha decay
        self.epsilons = []  # Monitor epsilon decay

        self.losses = []

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
            # NOTE excluding handoffs
            block_prob = self.n_curr_rejected / (self.n_curr_incoming + 1)
            self.block_probs.append(block_prob)
            block_prob_cum = self.n_rejected / (self.n_incoming + 1)
            self.block_probs_cum.append(block_prob_cum)
            self.logger.info(
                f"\nBlocking probability events"
                f" {self.i-niter}-{self.i}:"
                f" {block_prob:.4f}, cumulative {block_prob_cum:.4f}")
            self.n_curr_rejected = 0
            self.n_curr_incoming = 0

    def report_rl(self, epsilon, alpha):
        self.alphas.append(alpha)
        self.epsilons.append(epsilon)
        self.logger.info(f"Epsilon: {epsilon:.5f}," f" Alpha: {alpha:.5f}")

    def report_net(self, losses):
        self.losses = losses
        niter = self.pp['log_iter']
        avg_loss = sum(losses[-niter:]) / niter
        self.logger.info(f"Avg. loss last {niter} events: {avg_loss:.2f}")

    def end_episode(self, n_inprogress):
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
        t = time.time() - self.start_time
        m, s = map(int, divmod(t, 60))
        self.logger.warn(f"\nSimulation duration: {self.t/60:.2f} sim hours,"
                         f" {m}m{s}s real,"
                         f" {self.i+1} episodes"
                         f" at {self.pp['n_events']/t:.0f}"
                         " episodes/second")
        # Avoid zero divisions by adding 1 do dividers
        self.logger.error(
            f"\nT{self.pid} Blocking probability:"
            f" {self.block_prob_cum:.4f} for new calls, "
            f"{self.n_handoffs_rejected/(self.n_handoffs+1):.4f} for handoffs")
        self.logger.warn(
            f"\nAverage number of calls in progress when blocking: "
            f"{self.n_inuse_rej/(self.n_rejected+1):.2f}")
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
        if self.losses:
            plt.subplot(222)
            plt.plot(self.losses)
            plt.ylabel("Loss")
            plt.xlabel(xlabel_iters)
        plt.show()
