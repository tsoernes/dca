import time


class Stats:
    """
    For monitoring (and warning about) simulation statistics
    """

    def __init__(self, logger, n_channels, log_iter, n_episodes):
        self.logger = logger
        self.n_channels = n_channels
        self.log_iter = log_iter
        self.n_episodes = n_episodes

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
            lgger = self.logger.warn
        else:
            lgger = self.logger.debug
        lgger(
                f"Rejected call to {cell} when {n_used}"
                f" of {self.n_channels} channels in use")

    def iter(self, t, i, epsilon, alpha):
        self.logger.info(
                f"\n{t:.2f}-{i}: Blocking probability events"
                f" {i-self.log_iter}-{self.log_iter}:"
                f" {self.n_curr_rejected/(self.n_curr_incoming+1):.4f}")
        if epsilon:
            self.logger.info(
                f"\nEpsilon: {epsilon:.5f},"
                f" Alpha: {alpha:.5f}")
        self.n_curr_rejected = 0
        self.n_curr_incoming = 0

    def endsim(self, t, i, n_inprogress):
        if (self.n_incoming + self.n_handoffs -
                self.n_ended - self.n_rejected -
                self.n_handoffs_rejected) != n_inprogress:
            self.logger.error(
                    f"\nSome calls were lost."
                    f" accepted: {self.n_incoming}, ended: {self.n_ended}"
                    f" rejected: {self.n_rejected},"
                    f" in progress: {n_inprogress}")
        self.logger.warn(
            f"\nSimulation duration: {t/24:.2f} hours?,"
            f" {self.n_episodes} episodes"
            f" at {self.n_episodes/(time.time()-self.start_time):.0f}"
            " episodes/second"
            f"\nRejected {self.n_rejected} of {self.n_incoming} new calls,"
            f" {self.n_handoffs_rejected} of {self.n_handoffs} handoffs"
            f"\nBlocking probability: {self.n_rejected/self.n_incoming:.4f}"
            " for new calls,"
            f" {self.n_handoffs_rejected/self.n_handoffs:.4f} for handoffs"
            f"\nAverage number of calls in progress when blocking: "
            f"{self.n_inuse_rej/(self.n_rejected+1):.2f}"
            f"\n{n_inprogress} calls in progress at simulation end\n")
