import logging
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from runners.runner import Runner


class AvgRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        t = time.time()
        n_runs = self.pp['avg_runs']
        simproc = partial(avg_proc, self.stratclass, self.pp, reseed=True)
        # Net runs use same np seed for all runs; other strats do not
        with Pool() as p:
            results = p.map(simproc, range(n_runs))
        # Filter out bad results (invalid loss etc)
        results = [r for r in results if r[0][0] != 1 and r[0][0] is not None]
        if not results:
            self.logger.error("NO RESULTS")
            return
        n_events = self.pp['n_events']
        cum_block_probs = np.array([r[0] for r in results])
        # For each thread, block prob (not cum) during each log iter
        block_probs = np.array([r[1] for r in results])
        # print(cum_block_probs, "\n", block_probs)
        bprobs = ", ".join("%.4f" % f for f in np.mean(block_probs, axis=0))
        self.logger.error(
            f"\n{n_runs}x{n_events} events finished with speed"
            f" {(n_runs*n_events)/(time.time()-t):.0f} events/second"
            f"\nAverage new call block each log iter: {bprobs}"
            f"\nAverage cumulative block probability over {n_runs} episodes:"
            f" {np.mean(cum_block_probs[:,0]):.4f}"
            f" with standard deviation {np.std(cum_block_probs[:,0]):.5f}"
            f"\nAverage cumulative handoff block probability"
            f" {np.mean(cum_block_probs[:,1]):.4f}"
            f" with standard deviation {np.std(cum_block_probs[:,1]):.5f}"
            f"\nAverage cumulative total block probability"
            f" {np.mean(cum_block_probs[:,2]):.4f}"
            f" with standard deviation {np.std(cum_block_probs[:,2]):.5f}"
            f"\n{cum_block_probs}")
        # TODO Plot average cumulative over time


def avg_proc(stratclass, pp, pid, reseed=True):
    """
    Allows for running simulation in separate process
    """
    logger = logging.getLogger('')
    if reseed:
        # Must reseed lest all results will be the same.
        # Reseeding is wanted for avg runs but not hopt
        np.random.seed()
        seed = np.random.get_state()[1][0]
        pp['rng_seed'] = seed  # Reseed for tf
    strat = stratclass(pp, logger=logger, pid=pid)
    result = strat.simulate()
    block_probs = strat.env.stats.block_probs
    return result, block_probs
