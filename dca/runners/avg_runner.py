import datetime
import logging
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from datahandler import next_filename
from plotter import ctypes_short, plot_bps
from runners.runner import Runner


class AvgRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        t = time.time()
        n_runs = self.pp['avg_runs']
        simproc = partial(avg_proc, self.stratclass, self.pp, reseed=True)
        # Net runs use same np seed for all runs; other strats do not
        with Pool(self.pp['threads']) as p:
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
        try:
            bprobs = ", ".join("%.4f" % f for f in np.mean(block_probs, axis=0))
        except TypeError:
            print(block_probs, results)
            raise
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

        # block_probs = np.array([r[2] for r in results])
        # block_probs_h = np.array([r[3] for r in results])
        # block_probs_t = np.array([r[4] for r in results])
        all_block_probs_cums = (np.array([r[i] for r in results])
                                for i in range(2, len(results[0])))
        if self.pp['save_cum_block_probs']:
            self.save_bps(all_block_probs_cums, self.pp['log_iter'], n_events)
        if self.pp['do_plot']:
            plot_bps(
                all_block_probs_cums,
                self.pp['log_iter'],
                n_events,
                fname=self.pp['plot_save'])

    def save_bps(self, all_block_probs_cum, log_iter, n_events):
        """ Log cumulative block probs, and save to file"""
        data = {
            'datetime': datetime.datetime.now(),
            'log_iter': log_iter,
            'n_events': n_events
        }
        for i, block_probs_cums in enumerate(all_block_probs_cum):
            self.fo_logger.error(f'{ctypes_short[i]} cumulative block probs')
            self.fo_logger.error(block_probs_cums)
            data[ctypes_short[i]] = block_probs_cums
        fname = next_filename('bps/' + self.pp['save_cum_block_probs'], '.pkl')
        if not os.path.exists("bps"):
            os.makedirs("bps")
        with open(fname, "wb") as f:
            pickle.dump(data, f)
        self.logger.error(f"Saved cumulative block probs to {fname}")


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
    block_probs_cums = strat.env.stats.block_probs_cum
    block_probs_cums_h = strat.env.stats.block_probs_cum_h
    block_probs_cums_t = strat.env.stats.block_probs_cum_t
    return result, block_probs, block_probs_cums, block_probs_cums_h, block_probs_cums_t
