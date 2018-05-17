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
        # print(results)
        if not results:
            self.logger.error("NO RESULTS")
            return

        # For each run; for each ctype (new-call/hoff/tot); cumulative block. prob
        block_probs_cum = []
        # For each run; newcall block prob (not cum) during each log iter
        newcall_block_probs = []
        # For each ctype; for each run; _for each log iter_; cumulative b.p.
        all_block_probs_cums = [[], [], []]
        # For each run/simulation
        for run in results:
            new_call_cum_bp = run[0][0]
            # Filter out bad results (invalid loss etc)
            if new_call_cum_bp != 1 and new_call_cum_bp is not None:
                block_probs_cum.append(np.array(run[0]))
                assert run[1] is not None and run[1] is not 1
                newcall_block_probs.append(np.array(run[1]))
                all_block_probs_cums[0].append(np.array(run[2]))
                all_block_probs_cums[1].append(np.array(run[3]))
                all_block_probs_cums[2].append(np.array(run[4]))

        block_probs_cum = np.array(block_probs_cum)
        newcall_block_probs = np.array(newcall_block_probs)
        all_block_probs_cums = np.array(all_block_probs_cums)

        n_events = self.pp['n_events']
        try:
            avgbprobs = ", ".join(
                "%.1f" % (f * 100) for f in np.mean(newcall_block_probs, axis=0))
        except (TypeError, ValueError):
            self.logger.error("Can't calculate per-logiter mean for incomplete runs")
            avgbprobs = 0
        self.logger.error(
            f"\n{n_runs}x{n_events} events finished with speed"
            f" {(n_runs*n_events)/(time.time()-t):.0f} events/second"
            f"\nAverage new call block (%) each log iter: {avgbprobs}"
            f"\nAverage cumulative block probability over {n_runs} episodes:"
            f" {np.mean(block_probs_cum[:,0]):.4f}"
            f" with standard deviation {np.std(block_probs_cum[:,0]):.5f}"
            f"\nAverage cumulative handoff block probability"
            f" {np.mean(block_probs_cum[:,1]):.4f}"
            f" with standard deviation {np.std(block_probs_cum[:,1]):.5f}"
            f"\nAverage cumulative total block probability"
            f" {np.mean(block_probs_cum[:,2]):.4f}"
            f" with standard deviation {np.std(block_probs_cum[:,2]):.5f}"
            f"\n{block_probs_cum}")

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
        pp['rng_seed'] = seed  # Reseed for TF
    strat = stratclass(pp, logger=logger, pid=pid)
    result = strat.simulate()
    # Block prob between each log iter
    block_probs = strat.env.stats.block_probs
    # Cumulative new call block prob for each log iter
    block_probs_cums = strat.env.stats.block_probs_cum
    # Cumulative hand-off block prob for each log iter
    block_probs_cums_h = strat.env.stats.block_probs_cum_h
    # Cumulative total block prob for each log iter
    block_probs_cums_t = strat.env.stats.block_probs_cum_t
    return result, block_probs, block_probs_cums, block_probs_cums_h, block_probs_cums_t
