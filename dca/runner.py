#! /usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import cProfile
import datetime
import logging
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool

import argcomplete
import numpy as np
from datadiff import diff
from hyperopt import Trials, fmin, hp, tpe  # noqa
from hyperopt.pyll.base import scope  # noqa

from gui import Gui  # noqa
from hopt_utils import (MongoConn, add_pp_pickle, hopt_best,
                        mongo_decide_gpu_usage, mongo_decrease_gpu_procs)
from params import get_pparams


class Runner:
    def __init__(self):
        self.pp, self.stratclass = get_pparams()

        logging.basicConfig(level=self.pp['log_level'], format='%(message)s')
        self.logger = logging.getLogger('')
        if self.pp['log_file']:
            fh = logging.FileHandler(self.pp['log_file'] + ".log")
            fh.setLevel(self.pp['log_level'])
            self.logger.addHandler(fh)
        self.logger.error(
            f"Starting simulation at {datetime.datetime.now()} with params:\n{self.pp}")

        if self.pp['hopt']:
            self.hopt()
        elif self.pp['dlib_hopt']:
            self.hopt_dlib()
        elif self.pp['avg_runs']:
            self.avg_run()
        elif self.pp['strat'] == 'show':
            self.show()
        elif self.pp['train_net']:
            self.train_net()
        elif self.pp['bench_batch_size']:
            self.bench_batch_size()
        else:
            self.run()

    def run(self):
        strat = self.stratclass(self.pp, logger=self.logger)
        if self.pp['gui']:
            # TODO Fix grid etc
            # gui = Gui(grid, strat.exit_handler, grid.print_cell)
            # strat.gui = gui
            raise NotImplementedError
        if self.pp['profiling']:
            cProfile.runctx('strat.simulate()', globals(), locals(), sort='tottime')
        else:
            strat.simulate()

    def avg_run(self):
        t = time.time()
        n_runs = self.pp['avg_runs']
        simproc = partial(self.sim_proc, self.stratclass, self.pp, reseed=True)
        # Net runs use same np seed for all runs; other strats do not
        with Pool() as p:
            results = p.map(simproc, range(n_runs))
        results = [r for r in results if r[0] != 1 and r[0] is not None]
        if not results:
            self.logger.error("NO RESULTS")
            return
        n_events = self.pp['n_events']
        results = np.array(results)
        self.logger.error(
            f"\n{n_runs}x{n_events} events finished with speed"
            f" {(n_runs*n_events)/(time.time()-t):.0f} events/second"
            f"\nAverage cumulative block probability over {n_runs} episodes:"
            f" {np.mean(results[:,0]):.4f}"
            f" with standard deviation {np.std(results[:,0]):.5f}"
            f"\nAverage cumulative handoff block probability"
            f" {np.mean(results[:,1]):.4f}"
            f" with standard deviation {np.std(results[:,1]):.5f}"
            f"\n{results}")
        # TODO Plot average cumulative over time

    def train_net(self):
        strat = self.stratclass(self.pp, logger=self.logger)
        strat.net.train()

    @staticmethod
    def sim_proc(stratclass, pp, pid, reseed=True):
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
        return result

    def hopt_dlib(self):
        import dlib
        lo_bounds = [8e-6, 0.85]  # Lower bound constraints on each var respectively
        up_bounds = [9e-5, 0.98]
        # lo_bounds = [0.65]
        # up_bounds = [0.85]
        n = 50  # The number of times find_min_global() will call holder_table()
        self.logger.error(
            f"Dlib hopt for {n} iterations, bounds {lo_bounds}, {up_bounds}")
        self.i = 0
        space = ['net_lr', 'net_lr_decay']
        is_integer_variable = [False, False]
        results = []

        def dlib_proc(*args):
            self.logger.error(f"\nIter {self.i}, testing {space}: {args}")
            for j, key in enumerate(space):
                self.pp[key] = args[j]
            strat = self.stratclass(self.pp, logger=self.logger, pid=self.i)
            result = strat.simulate()
            res = result[0]
            if res is None:
                res = 1
            results.append((res, args))
            # If user quits sim, need to abort further calls to dlib_proc
            if strat.quit_sim and not strat.invalid_loss and not strat.exceeded_bthresh:
                results.sort()
                self.logger.error(f"Top 5: {results[:5]}")
                sys.exit(0)
            self.i += 1
            return res

        """ the search will only attempt to find a global minimizer to at most
        solver_epsilon accuracy. Once a local minimizer is found to that
        accuracy the search will focus entirely on finding other minima
        elsewhere rather than on further improving the current local optima
        found so far. That is, once a local minima is identified to about
        solver_epsilon accuracy, the algorithm will spend all its time
        exploring the functions to find other local minima to investigate. An
        epsilon of 0 means it will keep solving until it reaches full floating
        point precision. Larger values will cause it to switch to pure global
        exploration sooner and therefore might be more effective if your
        objective function has many local minima and you don't care about a
        super high precision solution.

        On even iterations we pick the next x according to our upper bound while
        on odd iterations we pick the next x according to the trust region model
        """
        solver_epsilon = 0.00005
        x = dlib.find_min_global(
            dlib_proc, lo_bounds, up_bounds, is_integer_variable, n, solver_epsilon=0)
        results.sort()
        self.logger.error(f"Top 5: {results[:5]}")
        self.logger.error(f"Min x: {x}")

    def hopt(self, net=False):
        """
        Hyper-parameter optimization with hyperopt.
        """
        if self.pp['net']:
            space = {
                # Qlearnnet
                'net_lr': hp.uniform('net_lr', 1e-5, 5e-5),
                'net_lr_decay': hp.uniform('net_lr_decay', 0.90, 0.98),
                # Singh
                # 'net_lr': hp.loguniform('net_lr', np.log(1e-7), np.log(5e-4)),
                'beta': hp.loguniform('beta', np.log(7), np.log(23)),
                # Double
                'net_copy_iter': hp.loguniform('net_copy_iter', np.log(5), np.log(150)),
                'net_creep_tau': hp.loguniform('net_creep_tau', np.log(0.01),
                                               np.log(0.7)),
                # Exp. replay
                'batch_size': scope.int(hp.uniform('batch_size', 8, 16)),
                'buffer_size': scope.int(hp.uniform('buffer_size', 2000, 10000)),
                # N-step
                'n_step': scope.int(hp.uniform('n_step', 3, 40)),
                # Policy
                'vf_coeff': hp.uniform('vf_coeff', 0.005, 0.5),
                'entropy_coeff': hp.uniform('entropy_coeff', 1.0, 100.0)
            }
        else:
            space = {
                'beta': hp.loguniform('beta', np.log(7), np.log(23)),
                'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.4)),
                'alpha_decay': hp.uniform('alpha_decay', 0.9999, 0.9999999),
                'epsilon': hp.loguniform('epsilon', np.log(0.2), np.log(0.8)),
                'epsilon_decay': hp.uniform('epsilon_decay', 0.9995, 0.9999999),
                'gamma': hp.uniform('gamma', 0.7, 0.90),
                'lambda': hp.uniform('lambda', 0.0, 1.0)
            }
        # Only optimize parameters specified in args
        space = {param: space[param] for param in self.pp['hopt']}
        if self.pp['hopt_fname'].startswith('mongo:'):
            self._hopt_mongo(space)
        else:
            self._hopt_pickle(space)

    def _hopt_mongo(self, space):
        """Find previous best trial and pp from MongoDB, if any, then run hopt job server"""
        trials = MongoConn(self.pp['hopt_fname'])
        try:
            self.logger.error("Prev best:")
            hopt_best(trials, n=1, view_pp=False)
            prev_pps = trials.get_pps()
            # If given pp equals the last one found in MongoDB, don't add it.
            # Otherwise, ask whether to use the one found in DB instead,
            # and if not, store given pp in DB.
            if prev_pps:
                mongo_pp = prev_pps[-1]
                dt = mongo_pp['dt']
                del mongo_pp['dt']
                del mongo_pp['_id']
                pp_diff = diff(mongo_pp, self.pp)
                if len(pp_diff.diffs) > 1:
                    # 'diffs' list has more than 1 entry, i.e. pps are not equal
                    self.logger.error(f"Found old problem params in MongoDB added "
                                      f"at {dt}. Diff(a:old, from db. b: from args):\n{pp_diff}")
                    ans = ''
                    while ans not in ['y', 'n']:
                        ans = input("Use old pp instead? Y/N:").lower()
                    if ans == 'y':
                        self.pp = mongo_pp
                    else:
                        trials.add_pp(self.pp)
            else:
                trials.add_pp(self.pp)
        except ValueError:
            self.logger.error("No existing trials, starting from scratch")
            trials.add_pp(self.pp)
        mongo_uri = self.pp['hopt_fname'].replace('mongo:', '')
        fn = partial(hopt_proc, self.stratclass, self.pp, mongo_uri=mongo_uri)
        self.logger.error("Started hyperopt job server")
        fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)
        trials.client.close()

    def _hopt_pickle(self, space):
        """
        Saves progress to 'pp['hopt_fname'].pkl' and
        automatically resumes if file already exists.
        """
        if self.pp['net']:
            trials_step = 1  # Number of trials to run before saving
        else:
            trials_step = 4
        f_name = self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                prev_best = trials.argmin
                self.logger.error(f"Found {len(trials.trials)} saved trials")
        except FileNotFoundError:
            trials = Trials()
            prev_best = None

        add_pp_pickle(trials, self.pp)
        fn = partial(hopt_proc, self.stratclass, self.pp, mongo_uri=None)
        while True:
            n_trials = len(trials)
            self.logger.error(f"Running trials {n_trials+1}-{n_trials+trials_step}")
            best = fmin(
                fn=fn,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials + trials_step,
                trials=trials)
            if prev_best != best:
                bp = trials.best_trial['result']['loss']
                self.logger.error(f"Found new best params: {best} with block prob: {bp}")
                prev_best = best
            with open(f_name, "wb") as f:
                pickle.dump(trials, f)

    def show(self):
        # g = grid.RhombusAxialGrid(logger=self.logger, **self.pp)
        # gui = Gui(g, self.logger, g.print_neighs, "rhomb")
        # grid = RectOffGrid(logger=self.logger, **self.pp)
        # gui = Gui(grid, self.logger, grid.print_neighs, "rect")
        # gui.test()
        raise NotImplementedError


def hopt_proc(stratclass, pp, space, mongo_uri=None):
    """
    If 'mongo_uri' is present, determine whether to use GPU or not based
    on the number of processes that already utilize it.
    """
    using_gpu_and_mongo = False
    # Don't override user-given arg for disabling GPU-usage
    if pp['use_gpu'] and mongo_uri is not None:
        using_gpu_and_mongo = mongo_decide_gpu_usage(mongo_uri, pp['max_gpu_procs'])
        pp['use_gpu'] = using_gpu_and_mongo
    for key, val in space.items():
        pp[key] = val
    # import psutil
    # n_avg = psutil.cpu_count(logical=False)
    # Use same constant numpy seed for all sims
    # np.random.seed(pp['rng_seed'])
    logger = logging.getLogger('')
    logger.error(space)
    result = Runner.sim_proc(stratclass, pp, pid='', reseed=True)
    if using_gpu_and_mongo:
        # Finished using the GPU, so reduce the 'gpu_procs' count
        mongo_decrease_gpu_procs(mongo_uri)
    res = result[0]
    if res is None:
        # Loss is inf or nan
        return {"status": "fail"}
    elif res is 1:
        # User quit, e.g. ctrl-c
        return {"status": "suspended"}
    return {'status': "ok", "loss": res, "hoff_loss": result[1]}


if __name__ == '__main__':
    r = Runner()
