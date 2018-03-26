#! /usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import cProfile
import datetime
import logging
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool, Process, Queue, cpu_count

#  import argcomplete
import numpy as np
from hyperopt import Trials, fmin, hp, tpe  # noqa
from hyperopt.pyll.base import scope  # noqa
from icecream import ic  # noqa

from gui import Gui  # noqa
from hopt_utils import (MongoConn, add_pp_pickle, compare_pps, dlib_load,
                        dlib_save, hopt_best, mongo_decide_gpu_usage,
                        mongo_decrease_gpu_procs)
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
        elif self.pp['random_hopt']:
            self.hopt_random()
        elif self.pp['avg_runs']:
            self.avg_run()
        elif self.pp['strat'] == 'show':
            self.show()
        elif self.pp['train_net']:
            self.train_net()
        elif self.pp['bench_batch_size']:
            self.bench_batch_size()
        elif self.pp['analyze_net']:
            self.analyze_net()
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

    def analyze_net(self):
        from gridfuncs import GF
        from eventgen import CEvent
        strat = self.stratclass(self.pp, logger=self.logger)
        as_freps = GF.afterstate_freps(strat.grid, (3, 4), CEvent.NEW, range(70))
        as_freps_vals = strat.net.forward(as_freps)
        self.logger.error(as_freps_vals)
        self.logger.error(GF.nom_chs[3, 4])
        as_freps = GF.afterstate_freps(strat.grid, (4, 4), CEvent.NEW, range(70))
        as_freps_vals = strat.net.forward(as_freps)
        self.logger.error(as_freps_vals)
        self.logger.error(GF.nom_chs[4, 4])

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
        import dlib
        n_sims = 4000  # The number of times to sample and test params
        n_concurrent = 14  # int(cpu_count() / 2) - 1  # Number of concurrent procs
        solver_epsilon = 0.0005
        relative_noise_magnitude = 0.001  # Default
        space = {
            # parameter: [IsInteger, Low-Bound, High-Bound]
            # 'gamma': [False, 0.60, 0.99],
            'net_lr': [False, 8e-8, 1e-5],
            'beta': [True, 10, 3000],
            # 'net_lr_decay': [False, 0.65, 1.0],
            # 'weight_beta': [False, 1e-10, 1e-5]
            # 'epsilon': [True, 10, 2000],
            # 'alpha': [False, 0.00001, 0.3]
        }
        params, is_int, lo_bounds, hi_bounds = [], [], [], []
        for p, li in space.items():
            params.append(p)
            is_int.append(li[0])
            lo_bounds.append(li[1])
            hi_bounds.append(li[2])
        fname = self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        try:
            old_spec, evals, info, prev_best = dlib_load(fname)
            # Restore saved params and settings if they differ from current/specified
            saved_params = info['params']
            if saved_params != params:
                self.logger.error(
                    f"Saved params {saved_params} differ from specified ones {params}; using saved"
                )
                # TODO could check if bounds match as well
                params = saved_params
            saved_solver_epsilon = info['solver_epsilon']
            if saved_solver_epsilon != solver_epsilon:
                self.logger.error(
                    f"Saved solver_epsilon {saved_solver_epsilon} differ from"
                    " specified one {solver_epsilon}, using specified")
                # solver_epsilon = saved_solver_epsilon
            _, self.pp = compare_pps(info['pp'], self.pp)
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[evals],
                relative_noise_magnitude=info['relative_noise_magnitude'])
            self.logger.error(
                f"Restored {len(evals)} trials, prev best {saved_params} {prev_best}")
        except FileNotFoundError:
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(spec)
            optimizer.set_relative_noise_magnitude(relative_noise_magnitude)
        optimizer.set_solver_epsilon(solver_epsilon)

        result_queue = Queue()
        simproc = partial(dlib_proc, self.stratclass, self.pp, params, result_queue)
        evals = [None] * n_sims

        def quit_opt():
            # Store results of finished evals to file; print best eval
            finished_evals = optimizer.get_function_evaluations()[1][0]
            dlib_save(spec, finished_evals, params, solver_epsilon,
                      relative_noise_magnitude, self.pp, fname)
            best_eval = optimizer.get_best_function_eval()
            self.logger.error(f"Finished {len(finished_evals)} trials."
                              f" Best eval this session: {best_eval}")

        def spawn_eval(i):
            # Spawn a new sim process
            eeval = optimizer.get_next_x()
            evals[i] = eeval  # Store eval object to be set with result later
            Process(target=simproc, args=(i, list(eeval.x))).start()

        def store_result():
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                inp = ""
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save? Y/N: ").upper()
                if inp == "Y":
                    quit_opt()
                sys.exit(0)
            else:
                evals[i].set(result)

        self.logger.error(
            f"Dlib hopt for {n_sims} sims with {n_concurrent} procs"
            f" on params {params} with low bounds {lo_bounds}, high {hi_bounds}")
        # Spawn initial processes
        for i in range(n_concurrent):
            spawn_eval(i)
        # When a thread returns a result, start a new sim
        for i in range(n_concurrent, n_sims):
            store_result()
            spawn_eval(i)
        # Get remaining results
        for _ in range(n_concurrent):
            store_result()
        quit_opt()

    def hopt_random(self):
        """
        Hyper-parameter optimization with hyperopt.
        """
        space = {
            'net_lr': [1e-6, 4e-5],
            'net_lr_decay': [0.80, 0.999],
            'beta': [0.1, 100]
        }
        # Only optimize parameters specified in args
        space = {param: space[param] for param in self.pp['random_hopt']}
        simproc = partial(self.hopt_random, self.stratclass, self.pp, space)

        with Pool() as p:
            p.map(simproc, range(10000))

    def hopt(self):
        """
        Hyper-parameter optimization with hyperopt.
        """
        if self.pp['net']:
            space = {
                # Qlearnnet
                'net_lr': hp.loguniform('net_lr', np.log(5e-7), np.log(1e-4)),
                'net_lr_decay': hp.loguniform('net_lr_decay', np.log(0.90), np.log(0.99)),
                # Singh
                # 'net_lr': hp.loguniform('net_lr', np.log(1e-7), np.log(5e-4)),
                'beta': hp.uniform('beta', 16, 30),
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
                'beta': hp.uniform('beta', 7, 23),
                'alpha': hp.uniform('alpha', 0.0001, 0.4),
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
                use_mongo_pp, new_pp = compare_pps(mongo_pp, self.pp)
                if use_mongo_pp:
                    self.pp = new_pp
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


def hopt_random_proc(stratclass, pp, space, n):
    # import psutil
    # n_avg = psutil.cpu_count(logical=False)
    # Use same constant numpy seed for all sims
    # np.random.seed(pp['rng_seed'])
    print(space)
    for param, (lb, ub) in space.items():
        pp[param] = np.random.uniform(lb, ub)
    result = Runner.sim_proc(stratclass, pp, pid=n, reseed=True)
    res = result[0]
    if res is None:
        # Loss is inf or nan
        return {"status": "fail"}
    elif res is 1:
        # User quit, e.g. ctrl-c
        return {"status": "suspended"}
    return {'status': "ok", "loss": res, "hoff_loss": result[1]}


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


def dlib_proc(stratclass, pp, space_params, result_queue, i, space_vals):
    logger = logging.getLogger('')
    logger.error(f"T{i} Testing {space_params}: {space_vals}")
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    strat = stratclass(pp=pp, logger=logger, pid=i)
    res = strat.simulate()[0]
    if res is None:
        res = 1
    if strat.quit_sim and not strat.invalid_loss and not strat.exceeded_bthresh:
        # If user quits sim, don't want to return result
        return
    # result_queue.put(None)
    # else:
    # Must negate result as dlib performs maximization by default
    result_queue.put((i, -res))


if __name__ == '__main__':
    r = Runner()
