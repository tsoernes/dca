import cProfile
import datetime
import logging
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from hyperopt import Trials, fmin, hp, tpe  # noqa
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll.base import scope  # noqa
from matplotlib import pyplot as plt

from gui import Gui
from hopt_utils import (add_pp_mongo, add_pp_pickle, hopt_results, hopt_trials,
                        mongo_decide_gpu_usage, mongo_decrease_gpu_procs,
                        mongo_list_dbs)
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
        elif self.pp['hopt_best'] > 0:
            self.hopt_best(None, self.pp['hopt_best'], view_pp=True)
        elif self.pp['hopt_plot']:
            self.hopt_plot()
        elif self.pp['hopt_list']:
            mongo_list_dbs()
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
        simproc = partial(self.sim_proc, self.stratclass, self.pp)
        # Net runs use same np seed for all runs; other strats do not
        if self.pp['net'] and not self.pp['no_gpu']:
            results = []
            for i in range(n_runs):
                res = simproc(i, reseed=False)
                results.append(res)
        else:
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
        if reseed:
            # Must reseed lest all results will be the same.
            # Reseeding is wanted for avg runs but not hopt
            np.random.seed()
        logger = logging.getLogger('')
        strat = stratclass(pp, logger=logger, pid=pid)
        result = strat.simulate()
        return result

    def hopt(self, net=False):
        """
        Hyper-parameter optimization with hyperopt.
        """
        if self.pp['net']:
            space = {
                # Qlearnnet
                'net_lr': hp.uniform('net_lr', 1e-6, 1e-4),
                'net_lr_decay': hp.uniform('net_lr_decay', 0.80, 0.9999999),
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
                # Policy
                'vf_coeff': hp.uniform('vf_coeff', 0.005, 0.5),
                'entropy_coeff': hp.uniform('entropy_coeff', 1.0, 100.0)
            }
        else:
            space = {
                'alpha': hp.loguniform('alpha', np.log(0.001), np.log(0.1)),
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
        name = self.pp['hopt_fname'].replace('mongo:', '')
        trials = MongoTrials('mongo://localhost:1234/' + name + '/jobs')
        add_pp_mongo(name, self.pp)
        try:
            self.logger.error("Prev best:")
            self.hopt_best(trials, n=1, view_pp=False)
        except ValueError:
            self.logger.error("No existing trials, starting from scratch")
        fn = partial(hopt_proc, self.stratclass, self.pp, mongo_uri=name)
        fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

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

    def hopt_best(self, trials=None, n=1, view_pp=True):
        if trials is None:
            try:
                trials = hopt_trials(self.pp)
            except (FileNotFoundError, ValueError):
                sys.exit(1)
        # Something below here might throw AttributeError when mongodb exists but is empty
        if n == 1:
            b = trials.best_trial
            params = b['misc']['vals']
            fparams = ' '.join([f"--{key} {value[0]}" for key, value in params.items()])
            self.logger.error(f"Loss: {b['result']['loss']}\n{fparams}")
            return
        valid_results, params, attachments = hopt_results(self.pp, trials)
        sorted_results = sorted(valid_results)
        self.logger.error(f"Found {len(sorted_results)} valid trials")
        if view_pp and attachments:
            self.logger.error(attachments)
        for lt in sorted_results[:n]:
            fparams = ' '.join(
                [f"--{key} {value[lt[1]]}" for key, value in params.items()])
            self.logger.error(f"Loss {lt[0]:.6f}: {fparams}")

    def hopt_plot(self):
        trials = hopt_trials(self.pp)
        valid_results, params, attachments = hopt_results(self.pp, trials)
        losses = [x[0] for x in valid_results]
        n_params = len(params.keys())
        for i, (param, values) in zip(range(n_params), params.items()):
            pl1 = plt.subplot(n_params, 1, i + 1)
            pl1.plot(values, losses, 'ro')
            plt.xlabel(param)
        plt.show()

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
    if not pp['no_gpu'] and mongo_uri is not None:
        using_gpu_and_mongo = mongo_decide_gpu_usage(mongo_uri, pp['max_gpu_procs'])
        pp['no_gpu'] = not using_gpu_and_mongo
    for key, val in space.items():
        pp[key] = val
    # import psutil
    # n_avg = psutil.cpu_count(logical=False)
    # Use same constant numpy seed for all sims
    np.random.seed(pp['rng_seed'])
    logger = logging.getLogger('')
    logger.error(space)
    result = Runner.sim_proc(stratclass, pp, pid='', reseed=False)
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
