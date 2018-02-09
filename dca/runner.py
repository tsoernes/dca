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

import grid
from gui import Gui
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

    def avg_run(self):
        t = time.time()
        n_runs = self.pp['avg_runs']
        simproc = partial(self.sim_proc, self.stratclass, self.pp)
        if self.pp['net']:
            # Use constant tf seed
            np.random.seed(0)
            results = []
            for i in range(n_runs):
                # Use constant np seed
                res = simproc(i, reseed=False)
                results.append(res)
        else:
            with Pool() as p:
                results = p.map(simproc, range(n_runs))
        results = list(filter(lambda r: r[0] != 1, results))
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
        Saves progress to 'results-{stratname}.pkl' and
        automatically resumes if file already exists.
        """
        if self.pp['net']:
            space = {
                'net_lr': hp.loguniform('net_lr', np.log(1e-7), np.log(5e-4)),
                # Singh
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
        np.random.seed(0)
        if self.pp['hopt_fname'].startswith('mongo:'):
            self._hopt_mongo(space)
        else:
            self._hopt_pickle(space)

    def _hopt_mongo(self, space):
        name = self.pp['hopt_fname'].replace('mongo:', '')
        trials = MongoTrials('mongo://localhost:1234/' + name + '/jobs')
        self._hopt_add_attachments(trials)
        if len(trials) > 0:
            self.hopt_best(trials, n=1, view_pp=False)
        else:
            self.logger.errors("No existing trials, starting from scratch")
        fn = partial(hopt_proc, self.stratclass, self.pp)
        fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    def _hopt_pickle(self, space):
        """
        Hyper-parameter optimization with hyperopt.
        Saves progress to 'pp['hopt_fname'].pkl' and
        automatically resumes if file already exists.
        """
        if self.pp['net']:
            trials_step = 1  # Number of trials to run before saving
        else:
            trials_step = 2
        f_name = self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                prev_best = trials.argmin
                self.logger.error(f"Found {len(trials.trials)} saved trials")
        except FileNotFoundError:
            trials = Trials()
            prev_best = {}

        self._hopt_add_attachments(trials)
        fn = partial(hopt_proc, self.stratclass, self.pp)
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

    def _hopt_add_attachments(self, trials):
        att = trials.attachments
        if 'pp' in att:
            if att['pp'][-1] != self.pp:
                att['pp'].append(self.pp)
        else:
            att['pp'] = [self.pp]

    def _hopt_trials(self, f_name=None):
        if f_name is None:
            f_name = self.pp['hopt_fname']
        try:
            if f_name.startswith("mongo"):
                # 'mongo://localhost:1234/results-singhnet-net_lr-beta/jobs'
                f_name = f"mongo://localhost:1234/{f_name.replace('mongo:', '')}/jobs"
                self.logger.error(f"Attempting to connect to mongodb with url {f_name}")
                return MongoTrials(f_name)
            else:
                fname = self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
                with open(fname, "rb") as f:
                    return pickle.load(f)
        except (FileNotFoundError, AttributeError):
            self.logger.error(f"Could not find {f_name}")
            raise
        except:
            if f_name.startswith("mongo"):
                self.logger.error("Have you started mongod server in 'db' dir? \n"
                                  "mongod --dbpath . --directoryperdb"
                                  " --journal --nohttpinterface --port 1234")
            raise

    def hopt_best(self, trials=None, n=1, view_pp=True):
        if trials is None:
            trials = self._hopt_trials()
        self.logger.error(f"Found {len(trials)} trials")
        if view_pp:
            a = trials.attachments
            if a:
                self.logger.error(a)
        if n == 1:
            b = trials.best_trial
            params = b['misc']['vals']
            fparams = ' '.join([f"--{key} {value[0]}" for key, value in params.items()])
            self.logger.error(f"Loss: {b['result']['loss']}\n{fparams}")
        else:
            # Gather losses, statuses and params for all results
            statuses = [e['status'] for e in trials.results]
            losses = [e['loss'] for e in trials.results]
            sorted_losses = sorted(zip(losses, range(len(losses))))
            for lt in sorted_losses[:n]:
                fparams = ' '.join(
                    [f"--{key} {value[lt[1]]:.7f}" for key, value in trials.vals.items()])
                self.logger.error(f"Loss {lt[0]:.6f} {statuses[lt[1]]}: {fparams}")

    def hopt_plot(self, trials=None):
        if trials is None:
            trials = self._hopt_trials()
        losses = trials.losses()
        n_params = len(trials.vals.keys())
        for i, (param, values) in zip(range(n_params), trials.vals.items()):
            pl1 = plt.subplot(n_params, 1, i + 1)
            pl1.plot(values, losses, 'ro')
            plt.xlabel(param)
        plt.show()

    def show(self):
        g = grid.RhombusAxialGrid(logger=self.logger, **self.pp)
        gui = Gui(g, self.logger, g.print_neighs, "rhomb")
        # grid = RectOffGrid(logger=self.logger, **self.pp)
        # gui = Gui(grid, self.logger, grid.print_neighs, "rect")
        gui.test()


def hopt_proc(stratclass, pp, space):
    """
    n_avg: Number of runs to take the average of.
    For non-net strats, these are run in parallell.
    """
    for key, val in space.items():
        pp[key] = val
    # import psutil
    # n_avg = psutil.cpu_count(logical=False)
    np.random.seed(pp['rng_seed'])
    simproc = partial(Runner.sim_proc, stratclass, pp, reseed=False)
    logger = logging.getLogger('')
    logger.error(space)
    result = simproc('')
    res = result[0]
    if res is None:
        return {"status": "fail"}
    if res is 1:
        return {"status": "suspended"}
    return {'status': "ok", "loss": res, "hoff_loss": result[1]}


if __name__ == '__main__':
    r = Runner()
