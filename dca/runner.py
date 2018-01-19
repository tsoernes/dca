import cProfile
import datetime
import logging
import pickle
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
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
            fh = logging.FileHandler(self.pp['log_file'])
            fh.setLevel(self.pp['log_level'])
            self.logger.addHandler(fh)
        self.logger.error(f"Starting simulation at {datetime.datetime.now()}"
                          f" with params:\n{self.pp}")

        if self.pp['hopt']:
            self.hopt()
        elif self.pp['hopt_best']:
            self.hopt_best()
        elif self.pp['hopt_plot']:
            self.hopt_plot()
        elif self.pp['avg_runs'] > 1:
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
            results = []
            for i in range(n_runs):
                res = simproc(i)
                if not (np.isnan(res[0]) or np.isinf(res[0]) or res[0] == 1):
                    results.append(res)
        with Pool() as p:
            results = p.map(simproc, range(n_runs))
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
        # TODO Run until ctrl-c, then present results

    def run(self):
        strat = self.stratclass(self.pp, logger=self.logger)
        if self.pp['gui']:
            # TODO Fix grid etc
            # gui = Gui(grid, strat.exit_handler, grid.print_cell)
            # strat.gui = gui
            raise NotImplementedError
        if self.pp['profiling']:
            cProfile.runctx(
                'strat.simulate()', globals(), locals(), sort='tottime')
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

    @staticmethod
    def hopt_proc(stratclass, pp, space, n_avg=4):
        """
        n_avg: Number of runs to take the average of.
        For non-net strats, these are run in parallell.
        """
        for key, val in space.items():
            pp[key] = val
        # import psutil
        # n_avg = psutil.cpu_count(logical=False)
        simproc = partial(Runner.sim_proc, stratclass, pp, reseed=False)
        logger = logging.getLogger('')
        logger.error(space)
        if pp['net']:
            results = []
            for i in range(n_avg):
                res = simproc(pid=i)[0]
                if np.isnan(res) or np.isinf(res):
                    break
                results.append(res)
        else:
            # No need to average for table-based methods since they use the same
            # numpy seed
            results = [simproc(0)[0]]
        for res in results:
            if np.isnan(res) or np.isinf(res):
                return {"status": "fail"}
        result = sum(results) / len(results)
        return result

    def show(self):
        g = grid.RhombusAxialGrid(logger=self.logger, **self.pp)
        gui = Gui(g, self.logger, g.print_neighs, "rhomb")
        # grid = RectOffGrid(logger=self.logger, **self.pp)
        # gui = Gui(grid, self.logger, grid.print_neighs, "rect")
        gui.test()

    def hopt(self, net=False):
        """
        Hyper-parameter optimization with hyperopt.
        Saves progress to 'results-{stratname}.pkl' and
        automatically resumes if file already exists.
        """
        from hyperopt import fmin, tpe, hp, Trials
        from hyperopt.pyll.base import scope  # noqa
        if self.pp['net']:
            space = {
                'net_lr':
                hp.loguniform('net_lr', np.log(5e-6), np.log(9e-4)),
                'net_copy_iter':
                hp.loguniform('net_copy_iter', np.log(5), np.log(150)),
                # 'batch_size':
                # scope.int(hp.uniform('batch_size', 8, 16)),
                # 'buffer_size':
                # scope.int(hp.uniform('buffer_size', 2000, 10000))
            }
            self.pp['n_events'] = 100000
            trials_step = 1  # Number of trials to run before saving
        else:
            space = {
                # 'alpha': hp.loguniform('alpha', np.log(0.001), np.log(0.1)),
                # 'alpha_decay': hp.uniform('alpha_decay', 0.9999, 0.9999999),
                # 'epsilon': hp.loguniform('epsilon', np.log(0.2), np.log(0.8)),
                # 'epsilon_decay': hp.uniform('epsilon_decay', 0.9995,
                #                             0.9999999),
                'gamma': hp.uniform('gamma', 0.7, 0.90),
                'lambda': hp.uniform('lambda', 0.0, 1.0)
            }
            trials_step = 2
        np.random.seed(0)
        f_name = f"results-{self.pp['strat']}.pkl"
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                self.logger.error(f"Found {len(trials.trials)} saved trials")
        except:
            trials = Trials()

        fn = partial(Runner.hopt_proc, self.stratclass, self.pp)
        prev_best = {}
        while True:
            n_trials = len(trials)
            self.logger.error(
                f"Running trials {n_trials+1}-{n_trials+trials_step}")
            best = fmin(
                fn=fn,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials + trials_step,
                trials=trials)
            if prev_best != best:
                self.logger.error(f"Found new best params: {best}")
                prev_best = best
            with open(f_name, "wb") as f:
                pickle.dump(trials, f)

    def hopt_best(self):
        f_name = f"results-{self.pp['strat']}.pkl"
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                b = trials.best_trial
                self.logger.error(
                    f"Loss: {b['result']['loss']}\n{b['misc']['vals']}")
        except FileNotFoundError:
            self.logger.error(f"Could not find {f_name}")
            raise

    def hopt_plot(self):
        f_name = f"results-{self.pp['strat']}.pkl"
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
        except FileNotFoundError:
            self.logger.error(f"Could not find {f_name}")
            raise
        losses = trials.losses()
        n_params = len(trials.vals.keys())
        for i, (param, values) in zip(range(n_params), trials.vals.items()):
            pl1 = plt.subplot(n_params, 1, i + 1)
            pl1.plot(values, losses, 'ro')
            plt.xlabel(param)
        plt.show()


if __name__ == '__main__':
    r = Runner()
