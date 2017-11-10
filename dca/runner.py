from gui import Gui
from net import Net
import grid
from params import get_pparams

import cProfile
import datetime
from functools import partial
import logging
from multiprocessing import Pool
import pickle
import time

import numpy as np


class Runner:
    def __init__(self):
        self.pp, self.stratclasses = get_pparams()

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

    def get_strat_class(self):
        # Override stratname with class reference
        for name, cls in self.stratclasses:
            if self.pp['strat'].lower() == name.lower():
                return cls

    def avg_run(self):
        t = time.time()
        n_eps = self.pp['avg_runs']
        stratclass = self.get_strat_class()
        simproc = partial(
            self.sim_proc, stratclass, self.pp)
        with Pool() as p:
            results = p.map(simproc, range(n_eps))
        n_events = self.pp['n_events']
        self.logger.error(
            f"\n{n_eps}x{n_events} events finished with speed"
            f" {(n_eps*n_events)/(time.time()-t):.0f} episodes/second"
            f"\nAverage cumulative block probability over {n_eps} episodes:"
            f" {np.mean(results):.4f}"
            f" with standard deviation {np.std(results):.5f}"
            f"\n{results}")
        # TODO Plot average cumulative over time

    def train_net(self):
        n = Net(self.pp, self.logger)
        n.train()

    def bench_batch_size(self):
        for bs in [256, 512, 1024, 2048]:
            self.pp['batch_size'] = bs
            net = Net(self.pp, self.logger, restore=False, save=False)
            t = time.time()
            net.train()
            self.logger.error(
                f"Batch size {bs} took {time.time()-t:.2f} seconds")

    def run(self):
        stratclass = self.get_strat_class()
        strat = stratclass(self.pp, logger=self.logger)
        if self.pp['gui']:
            # TODO Fix grid etc
            # gui = Gui(grid, strat.exit_handler, grid.print_cell)
            # strat.gui = gui
            raise NotImplementedError
        if self.pp['profiling']:
            cProfile.runctx('strat.simulate()', globals(), locals(),
                            sort='tottime')
        else:
            strat.simulate()

    @staticmethod
    def sim_proc(stratclass, pp, pid):
        """
        Allows for running simulation in separate process
        """
        np.random.seed()  # Must reseed lest all results will be the same
        logger = logging.getLogger('')
        strat = stratclass(pp, logger=logger, pid=pid)
        result = strat.simulate()
        return result

    @staticmethod
    def hopt_proc(stratclass, pp, space, n_avg=6):
        """
        n_avg: Number of runs to run in parallell and take the average of
        """
        for key, val in space.items():
            pp[key] = val
        simproc = partial(
            Runner.sim_proc, stratclass, pp)
        logger = logging.getLogger('')
        logger.error(space)
        with Pool() as p:
            results = p.map(simproc, range(n_avg))
        for res in results:
            if np.isnan(res) or np.isinf(res):
                return {"status": "fail"}
        result = sum(results) / len(results)
        return result

    def show(self):
        g = grid.RhombAxGrid(logger=self.logger, **self.pp)
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
        stratclass = self.get_strat_class()
        if 'net' in self.pp['strat'].lower():
            space = {
                'net_lr': hp.loguniform(
                    'net_lr', np.log(1e-7), np.log(1e-3)),
            }
            self.pp['n_events'] = 100000
            trials_step = 1  # Number of trials to run before saving
            n_avg = 1
        else:
            space = {
                'alpha': hp.loguniform(
                    'alpha', np.log(0.001), np.log(0.1)),
                'alpha_decay': hp.uniform(
                    'alpha_decay', 0.9999, 0.9999999),
                'epsilon': hp.loguniform(
                    'epsilon', np.log(0.2), np.log(0.8)),
                'epsilon_decay': hp.uniform(
                    'epsilon_decay', 0.9995, 0.9999999),
                'gamma': hp.uniform('gamma', 0.7, 0.90)
            }
            n_avg = 6
            trials_step = 4

        f_name = f"results-{self.pp['strat']}.pkl"
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                self.logger.error(f"Found {len(trials.trials)} saved trials")
        except:
            trials = Trials()

        fn = partial(
            Runner.hopt_proc, stratclass, self.pp, n_avg=n_avg)
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
                trials=trials
            )
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


if __name__ == '__main__':
    r = Runner()
