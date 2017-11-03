from gui import Gui
from grid import FixedGrid, RhombAxGrid, RectOffGrid
from strats import FixedAssign, strat_classes
from params import get_pparams, sample_params, sample_gamma

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
        self.pp = get_pparams()

        logging.basicConfig(level=self.pp['log_level'], format='%(message)s')
        self.logger = logging.getLogger('')
        if self.pp['log_file']:
            fh = logging.FileHandler(self.pp['log_file'])
            fh.setLevel(self.pp['log_level'])
            self.logger.addHandler(fh)
        self.logger.error(f"Starting simulation at {datetime.datetime.now()}"
                          f" with params:\n{self.pp}")

        if self.pp['test_params']:
            self.test_params()
        elif self.pp['hopt']:
            self.hopt()
        elif self.pp['avg_runs'] > 1:
            self.avg_run()
        elif self.pp['strat'] == 'show':
            self.show()
        else:
            self.run()

    def test_params(self):
        gridclass, stratclass = self.get_class(self.pp)
        simproc = partial(
            self.sim_proc, gridclass, stratclass, self.pp, do_sample=True)
        with Pool() as p:
            p.map(simproc, range(self.pp['param_iters']))

    def avg_run(self):
        t = time.time()
        n_eps = self.pp['avg_runs']
        gridclass, stratclass = self.get_class(self.pp)
        simproc = partial(
            self.sim_proc, gridclass, stratclass, self.pp)
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

    @staticmethod
    def get_class(pp):
        s = pp['strat']
        if s == 'fixed':
            return (FixedGrid, FixedAssign)
        stratcls = strat_classes()
        for name, cls in stratcls:
            if s == name.lower():
                return(RhombAxGrid, cls)

    def run(self):
        gridclass, stratclass = self.get_class(self.pp)
        grid = gridclass(logger=self.logger, **self.pp)
        strat = stratclass(self.pp, grid=grid, logger=self.logger)
        if self.pp['gui']:
            gui = Gui(grid, strat.exit_handler, grid.print_cell)
            strat.gui = gui
        if self.pp['profiling']:
            cProfile.runctx('strat.init_sim()', globals(), locals(),
                            sort='tottime')
        else:
            strat.init_sim()

    @staticmethod
    def sim_proc(gridclass, stratclass, pp, pid,
                 do_sample=False, do_sample_gamma=False):
        """
        Allows for running simulation in separate process
        """
        np.random.seed()
        if do_sample:
            pp = sample_params(pp)
        if do_sample_gamma:
            pp = sample_gamma(pp)
        logger = logging.getLogger('')
        grid = gridclass(logger=logger, **pp)
        strat = stratclass(pp, grid=grid, logger=logger, pid=pid)
        result = strat.init_sim()
        return result

    def show(self):
        grid = RhombAxGrid(logger=self.logger, **self.pp)
        gui = Gui(grid, self.logger, grid.print_neighs, "rhomb")
        # grid = RectOffGrid(logger=self.logger, **self.pp)
        # gui = Gui(grid, self.logger, grid.print_neighs, "rect")
        gui.test()

    def hopt(self, net=False):
        """
        Hyper-parameter optimization with hyperopt.
        Saves progress to 'results-{stratname}.pkl' and
        automatically resumes if file already exists.
        """
        # TODO parallell execution
        from hyperopt import fmin, tpe, hp, Trials
        gridclass, stratclass = Runner.get_class(self.pp)
        if net:
            space = {
                'alpha': hp.loguniform(
                    'alpha', np.log(0.000001), np.log(0.01)),
            }
            self.pp['n_events'] = 30000
            trials_step = 1
        else:
            space = {
                'alpha': hp.loguniform(
                    'alpha', np.log(0.001), np.log(0.3)),
                'alpha_decay': hp.uniform(
                    'alpha_decay', 0.9999, 0.9999999),
                'epsilon': hp.loguniform(
                    'epsilon', np.log(0.1), np.log(0.8)),
                'epsilon_decay': hp.uniform(
                    'epsilon_decay', 0.9995, 0.9999999),
                'gamma': hp.uniform('gamma', 0.6, 1)
            }
            trials_step = 4

        f_name = f"results-{self.pp['strat']}.pkl"
        try:
            trials = pickle.load(open(f_name, "rb"))
            self.logger.error(f"Found {len(trials.trials)} saved trials")
        except:
            trials = Trials()

        f = partial(Runner.hopt_proc, gridclass, stratclass, self.pp)
        prev_best = {}
        while True:
            n_trials = len(trials)
            self.logger.error(
                f"Running trials {n_trials+1}-{n_trials+trials_step}")
            best = fmin(
                fn=f,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials + trials_step,
                trials=trials
            )
            if prev_best != best:
                self.logger.error(f"Found new best params: {best}")
                prev_best = best
            with open(f_name, "wb") as fil:
                pickle.dump(trials, fil)

    @staticmethod
    def hopt_proc(gridclass, stratclass, pp, space):
        for key, val in space.items():
            pp[key] = val
        logger = logging.getLogger('')
        logger.error(space)
        grid = gridclass(logger=logger, **pp)
        strat = stratclass(pp, grid=grid, logger=logger)
        result = strat.init_sim()
        return result


if __name__ == '__main__':
    r = Runner()
