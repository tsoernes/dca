from gui import Gui
from grid import Grid, FixedGrid
from strats import RandomAssign, FixedAssign, \
        SARSA, TT_SARSA, RS_SARSA, \
        SARSAQNet_idx_nused
from params import get_pparams, sample_params, sample_gamma

import cProfile
import datetime
from functools import partial
import logging
import time
import pickle
from multiprocessing import Pool

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
        elif s == 'random':
            return (Grid, RandomAssign)
        elif s == 'sarsa':
            return (Grid, SARSA)
        elif s == 'tt_sarsa':
            return (Grid, TT_SARSA)
        elif s == 'rs_sarsa':
            return (Grid, RS_SARSA)
        elif s == 'sarsaqnet':
            return (Grid, SARSAQNet_idx_nused)

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
        grid = FixedGrid(logger=self.logger, **self.pp)
        gui = Gui(grid, self.logger)
        gui.test()

    def hopt(self):
        """
        Hyper-parameter optimization with hyperopt.
        Saves progress to 'results-{stratname}.pkl' and
        automatically resumes if file already exists.
        """
        # TODO parallell execution
        from hyperopt import fmin, tpe, hp, Trials
        gridclass, stratclass = Runner.get_class(self.pp)
        space = {
            'alpha': hp.loguniform('alpha', np.log(0.005), np.log(0.2)),
            'alpha_decay': hp.loguniform('alpha_decay', np.log(0.999), np.log(0.9999999)),  # noqa
            'epsilon': hp.loguniform('epsilon', np.log(0.1), np.log(0.8)),
            'epsilon_decay': hp.loguniform('epsilon_decay', np.log(0.999), np.log(0.9999999)),  # noqa
            'gamma': hp.uniform('gamma', 0.4, 1)
        }
        f = partial(Runner.hopt_proc, gridclass, stratclass, self.pp)

        f_name = f"results-{self.pp['strat']}.pkl"

        # how many additional trials to do after loading saved
        # trials. 1 = save after iteration
        trials_step = 1

        while True:
            # initial max_trials. put something small to not have to wait
            max_trials = 5
            try:
                trials = pickle.load(open(f_name, "rb"))
                max_trials = len(trials.trials) + trials_step
                self.logger.error(
                    "Found saved Trials! Loading..."
                    f"\nRunning from {len(trials.trials)} trials"
                    f" to {max_trials+trials_step}")
            except:
                trials = Trials()

            best = fmin(
                fn=f,
                space=space,
                algo=tpe.suggest,
                max_evals=max_trials,
                trials=trials,
            )

            # How to get the best over ALL trials, including loaded?
            # Does 'best' return the best for all trials, or just the new ones
            self.logger.error(f"Best found hopt: {best}")
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
