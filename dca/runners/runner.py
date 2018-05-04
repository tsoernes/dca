import cProfile
import datetime
import logging
import sys

import numpy as np

from gui import Gui  # noqa


class Runner:
    def __init__(self, pp, stratclass):
        self.pp, self.stratclass = pp, stratclass

        if self.pp['log_file']:
            fname = self.pp['log_file'] + ".log"
            logging.basicConfig(
                filename=fname, level=self.pp['log_level'], format='%(message)s')
            self.logger = logging.getLogger('')
            self.fo_logger = logging.getLogger('file')  # File only
            th = logging.StreamHandler(sys.stdout)
            th.setLevel(self.pp['log_level'])
            self.logger.addHandler(th)  # Log to file + stdout
        else:
            logging.basicConfig(level=self.pp['log_level'], format='%(message)s')
            self.logger = self.fo_logger = logging.getLogger('')
        cmdparms = " ".join(sys.argv[1:])
        self.logger.error(
            f"{cmdparms}\nStarting simulation at {datetime.datetime.now()} with params:\n{self.pp}"
        )

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


class ShowRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        # g = grid.RhombusAxialGrid(logger=self.logger, **self.pp)
        # gui = Gui(g, self.logger, g.print_neighs, "rhomb")
        # grid = RectOffGrid(logger=self.logger, **self.pp)
        # gui = Gui(grid, self.logger, grid.print_neighs, "rect")
        # gui.test()
        raise NotImplementedError


class AnalyzeNetRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
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


class TrainNetRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_net(self):
        strat = self.stratclass(self.pp, logger=self.logger)
        strat.net.train()
