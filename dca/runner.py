from strats import FixedAssign, SARSA, TT_SARSA, RS_SARSA
from gui import Gui
from grid import Grid, FixedGrid

import argparse
import cProfile
import datetime
import logging
from multiprocessing import Process

import numpy as np


def get_pparams():
    """
    Problem parameters
    """
    parser = argparse.ArgumentParser(
        description='DCA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--strat',
        choices=['show', 'fixed', 'sarsa', 'tt_sarsa', 'rs_sarsa'],
        help="show: just show gui",
        default='fixed')
    parser.add_argument(
        '--rows', type=int, help="number of rows in grid", default=7)
    parser.add_argument(
        '--cols', type=int, help="number of cols in grid", default=7)
    parser.add_argument(
        '--n_channels', type=int, help="number of channels", default=70)
    parser.add_argument(
        '--erlangs',
        type=int,
        help="erlangs = call_rate * call_duration"
        "\n 10 erlangs = 200 call rate, given 3 call duration"
        "\n 7.5 erlangs = 150cr, 3cd"
        "\n 5 erlangs = 100cr, 3cd",
        default=10)
    parser.add_argument(
        '--call_rates', type=int, help="in calls per minute", default=None)
    parser.add_argument(
        '--call_duration', type=int, help="in minutes", default=3)
    parser.add_argument(
        '--p_handoff', type=float, help="handoff probability", default=0.15)
    parser.add_argument(
        '--hoff_call_duration',
        type=int,
        help="handoff call duration, in minutes",
        default=1)
    parser.add_argument(
        '--n_episodes',
        type=int,
        help="number of events to simulate",
        default=200000)
    parser.add_argument(
        '--n_hours',
        type=int,
        help="number hours in simulation time to run",
        default=2)

    parser.add_argument(
        '--alpha', type=float, help="(RL) learning rate",
        default=0.02)
    parser.add_argument(
        '--alpha_decay',
        type=float,
        help="(RL) factor by which alpha is multiplied each iter",
        default=0.99999)
    parser.add_argument(
        '--epsilon',
        type=float,
        help="(RL) probability of choosing random action",
        default=0.3)
    parser.add_argument(
        '--epsilon_decay',
        type=float,
        help="(RL) factor by which epsilon is multiplied each iter",
        default=0.99999)
    parser.add_argument(
        '--gamma', type=float, help="(RL) discount factor",
        default=0.9)
    parser.add_argument(
        '--test_params',
        action='store_true',
        help="(RL) override default params by sampling in logspace"  # noqa
        "store results to logfile 'paramtest-MM.DD-hh.mm'.",
        default=False)
    parser.add_argument(
        '--param_iters',
        type=int,
        help="number of parameter iterations",
        default=1)

    parser.add_argument(
        '--verify_grid',
        action='store_true',
        help="verify reuse constraint each iteration",
        default=False)
    parser.add_argument(
        '--prof',
        dest='profiling',
        action='store_true',
        help="performance profiling",
        default=False)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument(
        '--plot', action='store_true', dest='do_plot', default=False)
    parser.add_argument(
        '--log_level',
        type=int,
        help="10: Debug,\n20: Info,\n30: Warning",
        default=logging.INFO)
    parser.add_argument(
        '--log_file', type=str,
        help="enable logging to file by entering file name")
    parser.add_argument(
        '--log_iter',
        type=int,
        help="Show blocking probability every n iterations",  # noqa
        default=100000)

    # iterations can be approximated from hours with:
    # iters = 7821* hours - 2015
    args = parser.parse_args()
    params = vars(args)

    if not params['call_rates']:
        params['call_rates'] = params['erlangs'] / params['call_duration']
    if params['test_params']:
        params['log_level'] = logging.ERROR
        params['gui'] = False
        params['plot'] = False
        params['log_iter'] = 1000
        now = datetime.datetime.now()
        date = now.date()
        time = now.time()
        params['log_file'] = f"logs/paramtest-{date.month}.{date.day}-" \
                             f"{time.hour}.{time.minute}.log"
    return params


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
        elif self.pp['strat'] == 'show':
            self.show()
        else:
            self.run()

    @staticmethod
    def sample_gamma(pp):
        npp = dict(pp)
        npp['gamma'] = np.random.uniform()
        return npp

    @staticmethod
    def sample_params(pp):
        """
        Sample parameters randomly,
        return copy of dict
        """
        # It's best to optimize in log space
        npp = dict(pp)
        npp['alpha'] = 10**np.random.uniform(-2, -0.3)
        npp['alpha_decay'] = 10**np.random.uniform(-0.0001, -0.00000001)
        npp['epsilon'] = 10**np.random.uniform(-1.3, -0.1)
        npp['epsilon_decay'] = 10**np.random.uniform(-0.0001, -0.00000001)
        return npp

    def test_params(self):
        n = 14  # number of concurrent processes
        gridclass, stratclass = self.get_class(self.pp)
        for i in range(1, self.pp['param_iters'] + 1):
            self.logger.error(f"Starting iters {i}-{i+n}")
            for j in range(1, n + 1):
                k = j * i
                pp = self.sample_gamma(self.pp)
                # pp = self.sample_params(self.pp)
                p = Process(
                    target=self.sim_proc, args=(gridclass, stratclass, pp, k))
                p.start()
            p.join()
        p.join()

    @staticmethod
    def get_class(pp):
        s = pp['strat']
        if s == 'fixed':
            return (FixedGrid, FixedAssign)
        elif s == 'sarsa':
            return (Grid, SARSA)
        elif s == 'tt_sarsa':
            return (Grid, TT_SARSA)
        elif s == 'rs_sarsa':
            return (Grid, RS_SARSA)

    def run(self):
        gridclass, stratclass = self.get_class(self.pp)
        grid = gridclass(logger=self.logger, **self.pp)
        self.strat = stratclass(self.pp, grid=grid, logger=self.logger)
        if self.pp['gui']:
            gui = Gui(grid, self.strat.exit_handler, grid.print_cell)
            self.strat.gui = gui
        if self.pp['profiling']:
            cProfile.runctx('self.strat.init_sim()', globals(), locals())
            #                 sort='tottime')
        else:
            self.strat.init_sim()

    def end_sim(self, e):
        """
        Handle key events from Tkinter and quit
        simulation gracefully on 'q'-key
        """
        self.strat.quit_sim = True

    def show(self):
        grid = FixedGrid(logger=self.logger, **self.pp)
        gui = Gui(grid, self.logger)
        gui.test()

    @staticmethod
    def sim_proc(gridclass, stratclass, pp, pid):
        logger = logging.getLogger('')
        grid = gridclass(logger=logger, **pp)
        strat = stratclass(pp, grid=grid, logger=logger, pid=pid)
        strat.init_sim()


if __name__ == '__main__':
    r = Runner()
