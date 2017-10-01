from strats import FixedAssign, SARSA, TT_SARSA, RS_SARSA
from gui import Gui
from grid import Grid, FixedGrid
from eventgen import EventGen

import argparse
import cProfile
import logging


def get_pparams():
    """
    Problem parameters
    # 10 erlangs = 200 call rate, 3 call duration
    # 7.5 erlangs = 150cr, 3cd
    # 5 erlangs = 100cr, 3cd
    """
    parser = argparse.ArgumentParser(
            description='DCA',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--strat',
                        choices=['none', 'fixed', 'sarsa', 'tt_sarsa', 'rs_sarsa'],  # noqa
                        help="none: just show gui",
                        default='fixed')
    parser.add_argument('--rows', type=int,
                        help="number of rows in grid",
                        default=7)
    parser.add_argument('--cols', type=int,
                        help="number of cols in grid",
                        default=7)
    parser.add_argument('--n_channels', type=int,
                        help="number of channels",
                        default=70)
    parser.add_argument('--erlangs', type=int,
                        help="erlangs = call_rate * call_duration",
                        default=10)
    parser.add_argument('--call_rates', type=int, help="in calls per minute",
                        default=None)
    parser.add_argument('--call_duration', type=int, help="in minutes",
                        default=3)
    parser.add_argument('--p_handoff', type=float, help="handoff probability",
                        default=0.15)
    parser.add_argument('--hoff_call_duration', type=int,
                        help="handoff call duration, in minutes",
                        default=1)
    parser.add_argument('--n_episodes', type=int,
                        help="number of events to simulate",
                        default=100000)
    parser.add_argument('--n_hours', type=int,
                        help="number hours in simulation time to run",
                        default=2)
    parser.add_argument('--alpha', type=float,
                        help="(RL) learning rate",
                        default=0.05)
    parser.add_argument('--alpha_decay', type=float,
                        help="(RL) factor by which alpha is multiplied each iter",  # noqa
                        default=0.999999)
    parser.add_argument('--epsilon', type=float,
                        help="(RL) probability of choosing random action",  # noqa
                        default=0.2)
    parser.add_argument('--epsilon_decay', type=float,
                        help="(RL) factor by which epsilon is multiplied each iter",  # noqa
                        default=0.999999)
    parser.add_argument('--gamma', type=float,
                        help="(RL) discount factor",
                        default=0.9)
    parser.add_argument('--prof', dest='profiling', action='store_true',
                        help="performance profiling",
                        default=False)
    parser.add_argument('--gui', action='store_true',
                        default=False)
    parser.add_argument('--verify_grid', action='store_true',
                        help="verify reuse constraint each iteration",
                        default=False)
    parser.add_argument('--log_level', type=int,
                        help="10: Debug, 20: Info, 30: Warning",
                        default=logging.INFO)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--log_iter', type=int,
                        help="Show blocking probability for the last n iterations",   # noqa
                        default=100000)

    args = parser.parse_args()
    params = vars(args)

    if not params['call_rates']:
        params['call_rates'] = params['erlangs'] / params['call_duration']
    return params


class Runner:
    def __init__(self):
        self.pp = get_pparams()

        logging.basicConfig(
                level=self.pp['log_level'], format='%(message)s')
        self.logger = logging.getLogger('')
        if self.pp['log_file']:
            fh = logging.FileHandler(self.pp['log_file'])
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
        self.logger.info(f"Starting simulation with params {self.pp}")

    def test_params(self):
        pass
        # alpha_range = [0.01, 0.2]
        # alpha_decay_range = [0.9999, 0.9999999]
        # epsilon_range = [0.05, 0.4]
        # epsilon_decay_range = [0.9999, 0.9999999]
        # TODO check karpathy lecs on how to sample

    def run(self):
        s = self.pp['strat']
        if s == 'fixed':
            self.run_strat(FixedGrid, FixedAssign)
        elif s == 'sarsa':
            self.run_strat(Grid, SARSA)
        elif s == 'tt_sarsa':
            self.run_strat(Grid, TT_SARSA)
        elif s == 'rs_sarsa':
            self.run_strat(Grid, RS_SARSA)
        elif s == 'none':
            self.show()

    def run_strat(self, gridclass, stratclass):
        grid = gridclass(logger=self.logger, **self.pp)
        eventgen = EventGen(**self.pp)
        self.strat = stratclass(
                    self.pp, grid=grid,
                    eventgen=eventgen, logger=self.logger)
        if self.pp['gui']:
            gui = Gui(grid, self.strat.exit_handler)
            self.strat.gui = gui
        if self.pp['profiling']:
            cProfile.runctx('self.strat.init_sim()', globals(), locals())
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


if __name__ == '__main__':
    r = Runner()
    r.run()
