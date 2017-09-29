from strats import RLStrat, FAStrat, SARSAStrat, TTSARSAStrat, RSSARSAStrat
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
    parser = argparse.ArgumentParser(description='DCA')

    parser.add_argument('--rows', type=int,
                        default=7)
    parser.add_argument('--cols', type=int,
                        default=7)
    parser.add_argument('--n_channels', type=int,
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
    parser.add_argument('--alpha', type=float, help="learning rate",
                        default=0.05)
    parser.add_argument('--alpha_decay', type=float,
                        help="factor by which alpha is multiplied each iter",
                        default=0.999999)
    parser.add_argument('--epsilon', type=float,
                        help="probability of choosing random action",
                        default=0.2)
    parser.add_argument('--epsilon_decay', type=float,
                        help="factor by which epsilon is multiplied each iter",
                        default=0.999999)
    parser.add_argument('--gamma', type=float,
                        help="discount factor",
                        default=0.9)
    parser.add_argument('--profiling', type=bool,
                        help="run performance profiling",
                        default=True)
    parser.add_argument('--show_gui', type=bool,
                        default=False)
    parser.add_argument('--sanity_check', type=bool,
                        help="verify reuse constraint each iteration",
                        default=False)
    parser.add_argument('--log_level', type=int,
                        default=logging.INFO)

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
        # fh = logging.FileHandler('out.log')
        # fh.setLevel(logging.INFO)
        # self.logger.addHandler(fh)
        self.logger.info(f"Starting simulation with params {self.pp}")

    def test_params(self):
        pass
        # alpha_range = [0.01, 0.2]
        # alpha_decay_range = [0.9999, 0.9999999]
        # epsilon_range = [0.05, 0.4]
        # epsilon_decay_range = [0.9999, 0.9999999]
        # TODO check karpathy lecs on how to sample

    def setup(self, grid):
        self.eventgen = EventGen(**self.pp)
        if self.pp['show_gui']:
            self.gui = Gui(grid, self.end_sim)
        else:
            self.gui = None

    def runFA(self):
        grid = FixedGrid(logger=self.logger, **self.pp)
        self.setup(grid)
        self.strat = FAStrat(
                self.pp, grid=grid, gui=self.gui,
                eventgen=self.eventgen,
                sanity_check=True, logger=self.logger)
        self.run(grid)

    def runTTSARSA(self):
        grid = Grid(logger=self.logger, **self.pp)
        self.setup(grid)
        self.strat = TTSARSAStrat(
                self.pp, grid=grid, gui=self.gui,
                eventgen=self.eventgen,
                sanity_check=self.pp['sanity_check'], logger=self.logger)
        self.run(grid)

    def run(self, grid):
        if self.pp['profiling']:
            cProfile.runctx('self.strat.simulate()', globals(), locals())
        else:
            self.strat.simulate()

    def end_sim(self, e):
        """
        Handle key events from Tkinter and quit
        simulation gracefully on 'q'-key
        """
        self.strat.quit_sim = True

    def show(self):
        grid = FixedGrid(**self.pp)
        gui = Gui(grid)
        gui.test()


if __name__ == '__main__':
    r = Runner()
    r.runTTSARSA()
