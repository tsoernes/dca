from main import RLStrat, FAStrat
from gui import Gui
from grid import Grid, FixedGrid
from eventgen import EventGen

import argparse
import cProfile
import logging


def def_pparams(
        rows=7,
        cols=7,
        n_channels=70,
        erlangs=8,
        call_rates=None,
        call_duration=3,
        n_episodes=2000000,
        n_hours=2,
        epsilon=0.2,
        epsilon_decay=0.999999,
        alpha=0.05,
        alpha_decay=0.999999,
        gamma=0.9):
    """
    n_hours: If n_episodes is not specified, run simulation for n_hours
        of simulation time
    Call rates in calls per minute
    Call duration in minutes
    gamma:
    """
    # erlangs = call_rate * duration
    # 10 erlangs = 200cr, 3cd
    # 7.5 erlangs = 150cr, 3cd
    # 5 erlangs = 100cr, 3cd
    if not call_rates:
        call_rates = erlangs / call_duration
    return {
            'rows': rows,
            'cols': cols,
            'n_channels': n_channels,
            'call_rates': call_rates,  # Avg. call rate, in calls per minute
            'call_duration': call_duration,  # Avg. call duration in minutes
            'n_episodes': n_episodes,
            'n_hours': n_hours,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'alpha': alpha,
            'alpha_decay': alpha_decay,
            'gamma': gamma
           }


def get_pparams():
    parser = argparse.ArgumentParser(description='DCA')
    parser.add_argument('--eps', dest='epsilon', type=float,
                        help='epsilon)')
    parser.add_argument('--epsdec', dest='epsilon_decay', type=float,
                        help='epsilon decay)')
    parser.add_argument('--alpha', dest='alpha', type=float)
    parser.add_argument('--alpha_dec', dest='alpha_decay', type=float,
                        help='alpa decay')
    args = parser.parse_args()
    present = dict()
    for k, v in vars(args).items():
        if v:
            present[k] = v
    params = def_pparams(**present)
    return params


class Runner:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger('')
        fh = logging.FileHandler('out.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.pp = get_pparams()
        self.logger.info(f"Starting simulation with params {self.pp}")

    def run(self, show_gui=False):
        grid = Grid(logger=self.logger, **self.pp)
        eventgen = EventGen(**self.pp)
        if show_gui:
            gui = Gui(grid, self.end_sim)
        else:
            gui = None
        self.strat = RLStrat(
                self.pp, grid=grid, gui=gui, eventgen=eventgen,
                version='trimmed',
                sanity_check=False, logger=self.logger)
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

    def run_fa(self):
        grid = FixedGrid(**self.pp)
        grid.assign_chs()
        eventgen = EventGen(**self.pp)
        gui = Gui(grid)
        fa_strat = FAStrat(self.pp, grid=grid, gui=gui, eventgen=eventgen)
        fa_strat.simulate(self.pp, grid, fa_strat, eventgen, gui)


if __name__ == '__main__':
    r = Runner()
    # cProfile.run('r.run()')
    r.run()
