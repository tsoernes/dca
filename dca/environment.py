import numpy as np

from eventgen import CEvent, EventGen, ce_str
from stats import Stats


class Env:
    def __init__(self, pp, grid, logger, pid="", gui=None, *args, **kwargs):
        self.rows, self.cols = pp['rows'], pp['cols']
        self.p_handoff = pp['p_handoff']
        self.verify_grid = pp['verify_grid']
        self.save = pp['save_exp_data']
        self.log_iter = pp['log_iter']
        self.dt_rewards = pp['dt_rewards']
        self.beta = pp['beta']
        self.grid = grid
        self.logger = logger
        self.gui = gui

        self.stats = Stats(pp=pp, logger=logger, pid=pid)
        self.eventgen = EventGen(logger=logger, **pp)

        # Current call event for which an action must be taken
        self.cevent = None
        # self.init_calls = self.init_marker_calls

    def init_marker_calls(self):
        """
        Generate DETERMINISTIC initial call events, in
        and return the first event.
        """
        self.logger.error("USING MARKER CALLS")
        dt = 0
        for r in range(self.rows):
            for c in range(self.cols):
                self.eventgen.event_new(0, (r, c), dt)
                dt += 1
        self.cevent = self.eventgen.pop()
        return self.cevent

    def init_calls(self):
        """
        Generate initial call events; one for each cell,
        and return the first event.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.eventgen.event_new(0, (r, c))
        self.cevent = self.eventgen.pop()
        return self.cevent

    def step(self, ch: int):
        """
        Execute action 'ch' in the environment and return the
        resulting reward and the next event
        """
        assert type(ch) is np.int64 or ch is None
        t, ce_type, cell = self.cevent[0:3]
        self.stats.iter(t, self.cevent)

        # Generate next event, log statistics and update the GUI
        n_used = np.count_nonzero(self.grid.state[cell])
        if ce_type == CEvent.NEW:
            self.stats.event_new()
            # Generate next incoming call
            self.eventgen.event_new(t, cell)
            if ch is None:
                self.stats.event_new_reject(cell, n_used)
                if self.gui:
                    self.gui.hgrid.mark_cell(*cell)
            else:
                # With some probability, hand off call instead
                # of generating the usual END event
                if np.random.random() < self.p_handoff:
                    self.eventgen.event_new_handoff(t, cell, ch,
                                                    self.grid.neighbors1(*cell))
                else:
                    self.eventgen.event_end(t, cell, ch)
        elif ce_type == CEvent.HOFF:
            self.stats.event_hoff_new()
            if ch is None:
                self.stats.event_hoff_reject(cell, n_used)
                if self.gui:
                    self.gui.hgrid.mark_cell(*cell)
            else:
                self.eventgen.event_end_handoff(t, cell, ch)
        elif ce_type == CEvent.END:
            self.stats.event_end()
            if ch is None:
                self.logger.error("No channel assigned for end event")
                raise Exception
            if self.gui:
                self.gui.hgrid.unmark_cell(*cell)

        if ch is not None:
            self.execute_action(self.cevent, ch)
        if self.verify_grid and not self.grid.validate_reuse_constr():
            self.logger.error(f"Reuse constraint broken")
            raise Exception
        if self.gui:
            self.gui.step()

        self.cevent = self.eventgen.pop()
        dt = self.cevent[0] - t
        disc = np.exp(-self.beta * dt)
        return (self.reward(dt), disc, self.cevent)

    def execute_action(self, cevent, ch: int):
        """
        Change the grid state according to the given action.

        For NEW or HOFF events, 'ch' specifies the channel to be assigned
        in the cell.
        For END events, 'ch' specifies the channel to be terminated. If 'ch'
        is not equal to the channel specified to by the END event,
        the call on 'ch' will be reassigned to that channel.
        """
        ce_type, cell = cevent[1:3]
        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            if self.grid.state[cell][ch]:
                self.logger.error(f"Tried assigning new call {ce_str(cevent)} to"
                                  f" ch {ch} which is already in use")
                raise Exception
            self.logger.debug(f"Assigned call in cell {cell} to ch {ch}")
            self.grid.state[cell][ch] = 1
        elif ce_type == CEvent.END:
            reass_ch = cevent[3]
            if not self.grid.state[cell][reass_ch]:
                self.logger.error(f"Tried to end call {ce_str(cevent)}"
                                  f" which is not in progress")
                raise Exception
            if reass_ch != ch:
                if not self.grid.state[cell][ch]:
                    self.logger.error(f"Tried to reassign to {ce_str(cevent)}"
                                      f" from ch {ch} which is not in use")
                    raise Exception
                self.logger.debug(f"Reassigned call in cell {cell}"
                                  f" on ch {ch} to ch {reass_ch}")
                self.eventgen.reassign(cevent[2], ch, reass_ch)
            else:
                self.logger.debug(f"Ended call cell in {cell} on ch {ch}")
            self.grid.state[cell][ch] = 0

    def reward(self, dt):
        """
        Immediate reward, which is the total number of calls
        currently in progress system-wide
        """
        count = np.count_nonzero(self.grid.state)
        if not self.dt_rewards:
            return count
        b = self.pp['beta']
        return ((1 - np.exp(-b * dt)) / b) * count
