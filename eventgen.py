from enum import Enum, auto

import numpy as np


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # End a current call


class EventGen:
    def __init__(self, rows, cols, call_rates, call_duration):
        self.rows = rows
        self.cols = cols
        self.call_rates = call_rates
        self.call_duration = call_duration

    def event_new(self, t):
        """
        Generate a new call event at a random cell at an
        exponentially distributed time dt from t
        """
        # Pick a random cell for new incoming call
        cell_r = np.random.randint(0, self.rows)
        cell_c = np.random.randint(0, self.cols)
        e_time = np.random.exponential(self.call_rates[cell_r][cell_c]) + t
        return (e_time, CEvent.NEW, (cell_r, cell_c))

    def event_end(self, t, cell, ch):
        """
        Generate end event for a call
        """
        e_time = np.random.exponential(self.call_duration) + t
        return (e_time, CEvent.END, cell, ch)
