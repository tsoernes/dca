from enum import Enum, auto

import numpy as np


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # End a current call


class EventGen:
    def __init__(self, rows, cols, call_rates, call_duration):
        self.rows = rows
        self.cols = cols
        if type(call_rates) == int:
            self.call_rates = np.ones((rows, cols)) * call_rates
        else:
            self.call_rates = call_rates
        self.call_duration = call_duration

    def event_new(self, t, cell_row, cell_col):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time dt from t.
        """
        e_time = np.random.exponential(self.call_rates[cell_row][cell_col]) + t
        return (e_time, CEvent.NEW, (cell_row, cell_col))

    def event_end(self, t, cell, ch):
        """
        Generate end event for a call
        """
        e_time = np.random.exponential(self.call_duration) + t
        return (e_time, CEvent.END, cell, ch)
