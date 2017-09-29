from enum import Enum, auto

import numpy as np


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # End a current call
    HOFF = auto()  # Handoff a call from one cell to another


class EventGen:
    def __init__(self, rows, cols, call_rates, call_duration,
                 hoff_call_duration,
                 *args, **kwargs):
        self.rows = rows
        self.cols = cols
        if type(call_rates) != np.ndarray:
            # Use uniform call rates through grid
            self.call_rates = np.ones((rows, cols)) * call_rates
        else:
            self.call_rates = call_rates
        # Avg. time between arriving calls
        self.call_intertimes = 1/self.call_rates
        self.call_duration = call_duration
        self.handoff_call_duration = hoff_call_duration

    def event_new(self, t, cell):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time dt from t.
        """
        dt = np.random.exponential(self.call_intertimes[cell])
        return (dt + t, CEvent.NEW, cell)

    def event_end(self, t, cell, ch):
        """
        Generate end event for a call
        """
        e_time = np.random.exponential(self.call_duration) + t
        return (e_time, CEvent.END, cell, ch)

    def event_handoff(self, t, cell, neighs, ch):
        """
        Pick a neighbor of cell randomly and hand off call.
        :param 1D ndarray neighs - indices of neighbors

        How should this be integrated?
        Should newly accepted calls generate a handoff event
        with some probability instead of an end event?
        If so, they will have an average duration in total
        that's longer than normal calls.
        """
        neigh = np.random.randint(0, len(neighs))
        e_time = np.random.exponential(self.handoff_call_duration) + t
        return (e_time, CEvent.HOFF, neighs[neigh], ch)


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
