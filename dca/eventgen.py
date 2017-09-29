from enum import Enum
from functools import total_ordering
import numpy as np


@total_ordering
class CEvent(Enum):
    NEW = 0  # Incoming call
    END = 1  # End a current call
    HOFF = 2  # Handoff a call from one cell to another

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


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
        # TODO Could possible simplify eventgen implementation,
        # by generating end events simultaneously,
        # and handling handoff generation here instead
        # of in strats

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

    def event_new_handoff(self, t, cell, neighs, ch):
        """
        Pick a neighbor of cell randomly and hand off call.
        :param 1D ndarray neighs - indices of neighbors

        A new call is handed off instead of terminated with
        some probability, in which case it generates
        an end event at the leaving cell and a hoff event
        at the entering cell, both with the same event time.
        The end event, having lower Enum(1) than the handoff
        event (2), is handled first due to minheap sorting,
        and keeping handoff events
        separate from new events makes it possible to reward
        handoff acceptance/rejectance different from new calls.

        In total, a handoff generates 3 events:
        - end call in current cell
        - new call in neighboring cell
        - end call in neighboring cell
        """
        neigh = np.random.randint(0, len(neighs))
        end_event = (t, CEvent.END, cell, ch)
        new_event = (t, CEvent.HOFF, neighs[neigh])
        return (end_event, new_event)

    def event_end_handoff(self, t, cell, ch):
        e_time = np.random.exponential(self.handoff_call_duration) + t
        return (e_time, CEvent.END, cell, ch)


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
