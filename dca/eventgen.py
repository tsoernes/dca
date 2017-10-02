from enum import Enum
from functools import total_ordering
from heapq import heappush, heappop
import sys

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

        self.pq = []  # min-heap of event times
        # mapping from time to events
        self.events = {}
        # mapping from (cell_r, cell_c, ch) to end event timestamps
        self.end_events = {}
        self.event_id = 0

    def event_new(self, t, cell):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time dt from t.
        """
        dt = np.random.exponential(self.call_intertimes[cell])
        self._push((t + dt, CEvent.NEW, cell))

    def event_end(self, t, cell, ch):
        """
        Generate end event for a call
        """
        dt = np.random.exponential(self.call_duration)
        event = (t + dt, CEvent.END, cell, ch)
        self._push(event)
        return event

    def event_new_handoff(self, t, cell, neighs, ch):
        """
        Pick a neighbor of cell randomly and hand off call.
        :param 1D ndarray neighs - indices of neighbors

        A new call is handed off instead of terminated with
        some probability, in which case it generates
        an end event at the leaving cell and a hoff event
        at the entering cell, both with the same event time
        some time dt from now.
        The end event, having lower Enum (1) than the handoff
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
        end_event = self.event_end(t, cell, ch)
        # make hoff event infinitesimally later than end event
        new_event = (end_event[0]+0.1,  #+sys.float_info.epsilon,
                     CEvent.HOFF, neighs[neigh])
        self._push(end_event)
        self._push(new_event)

    def event_end_handoff(self, t, cell, ch):
        dt = np.random.exponential(self.handoff_call_duration)
        self._push((t + dt, CEvent.END, cell, ch))

    def reassign(self, cell, from_ch, to_ch):
        """
        Reassign the event at time 't' to channel 'ch'
        """
        key = (*cell, from_ch)
        try:
            t = self.end_events[key]
        except KeyError:
            print(self.events)
            raise
        event = self.events[t]
        self.events[t] = (event[0], event[1], event[2], to_ch)
        del self.end_events[key]
        self.end_events[(*cell, to_ch)] = t

    def _push(self, event):
        t = event[0]
        self.events[t] = event
        if event[1] == CEvent.END:
            self.end_events[(*event[2], event[3])] = t
        heappush(self.pq, t)

    def pop(self):
        if self.pq:
            t = heappop(self.pq)
            try:
                event = self.events[t]
            except KeyError:
                print(t)
                print(self.events)
                raise
            if event[1] == CEvent.END:
                del self.end_events[(*event[2], event[3])]
            del self.events[t]
            return event
        raise KeyError('pop from an empty priority queue')


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
