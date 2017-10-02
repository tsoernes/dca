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
        # mapping of event time to events
        self.cevents = {}
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
        self._push((t + dt, CEvent.END, cell, ch))

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
        new_event = (end_event[0]+sys.float_info.epsilon,
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
            cevent = self.cevents[key]
        except KeyError:
            print(self.cevents)
            raise
        assert cevent[1] == CEvent.END
        self.cevents[key] = (cevent[0], cevent[1], cevent[2], to_ch)

    def _push(self, cevent):
        # TODO need a way to find and change an event in the
        # heapq with a given cell and channel (for it to be
        # reassigned to another ch).
        # problem: new-events don't have channel, can't sort on it
        # only store end events?

        self.event_id += 1
        eid =
        self.cevents[t] = (*cevent, self.event_id)
        heappush(self.pq, t)

    def pop(self):
        if self.pq:
            t = heappop(self.pq)
            cevent = self.cevents[t]
            del self.cevents[t]
            return cevent
        raise KeyError('pop from an empty priority queue')


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
