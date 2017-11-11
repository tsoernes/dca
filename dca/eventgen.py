from enum import Enum
from functools import total_ordering
from heapq import heappop, heappush

import numpy as np


@total_ordering
class CEvent(Enum):
    NEW = 0  # Incoming call
    END = 1  # End a current call
    # Denotes the incoming part to the receiving cell of a handoff event
    HOFF = 2

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class EventGen:
    def __init__(self, rows, cols, call_rates, call_duration,
                 hoff_call_duration, logger, *args, **kwargs):
        self.rows = rows
        self.cols = cols
        if type(call_rates) != np.ndarray:
            # Use uniform call rates through grid
            call_rates = np.ones((rows, cols)) * call_rates
        # Avg. time between arriving calls
        self.call_intertimes = 1 / call_rates
        self.call_duration = call_duration
        self.handoff_call_duration = hoff_call_duration
        self.logger = logger

        self.pq = []  # min-heap of event timestamps
        # mapping from timestamps to events
        self.events = {}
        # mapping from (cell_row, cell_col, ch) to end event timestamps
        self.end_events = {}

    def event_new(self, t, cell):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time dt from t.
        """
        dt = np.random.exponential(self.call_intertimes[cell])
        self._push((t + dt, CEvent.NEW, cell))

    def event_end(self, t, cell, ch):
        """
        Generate and return an end event for a call
        """
        dt = np.random.exponential(self.call_duration)
        event = (t + dt, CEvent.END, cell, ch)
        self._push(event)
        return event

    def event_new_handoff(self, t, cell, ch, neighs):
        """
        Pick a neighbor of cell randomly and hand off call.
        :param 1D ndarray neighs - indices of neighbors

        A new call is handed off instead of terminated with
        some probability, in which case it generates
        an end event at the leaving cell and a hoff event
        at the entering cell, both with the same event time
        some time dt from now.

        In total, a handoff generates 3 events:
        - end call in current cell (END)
        - new call in neighboring cell (HOFF)
        - end call in neighboring cell (END)
        """
        neigh = np.random.randint(0, len(neighs))
        end_event = self.event_end(t, cell, ch)
        # The end event, having lower Enum (1) than the handoff
        # event (2), is handled first due to the minheap sorting
        # on the second tuple element if the first is equal.
        # Keeping handoff events separate from new events makes it possible
        # to reward handoff acceptance/rejectance different from new calls.
        new_event = (end_event[0], CEvent.HOFF, neighs[neigh])
        self.logger.debug(
            f"Created handoff event for cell {cell} ch {ch}"
            f" to cell {neighs[neigh]} scheduled for time {end_event[0]}")
        self._push(new_event)

    def event_end_handoff(self, t, cell, ch):
        dt = np.random.exponential(self.handoff_call_duration)
        self._push((t + dt, CEvent.END, cell, ch))

    def reassign(self, cell, from_ch, to_ch):
        """
        Reassign the event at time 't' to channel 'ch'
        """
        # If the channels are equal, that means that
        # 'to_ch' belongs to a call (end_event) that is
        # already popped from the heapq, which means
        # that it will trigger a key-error.
        assert from_ch != to_ch
        end_key = (*cell, from_ch)
        try:
            key = self.end_events[end_key]
            event = self.events[key]
        except KeyError:
            self.logger.error(self.events)
            self.logger.error(self.end_events)
            raise
        self.events[key] = (event[0], event[1], event[2], to_ch)
        del self.end_events[end_key]
        self.end_events[(*cell, to_ch)] = key

    def _push(self, event):
        key = (event[0], event[1])
        self.events[key] = event
        if event[1] == CEvent.END:
            self.end_events[(*event[2], event[3])] = key
        heappush(self.pq, key)

    def pop(self):
        if self.pq:
            key = heappop(self.pq)
            try:
                event = self.events[key]
            except KeyError:
                print(self.events)
                raise
            if event[1] == CEvent.END:
                del self.end_events[(*event[2], event[3])]
            del self.events[key]
            return event
        raise KeyError('Pop from an empty priority queue')


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
