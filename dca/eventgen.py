from enum import IntEnum
from functools import total_ordering
from heapq import heappop, heappush

import numpy as np


@total_ordering
class CEvent(IntEnum):
    NEW = 0  # Incoming call
    END = 1  # End a current call
    HOFF = 2  # Incoming call, handed off from another cell

    def __lt__(self, other):
        """Allows for handling the END event part of a handoff before the
        HOFF part (i.e. incoming call)"""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplementedError


class EventGen:
    def __init__(self, rows, cols, call_rates, call_duration, hoff_call_duration, logger,
                 *args, **kwargs):
        self.rows, self.cols = rows, cols
        if type(call_rates) == float or type(call_rates) == int:
            # Use uniform call rates throughout grid
            call_rates = np.ones((rows, cols)) * call_rates
        # Avg. time between arriving calls
        self.call_intertimes = 1 / call_rates
        self.call_duration = call_duration
        self.handoff_call_duration = hoff_call_duration
        self.logger = logger

        # min-heap of event timestamps
        self.event_times = []
        # mapping from timestamps to events
        self.events = {}
        # mapping from (cell_row, cell_col, ch) to end event timestamps
        self.end_event_times = {}

    def event_new(self, t, cell, dt=None):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time 'dt' from 't'.
        """
        if dt is None:
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
        Hand off a call to a neighbor picked uniformly at random
        :param 1D ndarray neighs - indices of neighbors

        When a call is handed off, an END event at the leaving cell and a HOFF
        event at the entering cell is generated, both with the same event time
        some time 'dt' from now.

        The END event, having lower enum (1) than the HOFF
        event (2), is handled first due to the minheap sorting
        on the second tuple element if the first element (timestamp) is equal.
        Keeping handoff events separate from new events makes it possible
        to reward or log handoff acceptance/rejectance differently from regular
        new calls.
        """
        neigh_idx = np.random.randint(0, len(neighs))
        end_event = self.event_end(t, cell, ch)
        new_event = (end_event[0], CEvent.HOFF, neighs[neigh_idx])
        self.logger.debug(
            f"Created handoff event for cell {cell} ch {ch}"
            f" to cell {neighs[neigh_idx]} scheduled for time {end_event[0]}")
        self._push(new_event)

    def event_end_handoff(self, t, cell, ch):
        dt = np.random.exponential(self.handoff_call_duration)
        self._push((t + dt, CEvent.END, cell, ch))

    def reassign(self, cell, from_ch, to_ch):
        """
        Reassign the call in cell @cell@ on ch @from_ch@ to channel $to_ch$
        """
        # If the channels are equal, then
        # 'to_ch' belongs to a call (end_event) that is
        # already popped from the heap queue, which means
        # that it will trigger a key-error.
        assert from_ch != to_ch
        end_key = (*cell, from_ch)
        try:
            key = self.end_event_times[end_key]
            event = self.events[key]
        except KeyError:
            self.logger.error(self.events)
            self.logger.error(self.end_event_times)
            raise
        self.events[key] = (event[0], event[1], event[2], to_ch)
        del self.end_event_times[end_key]
        self.end_event_times[(*cell, to_ch)] = key

    def _push(self, event):
        key = (event[0], event[1])
        self.events[key] = event
        if event[1] == CEvent.END:
            self.end_event_times[(*event[2], event[3])] = key
        heappush(self.event_times, key)

    def pop(self):
        if not self.event_times:
            raise KeyError('No events to pop')
        key = heappop(self.event_times)
        try:
            event = self.events[key]
        except KeyError:
            print(self.events)
            raise
        if event[1] == CEvent.END:
            del self.end_event_times[(*event[2], event[3])]
        del self.events[key]
        return event


def ce_str(cevent):
    string = f"{cevent[0]:.3f}: {cevent[1].name} {cevent[2]}"
    if cevent[1] == CEvent.END:
        string += f" ch{cevent[3]}"
    return string
