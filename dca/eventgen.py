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


class CallSchedule:
    def __init__(self, call_intertimes, logger):
        self.call_intertimes = call_intertimes
        self.logger = logger

    def call_intertime(self, t, cell):
        """Return the call intertime parameter for a cell at time t"""
        pass


class UniformCallSchedule(CallSchedule):
    """Constant and identical inter-arrival time for every cell"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call_intertime(self, t, cell):
        return self.call_intertimes


class NonUniformCallSchedule(CallSchedule):
    """Constant inter-arrival time which may differ between cells"""

    def __init__(self, *args, **kwargs):
        """
        :param call_intertimes: Array of same size as grid, where each element contains
            the call_rate for the corresponding times
        """
        super().__init__(*args, **kwargs)

    def call_intertime(self, t, cell):
        return self.call_intertimes[cell]


class LinearCallSchedule(CallSchedule):
    """
    Create a piece-wise linear call schedule which is uniform over all cells
    """

    def __init__(self, hours, *args, **kwargs):
        """For example, given call_intertimes = [0, 4, 2] and hours = [2, 3],
        the call rate increases linearly from 0 to 4 calls per minute over the period
        t0 = 0 hours to t1 = 2 hours. The call rate then decrease linearly from 4 to 2 calls
        per minute over the period t1 = 2h to t2 = 3h.
        If the last given 'hours' is before simulation end, the call rate stays constant.

        :param call_intertimes: List of call rates
        :param hours: A strictly increasing list of hours.
            Must be length of 1 less than 'call_intertimes'
        """
        super().__init__(*args, **kwargs)
        self.hours = [0] + hours
        self.slopes = []
        for i in range(len(self.call_intertimes) - 1):
            dt_h = self.hours[i + 1] - self.hours[i]
            slope = (self.call_intertimes[i + 1] - self.call_intertimes[i]) / dt_h
            self.slopes.append(slope)
        # Constant after last given hour
        self.slopes.append(0)
        self.hours.append(0)
        self.call_intertimes.append(self.call_intertimes[-1])

        self.left_break = 0
        self.right_break = 1

        it1 = self.call_intertimes[0]
        it2 = self.call_intertimes[1]
        h = self.hours[1] - self.hours[0]
        self.logger.error(
            f"Changing call rate from {1/it1:.2f} to {1/it2:.2f} over {h} hours")

    def call_intertime(self, t, cell):
        t_hours = t / 60
        if t_hours > self.hours[self.right_break]:
            if self.left_break < len(self.call_intertimes) - 3:
                self.left_break += 1
                self.right_break += 1
                it1 = self.call_intertimes[self.left_break]
                it2 = self.call_intertimes[self.right_break]
                h = self.hours[self.right_break] - self.hours[self.left_break]
                self.logger.error(
                    f"{t_hours}h: Changing it from {1/it1:2f} to {1/it2:.2f} over {h} hours"
                )
            if self.left_break == len(self.call_intertimes) - 3:
                self.left_break += 1
                self.right_break += 1
                it = self.call_intertimes[self.right_break]
                self.logger.error(f"{t_hours}h: Setting constant call rate: {1/it}")
            # else:
            #     cr = self.call_intertimes[self.right_break]
            #     # self.logger.error(f"{t_hours}h: Setting constant call rate: {cr}")
            #     return cr
        dt_hours = t_hours - self.hours[self.left_break]
        return self.call_intertimes[self.left_break] \
            + dt_hours * self.slopes[self.left_break]


class EventGen:
    """
    Generates event tuples of the format (time, type, cell) for NEW/HOFF events
    and (time, type, cell, ch) for END events.
    """

    def __init__(self, rows, cols, traffic_preset, call_rate, call_duration,
                 hoff_call_duration, logger, *args, **kwargs):
        self.rows, self.cols = rows, cols
        # Avg. time between arriving calls
        self.call_duration = call_duration
        self.handoff_call_duration = hoff_call_duration
        self.logger = logger
        self.intertime_sched = self.presets(traffic_preset, call_rate, logger)

        # min-heap of event timestamps
        self.event_times = []
        # mapping from timestamps to events
        self.events = {}
        # mapping from (cell_row, cell_col, ch) to end event timestamps
        self.end_event_times = {}

    @staticmethod
    def presets(traffic_preset, call_rate, logger):
        intertime = 1 / call_rate
        if traffic_preset == 'uniform':
            # Use uniform call rates throughout grid
            intertime_sched = UniformCallSchedule(intertime, logger)
        elif traffic_preset == 'nonuniform':
            # Use constant non-uniform call rates
            raise NotImplementedError
            intertime_sched = NonUniformCallSchedule(intertime, logger)
        elif traffic_preset == 'linear24':
            # Use piecewise linear, uniform call rates
            hours = [24]
            call_intertimes = [intertime * 2, intertime]
            intertime_sched = LinearCallSchedule(hours, call_intertimes, logger)
        else:
            raise NotImplementedError(traffic_preset)
        return intertime_sched

    def event_new(self, t, cell, dt=None):
        """
        Generate a new call event at the given cell at an
        exponentially distributed time 'dt' from 't'.
        """
        if dt is None:
            dt = np.random.exponential(self.intertime_sched.call_intertime(t, cell))
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
