from enum import Enum, auto
from heapq import heappush, heappop
import numpy as np
import math
# Implement RL with TD(0) and table lookup


lr = 0.8  # Learning rate
y = 0.95  # Gamma (discount factor)


class ProblemParams:
    def __init__(self, rows, cols, n_channels, call_rates, call_duration,
                 n_episodes):
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels
        self.call_rates = call_rates
        self.call_duration = call_duration
        self.n_episodes = n_episodes


params = ProblemParams(
        rows=7,
        cols=7,
        n_channels=49,
        call_rates=0,
        call_duration=3/60,
        n_episodes=100)


class State:
    def __init__(self, rows, cols, n_channels, n_episodes):
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels
        self.n_episodes = n_episodes
        self.state = np.zeros((rows, cols, n_channels), dtype=bool)

    def fn_new(self, row, col):
        """
        Assign incoming call in cell in row @row@ column @col@ to a channel.
        Possibly rearrange existing calls.
        """
        raise NotImplementedError()

    def fn_end(self):
        raise NotImplementedError()

    def simulate(self):
        t = 0  # Current time, in minutes
        cevents = [self.event_new(t)]
        # Discrete event simulation
        for _ in range(self.n_episodes):
            event = heappop(cevents)
            t = event[0]
            # Accept incoming call
            if event[1] == CEvent.NEW:
                # Assign channel to call
                ch = self.fn_new()
                # TODO: Handle if there's no available ch
                if ch == -1:
                    pass
                else:
                    # Generate call duration for incoming call and add event
                    heappush(cevents, self.event_end(t, event[2], ch))
                    # Generate next incoming call
                    heappush(cevents, self.event_new(t))
                    # Add incoming call to current state
                    self.state[event[2][0]][event[2][1]][ch] = 0
            elif event[1] == CEvent.END:
                self.fn_end()
                # Remove call from current state
                self.state[event[2][0]][event[2][1]][event[3]] = 0

    def partition_cells(self):
        """
        Partition cells into 7 lots such that the minimum distance
        between cells with the same label ([0..6]) is at least 2
        (which corresponds to a minimum reuse distance of 3).
        Returns an n by m array with the label for each cell.
        """
        labels = np.zeros((self.rows, self.cols), dtype=int)

        def right_up(x, y):
            x_new = x + 3
            y_new = y
            if x % 2 != 0:
                # Odd column
                y_new = y - 1
            return (x_new, y_new)

        def down_left(x, y):
            x_new = x - 1
            if x % 2 == 0:
                # Even column
                y_new = y + 3
            else:
                # Odd Column
                y_new = y + 2
            return (x_new, y_new)

        def label(l, x, y):
            # A center and some part of its subgrid may be out of bounds.
            if (x >= 0 and x < self.cols
                    and y >= 0 and y < self.rows):
                labels[y][x] = l

        # Center of a 'circular' 7-cell subgrid where
        # each cell has a unique label
        center = (0, 0)
        # First center in current row which has neighbors inside grid
        first_row_center = (0, 0)
        # Move center down-left until subgrid goes out of bounds
        while (center[0] >= -1) and (center[1] <= self.rows):
            # Move center right-up until subgrid goes out of bounds
            while (center[0] <= self.cols) and (center[1] >= -1):
                # Label cells 0..6 with given center as 0
                label(0, *center)
                for i, neigh in enumerate(
                        self.neighbors1sparse(center[1], center[0])):
                    label(i+1, neigh[1], neigh[0])
                center = right_up(*center)
            center = down_left(*first_row_center)
            # Move right until x >= -1
            while center[0] < -1:
                center = right_up(*center)
            first_row_center = center
        return labels

    def assign_chs(self):
        # Number the channels from 1 to N , partition and assign them to cells
        partitions = self.partition_cells()
        channels_per_subgrid_cell = []
        channels_per_subgrid_cell_accu = [0]
        channels_per_cell = self.n_channels/7
        ceil = math.ceil(channels_per_cell)
        floor = math.floor(channels_per_cell)
        tot = 0
        for i in range(7):
            if tot + ceil + (6-i) * floor > self.n_channels:
                tot += ceil
                cell_channels = ceil
            else:
                tot += floor
                cell_channels = floor
            channels_per_subgrid_cell.append(cell_channels)
            channels_per_subgrid_cell_accu.append(tot)
        # The channels assigned to a cell are its nominal channels.
        # 1 if nominal, 0 otherwise
        nominal_channels = np.zeros((self.rows, self.cols, self.n_channels))
        for r in self.rows:
            for c in self.cols:
                label = partitions[r][c]
                lo = channels_per_subgrid_cell_accu[label]
                hi = channels_per_subgrid_cell_accu[label+1]
                nominal_channels[r][c][lo:hi] = 1
        return nominal_channels

    @staticmethod
    def neighbors1sparse(row, col):
        """
        Not including self. May not be within grid.
        In clockwise order starting from up-right.
        """
        idxs = []
        if col % 2 == 0:
            idxs.append((row, col+1))
            idxs.append((row+1, col+1))
            idxs.append((row+1, col))
            idxs.append((row+1, col-1))
            idxs.append((row, col-1))
            idxs.append((row-1, col))
        else:
            idxs.append((row-1, col+1))
            idxs.append((row, col+1))
            idxs.append((row+1, col))
            idxs.append((row, col-1))
            idxs.append((row-1, col-1))
            idxs.append((row-1, col))
        return idxs

    def neighbors1(self, row, col):
        """
        Returns a list with indexes of neighbors within a radius of 1,
        not including self
        """
        idxs = []
        r_low = max(0, row-1)
        r_hi = min(self.rows, row+1)
        c_low = max(0, col-1)
        c_hi = min(self.cols, col+1)
        if col % 2 == 0:
            cross = row-1
        else:
            cross = row+1
        for r in range(r_low, r_hi+1):
            for c in range(c_low, c_hi+1):
                if not ((r, c) == (cross, col-1) or
                        (r, c) == (cross, col+1) or
                        (r, c) == (row, col)):
                    idxs.append((r, c))
        return idxs

    def neighbors2(self, row, col):
        """
        Returns a list with indexes of neighbors within a radius of 2,
        not including self
        """
        idxs = []
        r_low = max(0, row-2)
        r_hi = min(self.rows, row+2)
        c_low = max(0, col-2)
        c_hi = min(self.cols, col+2)
        if col % 2 == 0:
            cross1 = row-2
            cross2 = row+2
        else:
            cross1 = row+2
            cross2 = row-2
        for r in range(r_low, r_hi+1):
            for c in range(c_low, c_hi+1):
                if not ((r, c) == (row, col) or
                        (r, c) == (cross1, col-2) or
                        (r, c) == (cross1, col-1) or
                        (r, c) == (cross1, col+1) or
                        (r, c) == (cross1, col+2) or
                        (r, c) == (cross2, col-2) or
                        (r, c) == (cross2, col+2)):
                    idxs.append((r, c))
        return idxs

    def event_new(self, t):
        """
        Generate a new call event at random time and cell
        """
        cell_r = np.random.randint(0, self.rows)  # Cell row for new call
        cell_c = np.random.randint(0, self.cols)
        e_time = np.random.exponential(self.call_rates[cell_r][cell_c]) + t
        return (e_time, CEvent.NEW, (cell_r, cell_c))

    def event_end(self, t, cell, ch):
        """
        Generate end event for a call
        """
        e_time = np.random.exponential(self.call_duration) + t
        return (e_time, CEvent.END, cell, ch)


class FAState(State):

    """
    Fixed assignment (FA) channel allocation;
    the set of channels is partitioned, and the partitions are permanently
    assigned
    to cells so that all cells can use all the channels assigned to them
    simultaneously
    without interference
    """
    def __init__(self):
        self.nominal_channels = self.assign()

    def fn_new(self, row, col):
        # When a call arrives in a cell,
        # if any pre-assigned channel is unused;
        # it is assigned, else the call is blocked.
        ch = -1
        for idx, is_nom_ch in enumerate(self.nominal_channels[row][col]):
            if is_nom_ch and self.state[row][col][idx]:
                ch = idx
        return ch

    def fn_end():
        # No rearrangement is done when a call terminates.
        pass


class FCState(State):
    def state_frepr(self):
        """
        Feature representation of a state
        """
        frepr = np.zeros((self.rows, self.cols, self.n_channels+1))
        # Number of available channels for each cell
        frepr[:, :, -1] = self.n_channels - np.sum(self.state, axis=2)
        # The number of times each channel is used within a 1 cell radius,
        # not including self
        for i in range(self.rows):
            for j in range(self.cols):
                for ch in range(self.n_channels):
                    neighs = self.neighbors2(i, j)
                    for neigh in neighs:
                        frepr[i][j][ch] += self.state[neigh[0]][neigh[1]][ch]
        return frepr


class BCDLState(State):
    # Borrowing with Directional Channel Locking (BDCL) of Zhang & Yum (1989).
    def __init__(self):
        # TODO: Is there any way to call 'super' without having all the args
        # here
        self.nominal_chs = self.assign_chs()

    def fn_new(self, row, col):
        """
        ON NEW CALL:
        """
        ch = -1
        # If a nominal channel is available when a call arrives in a cell,
        # the smallest numbered such channel is assigned to the call.
        for idx, is_nom_ch in enumerate(self.nominal_channels[row][col]):
            if is_nom_ch and self.state[row][col][idx]:
                ch = idx
                break
        if ch != -1:
            return ch

        # If no nominal channel is available, then the largest numbered
        # free channel is borrowed from the neighbour with
        # the most free channels.
        neigh_idxs = self.neighbors1()
        best_neigh = None
        best_n_free = 0
        for n_idx in neigh_idxs:
            n_free = self.n_channels - np.sum(self.state[n_idx])
            if n_free > best_n_free:
                best_n_free = n_free
                best_neigh = n_idx
        # When a channel is borrowed, careful accounting of the directional
        # effect of which cells can no longer use that channel because
        # of interference is done.
        # The call is blocked if there are no free channels at all.

        # Changing state (assigning call to cell and ch) to the
        # incoming call should not be done here, only rearrangement
        # of existing calls
        # TODO: rearrange existing and keep track of borrowed
        return ch

    def fn_end(self):
        """
        ON CALL TERM:
        When a call terminates in a cell and the channel so freed is a nominal
        channel, say numbered i, of that cell, then if there is a call
        in that cell on a borrowed channel, the call on the smallest numbered
        borrowed channel is reassigned to i and the borrowed channel
        is returned to the
        appropriate cell. If there is no call on a borrowed channel,
        then if there is a call on a nominal channel numbered larger than i,
        the call on the highest numbered nominal channel is reassigned to i.
        If the call just terminated was itself on a borrowed channel, the
        call on the smallest numbered borrowed channel is reassigned to it
        and that
        channel is returned to the cell from which it was borrowed.
        Notice that when a
        borrowed channel is returned to its original cell, a nominal
        channel becomes
        free in that cell and triggers a reassignment.
        """
        pass


class RLState(State):
    def __init__(self):
        self.value = np.zeros((self.rows, self.cols, self.n_channels+1))

    def fn_new(self):
        pass

    def fn_end(self):
        pass

    def reward(self, state, action, dt):
        """
        Immediate reward
        dt: Time until next event
        """
        pass

    def discount(self, dt):
        """
        Discount factor (gamma)
        """
        pass

    def value(self, avail, pack):
        """
        avail: The number of available channels per cell.
        pack: Packing feature.
              The number of times each channel is used within a 1 cell radius.
        """
        pass


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # End a current call











