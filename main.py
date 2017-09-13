from enum import Enum, auto
from heapq import heappush, heappop
import numpy as np
# Implement RL with TD(0) and table lookup

n = 7  # Rows
m = 9  # Columns
n_channels = 2
# Indexed as: state[row][col][channel]
state = np.zeros((n, m, n_channels), dtype=bool)

value = np.zeros((n, m, n_channels+1))

def partition_cells(n, m):
    """
    Partition cells into 7 lots such that the minimum distance
    between cells with the same label ([0..6]) is at least 2
    (which corresponds to a minimum reuse distance of 3).
    """
    labels = np.zeros((n, m), dtype=int)

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
        if (x >= 0 and x < m and y >= 0 and y < n):
            labels[y][x] = l

    # Center of a 'circular' 7-cell subgrid where each cell has a unique label
    center = (0, 0)
    # First center in current row which has neighbors inside grid
    first_row_center = (0, 0)
    # Move center down-left until subgrid goes out of bounds
    while (center[0] >= -1) and (center[1] <= n):
        # Move center right-up until subgrid goes out of bounds
        while (center[0] <= m) and (center[1] >= -1):
            # Label cells 0..6 with given center as 0
            print(f"Labeled with center r{center[1]},c{center[0]}")
            label(0, *center)
            # TODO the order of neighbors should be the same regardless of
            # wheter center is in odd/even col or at boundary edge etc. Is it?
            for i, neigh in enumerate(
                    neighbors1(center[1], center[0], n, m, sparse=True)):
                label(i+1, neigh[1], neigh[0])
            center = right_up(*center)
        center = down_left(*first_row_center)
        # Move right until x >= -1
        while center[0] < -1:
            center = right_up(*center)
        first_row_center = center
    return labels


def neighbors1(row, col, rows=n, cols=m, sparse=False, fill=[0, 0]):
    """
    Returns a list with indexes of neighbors within a radius of 1,
    not including self
    :param int rows: Number of rows in grid
    :param int cols: Number of cols in grid
    :param bool sparse: If true, then always returns a 6-element array
    where neighbors outside bounds are given position @fill@
    """
    idxs = np.array([fill]*6) if sparse else []
    r_low = max(0, row-1)
    r_hi = min(rows, row+1)
    c_low = max(0, col-1)
    c_hi = min(cols, col+1)
    if col % 2 == 0:
        cross = row-1
    else:
        cross = row+1
    i = 0
    for r in range(r_low, r_hi+1):
        for c in range(c_low, c_hi+1):
            if not ((r, c) == (cross, col-1) or
                    (r, c) == (cross, col+1) or
                    (r, c) == (row, col)):
                if sparse:
                    idxs[i] = [r, c]
                else:
                    idxs.append((r, c))
                i += 1
    return idxs


def neighbors2(row, col):
    """
    Returns a list with indexes of neighbors within a radius of 2,
    not including self
    """
    idxs = []
    r_low = max(0, row-2)
    r_hi = min(n, row+2)
    c_low = max(0, col-2)
    c_hi = min(m, col+2)
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


def state_frepr(state):
    """
    Feature representation of a state
    """
    frepr = np.zeros((n, m, n_channels+1))
    # Number of available channels for each cell
    frepr[:, :, -1] = n_channels - np.sum(state, axis=2)
    # The number of times each channel is used within a 1 cell radius,
    # not including self
    for i in range(n):
        for j in range(m):
            for ch in range(n_channels):
                neighs = neighbors2(i, j)
                for neigh in neighs:
                    frepr[i][j][ch] += state[neigh[0]][neigh[1]][ch]
    return frepr


def reward(state, action, dt):
    """
    Immediate reward
    dt: Time until next event
    """
    pass


def discount(dt):
    """
    Discount factor (gamma)
    """
    pass


def value(avail, pack):
    """
    avail: The number of available channels per cell.
    pack: Packing feature.
            The number of times each channel is used within a 1 cell radius.
    """
    pass


lr = 0.8  # Learning rate
y = 0.95  # Gamma (discount factor)
n_episodes = 100
rewards = np.zeros((n_episodes))


def event_new(t):
    """
    Generate a new call at random time and cell
    """
    cell_r = np.random.randint(0, n)  # Cell row for new call
    cell_c = np.random.randint(0, m)
    e_time = np.random.exponential(call_rates[cell_r][cell_c]) + t
    return (e_time, CEvent.NEW, (cell_r, cell_c))


def event_end(t, cell, ch):
    """
    Generate end duration for a call
    """
    e_time = np.random.exponential(call_duration) + t
    return (e_time, CEvent.END, cell, ch)


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # End a current call


call_rates = np.zeros((n, m))  # Mean incoming calls per hour for each cell
call_duration = 3/60  # Mean call duration, in hours
t = 0  # Current time, in minutes
cevents = [event_new(t)]
# Discrete event simulation
for _ in range(n_episodes):
    event = heappop(cevents)
    t = event[0]
    # Accept incoming call
    if event[1] == CEvent.NEW:
        # Assign channel to call
        ch = -1
        # Get call duration for incoming call
        heappush(cevents, event_end(t, event[2], ch))
        # Generate next incoming call
        heappush(cevents, event_new(t))
        # Add incoming call to current state
        state[event[2][0]][event[2][1]][ch] = 0
    elif event[1] == CEvent.END:
        # Remove call from current state
        state[event[2][0]][event[2][1]][event[3]] = 0

"""
# Fixed assignment (FA) channel allocation;
# the set of channels is partitioned, and the partitions are permanently
assigned
# to cells so that all cells can use all the channels assigned to them
simultaneously
# without interference (see Figure 1a). When a call arrives in a cell, if any
pre-
# assigned channel is unused; it is assigned, else the call is blocked.
No rearrangement
# is done when a call terminates. Such a policy is static and cannot take
advantage of
# temporary stochastic variations in demand for service. More ecient are
dynamic
# channel allocation policies, which assign channels to di erent cells, so that
every
# channel is available to every cell on a need basis, unless the channel is
used in a
# nearby cell and the reuse constraint is violated.
"""

"""
# Borrowing with Directional Channel Locking (BDCL) of Zhang & Yum (1989). It
# numbers the channels from 1 to N , partitions and assigns them to cells as in
FA.
# The channels assigned to a cell are its nominal channels. If a nominal
channel
# is available when a call arrives in a cell, the smallest numbered such
channel is
# assigned to the call. If no nominal channel is available, then the largest
numbered
# free channel is borrowed from the neighbour with the most free channels.
When a
# channel is borrowed, careful accounting of the directional e ect of which
cells can
# no longer use that channel because of interference is done. The call is
blocked if
# there are no free channels at all. When a call terminates in a cell and the
channel
# so freed is a nominal channel, say numbered i, of that cell, then if there
is a call
# in that cell on a borrowed channel, the call on the smallest numbered
borrowed
# channel is reassigned to i and the borrowed channel is returned to the
appropriate
# cell. If there is no call on a borrowed channel, then if there is a call on
a nominal
# channel numbered larger than i, the call on the highest numbered nominal
channel
# is reassigned to i. If the call just terminated was itself on a borrowed
channel, the
# call on the smallest numbered borrowed channel is reassigned to it and that
channel
# is returned to the cell from which it was borrowed. Notice that when a
borrowed
# channel is returned to its original cell, a nominal channel becomes free in
that cell
# and triggers a reassignment. Thus, in the worst case a call termination in
one cell
# can sequentially cause reassignments in arbitrarily far away cells | making
BDCL
"""
