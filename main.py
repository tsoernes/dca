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


def neighbors1(row, col):
    """
    Returns a list with indexes of neighbors within a radius of 1,
    not including self
    """
    idxs = []
    r_low = max(0, row-1)
    r_hi = min(n, row+1)
    c_low = max(0, col-1)
    c_hi = min(m, col+1)
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


def event_new(t, call_rate):
    """
    Generate a new call at random time and cell
    """
    a_time = np.random.exponential(call_rate) + t
    a_cell_r = np.random.randint(0, n)
    a_cell_c = np.random.randint(0, m)
    return (a_time, CEvent.NEW, (a_cell_r, a_cell_c))


def event_end(t, service_rate, cell):
    e_time = np.random.exponential(service_rate) + t
    return (e_time, CEvent.END, cell)


def event_handoff(t, l_cell):
    """
    Hand off a call from leaving cell to a neighboring cell
    """
    neighs = neighbors1(*l_cell)
    a_cell = neighs[np.random.randint(0, len(neighs))]
    return (t, CEvent.END, a_cell)


class CEvent(Enum):
    NEW = auto()  # Incoming call
    END = auto()  # Ending call
    HANDOFF = auto()  # Transfer call from one cell to another


call_rate = 100  # Average incoming calls per minute
service_rate = 0
t = 0  # Current time, in minutes
cevents = []
# Discrete event simulation
for _ in range(n_episodes):
    event = heappop(cevents)
    t = event[0]
    # Accept incoming call
    if event[1] == CEvent.NEW:
        # Get call duration for incoming call
        heappush(cevents, event_end(t, service_rate, event[2]))
        # Generate next incoming call
        heappush(cevents, event_new(t, call_rate))
        # Add incoming call to current state
    elif event[1] == CEvent.END:
        # Remove call from current state
        pass
    elif event[1] == CEvent.HANDOFF:
        # Move call from one cell to another
        # TODO: When are these events supposed to be generated?
        pass

