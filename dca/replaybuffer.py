import operator
import random

import numpy as np

from datahandler import h5py_save_append


class ReplayBuffer():
    def __init__(self, size, rows, cols, n_channels):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.rows = rows
        self.cols = cols
        self.n_channels = n_channels

        self._storage = {
            'grids': [],
            'freps': [],
            'cells': [],
            'chs': [],
            'rewards': [],
            'values': [],
            'next_grids': [],
            'next_freps': [],
            'next_cells': []
        }
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage['rewards'])

    def add(self,
            grid=None,
            frep=None,
            cell=None,
            ch=None,
            reward=None,
            value=None,
            next_grid=None,
            next_frep=None,
            next_cell=None):
        lenn = len(self)

        def _add(name, item):
            if self._next_idx >= lenn:
                self._storage[name].append(item)
            else:
                self._storage[name][self._next_idx] = item

        if grid is not None:
            _add('grids', grid)
        if frep is not None:
            _add('freps', frep)
        if cell is not None:
            _add('cells', cell)
        if ch is not None:
            _add('chs', ch)
        if reward is not None:
            _add('rewards', reward)
        if value is not None:
            _add('values', value)
        if next_grid is not None:
            _add('next_grids', next_grid)
        if next_frep is not None:
            _add('next_freps', next_frep)
        if next_cell is not None:
            _add('next_cells', next_cell)

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        n_samples = len(idxes)
        include_g = len(self._storage['grids']) > 0
        include_f = len(self._storage['freps']) > 0
        include_ch = len(self._storage['chs']) > 0
        include_ce = len(self._storage['cells']) > 0
        include_re = len(self._storage['rewards']) > 0
        include_nf = len(self._storage['next_freps']) > 0
        include_val = len(self._storage['values']) > 0
        include_ng = len(self._storage['next_grids']) > 0
        include_nc = len(self._storage['next_cells']) > 0
        data = {}
        if include_g:
            data['grids'] = np.zeros(
                (n_samples, self.rows, self.cols, self.n_channels), dtype=np.bool)
        if include_f:
            data['freps'] = np.zeros(
                (n_samples, self.rows, self.cols, self.n_channels + 1), dtype=np.bool)
        if include_ce:
            data['cells'] = []
        if include_ch:
            data['chs'] = np.zeros(n_samples, dtype=np.int32),
        if include_re:
            data['rewards'] = np.zeros(n_samples, dtype=np.float32)
        if include_val:
            data['values'] = np.zeros(n_samples, dtype=np.float32)
        if include_ng:
            data['next_grids'] = np.zeros(
                (n_samples, self.rows, self.cols, self.n_channels), dtype=np.int8)
        if include_nf:
            data['next_freps'] = np.zeros(
                (n_samples, self.rows, self.cols, self.n_channels + 1), dtype=np.bool)
        if include_nc:
            data['next_cells'] = []
        for i, j in enumerate(idxes):
            if include_g:
                data['grids'][i][:] = self._storage['grids'][j]
            if include_f:
                data['freps'][i][:] = self._storage['freps'][j]
            if include_ce:
                data['cells'].append(self._storage['cells'][j])
            if include_ch:
                data['chs'][i] = self._storage['chs'][j]
            if include_re:
                data['rewards'][i] = self._storage['rewards'][j]
            if include_nf:
                data['next_freps'][i][:] = self._storage['next_freps'][j]
            if include_val:
                data['values'][i] = self._storage['values'][j]
            if include_ng:
                data['next_grids'][i][:] = self._storage['next_grids'][j]
            if include_nc:
                data['next_cells'].append(self._storage['next_cells'][j])
        return data

    def pop(self, batch_size):
        """Return freshest examples, in order

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        NOTE TODO should delete the experiences that are popped? for AC
        """
        idxs = range(len(self) - batch_size - 1, len(self) - 1)
        return self._encode_sample(idxs)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        """
        idxs = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxs)

    def save_experience_to_disk(self):
        idxs = np.arange((len(self)))
        np.random.shuffle(idxs)
        data = self._encode_sample(idxs)
        h5py_save_append("data-experience", **data)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, rows, cols, n_channels, alpha=0.6):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0: no prioritization, 1: full prioritization)

        --------
        """
        super().__init__(size, rows, cols, n_channels)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def pop(self):
        # Not sure if this works correctly
        raise NotImplementedError

    def add(self, *args, **kwargs):
        """Experiences are added with maximum priority,
        ensuring they are trained on at least once"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def add_with_pri(self, priority, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self.update_priorities([idx], priority)

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        ...
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage))**(-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self))**(-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        # return tuple(list(encoded_sample) + [weights, idxes])
        return (encoded_sample, weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
            capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end))

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx],
                                               self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
