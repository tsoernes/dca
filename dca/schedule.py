import numpy as np


class Schedule:
    def __init__(self, initial_val):
        self.inital_val = initial_val
        self.val = initial_val

    def get_val(self):
        raise NotImplementedError


class ExpDecaySchedule(Schedule):
    def __init__(self, initial_val, factor):
        super().__init__(initial_val)
        self.factor = factor

    def get_val(self):
        self.val *= self.factor
        return self.val


class InvRootTimeSchedule(Schedule):
    def __init__(self, initial_val, factor=256):
        super().__init__(initial_val)
        self.factor = factor

    def get_val(self, t):
        val = self.inital_val / np.sqrt(t / self.factor)
        return val
