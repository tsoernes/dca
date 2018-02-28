import numpy as np


def beta_r(beta):
    xp = np.exp(-beta * 0.003)
    rat = xp / (((1 - xp) / beta) + xp)
    return rat
