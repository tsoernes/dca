import sys

import numpy as np


def beta(beta, avg_dt=0.0034):
    """for a given beta, Return bootstrap/gamma discount,
    reward discount
    and ratio of bootstrap discount to reward_discount"""
    gamma_disc = np.exp(-beta * avg_dt)
    reward_disc = (1 - gamma_disc) / beta
    ratio = gamma_disc / (reward_disc + gamma_disc)
    return gamma_disc, reward_disc, 2 * reward_disc, ratio


"""
For erlangs 10
Without handoffs, avg_dt converges to 0.0034
With, 0.294

For avg_dt=0.0034
For gamma=0.9 (ratio=1/1.9=0.526)
beta 2237 yields same ratio (higher beta => lower ratio)

gamma=1.0 (ratio=1/2)
2273

avg_dt=0.0030
gamma=0.9 (ratio=1/1.9=0.526)
2585

gamma=1.0 (ratio=1/2)
2624
"""


def run(inp):
    res = beta(*map(float, inp))
    print(tuple(zip(("Gamma-disc", "Reward-disc", "x2", "G-to-R+G ratio"), res)))


if __name__ == "__main__":
    run(sys.argv[1:])
