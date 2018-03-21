import numpy as np

from exp_policies import BoltzmannGumbel

c = 5
epol = BoltzmannGumbel(c)
action_counts = np.random.randint(1, 100, 70)
epol.action_counts = action_counts

n = 60
qvals = np.random.randint(1, 100, n)
chs = np.arange(0, n, 1, np.int32)

beta = c / np.sqrt(action_counts[chs])
ps = [epol.action_prob(beta, qvals, ch) for ch in chs]

empirical_ps = np.zeros(n)
nsims = 10000
for _ in range(nsims):
    ch, idx, _ = epol.select_action(None, chs, qvals, None)
    empirical_ps[ch] += 1

empirical_ps /= nsims
print(ps, empirical_ps)
