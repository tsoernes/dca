# How to run: 'p3 -m div.tf-debug-qnet' from dca directory
import logging
import numpy as np
from nets.qnet import QNet
from replaybuffer import ReplayBuffer
from params import get_pparams

dim = 7, 7, 70
rows, cols, n_chs = dim
grid1 = np.random.choice([0, 1], size=dim)
grid2 = np.random.choice([0, 1], size=dim)
cell1 = (3, 4)
cell2 = (5, 2)
action1 = 30
action2 = 40
reward1 = 300
reward2 = 303
next_grid1 = np.random.choice([0, 1], size=dim)
next_grid2 = np.random.choice([0, 1], size=dim)
next_cell1 = (1, 4)
next_cell2 = (6, 2)

rbuf = ReplayBuffer(2, *dim)
pp, _ = get_pparams()
logger = logging.getLogger('')
qnet = QNet(True, pp, logger)

rbuf.add(grid1, cell1, action1, reward1, next_grid1, next_cell1)
rbuf.add(grid2, cell2, action2, reward2, next_grid2, next_cell2)

qnet.backward(grid1, cell1, [action1], [reward1], next_grid1, next_cell1)
sample = rbuf.sample(2, False)
# print(sample)
qnet.backward(*sample)
