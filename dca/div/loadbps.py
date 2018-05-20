import pickle
from os import getcwd, listdir
from os.path import isfile, join

import numpy as np

ext = ".0"
if getcwd()[-3:] == 'div':
    fdir = "../bps"
else:
    fdir = "bps"
ctypes = ["new", "hoff", "tot"]
# fnames = {
#     'grads': (['resid', 'semi', 'tdc-gam', 'tdc']),
#     'final': (['fca', 'rand'], ['hla-rssarsa', 'hla-vnet']),
#     'final-nohoff': ['fca', 'rand', 'rssarsa', 'vnet'],
#     'hla': ['rssarsa', 'vnet']
# }

fnames = [f for f in listdir(fdir) if isfile(join(fdir, f))]
fnames = sorted(fnames)
print(fnames)
for fname in fnames:
    with open(join(fdir, fname), "rb") as f:
        bps = pickle.load(f)
        cum_bps = [f"{np.mean(bps[ct], axis=1)[-1]:.5f}" for ct in ctypes]
        stds = [f"{np.std(bps[ct], axis=1)[-1]:.5f}" for ct in ctypes]
        print(f"\n {fname} \n {cum_bps} {stds}")
