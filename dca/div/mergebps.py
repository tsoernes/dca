import pickle
import sys
from os import getcwd, listdir
from os.path import isfile, join

import numpy as np

next_ext_ = 4
if len(sys.argv) > 1 and sys.argv[-1] == 'no_dry':
    print("RUNNING WET!")
    dry = False
else:
    print("Running dry")
    dry = True

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
added = np.zeros_like(fnames, int)
groups = []
# print(fnames, "\n")
i = 0
while i < len(fnames) - 1:
    group = []
    for j in range(i, len(fnames)):
        if fnames[i][:-5] == fnames[j][:-5]:
            group.append(fnames[j])
            added[j] += 1
        else:
            break
    groups.append(group)
    if i == len(fnames) - 1:
        i = j + 1
    else:
        i = j
incorrect_added = np.where(added != 1)
assert incorrect_added[0].shape == (0, ), (incorrect_added,
                                           np.array(fnames)[incorrect_added])

# for group in filter(lambda g: len(g) > 1, groups):
for group in groups:
    next_ext = int(group[-1][-5]) + 1 if next_ext_ is None else next_ext_
    shapes = []
    with open(join(fdir, group[0]), "rb") as f:
        bps_dict = pickle.load(f)
        shapes.append(bps_dict['new'].shape)
    for fname in group[1:]:
        with open(join(fdir, fname), "rb") as f:
            bp_dict = pickle.load(f)
            shapes.append(bp_dict['new'].shape)
            for k, v in bp_dict.items():
                if k in ctypes:
                    bps_dict[k] = np.vstack((bps_dict[k], v))
                else:
                    bps_dict[k] = v
    next_fname = f"{group[0][:-6]}.{next_ext}.pkl"
    print(f"Merged {len(group)} {group} -> {next_fname}"
          f"\n{shapes} -> {bps_dict['new'].shape}")
    next_fname = join(fdir, next_fname)
    if not dry:
        assert not isfile(next_fname), f"File name {next_fname} already exists!"
        with open(next_fname, "wb") as f:
            pickle.dump(bps_dict, f)
