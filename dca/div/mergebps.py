import pickle
from os import getcwd, listdir
from os.path import isfile, join

import numpy as np

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
groups = []
print(fnames, "\n")
i = 0
while i < len(fnames):
    group = []
    for j in range(i, len(fnames)):
        if fnames[i][:-5] == fnames[j][:-5]:
            group.append(fnames[j])
        else:
            break
    groups.append(group)
    if i == len(fnames) - 1:
        i = j + 1
    else:
        i = j

# for group in filter(lambda g: len(g) > 1, groups):
for group in groups:
    next_ext = int(group[-1][-5]) + 1
    with open(join(fdir, group[0]), "rb") as f:
        bps_dict = pickle.load(f)
    for fname in group[1:]:
        with open(join(fdir, fname), "rb") as f:
            bp_dict = pickle.load(f)
            for k, v in bp_dict.items():
                if k in ctypes:
                    bps_dict[k] = np.vstack((bps_dict[k], v))
                else:
                    bps_dict[k] = v
    next_fname = f"{group[0][:-6]}.{next_ext}.pkl"
    print(f"Merged {len(group)} {group} into {next_fname}")
    print(bps_dict['new'].shape)
    if len(group) > 1:
        print(bp_dict['new'].shape)
    if not dry:
        with open(next_fname, "rb") as f:
            bps_dict.append(pickle.dump(bps_dict, f))
