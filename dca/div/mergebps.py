import pickle
import sys
from os import getcwd, listdir
from os.path import isfile, join

import numpy as np

next_ext_ = 5
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


def merge_uneven(bp1, bp2):
    """"Merge uneven log_iter"""
    bl1, bl2 = bp1['log_iter'], bp2['log_iter']
    assert bl1 != bl2, ("Same log iter: ", bl1, bl2)
    assert bp1['n_events'] == bp2['n_events'], (bp1['n_events'], bp2['n_events'])
    if bl1 < bl2:
        min_li, max_li = bl1, bl2
        min_bp, max_bp = bp1, bp2
    else:
        min_li, max_li = bl2, bl1
        min_bp, max_bp = bp2, bp1
    assert max_li % min_li == 0, (min_li, max_li)
    period = max_li // min_li
    # print(f"Period: {period}, min/max li: {min_li}, {max_li}")
    for k, v in max_bp.items():
        if k in ctypes:
            # Skip some log steps
            periodic_bp = min_bp[k][:, period - 1::period]
            # print(v.shape, min_bp[k].shape, periodic_bp.shape)
            max_bp[k] = np.vstack((v, periodic_bp))
    return max_bp


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
            try:
                for k, v in bp_dict.items():
                    if k in ctypes:
                        bps_dict[k] = np.vstack((bps_dict[k], v))
                    elif k != 'log_iter':
                        bps_dict[k] = v
            except ValueError:
                # print(fname, bps_dict[k].shape, v.shape)
                bps_dict = merge_uneven(bps_dict, bp_dict)
    next_fname = f"{group[0][:-6]}.{next_ext}.pkl"
    print(f"Merged {len(group)} {group} -> {next_fname}"
          f"\n{shapes} -> {bps_dict['new'].shape}")
    next_fname = join(fdir, next_fname)
    if not dry:
        assert not isfile(next_fname), f"File name {next_fname} already exists!"
        with open(next_fname, "wb") as f:
            pickle.dump(bps_dict, f)
