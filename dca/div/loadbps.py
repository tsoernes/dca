import glob
import pickle
import sys
from os import getcwd, listdir
from os.path import isfile, join

import numpy as np


def ensure_pkl(fname):
    return fname.replace('.pkl', '') + '.pkl'


ctypes = ["new", "hoff", "tot"]
fnames = None
if getcwd()[-3:] == 'div':
    fdir = "../bps"
else:
    fdir = "bps"


def ensure_dir(fname):
    if fname.startswith(fdir):
        return fname
    return join(fdir, fname)


if len(sys.argv) > 1:
    try:
        # Print all files with given extension, e.g. 0 for "vnet.0.pkl"
        lin = sys.argv[-1]
        int(lin)
        ext = lin
    except ValueError:
        # Print all given file (full) names, not including folder
        fnames = []
        inp = sys.argv[1:]
        for fname in inp:
            if '*' in fname:
                gres = glob.glob(join(fdir, fname))
                if not gres:
                    gres = glob.glob(fname)
                fnames.extend(gres)
            else:
                fname = ensure_pkl(fname)
                if isfile(join(fdir, fname)):
                    fnames.append(fname)
                elif isfile(fname):
                    fnames.append(fname)
                else:
                    print(f"Did not find: {fname}")
        ext = None
else:
    ext = None

if not fnames:
    fnames = [f for f in listdir(fdir) if isfile(join(fdir, f))]
fnames = sorted(fnames)
# print(fnames)
for i, fname in enumerate(fnames):
    if ext is None or fname[-5] == ext:
        with open(ensure_dir(fname), "rb") as f:
            bps = pickle.load(f)
            cum_bps = [f"{np.mean(bps[ct], axis=1)[-1]:.5f}" for ct in ctypes]
            stds = [f"{np.std(bps[ct], axis=1)[-1]:.5f}" for ct in ctypes]
            print(
                f" {fname}: {bps['new'].shape[0]} runs, {bps['log_iter']} logiter, shape {bps['new'].shape}"
                f" \n mean:{cum_bps} std:{stds}")
            if i < len(fnames) - 1 and fname[:-5] != fnames[i + 1][:-5]:
                print("\n")
