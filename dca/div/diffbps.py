import glob
import pickle
import sys
from os import getcwd, listdir
from os.path import isfile, join

import datadiff
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
fnames_bps = []
for fname in fnames:
    if ext is None or fname[-5] == ext:
        with open(ensure_dir(fname), "rb") as f:
            bp = pickle.load(f)
            fnames_bps.append((fname, bp))


def diff_filt1(diff_el):
    return diff_el[0] != 'equal'


def diff_filt2(diff_el):
    return len(
        diff_el[1]
    ) > 0 and 'log_file' not in diff_el[1][0] and 'avg_runs' not in diff_el[1][0]


for i, (fname1, bp1) in enumerate(fnames_bps[:-1]):
    fname2, bp2 = fnames_bps[i + 1]
    if fname1[:-5] == fname2[:-5]:
        li1 = bp1['log_iter']
        li2 = bp2['log_iter']
        if li1 != li2:
            print(f"{fname1}, {fname2}: Different log iters: {li1}, {li2}")
        ne1 = bp1['n_events']
        ne2 = bp2['n_events']
        if ne1 != ne2:
            print(f"{fname1}, {fname2}: Different n_events: {ne1}, {ne2}")
        if 'pp' in bp1:
            if 'pp' in bp2:
                ppdiff = datadiff.diff(bp1['pp'], bp2['pp'])
                filt1 = filter(diff_filt1, ppdiff.diffs)
                filt2 = filter(diff_filt2, filt1)
                ppdiff_uneq = list(filt2)
                if ppdiff_uneq:
                    print(f"{fname1}, {fname2}: {ppdiff_uneq}")
                # print(f" {fname}: {bp['log_iter']} logiter")
        else:
            print(f"{fname1} has no pp")
    else:
        print("\n")
