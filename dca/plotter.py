import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

ctypes = ['New call', 'Hand-off', 'Total']
ctypes_short = ['new', 'hoff', 'tot']
ctypes_map = dict(zip(ctypes_short, ctypes))


def plot_ctypes(all_block_probs_cums, log_iter, n_events):
    """
    Plot for each call type, for each log_iter, cumulative block prob

    all_block_probs_cums: For each call type (new, hoff, tot), for each run,
    cumulative block prob for each log iter [[run1_log1, run1_log2, ..],
    [run2_log1, run2_log2], ..]

    """

    plt.plot()
    x = np.arange(log_iter, n_events + 1, log_iter)
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        # print(block_probs_cums, x)
        y = np.mean(block_probs_cums, axis=0)
        std_devs = np.std(block_probs_cums, axis=0)
        plt.errorbar(x, y, yerr=std_devs, fmt='-o', label=ctypes[i])
    plt.legend(loc='lower right')
    # plt.title('Cumulative call blocking probability')
    plt.ylabel("Cumulative call blocking probability")
    plt.xlabel("Call events")
    plt.show()


def plot_strats(data, labels=None, ctype='new'):
    """Plot for each strat, for a specific call type ctype, cumulative
    block prob for each log iter.

    data: For each strat, a dict with:
    - datetime,
    - log_iter,
    - n_events,
    - for one or more call types, for each log iter, cumulative block prob
    """
    log_iter, n_events = data[0]['log_iter'], data[0]['n_events']
    for strat in data:
        assert log_iter == strat['log_iter'], (log_iter, strat['log_iter'])
        assert n_events == strat['n_events'], (n_events, strat['n_events'])
    all_block_probs_cums = (d[ctype] for d in data)
    if labels is None:
        labels = [None] * len(data)

    plt.plot()
    x = np.arange(log_iter, n_events + 1, log_iter)
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        y = np.mean(block_probs_cums, axis=0)
        std_devs = np.std(block_probs_cums, axis=0)
        plt.errorbar(x, y, yerr=std_devs, fmt='-o', label=labels[i])
    plt.legend(loc='lower right')
    plt.ylabel(f"{ctypes_map[ctype]} cumulative call blocking probability")
    plt.xlabel("Call events")
    plt.show()


def runner():
    parser = argparse.ArgumentParser(
        description='DCA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'fnames',
        type=str,
        nargs='+',
        help="File name(s) of cum block prob pickle file(s)",
        default=None)
    parser.add_argument(
        '--labels',
        type=str,
        kargs='?',
        help="Optional labels for corresponding pickle files",
        default=None)
    parser.add_argument(
        '--ctype', type=str, choices=ctypes_short, help="Call type to plot", default=None)

    args = vars(parser.parse_args())
    data = []
    for fname in args['fnames']:
        with open(fname + '.pkl', "rb") as f:
            data.append(pickle.load(f))
    labels = args['labels']
    if labels is not None:
        assert len(data) == len(labels), (len(data), len(labels))
    if len(data) == 1:
        all_block_probs_cums = (data[0][ctype] for ctype in ctypes_short)
        plot_ctypes(all_block_probs_cums, data[0]['log_iter'], data[0]['n_events'])
    else:
        plot_strats(data)


if __name__ == '__main__':
    runner()
