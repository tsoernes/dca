import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

ctypes = ['New call', 'Hand-offs', 'Total']


def plot(all_block_probs_cums, log_iter, n_events):
    """For each call type (new, hoff, tot), for each run, cumulative block prob for each log iter
    [[run1_log1, run1_log2, ..], [run2_log1, run2_log2], ..]"""

    plt.plot()
    x = np.arange(log_iter, n_events + 1, log_iter)
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        # print(block_probs_cums, x)
        y = np.mean(block_probs_cums, axis=0)
        std_devs = np.std(block_probs_cums, axis=0)

        plt.errorbar(x, y, yerr=std_devs, fmt='-o', label=ctypes[i])
    plt.legend(loc='lower right')
    # plt.title('Cumulative call blocking probability')
    plt.ylabel("Call blocking probability")
    plt.xlabel("Call events")
    plt.show()


def plot_multi(data, labels=None, ctype='all'):
    """ Different strats"""
    all_block_probs_cums, log_iter, n_events
    """For each call type (new, hoff, tot), for each run, cumulative block prob for each log iter
    [[run1_log1, run1_log2, ..], [run2_log1, run2_log2], ..]"""

    plt.plot()
    x = np.arange(log_iter, n_events + 1, log_iter)
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        # print(block_probs_cums, x)
        y = np.mean(block_probs_cums, axis=0)
        std_devs = np.std(block_probs_cums, axis=0)

        plt.errorbar(x, y, yerr=std_devs, fmt='-o', label=ctypes[i])
    plt.legend(loc='lower right')
    # plt.title('Cumulative call blocking probability')
    plt.ylabel("Call blocking probability")
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
        '--ctype',
        type=str,
        choices=['all', 'new', 'hoff', 'total'],
        help="Call type to plot",
        default=None)

    args = vars(parser.parse_args())
    # For each strat (file), a dict with:
    # - datetime,
    # - log_iter,
    # - n_events,
    # - for each call type (new call, ..), for each log iter: cumulative block prob
    data = []
    for fname in args['fnames']:
        with open(fname + '.pkl', "rb") as f:
            data.append(pickle.load(f))
    labels = args['labels']
    if labels is not None:
        assert len(data) == len(labels), (len(data), len(labels))
    if len(data) == 1:
        plot(data[0][''])
    else:
        plot_multi(data)


if __name__ == '__main__':
    runner()
