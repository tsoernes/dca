import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import PercentFormatter

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (13, 11),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
plt.rcParams.update(params)
ctypes = ['New call', 'Hand-off', 'Total']
ctypes_short = ['new', 'hoff', 'tot']
ctypes_map = dict(zip(ctypes_short, ctypes))


def plot_bps(all_block_probs_cums, log_iter, n_events, labels=None, ylabel=None,
             title=''):
    """ If labels=None and ylabel=None:
    Plot for each call type, for each log_iter, cumulative block prob

    all_block_probs_cums: For each call type (new, hoff, tot), for each run,
    cumulative block prob for each log iter [[run1_log1, run1_log2, ..],
    [run2_log1, run2_log2], ..]

    """
    if ylabel is None:
        ylabel = "Cumulative call blocking probability"
    if labels is None:
        labels = ctypes
        loc = 'upper right'
    else:
        loc = 'lower right'

    fig, ax = plt.subplots(1, 1)
    plt.plot()
    x = np.arange(log_iter, n_events + 1, log_iter)
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        y = 100 * np.mean(block_probs_cums, axis=0)
        std_devs = np.std(block_probs_cums, axis=0)
        ax.errorbar(x, y, yerr=std_devs, fmt='-o', label=labels[i])
    ax.legend(loc=loc)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Call events")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50000))
    plt.show()


def plot_strats(data, labels=None, ctype='new', title=''):
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
        print(strat['datetime'])
    all_block_probs_cums = (d[ctypes_map[ctype]] for d in data)
    if labels is None:
        labels = [None] * len(data)
    ylabel = f"{ctypes_map[ctype]} cumulative blocking probability"
    plot_bps(all_block_probs_cums, log_iter, n_events, labels, ylabel, title)


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
        nargs='*',
        help="Optional labels for corresponding pickle files",
        default=None)
    parser.add_argument(
        '--ctype', type=str, choices=ctypes_short, help="Call type to plot", default=None)
    parser.add_argument('--title', type=str, help="Call type to plot", default='')

    args = vars(parser.parse_args())
    data = []
    for fname in args['fnames']:
        with open('bps/' + fname + '.pkl', "rb") as f:
            data.append(pickle.load(f))
    labels = args['labels']
    if labels is not None:
        assert len(data) == len(labels), (len(data), len(labels))
    title = args['title']
    if len(data) == 1:
        all_block_probs_cums = (data[0][ctype] for ctype in ctypes_short)
        plot_bps(
            all_block_probs_cums, data[0]['log_iter'], data[0]['n_events'], title=title)
    else:
        plot_strats(data, labels, args['ctype'], title=title)


if __name__ == '__main__':
    runner()
