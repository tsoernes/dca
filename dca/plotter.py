import argparse
from os.path import isfile, join
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import PercentFormatter

from datahandler import next_filename

# Increase font size, set default figure size
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


def plot_erlangs_(all_block_probs_cums,
                  labels=None,
                  ymin=None,
                  ylabel=None,
                  title='',
                  fname=None):
    """ If labels=None and ylabel=None:
    All_block_probs_cums:
    For each strat, for each level of offered traffic,
    for each log iter, cumulative block prob

    If fname is given, save plot to file instead of showing it
    """
    if ylabel is None:
        ylabel = "Cumulative call blocking probability"
    if labels is None:
        labels = ctypes
        loc = 'upper right'
    else:
        loc = 'lower right'

    fig, ax = plt.subplots(1, 1)
    x = np.arange(5, 11)
    # Shift x-axis to avoid overlapping err bars
    shift_perc = 7 * 0.0025
    x_shift = np.arange(0, len(all_block_probs_cums) + 1) * shift_perc
    fmts = ['-o', '--o', '-.o', '-x', '--x', '-.x']
    for i, strat in enumerate(all_block_probs_cums):
        ys, std_devs = [], []
        for erlangs in range(0, 6):
            # Convert to percent
            # print(erlangs)
            # strat[erlangs]
            # strat[erlangs]['new']
            # strat[erlangs]['new'][-1]
            y = 100 * np.mean(strat[erlangs], axis=0)
            std_dev = 100 * np.std(strat[erlangs], axis=0)
            ys.append(y)
            std_devs.append(std_dev)
        xs = x + x_shift[i]
        fmt = fmts[i % len(fmts)]
        ax.errorbar(xs, ys, yerr=std_devs, fmt=fmt, label=labels[i], capsize=5)
    ax.legend(loc=loc)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Offered new traffic, in Erlangs")
    ax.set_title(title)
    ax.yaxis.grid(True)

    ymin_, ymax = ax.get_ylim()
    if ymin is None and ymin_ < 0:
        ymin = 0
    print(ymin_, ymin)
    ax.set_ylim(ymin=ymin)
    if ymax - ymin > 20:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))

    if fname:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fname = next_filename("plots/" + fname, '.png')
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved fig to {fname}")
    else:
        plt.show()


def plot_bps(all_block_probs_cums,
             log_iter,
             n_events,
             labels=None,
             ymin=None,
             ylabel=None,
             title='',
             fname=None):
    """ If labels=None and ylabel=None:
    Plot for each call type, for each log_iter, cumulative block prob

    all_block_probs_cums: For each call type (new, hoff, tot), for each run,
    cumulative block prob for each log iter [[run1_log1, run1_log2, ..],
    [run2_log1, run2_log2], ..]

    If fname is given, save plot to file instead of showing it
    """
    if ylabel is None:
        ylabel = "Cumulative call blocking probability"
    if labels is None:
        labels = ctypes
        loc = 'upper right'
    else:
        loc = 'lower right'

    fig, ax = plt.subplots(1, 1)
    x = np.arange(log_iter, n_events + 1, log_iter)
    # Shift x-axis to avoid overlapping err bars
    shift_perc = n_events * 0.0025
    x_shift = np.arange(0, len(all_block_probs_cums) + 1) * shift_perc
    fmts = ['-o', '--o', '-.o', '-x', '--x', '-.x']
    for i, block_probs_cums in enumerate(all_block_probs_cums):
        # Convert to percent
        y = 100 * np.mean(block_probs_cums, axis=0)
        std_devs = 100 * np.std(block_probs_cums, axis=0)
        xs = x + x_shift[i]
        fmt = fmts[i % len(fmts)]
        ax.errorbar(xs, y, yerr=std_devs, fmt=fmt, label=labels[i], capsize=5)
    ax.legend(loc=loc)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Call events")
    ax.set_title(title)
    ax.yaxis.grid(True)

    ymin_, ymax = ax.get_ylim()
    if ymin is None and ymin_ < 0:
        ymin = 0
    ax.set_ylim(ymin=ymin)
    if ymax - ymin > 20:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    if n_events >= 400_000:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50_000))
    elif n_events <= 10_000:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1_000))

    if fname:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fname = next_filename("plots/" + fname, '.png')
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved fig to {fname}")
    else:
        plt.show()


def plot_erlangs(data, labels=None, ctype='new', ymin=None, title='', fname=None):
    """Plot for each strat, for each erlang level, for a specific call type ctype,
    cumulative block prob.

    data: For each strat, for each erlang level, a dict with:
    - datetime,
    - log_iter,
    - n_events,
    - for one or more call types, for each log iter, cumulative block prob
    """
    log_iter, n_events = data[0][0]['log_iter'], data[0][0]['n_events']
    all_block_prob_cums = []
    for strat in data:
        strat_cum = []
        for erl in strat:
            # All entries should be same format
            assert log_iter == erl['log_iter'], (log_iter, erl['log_iter'])
            assert n_events == erl['n_events'], (n_events, erl['n_events'])
            # print(erl['datetime'])
            strat_cum.append(erl[ctype][-1])
        all_block_prob_cums.append(strat_cum)

    if labels is None:
        labels = [None] * len(data)
    ylabel = f"{ctypes_map[ctype]} cumulative blocking probability"
    plot_erlangs_(
        all_block_prob_cums,
        labels=labels,
        ymin=ymin,
        ylabel=ylabel,
        title=title,
        fname=fname)


def plot_strats(data, labels=None, ctype='new', ymin=None, title='', fname=None):
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
    all_block_probs_cums = [d[ctype] for d in data]
    if labels is None:
        labels = [None] * len(data)
    ylabel = f"{ctypes_map[ctype]} cumulative blocking probability"
    plot_bps(
        all_block_probs_cums,
        log_iter=log_iter,
        n_events=int(n_events),
        labels=labels,
        ymin=ymin,
        ylabel=ylabel,
        title=title,
        fname=fname)


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
        '--ext', type=str, nargs='?', help="File name extension e.g. .0", default=None)
    parser.add_argument(
        '--labels',
        type=str,
        nargs='*',
        help="Optional labels for corresponding pickle files",
        default=None)
    parser.add_argument(
        '--ymins', type=int, nargs='*', help="Optional ymins, in percent", default=None)
    parser.add_argument(
        '--ctype',
        nargs='*',
        type=str,
        choices=ctypes_short,
        help="Call type to plot",
        default=None)
    parser.add_argument(
        '--erlangs', action='store_true', help="Erlangs plot", default=False)
    parser.add_argument('--title', type=str, help="Call type to plot", default='')
    parser.add_argument(
        '--plot_save',
        type=str,
        help="Save plot to given file name, don't show",
        default=None)

    args = vars(parser.parse_args())
    data = []
    fnames = args['fnames']
    pext = args['ext'] + '.pkl'
    erlangs = args['erlangs']
    if erlangs:
        # Load e.g. vnet-e5.0.pkl, vnet-e6.0.pkl ..
        # for sims with 5, 6.. 10 Erlangs. If a file with e10 is not found,
        # drop the extension because 10 erlangs is the default and non-fixed
        # strats are not necessarily saved with it.
        for fname in fnames:
            data_s = []  # All erlangs for a given strat
            for i in range(5, 11):
                fname_e = fname + "-e" + str(i)
                if i == 10:
                    if not isfile('bps/' + fname_e + pext):
                        fname_e = fname
                with open('bps/' + fname_e + pext, "rb") as f:
                    data_s.append(pickle.load(f))
            data.append(data_s)
        assert len(data) == len(fnames)
        for li in data:
            assert len(li) == 6, len(li)
    else:
        for fname in fnames:
            with open('bps/' + fname + pext, "rb") as f:
                data.append(pickle.load(f))
    labels = args['labels']
    if labels is not None:
        assert len(data) == len(labels), (len(data), len(labels))
    plot_ctypes = args['ctype']
    ymins = args['ymins']
    if ymins is not None:
        assert len(plot_ctypes) == len(ymins), (len(plot_ctypes), len(ymins))
    # else:
    #     ymins = [None] * len(plot_ctypes)
    title = args['title']
    if len(data) == 1:
        all_block_probs_cums = [data[0][ctype] for ctype in ctypes_short]
        plot_bps(
            all_block_probs_cums,
            log_iter=data[0]['log_iter'],
            n_events=data[0]['n_events'],
            title=title,
            fname=args['plot_save'])
    else:
        for i, ctype in enumerate(plot_ctypes):
            fext = + '-' + ctype if len(plot_ctypes) > 1 else ""
            fname = args['plot_save'] + fext
            if erlangs:
                plot_erlangs(
                    data, labels, ctype=ctype, ymin=ymins[i], title=title, fname=fname)
            else:
                plot_strats(
                    data, labels, ctype=ctype, ymin=ymins[i], title=title, fname=fname)


if __name__ == '__main__':
    runner()
