#! /usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import inspect
import logging
import random
import sys

import argcomplete
import numpy as np

import strats.fixedstrats  # noqa
import strats.net_rl  # noqa
import strats.table_rl  # noqa


def strat_classes(module_name):
    """
    Return a list with (name, class) for all the strats
    """

    def is_class_member(member):
        return inspect.isclass(member) and member.__module__ == module_name

    clsmembers = inspect.getmembers(sys.modules[module_name], is_class_member)
    return clsmembers


def get_pparams(defaults=False):
    """
    Return problem parameters and chosen strategy class. If 'defaults' is True,
    just return the default params.
    """
    parser = argparse.ArgumentParser(
        description='DCA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Get available strategies and format their names real nice
    stratclasses = strat_classes("strats.net_rl") + strat_classes(
        "strats.fixedstrats") + strat_classes("strats.table_rl")
    stratnames = ['show']
    for i, s in enumerate(stratclasses):
        s1 = s[0].lower()
        s2 = s1.replace("strat", "")
        if s2 not in ["net", "qnet", "rl", "qtable"]:
            stratnames.append(s2)
        stratclasses[i] = (s2, stratclasses[i][1])
    weight_initializers = ['zeros', 'glorot_unif', 'glorot_norm', 'norm_cols']

    parser.add_argument('strat', type=str, choices=stratnames, default="rs_sarsa")
    parser.add_argument('--rows', type=int, help="number of rows in grid", default=7)
    parser.add_argument('--cols', type=int, help="number of columns in grid", default=7)
    parser.add_argument('--n_channels', type=int, help="number of channels", default=70)
    parser.add_argument(
        '--erlangs',
        type=float,
        help="erlangs = call_rate * call_duration"
        "\n 10 erlangs = 200 call rate, given call duration of 3"
        "\n 7.5 erlangs = 150cr, 3cd"
        "\n 5 erlangs = 100cr, 3cd",
        default=10)
    parser.add_argument(
        '--call_rates', type=int, help="in calls per minute", default=None)
    parser.add_argument('--call_duration', type=int, help="in minutes", default=3)
    parser.add_argument(
        '-phoff',
        '--p_handoff',
        type=float,
        help="handoff probability. Default: 0.15",
        default=None)
    parser.add_argument(
        '--hoff_call_duration',
        type=int,
        help="handoff call duration, in minutes",
        default=1)
    parser.add_argument(
        '--n_events',
        '-i',
        dest='n_events',
        type=int,
        help="number of events to simulate",
        default=470000)
    parser.add_argument(
        '--n_hours',
        type=float,
        help="Number of hours to simulate (overrides n_events)",
        default=None)
    parser.add_argument(
        '--breakout_thresh',
        '-thresh',
        type=float,
        default=0.22,
        help="Break out early if cumuluative blocking probability "
        "exceeds given threshold")
    parser.add_argument(
        '--avg_runs',
        metavar='N',
        type=int,
        help="Run simulation N times, report average block probs",
        default=None)

    parser.add_argument(
        '--alpha', type=float, help="(RL/Table) learning rate", default=0.01938893)
    parser.add_argument(
        '--alpha_decay',
        type=float,
        help="(RL/Table) factor by which alpha is multiplied each iteration",
        default=0.9999999)

    parser.add_argument(
        '--epsilon',
        '-eps',
        dest='epsilon',
        type=float,
        help="(RL) (initial) probability of choosing random action",
        default=0.75443)
    parser.add_argument(
        '--epsilon_decay',
        type=float,
        help="(RL) factor by which epsilon is multiplied each iteration",
        default=0.99999)
    parser.add_argument('--gamma', type=float, help="(RL) discount factor", default=0.85)
    parser.add_argument(
        '--beta',
        nargs='?',
        type=float,
        help="(RL) Instead of using a constant discount factor gamma;"
        "integrate rewards over dt between events (see Singh96)",
        const=15,
        default=None)
    parser.add_argument(
        '--reward_scale',
        type=float,
        help="(RL) Factor by which rewards are scaled",
        default=1)
    parser.add_argument(
        '--lambda',
        type=float,
        help="(RL/Table) lower lambda weighs fewer step returns higher",
        default=None)
    parser.add_argument(
        '--min_alpha',
        type=float,
        help="(RL) stop decaying alpha beyond this point",
        default=0.0)
    parser.add_argument(
        '--save_exp_data',
        help="Save experience data to file",
        action='store_true',
        default=False)
    parser.add_argument(
        '--restore_qtable',
        nargs='?',
        type=str,
        help="(RL/Table) Restore q-values from given file",
        default="",
        const="qtable.npy")
    parser.add_argument(
        '--dlib_hopt',
        nargs='+',
        metavar="PARAM1, PARAM2, ..",
        choices=[
            'epsilon', 'epsilon_decay', 'alpha', 'alpha_decay', 'gamma', 'lambda',
            'net_lr', 'net_copy_iter', 'net_creep_tau', 'vf_coeff', 'entropy_coeff',
            'beta', 'net_lr_decay', 'n_step'
        ],
        help="(Hopt) Hyper-parameter optimization with dlib.",
        default=None)
    parser.add_argument(
        '--hopt',
        nargs='+',
        metavar="PARAM1, PARAM2, ..",
        choices=[
            'epsilon', 'epsilon_decay', 'alpha', 'alpha_decay', 'gamma', 'lambda',
            'net_lr', 'net_copy_iter', 'net_creep_tau', 'vf_coeff', 'entropy_coeff',
            'beta', 'net_lr_decay', 'n_step'
        ],
        help="(Hopt) Hyper-parameter optimization with hyperopt.",
        default=None)
    parser.add_argument(
        '--hopt_fname',
        type=str,
        help="(Hopt) File name or Mongo-DB data base name"
        "for hyperopt destination/source. Prepend 'mongo:' to Mongo-DB names",
        default=None)

    parser.add_argument(
        '--net_lr',
        '-lr',
        dest='net_lr',
        type=float,
        help="(Net) Learning rate. Overrides 'alpha'.",
        default=2.95e-5)
    parser.add_argument(
        '--net_lr_decay',
        '-lr_dec',
        type=float,
        help="(Net) Exponential Learning rate decay multiplier",
        default=0.96)
    parser.add_argument(
        '--optimizer',
        '-opt',
        dest='optimizer',
        choices=['sgd', 'sgd-m', 'adam', 'rmsprop'],
        default='sgd-m')
    parser.add_argument(
        '--huber_loss',
        nargs='?',
        type=float,
        help="(Net) Enable huble loss with given delta",
        default=None,
        const=100000)
    parser.add_argument(
        '--max_grad_norm',
        '-norm',
        dest='max_grad_norm',
        type=float,
        metavar='N',
        nargs='?',
        help="(Net) Clip gradient to N",
        default=None,
        const=100000)
    parser.add_argument(
        '--weight_init_conv', choices=weight_initializers, default='zeros')
    parser.add_argument(
        '--weight_init_dense', choices=weight_initializers, default='norm_cols')
    parser.add_argument('--n_step', type=int, help="(Net) N step returns", default=1)
    parser.add_argument(
        '--dueling_qnet',
        '-duel',
        dest='dueling_qnet',
        action='store_true',
        help="(Net/Duel) Dueling QNet",
        default=False)
    parser.add_argument(
        '--layer_norm',
        action='store_true',
        help="(Net) Use layer normalization",
        default=False)
    parser.add_argument(
        '--no_grid_split',
        action='store_true',
        help="(Net) Don't double the depth and represent empty channels "
        "as 1 on separate layer",
        default=False)
    parser.add_argument(
        '--qnet_freps',
        action='store_true',
        help="(Net) Include feature representation a la Singh in "
        "addition to grid as input to net",
        default=False)
    parser.add_argument(
        '--act_fn',
        help="(Net) Activation function",
        choices=['relu', 'elu', 'leaky_relu'],
        default='relu')
    parser.add_argument(
        '--save_net',
        action='store_true',
        help="(Net) Save network params",
        default=False)
    parser.add_argument(
        '--restore_net',
        action='store_true',
        help="(Net) Restore network params",
        default=False)
    parser.add_argument(
        '--batch_size',
        type=int,
        help="(Net/Exp) Batch size for experience replay or training."
        "A value of 1 disables exp. replay",
        default=1)
    parser.add_argument(
        '--buffer_size',
        type=int,
        help="(Net/Exp) Buffer size for experience replay",
        default=5000)
    # parser.add_argument(
    #     '-pri',
    #     '--prioritized_replay',
    #     action='store_true',
    # help="(Net) Prioritized Experience Replay",
    # default=False)
    parser.add_argument(
        '--bench_batch_size',
        action='store_true',
        help="(Net) Benchmark batch size for neural network",
        default=False)
    parser.add_argument(
        '--net_copy_iter',
        type=int,
        metavar='N',
        help="(Net/Double) Copy weights from online to target "
        "net every N iterations",
        default=45)
    parser.add_argument(
        '--net_copy_iter_decr',
        type=int,
        metavar='N',
        help="(Net/Double) Decrease 'net_copy_iter' every N iterations",
        default=None)
    parser.add_argument(
        '--net_creep_tau',
        '-tau',
        dest='net_creep_tau',
        type=float,
        nargs='?',
        metavar='tau',
        help="(Net/Double) Creep target net 'tau' (0, 1] "
        "towards online net every 'net_copy_iter' iterations. "
        "Net copy iter should be decreased as tau decreases. "
        "'tau' ~ 0.1 when 'net_copy_iter' is 5 is good starting point.",
        default=1,
        const=0.12)
    parser.add_argument(
        '--vf_coeff',
        type=float,
        help="(Net/Pol) Value function coefficient in policy gradient loss",
        default=0.02)
    parser.add_argument(
        '--entropy_coeff',
        type=float,
        help="(Net/Pol) Entropy coefficient in policy gradient loss",
        default=10.0)
    parser.add_argument(
        '--train_net',
        type=int,
        metavar='N',
        nargs="?",
        help="(Net) Train network for 'N' passes",
        default=0,
        const=1)
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help="(Net) Disable TensorFlow GPU usage",
        default=False)
    parser.add_argument(
        '--max_gpu_procs',
        type=int,
        help="(Net) Maximum concocurrent processes that utilize the GPU with tensorflow",
        default=3)

    parser.add_argument(
        '--rng_seed',
        type=int,
        metavar='N',
        nargs='?',
        help="By default, use seed 0. "
        "If specified without a value, use a random seed.",
        default=0,
        const=np.random.randint(2000))
    parser.add_argument(
        '--verify_grid',
        action='store_true',
        help="Verify channel reuse constraint each iteration",
        default=False)
    parser.add_argument(
        '--prof',
        dest='profiling',
        action='store_true',
        help="performance profiling",
        default=False)
    parser.add_argument(
        '--tfprof',
        dest='tfprofiling',
        metavar='DEST',
        type=str,
        help="(Net) performance profiling for TensorFlow."
        " Specify ouput file name",
        default="")
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', dest='do_plot', default=False)
    parser.add_argument(
        '--log_level',
        type=int,
        choices=[10, 20, 30],
        help="10: Debug,\n20: Info,\n30: Warning",
        default=None)

    parser.add_argument(
        '--log_file', metavar='DEST', type=str, help="enable logging to given file name")
    parser.add_argument(
        '--log_iter',
        metavar='N',
        type=int,
        help="Show blocking probability and stats such as "
        "epsilon, learning rate and loss every N iterations",
        default=None)

    if defaults:
        args = parser.parse_args(['rs_sarsa'])
    else:
        argcomplete.autocomplete(parser)
        args = parser.parse_args()
    pp = vars(args)

    # We don't want no double negatives
    pp['grid_split'] = not pp['no_grid_split']
    del pp['no_grid_split']
    pp['use_gpu'] = not pp['no_gpu']
    del pp['no_gpu']

    if pp['hopt'] and not pp['hopt_fname']:
        print("No file name specified for hyperopt ('hopt_fname')")
        sys.exit(0)

    # Sensible presets, overrides, convenience variables
    if pp['n_hours'] is not None:
        # Approximate iters from hours. Only used for calculating
        # log_iter percentages and param decay schedules if log iter is not given
        pp['n_events'] = 7821 * pp['n_hours'] - 2015
    pp['dt_rewards'] = pp['beta'] is not None
    pp['dims'] = (pp['rows'], pp['cols'], pp['n_channels'])
    if "net" in pp['strat'].lower():
        if not pp['log_iter']:
            pp['log_iter'] = 5000
        pp['net'] = True  # Whether net is in use or not
    else:
        if not pp['log_iter']:
            pp['log_iter'] = 50000
        pp['batch_size'] = 1
        pp['net'] = False
    if not pp['call_rates']:
        pp['call_rates'] = pp['erlangs'] / pp['call_duration']
    if pp['avg_runs']:
        pp['gui'] = False
        pp['use_gpu'] = False
        pp['log_iter'] = pp['n_events'] // 8
    if pp['dlib_hopt'] is not None or pp['hopt'] is not None:
        if pp['net']:
            pp['n_events'] = 100000
        pp['gui'] = False
        if pp['log_level'] is None:
            pp['log_level'] = logging.ERROR
        pp['breakout_thresh'] = 0.18
        # Since hopt only compares new call block rate,
        # handoffs are a waste of data/computational resources.
        if pp['p_handoff'] is None:
            pp['p_handoff'] = 0
        # Always log to file so that parameters are recorded
        if pp['dlib_hopt'] is not None:
            libname = "dlib"
        elif pp['hopt'] is not None:
            libname = "hopt"
        pnames = str.join("-", pp['hopt'])
        f_name = f"results-{libname}-{pp['strat']}-{pnames}"
        pp['log_file'] = f_name
    if pp['bench_batch_size']:
        if pp['log_level'] is None:
            pp['log_level'] = logging.WARN
    if pp['log_level'] is None:
        pp['log_level'] = logging.INFO
    if pp['p_handoff'] is None:
        pp['p_handoff'] = 0.15

    random.seed(pp['rng_seed'])
    np.random.seed(pp['rng_seed'])

    for name, cls in stratclasses:
        if pp['strat'].lower() == name.lower():
            stratclass = cls
    return pp, stratclass


def non_uniform_preset(pp):
    raise NotImplementedError  # Untested
    """
    Non-uniform traffic patterns for linear array of cells.
    Formål: How are the different strategies sensitive to
    non-uniform call patterns?
    rows = 1
    cols = 20
    call rates: l:low, m:medium, h:high
    For each pattern, the numeric values of l, h and m are chosen
    so that the average call rate for a cell is 120 calls/hr.
    low is 1/3 of high; med is 2/3 of high.
    """
    avg_cr = 120 / 60  # 120 calls/hr
    patterns = [
        "mmmm" * 5, "lhlh" * 5, ("llh" * 7)[:20], ("hhl" * 7)[:20], ("lhl" * 7)[:20],
        ("hlh" * 7)[:20]
    ]
    pattern_call_rates = []
    for pattern in patterns:
        n_l = pattern.count('l')
        n_m = pattern.count('m')
        n_h = pattern.count('h')
        cr_h = avg_cr * 20 / (n_h + 2 / 3 * n_m + 1 / 3 * n_l)
        cr_m = 2 / 3 * cr_h
        cr_l = 1 / 3 * cr_h
        call_rates = np.zeros((1, 20))
        for i, c in enumerate(pattern):
            if c == 'l':
                call_rates[0][i] = cr_l
            elif c == 'm':
                call_rates[0][i] = cr_m
            elif c == 'h':
                call_rates[0][i] = cr_h
        pattern_call_rates.append(call_rates)
