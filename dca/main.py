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
import strats.qnet_rl  # noqa
import strats.table_rl  # noqa
import strats.vnet_rl  # noqa
from runners.runner import AnalyzeNetRunner, Runner, ShowRunner, TrainNetRunner
from strats.exp_policies import exp_pol_funcs


def main():
    pp, stratclass = get_pparams()

    if pp['hopt']:
        from runners.hopt_runner import HoptRunner
        run_cls = HoptRunner
    elif pp['dlib_hopt']:
        from runners.dlib_runner import DlibRunner
        run_cls = DlibRunner
    elif pp['exp_policy_cmp']:
        from runners.exp_pol_runner import ExpPolRunner
        run_cls = ExpPolRunner
    elif pp['avg_runs']:
        from runners.avg_runner import AvgRunner
        run_cls = AvgRunner
    elif pp['strat'] == 'show':
        run_cls = ShowRunner
    elif pp['train_net']:
        run_cls = TrainNetRunner
    elif pp['analyze_net']:
        run_cls = AnalyzeNetRunner
    else:
        run_cls = Runner
    runner = run_cls(pp, stratclass)
    runner.run()


def get_classes(module_name):
    """
    Return a list with (name, class) for all the classes in 'module_name'
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
    stratclasses = get_classes("strats.vnet_rl") + get_classes(
        "strats.qnet_rl") + get_classes("strats.fixedstrats") + get_classes(
            "strats.table_rl")
    stratnames = ['show']
    for i, s in enumerate(stratclasses):
        s1 = s[0].lower()
        s2 = s1.replace("strat", "")
        if s2 not in ["net", "qnet", "rl", "qtable"]:
            stratnames.append(s2)
        stratclasses[i] = (s2, stratclasses[i][1])

    policy_func_names = list(exp_pol_funcs.keys()) + ['greedy']
    weight_initializers = [
        'zeros', 'glorot_unif', 'glorot_norm', 'norm_cols', 'nominal', 'norm_pos',
        'const_pos'
    ]
    hopt_opts = [
        'epsilon', 'epsilon_decay', 'alpha', 'alpha_decay', 'gamma', 'lambda', 'net_lr',
        'net_copy_iter', 'net_creep_tau', 'beta', 'net_lr_decay', 'n_step', 'weight_beta',
        'weight_beta_decay', 'grad_beta', 'grad_beta_decay', 'huber_loss'
    ]

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
        '--traffic_preset',
        type=str,
        help="""Choose between
    traffic patterns. linear24: increase linearly from half the given call
    rate to the given callrate over 24h""",
        choices=['uniform', 'nonuniform', 'linear24'],
        default='uniform')
    parser.add_argument('--call_rate', type=int, help="in calls per minute", default=None)
    parser.add_argument('--call_duration', type=int, help="in minutes", default=3)
    parser.add_argument(
        '-phoff',
        '--p_handoff',
        type=float,
        nargs='?',
        help="handoff probability. Default: 0.0",
        default=None,
        const=0.15)
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
        help="number of events to simulate (Default: 470_000)",
        default=None)
    parser.add_argument(
        '--n_hours',
        type=float,
        help="Number of hours to simulate (overrides n_events)",
        default=None)
    parser.add_argument(
        '--breakout_thresh',
        '-thresh',
        type=float,
        default=None,
        help="Break out early if cumuluative blocking probability "
        "exceeds given threshold (Default: 0.23")
    parser.add_argument(
        '--avg_runs',
        metavar='N',
        type=int,
        help="Run simulation N times, report average block probs",
        default=None)
    parser.add_argument(
        '--threads', metavar='M', type=int, help="Max threads", default=16)
    parser.add_argument(
        '--exp_policy_cmp',
        metavar='N',
        type=int,
        help="Run different exp pols, average each over N runs",
        default=None)
    parser.add_argument(
        '--alpha', type=float, help="(RL/Table) learning rate", default=0.01938893)
    parser.add_argument(
        '--alpha_decay',
        type=float,
        help="(RL/Table) factor by which alpha is multiplied each iteration",
        default=0.999_999_9)
    parser.add_argument(
        '-epol',
        '--exp_policy',
        type=str,
        help="Exploration policy (only used for NEW/HOFF events)",
        choices=policy_func_names,
        default="boltzmann")
    parser.add_argument(
        '-epolc',
        '--exp_policy_param',
        type=float,
        help="Exploration policy parameter",
        default=None)
    parser.add_argument(
        '--epsilon',
        '-eps',
        dest='epsilon',
        type=float,
        help="(RL) (initial) probability of choosing random action",
        default=4.8)
    parser.add_argument(
        '-edec',
        '--epsilon_decay',
        type=float,
        help="(RL) factor by which epsilon is multiplied each iteration",
        default=0.999_995)
    parser.add_argument(
        '--eps_log_decay',
        type=int,
        help="(RL) Decay epsilon a la Lilith instead of exponentially (give s parameter)",
        default=0)
    parser.add_argument(
        '--lilith',
        action='store_true',
        help="(RL) Lilith hyperparam preset",
        default=False)
    parser.add_argument(
        '--lilith_noexp',
        action='store_true',
        help="(RL) Lilith hyperparam preset (excluding Exploration)",
        default=False)
    parser.add_argument('--gamma', type=float, help="(RL) discount factor", default=0.845)
    parser.add_argument(
        '-wbeta',
        '--weight_beta',
        type=float,
        help="(RL) Avg. reward learning rate",
        default=6e-2)
    parser.add_argument(
        '-wbeta_dec', '--weight_beta_decay', type=float, help="(RL)", default=4.75e-5)
    parser.add_argument('-gbeta', '--grad_beta', type=float, help="(RL)", default=5e-6)
    parser.add_argument(
        '-gbeta_dec', '--grad_beta_decay', type=float, help="(RL)", default=9e-4)
    parser.add_argument(
        '-rtype',
        '--reward_type',
        help="new_block: +1 for accepted calls, -1 for blocked. "
        "callcount: Calls in progress. "
        "smdp_callcount: Calls in progress, integrated over time",
        choices=['new_block', 'callcount', 'smdp_callcount'],
        default='callcount'),
    parser.add_argument(
        '--beta',
        nargs='?',
        type=float,
        help="(RL) integrate rewards over dt between events (see Singh96)",
        const=15,
        default=None)
    parser.add_argument(
        '-bdisc',
        '--beta_disc',
        action='store_true',
        help="(RL) Semi-MDP Bootstrap discount",
        default=False)
    parser.add_argument(
        '-imp_sampl',
        '--importance_sampling',
        action='store_true',
        help="(RL)",
        default=False)
    parser.add_argument(
        '-hla', '--hoff_lookahead', action='store_true', help="(RL)", default=False)
    parser.add_argument(
        '--target', choices=['avg', 'avg_rsmart', 'discount'], default='avg')
    parser.add_argument(
        '--lambda',
        type=float,
        help="(RL/Table) lower lambda weighs fewer step returns higher",
        default=None)
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
        '--random_hopt',
        nargs='+',
        metavar="PARAM1, PARAM2, ..",
        choices=hopt_opts,
        help="(Hopt) Hyper-parameter optimization with dlib.",
        default=None)
    parser.add_argument(
        '--dlib_hopt',
        nargs='+',
        metavar="PARAM1, PARAM2, ..",
        choices=hopt_opts,
        help="(Hopt) Hyper-parameter optimization with dlib.",
        default=None)
    parser.add_argument(
        '--hopt',
        nargs='+',
        metavar="PARAM1, PARAM2, ..",
        choices=hopt_opts,
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
        default=2.52e-06)
    parser.add_argument(
        '--net_lr_decay',
        '-lr_dec',
        type=float,
        help="(Net) Exponential Learning rate decay multiplier",
        default=0.78)
    parser.add_argument(
        '--optimizer',
        '-opt',
        dest='optimizer',
        choices=['sgd', 'sgd-m', 'adam', 'rmsprop'],
        default='sgd')
    parser.add_argument(
        '--huber_loss',
        nargs='?',
        type=float,
        help="(Net) Enable huble loss with given delta",
        default=None,
        const=100_000)
    parser.add_argument(
        '--max_grad_norm',
        '-norm',
        dest='max_grad_norm',
        type=float,
        metavar='N',
        nargs='?',
        help="(Net) Clip gradient to N",
        default=None,
        const=100_000)
    parser.add_argument(
        '--weight_init_conv', choices=weight_initializers, default='glorot_unif')
    parser.add_argument(
        '--weight_init_dense', choices=weight_initializers, default='zeros')
    parser.add_argument(
        '-filters',
        '--conv_nfilters',
        nargs='+',
        type=int,
        help='(Net) Number of convolutional filters',
        default=[80, 70])
    parser.add_argument(
        '-kernels',
        '--conv_kernel_sizes',
        nargs='+',
        type=int,
        help='(Net) Convolutional kernel sizes',
        default=[4, 3])
    parser.add_argument(
        '--conv_bias',
        action='store_true',
        help="(Net) Bias for convolutional layers",
        default=False)
    parser.add_argument(
        '--pre_conv',
        action='store_true',
        help="(Singh) Conv layer(s) before dense",
        default=False)
    parser.add_argument('--prep_net', type=int, default=0)
    parser.add_argument('--n_step', type=int, help="(Net) N step returns", default=1)
    parser.add_argument(
        '--bighead',
        action='store_true',
        help="(Net) rows X cols X n_channels net output",
        default=False)
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
        '--l2_conv',
        action='store_true',
        help="(Net) Use L2 regularization for conv layers",
        default=False)
    parser.add_argument(
        '--l2_scale', type=float, help="(Net) Scale L2 loss", default=1e-5)
    parser.add_argument(
        '--l2_dense',
        action='store_true',
        help="(Net) Use L2 regularization for dense layers",
        default=False)
    parser.add_argument(
        '--top_stack',
        action='store_true',
        help="(Net) Stack cell before conv instead of after",
        default=False)
    parser.add_argument(
        '--no_grid_split',
        action='store_true',
        help="(Net) Don't double the depth and represent empty channels "
        "as 1 on separate layer",
        default=False)
    parser.add_argument(
        '-ftype', '--frep_type', choices=['vanilla', 'big', 'big2'], default='vanilla')
    parser.add_argument(
        '--singh_grid',
        action='store_true',
        help="(Net) Include grid representation in "
        "addition to freps as input to singh",
        default=False)
    parser.add_argument(
        '--qnet_freps',
        action='store_true',
        help="(Net) Include feature representation a la Singh in "
        "addition to grid as input to qnet",
        default=False)
    parser.add_argument(
        '--qnet_freps_only',
        action='store_true',
        help="(Net) Feature representation a la Singh in "
        "replacement to grid as input to qnet",
        default=False)
    parser.add_argument(
        '--scale_freps', action='store_true', help="(Net) Scale freps", default=False)
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
        default=1000)
    # parser.add_argument(
    #     '-pri',
    #     '--prioritized_replay',
    #     action='store_true',
    # help="(Net) Prioritized Experience Replay",
    # default=False)
    parser.add_argument(
        '--net_copy_iter',
        type=int,
        metavar='N',
        help="(Net/Double) Copy weights from online to target "
        "net every N iterations",
        default=5)
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
        default=0.12,
        const=0.12)
    parser.add_argument(
        '--train_net',
        type=int,
        metavar='N',
        nargs="?",
        help="(Net) Train network for 'N' passes",
        default=0,
        const=1)
    parser.add_argument(
        '--analyze_net',
        action='store_true',
        help="(Net) Run empty grid through net, print vals",
        default=False)
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="(Net) Enable TensorFlow GPU usage",
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
    parser.add_argument('--debug', action='store_true', help="Debug flag", default=False)
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
    parser.add_argument(
        '--print_weights',
        help="Print min/max/avg of weights every log iter",
        action='store_true',
        default=False)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', dest='do_plot', default=False)
    parser.add_argument(
        '--plot_save',
        type=str,
        help="Save plot to given file name, instead of showing (implies plot)",
        default=None)
    parser.add_argument(
        '--log_level',
        type=int,
        choices=[10, 20, 30],
        help="10: Debug,\n20: Info,\n30: Warning",
        default=None)

    parser.add_argument(
        '--log_file', metavar='DEST', type=str, help="enable logging to given file name")
    parser.add_argument(
        '-save_bp',
        '--save_cum_block_probs',
        type=str,
        help="For avg runs, output and save cum. block. prob. for each log iter at end",
        default=None)
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
    # pp['conv_bias'] = not pp['no_conv_bias']
    # del pp['no_conv_bias']

    assert len(pp['conv_kernel_sizes']) == len(pp['conv_nfilters'])

    pp['freps'] = pp['qnet_freps'] or pp['qnet_freps_only']
    if pp['hopt'] and not pp['hopt_fname']:
        print("No file name specified for hyperopt ('hopt_fname')")
        sys.exit(0)

    # Sensible presets, overrides, convenience variables
    if pp['n_hours'] is not None:
        # Approximate iters from hours. Only used for calculating
        # log_iter percentages and param decay schedules if log iter is not given
        pp['n_events'] = int(7821 * pp['n_hours'] - 2015)
    if not pp['call_rate']:
        pp['call_rate'] = pp['erlangs'] / pp['call_duration']
    pp['dims'] = (pp['rows'], pp['cols'], pp['n_channels'])
    if pp['lilith'] or pp['lilith_noexp']:
        pp['alpha'] = 0.05
        pp['alpha_decay'] = 1
        pp['target'] = 'discount'
        pp['gamma'] = 0.975
    if pp['lilith']:
        pp['eps_log_decay'] = 256
        pp['epsilon'] = 5
    if pp['plot_save']:
        pp['plot'] = True
    if pp['beta'] and not pp['beta_disc']:
        print("Using beta but not beta_disc!")
    if pp['target'] != 'discount':
        pp['gamma'] = None
        pp['beta'] = None
    if "net" in pp['strat'].lower():
        if not pp['log_iter']:
            pp['log_iter'] = 5000
        pp['net'] = True  # Whether net is in use or not
    else:
        if not pp['log_iter']:
            pp['log_iter'] = 50000
        pp['batch_size'] = 1
        pp['net'] = False
    if pp['avg_runs'] or pp['exp_policy_cmp']:
        pp['gui'] = False
        pp['gpu'] = False
        if pp['n_events'] is None:
            pp['n_events'] = 470000
        if pp['log_iter'] is None:
            pp['log_iter'] = int(pp['n_events'] // 8)
        if pp['exp_policy_cmp']:
            f_name = f"exp_pol_cmp-{pp['strat']}"
            pp['log_file'] = f_name
        else:
            if pp['log_file'] is None and (
                (pp['avg_runs'] is not None and pp['avg_runs'] >= 16) or
                    pp['exp_policy_cmp']):  # yapf: disable
                # Force file logging for big runs
                f_name = f"avg-{pp['strat']}"
                pp['log_file'] = f_name
    if pp['exp_policy_cmp']:
        if pp['log_level'] is None:
            pp['log_level'] = logging.ERROR
    if pp['dlib_hopt'] is not None or pp['hopt'] is not None:
        if pp['net']:
            if pp['n_events'] is None:
                pp['n_events'] = 100000
        pp['gui'] = False
        if pp['log_level'] is None:
            pp['log_level'] = logging.ERROR
        if pp['breakout_thresh'] is None:
            pp['breakout_thresh'] = 0.18
        # Since hopt only compares new call block rate,
        # handoffs are a waste of data/computational resources.
        if pp['p_handoff'] is None:
            pp['p_handoff'] = 0
        # Always log to file so that parameters are recorded
        if pp['dlib_hopt'] is not None:
            libname = "dlib"
            pnames = str.join("-", pp['dlib_hopt'])
        elif pp['hopt'] is not None:
            libname = "hopt"
            pnames = str.join("-", pp['hopt'])
        if pp['log_file'] is None:
            # Always log hopt runs
            f_name = f"results-{libname}-{pp['strat']}-{pnames}"
            pp['log_file'] = f_name
    if pp['log_level'] is None:
        pp['log_level'] = logging.INFO
    if pp['p_handoff'] is None:
        pp['p_handoff'] = 0.0
    if pp['n_events'] is None:
        pp['n_events'] = 470000
    if pp['breakout_thresh'] is None:
        pp['breakout_thresh'] = 0.23
    if pp['exp_policy'] == "greedy":
        pp['exp_policy'] = "eps_greedy"
        pp['epsilon'] = 0

    random.seed(pp['rng_seed'])
    np.random.seed(pp['rng_seed'])

    stratclass = None
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


if __name__ == '__main__':
    main()
