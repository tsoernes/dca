import logging
import pickle
from functools import partial

import numpy as np
from hyperopt import Trials, fmin, hp, tpe  # noqa
from hyperopt.pyll.base import scope  # noqa

from hopt_utils import (MongoConn, add_pp_pickle, compare_pps, hopt_best,
                        mongo_decide_gpu_usage, mongo_decrease_gpu_procs)
from runners.runner import Runner


class HoptRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        """
        Hyper-parameter optimization with hyperopt.
        """
        if self.pp['net']:
            space = {
                # Qlearnnet
                'net_lr': hp.loguniform('net_lr', np.log(5e-7), np.log(1e-4)),
                'net_lr_decay': hp.loguniform('net_lr_decay', np.log(0.90), np.log(0.99)),
                # Singh
                # 'net_lr': hp.loguniform('net_lr', np.log(1e-7), np.log(5e-4)),
                'beta': hp.uniform('beta', 16, 30),
                # Double
                'net_copy_iter': hp.loguniform('net_copy_iter', np.log(5), np.log(150)),
                'net_creep_tau': hp.loguniform('net_creep_tau', np.log(0.01),
                                               np.log(0.7)),
                # Exp. replay
                'batch_size': scope.int(hp.uniform('batch_size', 8, 16)),
                'buffer_size': scope.int(hp.uniform('buffer_size', 2000, 10000)),
                # N-step
                'n_step': scope.int(hp.uniform('n_step', 3, 40)),
                # Policy
                'vf_coeff': hp.uniform('vf_coeff', 0.005, 0.5),
                'entropy_coeff': hp.uniform('entropy_coeff', 1.0, 100.0)
            }
        else:
            space = {
                'beta': hp.uniform('beta', 7, 23),
                'alpha': hp.uniform('alpha', 0.0001, 0.4),
                'alpha_decay': hp.uniform('alpha_decay', 0.9999, 0.9999999),
                'epsilon': hp.loguniform('epsilon', np.log(0.2), np.log(0.8)),
                'epsilon_decay': hp.uniform('epsilon_decay', 0.9995, 0.9999999),
                'gamma': hp.uniform('gamma', 0.7, 0.90),
                'lambda': hp.uniform('lambda', 0.0, 1.0)
            }
        # Only optimize parameters specified in args
        space = {param: space[param] for param in self.pp['hopt']}
        if self.pp['hopt_fname'].startswith('mongo:'):
            self._hopt_mongo(space)
        else:
            self._hopt_pickle(space)

    def _hopt_mongo(self, space):
        """Find previous best trial and pp from MongoDB, if any, then run hopt job server"""
        trials = MongoConn(self.pp['hopt_fname'])
        try:
            self.logger.error("Prev best:")
            hopt_best(trials, n=1, view_pp=False)
            prev_pps = trials.get_pps()
            # If given pp equals the last one found in MongoDB, don't add it.
            # Otherwise, ask whether to use the one found in DB instead,
            # and if not, store given pp in DB.
            if prev_pps:
                mongo_pp = prev_pps[-1]
                use_mongo_pp, new_pp = compare_pps(mongo_pp, self.pp)
                if use_mongo_pp:
                    self.pp = new_pp
                else:
                    trials.add_pp(self.pp)
            else:
                trials.add_pp(self.pp)
        except ValueError:
            self.logger.error("No existing trials, starting from scratch")
            trials.add_pp(self.pp)
        mongo_uri = self.pp['hopt_fname'].replace('mongo:', '')
        fn = partial(hopt_proc, self.stratclass, self.pp, mongo_uri=mongo_uri)
        self.logger.error("Started hyperopt job server")
        fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)
        trials.client.close()

    def _hopt_pickle(self, space):
        """
        Saves progress to 'pp['hopt_fname'].pkl' and
        automatically resumes if file already exists.
        """
        if self.pp['net']:
            trials_step = 1  # Number of trials to run before saving
        else:
            trials_step = 4
        f_name = self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        try:
            with open(f_name, "rb") as f:
                trials = pickle.load(f)
                prev_best = trials.argmin
                self.logger.error(f"Found {len(trials.trials)} saved trials")
        except FileNotFoundError:
            trials = Trials()
            prev_best = None

        add_pp_pickle(trials, self.pp)
        fn = partial(hopt_proc, self.stratclass, self.pp, mongo_uri=None)
        while True:
            n_trials = len(trials)
            self.logger.error(f"Running trials {n_trials+1}-{n_trials+trials_step}")
            best = fmin(
                fn=fn,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials + trials_step,
                trials=trials)
            if prev_best != best:
                bp = trials.best_trial['result']['loss']
                self.logger.error(f"Found new best params: {best} with block prob: {bp}")
                prev_best = best
            with open(f_name, "wb") as f:
                pickle.dump(trials, f)


def hopt_proc(stratclass, pp, space, mongo_uri=None):
    """
    If 'mongo_uri' is present, determine whether to use GPU or not based
    on the number of processes that already utilize it.
    """
    using_gpu_and_mongo = False
    # Don't override user-given arg for disabling GPU-usage
    if pp['use_gpu'] and mongo_uri is not None:
        using_gpu_and_mongo = mongo_decide_gpu_usage(mongo_uri, pp['max_gpu_procs'])
        pp['use_gpu'] = using_gpu_and_mongo
    for key, val in space.items():
        pp[key] = val
    # import psutil
    # n_avg = psutil.cpu_count(logical=False)
    # Use same constant numpy seed for all sims
    # np.random.seed(pp['rng_seed'])
    logger = logging.getLogger('')
    logger.error(space)
    result = Runner.sim_proc(stratclass, pp, pid='', reseed=True)
    if using_gpu_and_mongo:
        # Finished using the GPU, so reduce the 'gpu_procs' count
        mongo_decrease_gpu_procs(mongo_uri)
    res = result[0]
    if res is None:
        # Loss is inf or nan
        return {"status": "fail"}
    elif res is 1:
        # User quit, e.g. ctrl-c
        return {"status": "suspended"}
    return {'status': "ok", "loss": res, "hoff_loss": result[1]}
