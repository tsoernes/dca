import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count  # noqa

import dlib
import numpy as np

from hopt_utils import compare_pps, dlib_load, dlib_save
from runners.runner import Runner


class DlibRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        pp = self.pp
        logger = self.logger

        # n_concurrent = cpu_count() - 4  # Number of concurrent procs
        n_concurrent = 12
        n_avg = 4
        assert n_concurrent % n_avg == 0, \
            f"n_avg {n_avg} does not evenly divide n_concurrent {n_concurrent}"
        n_step = n_concurrent // n_avg
        n_sims = 1000  # The number of times to sample and test params
        save_iter = 30
        eps = 0.0005  # solver_epsilon
        noise_mag = 0.005  # relative_noise_magnitude. Default setting: 0.001
        fname = "dlib-" + pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        space = {
            # parameter: [IsInteger, Low_Bound, High_Bound]
            'gamma': [False, 0.60, 0.99],
            'lambda': [False, 0.60, 0.99],
            'net_lr': [False, 1e-7, 1e-6],
            'beta': [True, 10, 3000],
            'net_lr_decay': [False, 0.70, 1.0],
            'weight_beta': [False, 1e-3, 9e-1],
            'weight_beta_decay': [False, 1e-8, 1e-4],
            'grad_beta': [False, 1e-3, 9e-1],
            'grad_beta_decay': [False, 1e-8, 1e-3],
            'epsilon': [False, 2, 5],
            'epsilon_decay': [False, 0.999_5, 0.999_999],
            'alpha': [False, 0.00001, 0.3]
        }
        space = {param: space[param] for param in pp['dlib_hopt']}
        params, is_int, lo_bounds, hi_bounds = [], [], [], []
        for p, li in space.items():
            params.append(p)
            is_int.append(li[0])
            lo_bounds.append(li[1])
            hi_bounds.append(li[2])
        try:
            old_raw_spec, old_spec, old_evals, info, prev_best = dlib_load(fname)
            saved_params = info['params']
            logger.error(f"Restored {len(old_evals)} trials, prev best: "
                         f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}")
            # Switching params being optimized over would throw off DLIB.
            # Have to assert that they are equal instead.
            # What happens if you introduce another variable in addition to the previously?
            # E.g. initialize dlib with evals over (eps, beta) then specify bounds for
            # (eps, beta, gamma)?

            # Restore saved params and settings if they differ from current/specified
            if params != saved_params:
                logger.error(
                    f"Saved params {saved_params} differ from currently specified "
                    f"{params}. Using saved.")
                params = saved_params
            raw_spec = cmp_and_choose('bounds', old_raw_spec,
                                      (is_int, lo_bounds, hi_bounds))
            spec = dlib.function_spec(
                bound1=raw_spec[1], bound2=raw_spec[2], is_integer=raw_spec[0])
            eps = cmp_and_choose('solver_epsilon', info['solver_epsilon'], eps)
            noise_mag = cmp_and_choose('relative_noise_magnitude',
                                       info['relative_noise_magnitude'], noise_mag)
            _, pp = compare_pps(info['pp'], pp)
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[old_evals],
                relative_noise_magnitude=noise_mag)
        except FileNotFoundError:
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(spec)
            optimizer.set_relative_noise_magnitude(noise_mag)
        optimizer.set_solver_epsilon(eps)
        # Becomes populated with results as simulations finished
        result_queue = Queue()
        simproc = partial(dlib_proc, self.stratclass, pp, params, result_queue)
        # Becomes populated with evaluation objects to be set later
        evals = [None] * n_sims
        # Becomes populates with losses. When n_avg losses for a particular
        # set of params are ready, their mean is set for the correponding eval.
        results = [[] for _ in range(n_sims)]

        def save_evals():
            """Store results of finished evals to file; print best eval"""
            finished_evals = optimizer.get_function_evaluations()[1][0]
            dlib_save(spec, finished_evals, params, eps, noise_mag, pp, fname)
            best_eval = optimizer.get_best_function_eval()
            prms = list(zip(params, list(best_eval[0])))
            logger.error(f"Saving {len(finished_evals)} trials. "
                         f"Best eval so far: {best_eval[1]}@{prms}")

        def spawn_evals(i):
            """Spawn a new sim process"""
            eeval = optimizer.get_next_x()
            evals[i] = eeval  # Store eval object to be set with result later
            vals = list(eeval.x)
            logger.error(f"T{i} Testing {params}: {vals}")
            for _ in range(n_avg):
                Process(target=simproc, args=(i, vals)).start()

        def store_result():
            """Block until a result is ready, then store it and report it to dlib"""
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                inp = ""
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save? Y/N: ").upper()
                if inp == "Y":
                    save_evals()
                sys.exit(0)
            else:
                if result is not None:
                    results[i].append(result)
                    if len(results[i]) == n_avg:
                        evals[i].set(np.mean(results[i]))
                if i > 0 and i % save_iter == 0 and len(results[i]) == n_avg:
                    save_evals()

        """ the search will only attempt to find a global minimizer to at most
        solver_epsilon accuracy. Once a local minimizer is found to that
        accuracy the search will focus entirely on finding other minima
        elsewhere rather than on further improving the current local optima
        found so far. That is, once a local minima is identified to about
        solver_epsilon accuracy, the algorithm will spend all its time
        exploring the functions to find other local minima to investigate. An
        epsilon of 0 means it will keep solving until it reaches full floating
        point precision. Larger values will cause it to switch to pure global
        exploration sooner and therefore might be more effective if your
        objective function has many local minima and you don't care about a
        super high precision solution.

        On even iterations we pick the next x according to our upper bound while
        on odd iterations we pick the next x according to the trust region model
        """

        logger.error(f"Dlib hopt for {n_sims} sims with {n_concurrent} procs"
                     f" on params {space} and s.eps {eps}")
        # Spawn initial processes
        for i in range(n_step):
            spawn_evals(i)
        # When a thread returns a result, start a new sim
        for i in range(n_step, n_sims):
            for _ in range(n_avg):
                store_result()
            spawn_evals(i)
        # Get remaining results
        for _ in range(n_step):
            for _ in range(n_avg):
                store_result()
        save_evals()


def cmp_and_choose(what, saved, specified):
    chosen = specified
    if saved != specified:
        print(f"Saved {what} {saved} differ from currently specified {specified}")
        inp = ""
        while inp not in ['N', 'Y']:
            inp = input(f"Use saved {what} (Y) instead of specified (N)?: ").upper()
        if inp == "Y":
            chosen = saved
    return chosen


def dlib_proc(stratclass, pp, space_params, result_queue, i, space_vals):
    logger = logging.getLogger('')
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    if pp['exp_policy'].lower().endswith(
            'boltzmann') and pp['epsilon'] < 2.1 and pp['epsilon_decay'] < 0.999_8:
        # Avoid overflow in boltzmann pol
        pp['exp_policy'] = 'eps_greedy'
        pp['epsilon'] = 0

    np.random.seed()
    strat = stratclass(pp=pp, logger=logger, pid=i)
    res = strat.simulate()[0]
    if res is None:
        res = 1
    if strat.quit_sim and not strat.invalid_loss and not strat.exceeded_bthresh:
        # If user quits sim, don't want to return result
        result_queue.put((i, None))
    else:
        # Must negate result as dlib performs maximization by default
        result_queue.put((i, -res))
