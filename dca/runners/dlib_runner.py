import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count

import dlib

from hopt_utils import compare_pps, dlib_load, dlib_save
from runners.runner import Runner


class DlibRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.n_concurrent = cpu_count() // 2 - 1  # Number of concurrent procs
        self.n_concurrent = cpu_count() - 1  # Number of concurrent procs
        self.n_sims = 4000  # The number of times to sample and test params
        self.save_iter = 50
        self.eps = 0.0005  # solver_epsilon
        self.noise_mag = 0.001  # relative_noise_magnitude. Default setting: 0.001
        self.fname = "dlib-" + self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        space = {
            # parameter: [IsInteger, Low-Bound, High-Bound]
            'gamma': [False, 0.60, 0.99],
            'lambda': [False, 0.60, 0.99],
            'net_lr': [False, 1e-7, 1e-6],
            'beta': [True, 10, 3000],
            'net_lr_decay': [False, 0.70, 1.0],
            'weight_beta': [False, 1e-3, 9e-1],
            'epsilon': [False, 2, 5],
            'epsilon_decay': [False, 0.999_5, 0.999_999],
            'alpha': [False, 0.00001, 0.3]
        }
        self.space = {param: space[param] for param in self.pp['dlib_hopt']}
        self.params, is_int, lo_bounds, hi_bounds = [], [], [], []
        for p, li in self.space.items():
            self.params.append(p)
            is_int.append(li[0])
            lo_bounds.append(li[1])
            hi_bounds.append(li[2])
        try:
            old_spec, self.evals, info, prev_best = dlib_load(self.fname)
            # Restore saved params and settings if they differ from current/specified
            saved_params = info['params']
            if saved_params != self.params:
                self.logger.error(
                    f"Saved params {saved_params} differ from specified ones {self.params}"
                    "; using saved")
                # TODO could check if bounds match as well
                self.params = saved_params
            saved_solver_epsilon = info['solver_epsilon']
            if saved_solver_epsilon != self.eps:
                self.logger.error(
                    f"Saved solver_epsilon {saved_solver_epsilon} differ from"
                    " specified one {self.eps}, using specified")
                # self.eps = saved_solver_epsilon
            _, self.pp = compare_pps(info['pp'], self.pp)
            self.spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            self.optimizer = dlib.global_function_search(
                [self.spec],
                initial_function_evals=[self.evals],
                relative_noise_magnitude=info['relative_noise_magnitude'])
            self.logger.error(f"Restored {len(self.evals)} trials, prev best: "
                              f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}")
        except FileNotFoundError:
            self.spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            self.optimizer = dlib.global_function_search(self.spec)
            self.optimizer.set_relative_noise_magnitude(self.noise_mag)
        self.optimizer.set_solver_epsilon(self.eps)
        # Becomes populated with results as simulations finished
        self.result_queue = Queue()
        self.simproc = partial(dlib_proc, self.stratclass, self.pp, self.params,
                               self.result_queue)
        # Becomes populated with evaluation objects to be set later
        self.evals = [None] * self.n_sims

    def run(self):
        if not self.pp['avg_runs']:
            self.run_single()
        else:
            raise NotImplementedError
            self.run_avg()

    def save_evals(self):
        """Store results of finished evals to file; print best eval"""
        finished_evals = self.optimizer.get_function_evaluations()[1][0]
        dlib_save(self.spec, finished_evals, self.params, self.eps, self.noise_mag,
                  self.pp, self.fname)
        best_eval = self.optimizer.get_best_function_eval()
        prms = list(zip(self.params, list(best_eval[0])))
        self.logger.error(f"Finished {len(finished_evals)} trials."
                          f" Best eval this session: {best_eval[1]}@{prms}")

    def spawn_eval(self, i):
        """Spawn a new sim process"""
        eeval = self.optimizer.get_next_x()
        self.evals[i] = eeval  # Store eval object to be set with result later
        Process(target=self.simproc, args=(i, list(eeval.x))).start()

    def store_result(self):
        """Block until a result is ready, then store it and report it to dlib"""
        try:
            # Blocks until a result is ready
            i, result = self.result_queue.get()
        except KeyboardInterrupt:
            inp = ""
            while inp not in ["Y", "N"]:
                inp = input("Premature exit. Save? Y/N: ").upper()
            if inp == "Y":
                self.save_evals()
            sys.exit(0)
        else:
            if result is not None:
                self.evals[i].set(result)
            if i > 0 and i % self.save_iter == 0:
                self.save_evals()

    def run_single(self):
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

        self.logger.error(
            f"Dlib hopt for {self.n_sims} sims with {self.n_concurrent} procs"
            f" on params {self.space}")
        # Spawn initial processes
        for i in range(self.n_concurrent):
            self.spawn_eval(i)
        # When a thread returns a result, start a new sim
        for i in range(self.n_concurrent, self.n_sims):
            self.store_result()
            self.spawn_eval(i)
        # Get remaining results
        for _ in range(self.n_concurrent):
            self.store_result()
        self.save_evals()

    def run_avg(self):
        pass


def dlib_proc(stratclass, pp, space_params, result_queue, i, space_vals):
    logger = logging.getLogger('')
    logger.error(f"T{i} Testing {space_params}: {space_vals}")
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    if pp['exp_policy'].lower().endswith(
            'boltzmann') and pp['epsilon'] < 2.1 and pp['epsilon_decay'] < 0.999_8:
        # Avoid overflow in boltzmann pol
        pp['exp_policy'] = 'eps_greedy'
        pp['epsilon'] = 0

    strat = stratclass(pp=pp, logger=logger, pid=i)
    res = strat.simulate()[0]
    if res is None:
        res = 1
    if strat.quit_sim and not strat.invalid_loss and not strat.exceeded_bthresh:
        # If user quits sim, don't want to return result
        result_queue.put(None)
    else:
        # Must negate result as dlib performs maximization by default
        result_queue.put((i, -res))
