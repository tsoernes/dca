import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count

from hopt_utils import compare_pps, dlib_load, dlib_save
from runners.runner import Runner


class DlibRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
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
        import dlib
        n_sims = 4000  # The number of times to sample and test params
        save_iter = 50
        n_concurrent = int(cpu_count() / 2) - 1  # Number of concurrent procs
        solver_epsilon = 0.0005
        relative_noise_magnitude = 0.001  # Default
        space = {
            # parameter: [IsInteger, Low-Bound, High-Bound]
            # 'gamma': [False, 0.60, 0.99],
            'net_lr': [False, 1e-7, 1e-6],
            # 'beta': [True, 10, 3000],
            # 'net_lr_decay': [False, 0.70, 1.0],
            'weight_beta': [False, 1e-3, 9e-1],
            # 'epsilon': [False, 2, 5],
            # 'epsilon_decay': [False, 0.999_5, 0.999_999],
            # 'alpha': [False, 0.00001, 0.3]
        }
        params, is_int, lo_bounds, hi_bounds = [], [], [], []
        for p, li in space.items():
            params.append(p)
            is_int.append(li[0])
            lo_bounds.append(li[1])
            hi_bounds.append(li[2])
        fname = "dlib-" + self.pp['hopt_fname'].replace('.pkl', '') + '.pkl'
        try:
            old_spec, evals, info, prev_best = dlib_load(fname)
            # Restore saved params and settings if they differ from current/specified
            saved_params = info['params']
            if saved_params != params:
                self.logger.error(
                    f"Saved params {saved_params} differ from specified ones {params}; using saved"
                )
                # TODO could check if bounds match as well
                params = saved_params
            saved_solver_epsilon = info['solver_epsilon']
            if saved_solver_epsilon != solver_epsilon:
                self.logger.error(
                    f"Saved solver_epsilon {saved_solver_epsilon} differ from"
                    " specified one {solver_epsilon}, using specified")
                # solver_epsilon = saved_solver_epsilon
            _, self.pp = compare_pps(info['pp'], self.pp)
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[evals],
                relative_noise_magnitude=info['relative_noise_magnitude'])
            self.logger.error(f"Restored {len(evals)} trials, prev best: "
                              f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}")
        except FileNotFoundError:
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(spec)
            optimizer.set_relative_noise_magnitude(relative_noise_magnitude)
        optimizer.set_solver_epsilon(solver_epsilon)

        result_queue = Queue()
        simproc = partial(dlib_proc, self.stratclass, self.pp, params, result_queue)
        evals = [None] * n_sims

        def save():
            finished_evals = optimizer.get_function_evaluations()[1][0]
            dlib_save(spec, finished_evals, params, solver_epsilon,
                      relative_noise_magnitude, self.pp, fname)
            self.logger.error("Saved progress")

        def quit_opt():
            # Store results of finished evals to file; print best eval
            finished_evals = optimizer.get_function_evaluations()[1][0]
            dlib_save(spec, finished_evals, params, solver_epsilon,
                      relative_noise_magnitude, self.pp, fname)
            best_eval = optimizer.get_best_function_eval()
            prms = list(zip(params, list(best_eval[0])))
            self.logger.error(f"Finished {len(finished_evals)} trials."
                              f" Best eval this session: {best_eval[1]}@{prms}")

        def spawn_eval(i):
            # Spawn a new sim process
            eeval = optimizer.get_next_x()
            evals[i] = eeval  # Store eval object to be set with result later
            Process(target=simproc, args=(i, list(eeval.x))).start()

        def store_result():
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                inp = ""
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save? Y/N: ").upper()
                if inp == "Y":
                    quit_opt()
                sys.exit(0)
            else:
                if result is not None:
                    evals[i].set(result)
                if i > 0 and i % save_iter == 0:
                    save()

        self.logger.error(f"Dlib hopt for {n_sims} sims with {n_concurrent} procs"
                          f" on params {space}")
        # Spawn initial processes
        for i in range(n_concurrent):
            spawn_eval(i)
        # When a thread returns a result, start a new sim
        for i in range(n_concurrent, n_sims):
            store_result()
            spawn_eval(i)
        # Get remaining results
        for _ in range(n_concurrent):
            store_result()
        quit_opt()


def dlib_proc(stratclass, pp, space_params, result_queue, avg_runs, i, space_vals):
    logger = logging.getLogger('')
    logger.error(f"T{i} Testing {space_params}: {space_vals}")
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    if pp['epsilon'] < 2.1 and pp['epsilon_decay'] < 0.999_8:
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
