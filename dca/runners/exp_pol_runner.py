import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count
from operator import itemgetter, neg

import numpy as np

from runners.runner import Runner


class ExpPolRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        include_hoffs = self.pp['p_handoff'] > 0
        pols = {
            'boltzmann': {'epsilon': [2, 4, 6]},
            'nom_boltzmann': {'epsilon': [2, 4, 6]},
            'eps_greedy': {'epsilon': [0.0, 0.2, 0.4, 0.7]},
            'nom_eps_greedy': {'epsilon': [0.0, 0.2, 0.4, 0.7]},
            'eps_nom_greedy': {'epsilon': [0.0, 0.2, 0.4, 0.7]},
            'nom_greedy': {'epsilon': [0]},
            'nom_fixed_greedy': {'epsilon': [0]},
            'bgumbel': {'exp_policy_param': [4.0, 4.5, 5.0, 5.5, 6.0]}
        }  # yapf: disable
        space, results = [], []
        for pol, polparams in pols.items():
            for param, pvals in polparams.items():
                for pval in pvals:
                    space.append({'pol': pol, param: pval})
                    results.append({'btresh': False, 'results': []})
        n_avg = self.pp['exp_policy_cmp']
        # n_concurrent = cpu_count() // 2
        n_concurrent = cpu_count() - 1
        self.logger.error(
            f"Running {n_concurrent} concurrent procs with {n_avg} average runs "
            f"for up to {n_avg*len(space)} sims on space:\n{pols}")
        result_queue = Queue()
        simproc = partial(exp_proc, self.stratclass, self.pp, result_queue)

        # If the first run of a set of params exceeds block prob, there's
        # no need to run multiple of them and take the average.
        def pprint(rr):
            return ", ".join([f"{-r:.4f}" for r in rr])

        avg_descs = ['avg', 'avg_h', 'avg_t']

        def print_results():
            for evaluation in results:
                res = np.array(evaluation['results'])
                for avg_typ in range(res.shape[1]):
                    evaluation[avg_descs[avg_typ]] = f"{-np.mean(res[:, avg_typ]):.4f}"
                evaluation['results'] = list(map(pprint, evaluation['results']))
            params_and_res = [{**p, **r} for p, r in zip(space, results)]
            self.logger.error("\n".join(map(repr, params_and_res)))
            best = min(params_and_res, key=itemgetter('avg'))
            self.logger.error(f"Best:\n{best}")
            if include_hoffs:
                best_h = min(params_and_res, key=itemgetter('avg_h'))
                best_t = sorted(params_and_res, key=itemgetter('avg_t'))
                self.logger.error(f"Best handoff:\n{best_h}")
                best_tot = "\n".join(map(repr(best_t[:5])))
                self.logger.error(f"Best 5 total:\n{best_tot}")

        def spawn_eval(i):
            j = i % len(space)
            if not results[j]['btresh']:
                Process(target=simproc, args=(i, space[j])).start()
                return True
            return False

        def store_result():
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                print_results()
                sys.exit(0)
            else:
                j = i % len(space)
                if result is None:
                    results[j]['btresh'] = True
                else:
                    results[j]['results'].append(result)

        for i in range(n_concurrent - 1):
            spawn_eval(i)
        for i in range(n_concurrent - 1, n_avg * len(space)):
            did_spawn = spawn_eval(i)
            if did_spawn:
                store_result()
        for _ in range(n_concurrent - 1):
            store_result()
        print_results()


def exp_proc(stratclass, pp, result_queue, i, space):
    logger = logging.getLogger('')
    logger.error(f"T{i} Testing {space}")
    for param, paramval in space.items():
        pp[param] = paramval
    np.random.seed()
    strat = stratclass(pp=pp, logger=logger, pid=i)
    res = strat.simulate()
    # if strat.quit_sim and not strat.invalid_loss and not strat.exceeded_bthresh:
    if strat.quit_sim:
        # If user quits sim or sim exceeded block thresh, don't store result
        res = None
    else:
        assert res is not None and res[0] is not None
        # Must negate result as dlib performs maximization by default
        res = tuple(map(neg, res))
    result_queue.put((i, res))
