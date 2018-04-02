import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count

import numpy as np

from runners.runner import Runner


class ExpPolRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        pols = {
            'boltzmann': {'epsilon': [2, 5, 10]},
            # 'nom_boltzmann': {'epsilon': [2, 5, 10]},
            # 'eps_greedy': {'epsilon': [0.0, 0.2, 0.4, 0.7]},
            # 'nom_eps_greedy': {'epsilon': [0.1, 0.4, 0.7]},
            # 'nom_greedy': {'epsilon': [0]},
            # 'nom_fixed_greedy': {'epsilon': [0]},
            # 'bgumbel': {'epolc': [4.5, 5.0, 5.5]}
        }  # yapf: disable
        space, results = [], []
        for pol, polparams in pols.items():
            for param, pvals in polparams.items():
                for pval in pvals:
                    space.append({'exp_policy': pol, param: pval})
                    results.append({'exp_policy': pol, param: pval, 'results': []})
        n_avg = self.pp['exp_policy_cmp']
        n_concurrent = cpu_count() // 2
        self.logger.error(f"Space: {space}\n navg:{n_avg}, n_concurrent:{n_concurrent}")
        result_queue = Queue()
        simproc = partial(exp_proc, self.stratclass, self.pp, result_queue)

        def spawn_eval(i):
            Process(target=simproc, args=(i, space[i])).start()

        def store_result():
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                print(results)
                sys.exit(0)
            else:
                if result is not None:
                    results[i]['results'].append(result)

        for i in range(n_concurrent):
            j = i % len(space)
            spawn_eval(j)
        for i in range(n_concurrent, n_avg * len(space)):
            j = i % len(space)
            store_result()
            spawn_eval(j)
        for _ in range(n_concurrent):
            store_result()
        for evaluation in results:
            evaluation['avg_result'] = np.mean(evaluation['results'])
        self.logger.error(results)


def exp_proc(stratclass, pp, result_queue, i, space):
    logger = logging.getLogger('')
    logger.error(f"T{i} Testing {space}")
    for param, paramval in space.items():
        pp[param] = paramval
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
