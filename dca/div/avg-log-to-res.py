"""
Read a log file, parse the results and save to pickle file.
Useful if computer crashes during run.
"""

import pickle
from operator import itemgetter

import numpy as np

FNAME_IN = "exppol.txt"

with open(FNAME_IN) as f:
    lines = f.readlines()
no_blank = list(filter(lambda l: l != '\n', lines))


def get_iter(entry):
    return int(entry[1:entry.find(" ")])


block_probs = []  # block prob strings
params = []  # param strings
for r in no_blank:
    if r.endswith("}\n"):
        params.append(r)
    else:
        block_probs.append(r)
# Sort by iter
block_probs = sorted(block_probs, key=get_iter)
params = sorted(params, key=get_iter)

avg_descs = ['avg', 'avg_h', 'avg_t']
pols = {
    'boltzmann': {'epsilon': [2.2, 4, 6]},
    'nom_boltzmann': {'epsilon': [2.2, 4, 6]},
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

for bp in block_probs:
    # T0 Blocking probability: 0.1509 for new calls, 0.0744 for handoffs, 0.1423 total
    # T146 Block prob threshold exceeded at 0.1605; breaking out early
    j = get_iter(bp) % len(space)
    if bp.endswith("early\n") or bp.endswith("calls\n"):
        results[j]['btresh'] = True
    else:
        bps = []
        for k, c in enumerate(bp):
            if c == '.':
                try:
                    bps.append(float(bp[k:k + 5]))
                except:
                    print(bp)
                    print(bp[k:k + 5])
                    raise
        assert len(bps) == 3, (bps, bp)
        results[j]['results'].append(bps)

# T13 Testing {'pol': 'nom_eps_greedy', 'epsilon': 0.7}


def pprint(rr):
    return ", ".join([f"{r:.4f}" for r in rr])


def print_results():
    for evaluation in results:
        res = np.array(evaluation['results'])
        # If the first run with a set of params fails, there won't be any
        # results and 'res' will be dim 1
        if len(res.shape) > 1:
            for avg_typ in range(res.shape[1]):
                evaluation[avg_descs[avg_typ]] = f"{np.mean(res[:, avg_typ]):.4f}"
            evaluation['results'] = list(map(pprint, evaluation['results']))
        else:
            for avg_typ in avg_descs:
                evaluation[avg_typ] = "1"
    params_and_res = [{**p, **r} for p, r in zip(space, results)]
    print("\n".join(map(repr, params_and_res)))
    best = min(params_and_res, key=itemgetter('avg'))
    print(f"Best new call:\n{best}")
    if True:
        best_h = min(params_and_res, key=itemgetter('avg_h'))
        best_t = sorted(params_and_res, key=itemgetter('avg_t'))
        print(f"Best handoff:\n{best_h}")
        best_tot = "\n".join(map(repr, best_t[:5]))
        print(f"Best 5 total:\n{best_tot}")


print_results()
