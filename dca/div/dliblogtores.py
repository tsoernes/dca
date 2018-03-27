import pickle

import numpy as np

FNAME_IN = "avgres.txt"
FNAME_OUT = "avgres.pkl"
N_PARAMS = 5

with open(FNAME_IN) as f:
    lines = f.readlines()
no_blank = list(filter(lambda l: l != '\n', lines))
no_block = list(filter(lambda l: not l.startswith("Block"), no_blank))


def get_iter(entry):
    return int(entry[1:entry.find(" ")])


block_probs = []  # block prob strings
params = []  # param strings
for r in no_block:
    if r.endswith("handoffs\n"):
        block_probs.append(r)
    else:
        params.append(r)
# Sort by iter
block_probs = sorted(block_probs, key=get_iter)
params = sorted(params, key=get_iter)
# Some trials may not have finished
params = params[:len(block_probs)]
results = np.zeros(shape=(len(block_probs), N_PARAMS + 1), dtype=np.float64)

# Gather block probs
min_bp = 1
for i, res in enumerate(block_probs):
    idx = res.find('.')
    bp = np.float64(res[idx - 1:idx + 5])
    results[i][0] = -bp  # dlib maxes, must negate block prob

# Gather params
for i, res in enumerate(params):
    idx = res.find("[", 20)
    if idx == -1:
        print(i, res)
    else:
        vals = list(map(np.float64, res[idx + 1:-2].split(", ")))
        results[i][1:] = vals

prev_best = results[np.argmax(results, axis=0)[0]]
print(prev_best)

with open(FNAME_OUT, "wb") as fb:
    pickle.dump(results, fb)
