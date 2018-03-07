import numpy as np

from gridfuncs import GF


def _nominal_eligible_idxs(chs, cell):
    """Return the indecies of 'chs' which correspond to nominal channels in 'cell'"""
    nominal_eligible_idxs = [i for i, ch in enumerate(chs) if GF.nom_chs[cell][ch]]
    return nominal_eligible_idxs


def policy_eps_greedy(epsilon, chs, qvals_dense, *args):
    """Epsilon greedy action selection with expontential decay"""
    if np.random.random() < epsilon:
        # Choose an eligible channel at random
        idx = np.random.randint(0, len(chs))
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_nom_eps_greedy(epsilon, chs, qvals_dense, cell):
    """Epsilon greedy where exploration actions are selected randomly from
    the cells nominal channels, if possible"""
    if np.random.random() < epsilon:
        # Choose at random, but prefer nominal channels
        nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
        if nom_elig_idxs:
            idx = np.random.choice(nom_elig_idxs)
        else:
            idx = np.random.randint(0, len(chs))
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_nom_eps_greedy2(epsilon, chs, qvals_dense, cell):
    """Epsilon greedy where exploration actions are selected greedily from
    the cells nominal channels, if possible"""
    if np.random.random() < epsilon:
        # Choose at greedily from nominal channels, else at random
        nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
        if nom_elig_idxs:
            nidx = np.argmax(qvals_dense[nom_elig_idxs])
            idx = nom_elig_idxs[nidx]
        else:
            idx = np.random.randint(0, len(chs))
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_nom_greedy(epsilon, chs, qvals_dense, cell):
    """Channel is always greedily selected from the cells nominal channels, if one
    is available, else greedily from the eligible channels.
    """
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        nidx = np.argmax(qvals_dense[nom_elig_idxs])
        idx = nom_elig_idxs[nidx]
    else:
        idx = np.argmax(qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_nom_greedy_fixed(temp, chs, qvals_dense, cell):
    """The lowest numbered nominal channel is selected, if any, else greedy selection"""
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        idx = nom_elig_idxs[0]
    else:
        idx = np.argmax(qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_boltzmann(temp, chs, qvals_dense, *args):
    """Boltzmann selection"""
    scaled = np.exp((qvals_dense - np.max(qvals_dense)) / temp)
    probs = scaled / np.sum(scaled)
    idx = np.random.choice(range(chs), p=probs)
    ch = chs[idx]
    return ch, idx


def policy_nom_boltzmann(temp, chs, qvals_dense, cell):
    """Boltzmann selection from the nominal channels, if any, else boltzmann
    selection from the eligible channels"""
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        nom_qvals = qvals_dense[nom_elig_idxs]
        idx = policy_boltzmann(temp, nom_elig_idxs, nom_qvals)
    else:
        _, idx = policy_boltzmann(temp, chs, qvals_dense)
    ch = chs[idx]
    return ch, idx


exp_pol_funcs = {
    'eps_greedy': policy_eps_greedy,
    'nom_greedy': policy_nom_greedy,
    'nom_eps_greedy': policy_nom_eps_greedy,
    'nom_eps_greedy2': policy_nom_eps_greedy2,
    'boltzmann': policy_boltzmann,
    'nom_boltzmann': policy_nom_boltzmann,
    'nom_greedy_fixed': policy_nom_greedy_fixed
}  # yapf: disable
