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
        ch = np.random.choice(chs)
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_nom_eps_greedy(epsilon, chs, qvals_dense, cell):
    """Epsilon greedy where exploration actions are selected randomly from
    the cells nominal channels, if possible"""
    if np.random.random() < epsilon:
        # Choose at random, but prefer nominal channels
        nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
        if nom_elig_idxs:
            idx = np.random.choice(nom_elig_idxs)
            ch = chs[idx]
        else:
            ch = np.random.choice(chs)
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_nom_eps_greedy2(epsilon, chs, qvals_dense, cell):
    """Epsilon greedy where exploration actions are selected greedily from
    the cells nominal channels, if possible"""
    if np.random.random() < epsilon:
        # Choose at greedily from nominal channels, else at random
        nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
        if nom_elig_idxs:
            idx = np.argmax(qvals_dense[nom_elig_idxs])
            ch = chs[nom_elig_idxs[idx]]
        else:
            ch = np.random.choice(chs)
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_nom_greedy(epsilon, chs, qvals_dense, cell):
    """Channel is always greedily selected from the cells nominal channels, if one
    is available, else greedily from the eligible channels.
    """
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        idx = np.argmax(qvals_dense[nom_elig_idxs])
        ch = chs[nom_elig_idxs[idx]]
    else:
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_nom_greedy_fixed(temp, chs, qvals_dense, cell):
    """The lowest numbered nominal channel is selected, if any, else greedy selection"""
    nom_elig = [ch for ch in chs if GF.nom_chs[cell][ch]]
    if nom_elig:
        ch = nom_elig[0]
    else:
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_boltzmann(temp, chs, qvals_dense, *args):
    """Boltzmann selection"""
    scaled = np.exp((qvals_dense - np.max(qvals_dense)) / temp)
    probs = scaled / np.sum(scaled)
    ch = np.random.choice(chs, p=probs)
    return ch


def policy_nom_boltzmann(temp, chs, qvals_dense, cell):
    """Boltzmann selection from the nominal channels, if any, else boltzmann
    selection from the eligible channels"""
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        nom_qvals = qvals_dense[nom_elig_idxs]
        idx = policy_boltzmann(temp, nom_elig_idxs, nom_qvals)
        ch = chs[idx]
    else:
        ch = policy_boltzmann(temp, chs, qvals_dense)
    return ch


exp_pol_funcs = {
    'eps_greedy': policy_eps_greedy,
    'nom_greedy': policy_nom_greedy,
    'nom_eps_greedy': policy_nom_eps_greedy,
    'nom_eps_greedy2': policy_nom_eps_greedy2,
    'boltzmann': policy_boltzmann,
    'nom_boltzmann': policy_nom_boltzmann,
    'nom_greedy_fixed': policy_nom_greedy_fixed
}  # yapf: disable
