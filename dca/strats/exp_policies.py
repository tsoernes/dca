import numpy as np

from gridfuncs import GF


def _nominal_eligible_chs(chs, cell):
    nominal_eligible_chs = []
    for ch in chs:
        if GF.nom_chs[cell][ch]:
            nominal_eligible_chs.append(ch)
    return nominal_eligible_chs


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
        nom_elig = _nominal_eligible_chs(chs, cell)
        if nom_elig:
            ch = np.random.choice(nom_elig)
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
        # Choose at random, but prefer nominal channels
        nom_elig = _nominal_eligible_chs(chs, cell)
        if nom_elig:
            idx = np.argmax(nom_elig)
            ch = chs[idx]
        else:
            ch = np.random.choice(chs)
    else:
        # Choose greedily
        idx = np.argmax(qvals_dense)
        ch = chs[idx]
    return ch


def policy_boltzmann(temp, chs, qvals_dense, *args):
    scaled = np.exp((qvals_dense - np.max(qvals_dense)) / temp)
    probs = scaled / np.sum(scaled)
    ch = np.random.choice(chs, p=probs)
    return ch


def policy_nom_boltzmann(temp, chs, qvals_dense, cell):
    nom_elig = _nominal_eligible_chs(chs, cell)
    if nom_elig:
        ch = policy_boltzmann(temp, nom_elig, qvals_dense)
    else:
        ch = policy_boltzmann(temp, chs, qvals_dense)
    return ch


exp_pol_funcs = {
    'eps_greedy': policy_eps_greedy,
    'nom_eps_greedy': policy_nom_eps_greedy,
    'nom_eps_greedy2': policy_nom_eps_greedy2,
    'boltzmann': policy_boltzmann,
    'nom_boltzmann': policy_nom_boltzmann
}  # yapf: disable
