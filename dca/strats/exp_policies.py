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
    idx = np.random.choice(range(len(chs)), p=probs)
    ch = chs[idx]
    return ch, idx


def policy_nom_boltzmann(temp, chs, qvals_dense, cell):
    """Boltzmann selection from the nominal channels, if any, else boltzmann
    selection from the eligible channels"""
    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    if nom_elig_idxs:
        nom_qvals = qvals_dense[nom_elig_idxs]
        scaled = np.exp((nom_qvals - np.max(nom_qvals)) / temp)
        probs = scaled / np.sum(scaled)
        idx = np.random.choice(nom_elig_idxs, p=probs)
    else:
        _, idx = policy_boltzmann(temp, chs, qvals_dense)
    ch = chs[idx]
    return ch, idx


def policy_nom_boltzmann2(temp, chs, qvals_dense, cell, c=1.5):
    nom_temps_mults = np.arange(c, 1, (1 - c) / 10)
    nom_prob_consts = np.arange(10, 1, -0.1)
    nominal_eligible_idxs = []
    nom_elig_temps_mults = []
    nom_elig_prob_consts = []
    i = 0
    for ch_i, ch in enumerate(chs):
        if GF.nom_chs[cell][ch]:
            nominal_eligible_idxs.append(ch_i)
            nom_elig_temps_mults.append(nom_temps_mults[i])
            nom_elig_prob_consts.append(nom_prob_consts[i])
            i += 1

    nom_elig_idxs = _nominal_eligible_idxs(chs, cell)
    temps = np.repeat(temp, len(chs))
    temps[nom_elig_idxs] *= nom_elig_temps_mults

    scaled = np.exp((qvals_dense - np.max(qvals_dense)) / temps)
    scaled[nom_elig_idxs] += nom_elig_prob_consts
    probs = scaled / np.sum(scaled)
    idx = np.random.choice(range(len(chs)), p=probs)
    ch = chs[idx]
    # print(nom_elig_idxs, temps, qvals_dense, probs, "\n")
    return ch, idx


class BoltzmannGumbelQPolicy():
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).
    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.
    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, n_channels=70, C=200.0):
        assert C > 0, "BoltzmannGumbelQPolicy C parameter must be > 0, not " + repr(C)
        self.C = C
        self.action_counts = np.ones(n_channels)

    def select_action(self, temp, chs, qvals_dense, cell):
        assert qvals_dense.ndim == 1, qvals_dense.ndim

        beta = self.C / np.sqrt(self.action_counts[chs])
        Z = np.random.gumbel(size=qvals_dense.shape)

        perturbation = beta * Z
        perturbed_q_values = qvals_dense + perturbation
        idx = np.argmax(perturbed_q_values)
        ch = chs[idx]

        self.action_counts[ch] += 1
        return ch, idx


exp_pol_funcs = {
    'eps_greedy': policy_eps_greedy,
    'nom_greedy': policy_nom_greedy,
    'nom_eps_greedy': policy_nom_eps_greedy,
    'nom_eps_greedy2': policy_nom_eps_greedy2,
    'boltzmann': policy_boltzmann,
    'nom_boltzmann': policy_nom_boltzmann,
    'nom_boltzmann2': policy_nom_boltzmann2,
    'nom_greedy_fixed': policy_nom_greedy_fixed,
    'bgumbel': BoltzmannGumbelQPolicy
}  # yapf: disable
