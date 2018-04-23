import random

import numpy as np

np.random.seed(0)

# 5 room example
# THIS IS THE REWARD MATRIX.. R(s,a); Dimension is S x A = number of states
# x number of actions


def ChooseAction(x, noise):
    """This function is used to choose a randomized feasible action, given the
    current state. Also return next state given action and current state."""

    r = random.random()
    if x == 0:
        if (r < 0.5):
            a = 0
        else:
            a = 4
    if x == 1:
        if (r < (1 / 3)):
            a = 1
        elif (r < (2 / 3)):
            a = 3
        else:
            a = 5
    if x == 2:
        if (r < 0.5):
            a = 2
        else:
            a = 3
    if x == 3:
        if (r < 0.25):
            a = 1
        elif (r < 0.5):
            a = 2
        elif (r < 0.75):
            a = 3
        else:
            a = 4
    if x == 4:
        if (r < 0.25):
            a = 0
        elif (r < 0.5):
            a = 3
        elif (r < 0.75):
            a = 4
        else:
            a = 5
    if x == 5:
        if (r < (1 / 3)):
            a = 1
        elif (r < (2 / 3)):
            a = 4
        else:
            a = 5

    rr = random.random()
    if (rr > noise):
        xtp1 = a
    else:
        rrr = random.random()
        if x == 0:
            if (rrr < 0.5):
                xtp1 = 0
            else:
                xtp1 = 4
        if x == 1:
            if (rrr < (1 / 3)):
                xtp1 = 1
            elif (rrr < (2 / 3)):
                xtp1 = 3
            else:
                xtp1 = 5
        if x == 2:
            if (rrr < 0.5):
                xtp1 = 2
            else:
                xtp1 = 3
        if x == 3:
            if (rrr < 0.25):
                xtp1 = 1
            elif (rrr < 0.5):
                xtp1 = 2
            elif (rrr < 0.75):
                xtp1 = 3
            else:
                xtp1 = 4
        if x == 4:
            if (rrr < 0.25):
                xtp1 = 0
            elif (rrr < 0.5):
                xtp1 = 3
            elif (rrr < 0.75):
                xtp1 = 4
            else:
                xtp1 = 5
        if x == 5:
            if (rrr < (1 / 3)):
                xtp1 = 1
            elif (rrr < (2 / 3)):
                xtp1 = 4
            else:
                xtp1 = 5

    xtp1at = [xtp1, a]
    return xtp1at


# This is the reward function; It is of dimensions X x A = 6 x 6; Its (x,a) entry corresponds to
# the reward for being in state x and taking action a
R = np.array([[0, -100, -100, -100, -5, -100], [-100, 0, -100, -5, -100, 100],
              [-100, -100, 0, -5, -100, -100], [-100, -5, -5, 0, -100, -100],
              [-5, -100, -100, -5, 0, 100], [-100, -5, -100, -100, -5, 100]])

T = 100_000  # Number of iterations for learning (number of data samples we observe)
T = 100

xt = 4
# start in state 0
beta = 0.8  # Discount factor
rho = 0.85  # Stepsize \gamma_n = (1/n)^\rho
noise = 0.2  # Probability of not going in the direction you want to go; Noisy transitions;

# Number of actions in each state; Ex: 2 = number of actions in state 0.
AlenVec = np.array([2, 3, 2, 4, 4, 3])

# We need a mapping from state-action pair to a number between 1 and 18.
# E.g. [State 0, Action 0] corresponds to the zeroeth state-action pair,
# [State 0, Action 4] corresponds to the first state-action pair.
# Location of the state-action pair indicates the unique number which represents this pair
# (e.g. [5 5] = 17)
NumbertoSAPair = np.array([[0, 0], [0, 4], [1, 1], [1, 3], [1, 5], [2, 2], [2, 3], [3, 1],
                           [3, 2], [3, 3], [3, 4], [4, 0], [4, 3], [4, 4], [4, 5], [5, 1],
                           [5, 4], [5, 5]])

# Inverse mapping: Given state-action pair, what is the curresponding number:
SAPairtoNumber = -1 * np.ones((6, 6), dtype=np.int32)

for i in range(len(NumbertoSAPair)):
    SAPairtoNumber[NumbertoSAPair[i, 0], NumbertoSAPair[i, 1]] = i

StaActLen = np.sum(AlenVec)  # Total number of state action pairs

# Initialize the Matrix gain for Zap Q
A_ZapQ = 10 * np.eye(StaActLen)  # d x d matrix

# Initialize the parameter vector for Zap Q
Qthetat_ZapQ = np.random.rand(StaActLen)  # d vector


def onehot(i):
    arr = np.zeros(StaActLen)
    arr[i] = 1
    return arr


def onehot2D(i):
    arr = np.zeros((StaActLen, StaActLen))
    arr[i] = 1
    return arr


for t in range(T):
    # ChooseAction will return the random action chosen in the current state AND the next state
    xtp1, at = ChooseAction(xt, noise)

    alphat = 1. / (t + 2)  # stepsize for parameter recursion
    gammat = np.power((t + 1), -rho)  # stepsize for matrix gain recursion

    num_sa_pair = SAPairtoNumber[xt][at]  # Mapping [S A] -> the curresponding number

    # Watkins' basis: Column vector with 1 at (x,a) and 0 elsewhere.
    # Basis are indicator functions
    _psixtat = onehot(num_sa_pair)

    # Watkins' basis evaluated at next state and the optimal policy.
    # Takes value 1 at state-action pairs corresponding to xtp1,
    # and feasible actions
    psixtp1 = np.zeros(StaActLen)
    for jj in range(6):
        if SAPairtoNumber[xtp1, jj] >= 0:  # Is the action feasible?
            psixtp1[SAPairtoNumber[xtp1, jj]] = 1  # Take value 1

    # Zap Q-learning step:
    # Q(X_t,A_t):
    Qxt_Zap = Qthetat_ZapQ[num_sa_pair]

    # Q(X_t+1,a) with all feasible a's; Need to take minimum over all 'a' later:
    Qxtp1_Zap = Qthetat_ZapQ * psixtp1

    # Optimal action for state xtp1, given the current parameter estimate:
    OptAct_xtp1 = np.argmax(Qxtp1_Zap)
    # Q-value at for xtp1 and the optimal action
    max_Qxtp1_Zap = Qxtp1_Zap[OptAct_xtp1]

    # Basis function evaluated at xtp1 and the corresponding optimal action
    psixtp1_OptAct = np.zeros(StaActLen)
    psixtp1_OptAct[OptAct_xtp1] = 1

    outer1 = onehot2D((num_sa_pair, num_sa_pair))
    outer2 = onehot2D((num_sa_pair, OptAct_xtp1))
    # Zap Q-learning begins here:
    # Estimating the A(\theta_t) matrix:
    A_ZapQ += gammat * ((-outer1 + beta * outer2) - A_ZapQ)
    Ainv_ZapQ = np.linalg.pinv(A_ZapQ)

    # Q update for SNR 2 (a) gain:
    a_inv_dot = Ainv_ZapQ[:, num_sa_pair]
    Qthetat_ZapQ -= alphat * a_inv_dot * (R[xt, at] + beta * max_Qxtp1_Zap - Qxt_Zap)

    xt = xtp1

# print(A_ZapQ, Qthetat_ZapQ)
print(A_ZapQ.shape, Qthetat_ZapQ.shape)
