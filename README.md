# fordyp
Dynamic Channel Allocation using different strategies, such as Fixed Assignment and Reinforcement Learning (SARSA)

BIO-INSPIRED: Deep Reinforcement Learning for Dynamic Channel Allocation
In cellular telephone networks, each new caller must be allocated a channel
(within the provider's frequency band) that does not interfere with those
allocated in nearby areas.  This quickly becomes a serious problem when
many callers congregate in particular areas, such as entertainment venues.

Reinforcement Learning (RL) has previously been applied to this problem:

Singh and Bertsekas (1997).  "Reinforcement Learning for Dynamic Channel
Allocation in Cellular Telephone Systems"" in Advances in Neural
Information Processing Systems (NIPS), pp. 974-980.

However, in that work, the critical features given to the learning system were determined by hand.
The goal of this project is to allow a deep neural network to discover
some of the key features needed to support RL.  In general, the system will
receive a more raw form of data than that used by Singh and Bertsekas and
then exploit Deep Learning's ability to discover salient patterns (a.k.a. features).

This project will use Google's Tensorflow system for Deep Learning and must
combine it with a reliable RL system and a simple simulator for multi-channel
telephone networks.

## Features and implementations:
Reward types: 
- MDP Average reward
- MDP Discounted reward
- RSMART
- SMDP Discounted reward


# Value nets
Using Singh and Bertsekas 1997 paper (see above) as base

Different gradients/targets
True Online TD Lambda (`singh_tdl.py`)
GTD2 (`singh_gtd2.py`)
TDC - TD0 with Gradient correction (`singh_tdc.py`, `singh_tdc_tf.py` for full TF version)
TDC-NL - TD0 with Gradient correction for Non-linear func. approx (`singh_tdc_nl.py`)
Naive Residual Gradients - (`singh_resid.py`)
LSTD - Least Squares Temporal Difference (`singh_lstd.py`)
