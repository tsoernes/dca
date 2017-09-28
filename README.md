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

