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

## Non-learning agents:
See `fixedstrats.py`
- Fixed Assign (`fixedassign`)
- Random Assign (`randomassign`)
- Fixed channel preference, random if none available (`fixedrandomassign`)

## Features and implementations:
Some of the features and agent implementations. Run `python3 main.py --help` for comprehensive list.

Target types: 
- MDP Average reward (`-target avg -wbeta 0.01`)
- MDP Discounted reward (`-target discount --gamma 0.8`)
- SMDP Discounted reward (`-target discount -bdisc --beta 40 -rtype smdp_callcount`)
- RSMART (`-target avg_rsmart`)

Reward types:
- +1 on accepted call, -1 on blocked call (`-rtype new_block`)
- Callcount (`-rtype callcount`)
- Callcount integrated over time (`-rtype smdp_callcount`)

Two step look-ahead on hand-offs (`-hla`)

Exploration strategies (`exp_policies.py`):
-Epsilon-greedy (`-epol eps_greedy -eps 0.5`)
-Boltzmann (`-epol boltzmann -eps 2`)
-Boltzmann-Gumbel (`-epol bgumbel`)
-Fixed nominal channel preference (`-epol nom_fixed_greedy`)
-Greedy nominal channel preference (`-epol nom_greedy`)
-Boltzmann with nominal channel preference (`-epol nom_boltzmann`)

# State Value nets
Using Singh and Bertsekas 1997 paper (see above) as base

Different gradients/RL methods:
-True Online TD Lambda (`singh_tdl.py`)
-GTD2 (`singh_gtd2.py`)
-TDC - TD0 with Gradient correction (`singh_tdc.py`, `singh_tdc_tf.py` for full TF version)
-TDC-NL - TD0 with Gradient correction for Non-linear func. approx (`singh_tdc_nl.py`)
-Naive Residual Gradients - (`singh_resid.py`)
-LSTD - Least Squares Temporal Difference (`singh_lstd.py`)

Feature representations (see `gridfuncs_numba.py`)
-As in Singh97 (`-ftype vanilla`)
-As in Singh97, include number of used chs with dist 3 or less (`-ftype big`)
-Number of used chs with dist 4, with dist 3, .. (`-ftype big2`)

# State-Action Value nets
RL Methods:
- Q-Learning (`qlearnnet`)
- SARSA (`sarsanet`)
- Update towards _eligible_ max action (``)

Options:
- n\_channels X n\_cells outputs (`--bighead`)
- Dueling Q-Net (`-duel`)
- Double Q-Net (`--net_copy_iter 50 --net_creep_tau 0.1`)
- Experience replay (`--batch_size 10 --buffer_size 5000`)

# State-Action Table lookup
See `table_rl.py`
- Lilith SARSA (`sarsa`)
- Lilith Table-Trimmed SARSA (`tt_sarsa`)
- Lilith Reduced-State SARSA (`rs_sarsa`)
- All of the above accept lambda returns, e.g. (`rs_sarsa --lambda 0.8`)
- Zap-Q learning (`zapq`) (Warning: SLOW!)
