# Dynamic Channel Allocation in cellular radio networks

This project implements a large variety of so-called agents for doing channel
allocation in cellular radio networks. In cellular networks, each base station
must choose (from a limited set) which radio channel to use for each service request. 
A channel cannot be in simultanous use at two nearby base stations without causing 
interference. If a mobile caller requests service from a base station but there
are no channels that can be used without interference,
then the call must be blocked.
The objective of the channel allocation policy (henceforth "agent") is to 
minimize the number of blocked calls.
![A 7x7 grid of cells](docs/axial-rhombus-grid.png)

See also the version written in [Rust](https://github.com/tsoernes/rustdca)
and [Haskell](https://github.com/tsoernes/haskelldca).

## Example usage

The agent presented in 
[Torstein Sørnes 2018](
https://brage.bibsys.no/xmlui/bitstream/handle/11250/2562774/19523_FULLTEXT.pdf),
provides state-of-the-art results for cellular networks with a centralized agent.
It uses TDC gradients with a state-value linear neural network with differential returns 
and hand-off look-ahead (HLA). To run it, with 10 Erlangs call traffic 
and 15% chance of hand-off:

`python3 main.py tftdcsinghnet --hoff_lookahead --erlangs 10 --p_handoff 0.15`

In general:

`python3 main.py <agent_name> --long_option_name -short_option_name`

To see the options used for the plots in the thesis above, see `plotscript.sh`.

Listed below are some of the features and agent implementations. Run `python3
main.py --help` for comprehensive list and default options.
## Non-learning agents
See `fixedstrats.py`
- Fixed Assign (`fixedassign`)
- Random Assign (`randomassign`)
- Fixed channel preference, random if none available (`fixedrandomassign`)

## RL agents
### Features and implementations
Target types: 
- MDP Average reward (e.g. `-target avg -wbeta 0.01`)
- MDP Discounted reward (e.g. `-target discount --gamma 0.8`)
- SMDP Discounted reward (e.g. `-target discount -bdisc --beta 40 -rtype smdp_callcount`)
- RSMART (`-target avg_rsmart`)

Reward types:
- +1 on accepted call, -1 on blocked call (`-rtype new_block`)
- Call count (`-rtype callcount`)
- Call count integrated over time (`-rtype smdp_callcount`)

Two step look-ahead on hand-offs (`-hla`)

Exploration strategies (`exp_policies.py`):
- Greedy (`-epol greedy`)
- Epsilon-greedy (e.g. `-epol eps_greedy -eps 0.5`)
- Boltzmann (e.g. `-epol boltzmann -eps 2`)
- Boltzmann-Gumbel (`-epol bgumbel`)
- Fixed nominal channel preference (`-epol nom_fixed_greedy`)
- Greedy nominal channel preference (`-epol nom_greedy`)
- Boltzmann with nominal channel preference (`-epol nom_boltzmann`)

# State Value Nets
Using Singh and Bertsekas 1997 paper as base implementation

Different gradients/RL methods:
- True Online TD Lambda (`tdlsinghnet`, `singh_tdl.py`)
- GTD2 (`gtd2singhnet`, `singh_gtd2.py`)
- TDC - TD0 with Gradient correction (`tdcsinghnet` for `singh_tdc.py`, `tftdcsinghnet` for `singh_tdc_tf.py` which is equivalent version implemented in all TensorFlow)
- TDC-NL - TD0 with Gradient correction for Non-linear func. approx (`singh_tdc_nl.py`)
- Naive Residual Gradients - (`singh_resid.py`)
- LSTD - Least Squares Temporal Difference (`singh_lstd.py`)

Feature representations (see `gridfuncs_numba.py`)
- As in Singh97 (`-ftype vanilla`)
- As in Singh97, also include number of used chs with dist 3 or less (`-ftype big`)
- Separate features for number of used chs with dist 4, with dist 3, .. (`-ftype big2`)

# State-Action Value Nets
RL Methods:
- Q-Learning (`qlearnnet`)
- SARSA (`sarsanet`)
- Update towards _eligible_ max action (`qlearneligiblenet`)

Options:
- Use n\_channels X n\_cells outputs (`--bighead`) instead of default where cell is input to network and output is of size n\_channels_
- Dueling Q-Net (`-duel`)
- Double Q-Net (e.g. `--net_copy_iter 50 --net_creep_tau 0.1`)
- Experience replay (e.g. `--batch_size 10 --buffer_size 5000`)

Other:
- N-step returns (e.g. `nqlearnnet --n_step 4`)
- GAE (Generalized Advantage Estimator) returns (`gaeqlearnnet`)
- Feature representation as network input (raw grid is default) (`--qnet_freps` or `--qnet_freps_only`)
- ++

# State-Action Table Lookup
See `table_rl.py` for implementation
- Lilith SARSA (`sarsa`)
- Lilith Table-Trimmed SARSA (`tt_sarsa`)
- Lilith Reduced-State SARSA (`rs_sarsa`)
- All of the above accept lambda returns, (e.g. `rs_sarsa --lambda 0.8`)
- Zap-Q learning with RS-SARSA feature rep. (`zapq`) (Warning: __Very_ SLOW!)

## Misc
Extensive scaffolding for hyperparameter testing using either Dlib (see `runners/dlib_runner.py`) or Hyperopt (see `runners/hopt_runner.py`)
- Parallel optimization processes
- Save and resume optimization results and parameters to file or database (MongoDB, for Hyperopt only)
- Limit GPU usage to N<M of the simultaneous processes M, use CPU only for remaining M-N processes
- Run N<=M processes with same hyperparameters but different random seeds and take average, with a total of M concurrent processes

Average over multiple runs (e.g. `python3 rs_sarsa --avg_runs 8`, see `runners/avg_runner.py`)

Exploration comparisons (`e.g. python3 rs_sarsa --exp_policy_cmp N`, see `runners/exp_pol_runner.py`)
- Compare different exploration strategies, each using a different range of exploration parameters (e.g. epsilon)
- For each exploration strategy and parameter choice, average over up to N runs if all runs so far have yielded block. prob. less than threshold (e.g. `--breakout_thresh 0.15`)
 
