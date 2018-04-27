## Dynamic Channel Allocation with Reinforcement Learning
Dynamic Channel Allocation using different strategies, such as Fixed Assignment and Reinforcement Learning. 

Example usage, TDC gradient value net on simulation with 10 Erlangs and 15% chance of hand-offs using hand-off look-ahead:
`python3 main.py tftdcsinghnet --hoff_lookahead --erlangs 10 --p_handoff 0.15`

# Non-learning agents
See `fixedstrats.py`
- Fixed Assign (`fixedassign`)
- Random Assign (`randomassign`)
- Fixed channel preference, random if none available (`fixedrandomassign`)

## RL agents
# Features and implementations
Some of the features and agent implementations. Run `python3 main.py --help` for comprehensive list.

Target types: 
- MDP Average reward (`-target avg -wbeta 0.01`)
- MDP Discounted reward (`-target discount --gamma 0.8`)
- SMDP Discounted reward (`-target discount -bdisc --beta 40 -rtype smdp_callcount`)
- RSMART (`-target avg_rsmart`)

Reward types:
- +1 on accepted call, -1 on blocked call (`-rtype new_block`)
- Call count (`-rtype callcount`)
- Call count integrated over time (`-rtype smdp_callcount`)

Two step look-ahead on hand-offs (`-hla`)

Exploration strategies (`exp_policies.py`):
- Greedy (`-epol greedy`)
- Epsilon-greedy (`-epol eps_greedy -eps 0.5`)
- Boltzmann (`-epol boltzmann -eps 2`)
- Boltzmann-Gumbel (`-epol bgumbel`)
- Fixed nominal channel preference (`-epol nom_fixed_greedy`)
- Greedy nominal channel preference (`-epol nom_greedy`)
- Boltzmann with nominal channel preference (`-epol nom_boltzmann`)

# State Value Nets
Using Singh and Bertsekas 1997 paper (see above) as base

Different gradients/RL methods:
- True Online TD Lambda (`singh_tdl.py`)
- GTD2 (`singh_gtd2.py`)
- TDC - TD0 with Gradient correction (`singh_tdc.py`, `singh_tdc_tf.py` for full TF version)
- TDC-NL - TD0 with Gradient correction for Non-linear func. approx (`singh_tdc_nl.py`)
- Naive Residual Gradients - (`singh_resid.py`)
- LSTD - Least Squares Temporal Difference (`singh_lstd.py`)

Feature representations (see `gridfuncs_numba.py`)
- As in Singh97 (`-ftype vanilla`)
- As in Singh97, include number of used chs with dist 3 or less (`-ftype big`)
- Number of used chs with dist 4, with dist 3, .. (`-ftype big2`)

# State-Action Value Nets
RL Methods:
- Q-Learning (`qlearnnet`)
- SARSA (`sarsanet`)
- Update towards _eligible_ max action (``)

Options:
- n\_channels X n\_cells outputs (`--bighead`)
- Dueling Q-Net (`-duel`)
- Double Q-Net (`--net_copy_iter 50 --net_creep_tau 0.1`)
- Experience replay (`--batch_size 10 --buffer_size 5000`)

# State-Action Table Lookup
See `table_rl.py`
- Lilith SARSA (`sarsa`)
- Lilith Table-Trimmed SARSA (`tt_sarsa`)
- Lilith Reduced-State SARSA (`rs_sarsa`)
- All of the above accept lambda returns, e.g. (`rs_sarsa --lambda 0.8`)
- Zap-Q learning (`zapq`) (Warning: SLOW!)

## Misc
Extensive scaffolding for hyperparameter testing using either Dlib (`runners/dlib_runner.py`) or Hyperopt (`runners/hopt_runner.py`)
- Parallel optimization
- Save and restore optimization results and parameters to file or database (MongoDB)
- Limit GPU usage to N simultaneous processes, use CPU only for remaining
- Run N<=M processes with same hyperparameters but different random seeds and take average, with a total of M concurrent processes

Other runners
- Average over multiple runs (e.g. `python3 rs_sarsa --avg_runs 8`, see `runners/avg_runner.py`)
- Exploration comparisons (`e.g. python3 rs_sarsa --exp_policy_cmp`, see `runners/exp_pol_runner.py`)
 
