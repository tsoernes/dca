for t in range(max_timesteps):
    kwargs = {}
    if param_noise:
        update_eps = 0.
        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
        # for detailed explanation.
        update_param_noise_threshold = -np.log(
            1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
        kwargs['reset'] = reset
        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noise_scale'] = True
