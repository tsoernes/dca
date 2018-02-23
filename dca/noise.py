for t in range(max_timesteps):
    kwargs = {}
    if param_noise:
        update_eps = 0.
        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
        update_param_noise_threshold = -np.log(
            1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
        kwargs['reset'] = reset
        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noise_scale'] = True

# parameter space noise exploration (https://arxiv.org/abs/1706.01905):


def default_param_noise_filter(var):
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False
