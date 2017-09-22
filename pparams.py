

def mk_pparams(rows=7,
               cols=7,
               n_channels=7,
               erlangs=8,
               call_rates=None,
               call_duration=3,
               n_episodes=10000,
               epsilon=0.1,
               alpha=0.01,
               gamma=0.9):
    """
    Call rates in calls per minute
    Call duration in minutes
    gamma:
    """
    # erlangs = call_rate * duration
    # 10 erlangs = 200cr, 3cd
    # 7.5 erlangs = 150cr, 3cd
    # 5 erlangs = 100cr, 3cd
    if not call_rates:
        call_rates = erlangs / call_duration
        # is this right unit
    return {
            'rows': rows,
            'cols': cols,
            'n_channels': n_channels,
            'call_rates': call_rates,  # Avg. call rate, in calls per minute
            'call_duration': call_duration,  # Avg. call duration in minutes
            'n_episodes': n_episodes,
            'epsilon': epsilon,
            'alpha': alpha,
            'gamma': gamma
           }
