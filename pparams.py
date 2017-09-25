

def mk_pparams(rows=7,
               cols=7,
               n_channels=70,
               erlangs=8,
               call_rates=None,
               call_duration=3,
               n_episodes=2000000,
               n_hours=2,
               epsilon=0.25,
               epsilon_decay=0.999999,
               alpha=0.1,
               alpha_decay=0.999999,
               gamma=0.9):
    """
    n_hours: If n_episodes is not specified, run simulation for n_hours
        of simulation time
    Call rates in calls per minute
    Call duration in minutes
    gamma:
    """
    # erlangs = call_rate * duration
    # 10 erlangs = 200cr, 3cd
    # 7.5 erlangs = 150cr, 3cd
    # 5 erlangs = 100cr, 3cd
    if not call_rates:
        call_rates = erlangs / call_duration
    return {
            'rows': rows,
            'cols': cols,
            'n_channels': n_channels,
            'call_rates': call_rates,  # Avg. call rate, in calls per minute
            'call_duration': call_duration,  # Avg. call duration in minutes
            'n_episodes': n_episodes,
            'n_hours': n_hours,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'alpha': alpha,
            'alpha_decay': alpha_decay,
            'gamma': gamma
           }
