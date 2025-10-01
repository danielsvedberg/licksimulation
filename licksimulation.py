import numpy as np
import pandas as pd

default_params = {
    'mu': 3, # mean of the log-normal distribution
    'sigma': 1, # standard deviation of the log-normal distribution
    'num_trials': 10000, # number of trials to simulate
    'iti': 10, # inter-trial interval in seconds
    'abort_end': 3.33, # time after cue when reward window starts
    'trial_end': 7, # time after cue when reward window ends
    'random_delay_min': 0.4, # minimum random delay between lamp and cue
    'random_delay_max': 1.5, # maximum random delay between lamp and cue
}

def sim_STMT(params=None):
    if params is None:
        params = default_params
    else:
        # fill in any missing parameters with defaults
        for key in default_params:
            if key not in params:
                params[key] = default_params[key]

    # unpack parameters
    mu = params['mu']
    sigma = params['sigma']
    num_trials = params['num_trials']
    iti = params['iti']
    abort_end = params['abort_end']
    trial_end = params['trial_end']
    random_delay_min = params['random_delay_min']
    random_delay_max = params['random_delay_max']

    #generate an array of log-normally distributed numbers with a mean of 3 and a standard deviation of 1
    mu = np.log(mu)
    lick_ili = np.random.lognormal(mu, sigma, size=200000)
    # make a cumulative sum of the array
    lick_times = np.cumsum(lick_ili)

    # add some random number to lick_times
    # generate one random number between 0 and 1.5
    random_offset = np.random.uniform(0, 1.5)
    lick_times = lick_times + random_offset

    #STMT parameters:
    #10s iti
    #lamp to cue iti is 400-1500ms uniform random
    #time from cue to lick window start is 3s
    #time from cue to lick window end is 7s
    #make a loop simulating successive trials of the STMT
    #generate 100 trials

    cue_to_lamp_iti = np.random.uniform(random_delay_min, random_delay_max, num_trials) # seconds
    reward_window_starts = []
    reward_window_ends = []
    trial_start_times = []
    cue_times = []
    rewards_earned = []
    movement_times = []
    trial_licks = []
    current_time = 0
    for i in range(num_trials):
        trial_start_times.append(current_time)
        #get the index in the lick_times that is just after the current time
        startidx = np.searchsorted(lick_times, current_time)

        current_time += iti
        current_time += cue_to_lamp_iti[i]
        cue_times.append(current_time)
        #get the time of the first lick after the cue
        first_lick_after_cue = lick_times[lick_times > current_time][0]

        movement_times.append(first_lick_after_cue - current_time)
        reward_window_start = current_time + abort_end
        reward_window_end = current_time + trial_end
        #figure out if the first lick after the cue is within the reward window
        in_window = (first_lick_after_cue >= reward_window_start) and (first_lick_after_cue <= reward_window_end)
        rewards_earned.append(1 if in_window else 0)
        reward_window_starts.append(reward_window_start)
        reward_window_ends.append(reward_window_end)
        current_time += trial_end # move time to end of lick window for next trial
        #get the index of lick_times that is 5s after the current time
        endidx = np.searchsorted(lick_times, current_time + 5)
        #get all the licks that happened during this trial
        licks = lick_times[startidx:endidx]
        #subtract the time of the cue from the licks
        licks = licks - cue_times[-1]
        trial_licks.append(licks)

    #make a pandas dataframe with the results
    df = pd.DataFrame({
        'trial_start_time': trial_start_times,
        'cue_time': cue_times,
        'reward_window_start': reward_window_starts,
        'reward_window_end': reward_window_ends,
        'movement_time': movement_times,
        'reward_earned': rewards_earned,
        'trial_licks': trial_licks
    })
    total = sum(rewards_earned)
    print(f'Total Rewards Earned: {total} out of {num_trials}')
    return df, lick_ili, lick_times
