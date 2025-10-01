import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('macosx')

from licksimulation import default_params

def plot_ili_histogram(array):
    # plot a histogram of the array
    fig = plt.figure()
    plt.hist(array, bins=30)
    # save the figure
    fig.savefig('histogram.png')
def plot_stmt_histogram(df, params):
    if params is None:
        params = default_params

    movement_times = df['movement_time'].values
    rewards_earned = df['reward_earned'].values
    num_trials = len(movement_times)
    total = sum(rewards_earned)

    cue_to_lick_window_start = params['abort_end'] # seconds
    cue_to_lick_window_end = params['trial_end'] # seconds

    #plot a histogram of the movement times
    fig = plt.figure()
    plt.hist(movement_times)
    #plot a vertical line at 3.33s
    plt.axvline(x=cue_to_lick_window_start, color='r', linestyle='--')
    plt.axvline(x=cue_to_lick_window_end, color='r', linestyle='--')
    plt.xlabel('Movement Time after cue (s)')
    plt.title(f'Total Rewards Earned: {total} out of {num_trials}')
    fig.savefig('movement_times.png')

def plot_stmt_raster(df, params):
    if params is None:
        params = default_params

    num_trials = len(df)
    movement_times = df['movement_time'].values
    rewards_earned = df['reward_earned'].values
    trial_licks = df['trial_licks'].values
    cue_to_lick_window_start = params['abort_end'] # seconds
    cue_to_lick_window_end = params['trial_end'] # seconds

    #plot a raster aligned to each reward window
    #the x axis should be from -2s to +10s around the reward window start
    fig = plt.figure()
    for i in range(num_trials):
        #plot all the licks for this trial as gray dots
        trl_lks = trial_licks[i]
        #make an array of the same length as trl_lks with the value i
        i_arr = np.full_like(trl_lks, i)
        plt.plot(trl_lks, i_arr, 'k.', markersize=2, alpha=0.5)
        #plot a black vertical line at the reward window start
        plt.axvline(x=cue_to_lick_window_start, color='black', linestyle='--')
        plt.axvline(x=cue_to_lick_window_end, color='black', linestyle='--')
        #if the trial was rewarded, plot a green dot at the movement time
        if rewards_earned[i] == 1:
            plt.plot(movement_times[i], i, 'go', markersize=2)
    #save the figure
    plt.xlim(-2, 10)
    plt.ylim(-1, num_trials)
    plt.xlabel('Time from Cue (s)')
    plt.ylabel('Trial')
    plt.title('Raster of Movement Times Aligned to Cue')
    fig.savefig('raster.png')
