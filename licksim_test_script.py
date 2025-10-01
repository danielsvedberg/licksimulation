import licksimulation as ls
import plotting as pl
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

df, licktimes, lickili = ls.sim_STMT()
pl.plot_ili_histogram(lickili)
pl.plot_stmt_histogram(df, ls.default_params)
pl.plot_stmt_raster(df, ls.default_params)


params = ls.default_params.copy()

def oneiter(combo):
    ili, sd = combo
    params['mu'] = ili
    params['sigma'] = sd
    df, licktimes, lickili = ls.sim_STMT(params)
    df['ili'] = ili
    df['sd'] = sd
    total_rewards = df['reward_earned'].sum()
    reward_rate = total_rewards / params['num_trials']
    df['reward_rate'] = reward_rate
    return df

#generate ili values from 1 to 10 in increments of 0.1
ili_values = np.arange(3.3, 10.1, 0.2)
sd_values = np.arange(0.01, 0.51, 0.02)

#generate all combinations of ili and sd values
combinations = list(itertools.product(ili_values, sd_values))
#parallelize the above loop using joblib
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(delayed(oneiter)(out) for out in combinations)
#combine the results into a single dataframe
bigdf = pd.concat(results)
bigdf.to_csv('parameter_sweep_results.csv')

summarydf = bigdf[['ili', 'sd', 'reward_rate']].drop_duplicates().reset_index(drop=True)
#round everything to 2 decimal places
summarydf = summarydf.round(2)

#for each sd value, find the ili value that gives the highest reward rate
best_params = summarydf.loc[summarydf.groupby('sd')['reward_rate'].idxmax()]





#plot a heatmap of the reward rate as a function of ili and sd
import seaborn as sns
import matplotlib.pyplot as plt
pivot_table = summarydf.pivot(index='sd', columns='ili', values='reward_rate')
#convert entries in best_params to integers that correspond to the indices of sd_values and ili_values
best_params_converted = best_params.copy()
#get indices of sd_values where best_params['sd'] matches
best_params_converted['sd'] = best_params_converted['sd'].apply(lambda x: np.where(np.isclose(sd_values, x))[0][0])
#get indices of ili_values where best_params['ili'] matches
best_params_converted['ili'] = best_params_converted['ili'].apply(lambda x: np.where(np.isclose(ili_values, x))[0][0])
#add 0.5 to each value to center the dot in the heatmap cell
best_params_converted['sd'] = best_params_converted['sd'] + 0.5
best_params_converted['ili'] = best_params_converted['ili'] + 0.5

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(pivot_table, cmap='viridis', ax=ax)

#for each point in best_params, plot a green dot on the heatmap
ax.scatter(best_params_converted['ili'], best_params_converted['sd'], color='green', label='Best Params')
#use an ascending y axis
ax.invert_yaxis()

plt.title('Reward Rate Heatmap')
plt.xlabel('Mean Inter-Lick Interval (s)')
plt.ylabel('Standard Deviation of Inter-Lick Interval (s)')
#save the figure
plt.savefig('reward_rate_heatmap.png')


from joblib import Parallel, delayed
#generate ili values from 1 to 10 in increments of 0.1
ili_values = np.arange(3.3, 10.1, 0.1)
sd_values = np.arange(0.41, 0.51, 0.01)

#generate all combinations of ili and sd values
combinations = list(itertools.product(ili_values, sd_values))
results = Parallel(n_jobs=-1)(delayed(oneiter)(out) for out in combinations)
#combine the results into a single dataframe
bigdf = pd.concat(results)
bigdf.to_csv('parameter_sweep_results2.csv')

summarydf = bigdf[['ili', 'sd', 'reward_rate']].drop_duplicates().reset_index(drop=True)
#round everything to 2 decimal places
summarydf = summarydf.round(2)

#for each sd value, find the ili value that gives the highest reward rate
best_params = summarydf.loc[summarydf.groupby('sd')['reward_rate'].idxmax()]

#plot a heatmap of the reward rate as a function of ili and sd
import seaborn as sns
import matplotlib.pyplot as plt
pivot_table = summarydf.pivot(index='sd', columns='ili', values='reward_rate')
#convert entries in best_params to integers that correspond to the indices of sd_values and ili_values
best_params_converted = best_params.copy()
#get indices of sd_values where best_params['sd'] matches
best_params_converted['sd'] = best_params_converted['sd'].apply(lambda x: np.where(np.isclose(sd_values, x))[0][0])
#get indices of ili_values where best_params['ili'] matches
best_params_converted['ili'] = best_params_converted['ili'].apply(lambda x: np.where(np.isclose(ili_values, x))[0][0])
#add 0.5 to each value to center the dot in the heatmap cell
best_params_converted['sd'] = best_params_converted['sd'] + 0.5
best_params_converted['ili'] = best_params_converted['ili'] + 0.5

#for i = 1 to 6:
periods = []
for i in range(2, 7):
    beats = 19.75/i
    periods.append(beats)

#get ili_values indices where ili_values equals any of the periods
period_indices = []
for p in periods:
    idx = np.where(np.isclose(ili_values, p, atol=0.05))[0]
    if len(idx) > 0:
        period_indices.append(idx[0] + 0.5) #add 0.5 to center the dot in the heatmap cell


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(pivot_table, cmap='viridis', ax=ax,cbar_kws={'label': 'Reward Rate'})


#for each point in best_params, plot a green dot on the heatmap
ax.scatter(best_params_converted['ili'], best_params_converted['sd'], color='green', label='Best Params')
#add a dashed gray vertical line at each of the period_indices
for pi in period_indices:
    ax.axvline(x=pi, color='gray', linestyle='--', alpha=0.5)
#use an ascending y axis
ax.invert_yaxis()

plt.title('Reward Rate Heatmap')
plt.xlabel('Mean Inter-Lick Interval (s)')
plt.ylabel('Standard Deviation of log[Inter-Lick Interval (s)]')
#save the figure
plt.savefig('reward_rate_heatmap2.png')
