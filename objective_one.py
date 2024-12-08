import markov_chain_functions as mcf
import pandas as pd
import numpy as np

# import data
df = pd.read_csv(r'simulation_data.csv')
# convert outcome to binary (0,1 -> 0), (2,3 -> 1)
df['Outcome'].replace({1:0, 2:1, 3:1}, inplace=True)

# name the healthy vs icpr states consecutively (h0=0, i1=1, h1=2, i2=3, h2=4, i3=5, h4=6 ...)
# manipulations to create the variable outcome_status which describes the state according to the task
df['Outcome_diff'] = df.groupby('Athlete')['Outcome'].diff().fillna(df['Outcome']).replace({-1:1})
df['Outcome_status'] = df.groupby('Athlete')['Outcome_diff'].cumsum()
# create Markov chains for the first 5 states [0: healthy 0, 1: injury 1, 2: healthy 1, 3: imjury 2, 4: healthy 3]
mcm_recur_observed, mcm_recur_summary, mcm_recur_matrices = mcf.mc_from_data_with_ci(df, 'Athlete', 'Outcome_status', list(np.arange(0,6)), n_bootstrap=1000)
mcm_recur_diffs, _, mcm_recur_ratios, _ = mcf.prob_diffs_ratios_btstrp_stat(mcm_recur_observed, mcm_recur_matrices, 
                                                                                  comparison_lists=[['01', '23'], ['11', '33']], alpha=0.05)

print(mcm_recur_summary)