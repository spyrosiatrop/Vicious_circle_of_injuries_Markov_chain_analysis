import markov_chain_functions as mcf
import pandas as pd
import numpy as np

# import data
df = pd.read_csv(r'simulation_data.csv')

mcm0123_observed, mcm0123_summary, mcm0123_matrices = mcf.mc_from_data_with_ci(df, 'Athlete', 'Outcome', [0,1,2,3], n_bootstrap=1000)
mcm0123_diffs, _, mcm0123_ratios, _ = mcf.prob_diffs_ratios_btstrp_stat(mcm0123_observed, mcm0123_matrices, [['01', '02', '03', '12', '13', '23'], ['11', '22', '33']], bootstrap_method='bca', alpha=0.05)

print(mcm0123_summary)