import numpy as np
import pandas as pd
import random
from scipy import stats

# MC function 1
# from a time-series of data claculates the frequencies of transitions
def my_mc_from_data(data, states):
    matrix =pd.DataFrame(np.zeros(shape=(len(states), len(states))), columns=states, index=states)
    for i in range(len(data)-1):
        try:
            matrix.loc[data[i], data[i+1]] +=1
        except:
            pass
    return matrix  
 
# MC function 2
# from individual transition matrices add them all to derive the group matrix
def add_mc_matrices(matrices, states):
    matrix = pd.DataFrame(np.zeros(shape=(len(states), len(states))), columns=states, index=states)
    for m in matrices:
        try:
            matrix += m.loc[states, states]
        except:
            pass
        
    return matrix

# MC function 3
# from the transition matrix of frequencies (0 -> inf) calculate the transition matrix of probabilities (0 -> 1)
def mc_freq_to_prob(matrix):
    row_sums = matrix.sum(axis=1)
    probs = matrix.div(row_sums, axis=0)
    
    return probs

# To calculate Bias-Corrected Accelerated confidence intervals instead of quantiles
import numpy as np
from scipy.stats import norm

def bca_confidence_interval(distribution, alpha=0.05, stat=None):
    """
    Calculate the BCa confidence interval for a given bootstrap distribution.

    Parameters:
    - distribution: A 1D array of bootstrap samples of the statistic.
    - alpha: Significance level (default is 0.05 for a 95% confidence interval).
    - stat: The observed value of the statistic (optional). If None, the median of the distribution will be used.

    Returns:
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
    """
    # Sort the bootstrap distribution
    distribution = np.sort(distribution)
    
    # Number of bootstrap samples
    n = len(distribution)

    # Observed value of the statistic (by default, we use the median of the bootstrap distribution)
    if stat is None:
        stat = np.median(distribution)

    # Proportion of bootstrap samples less than the observed statistic
    z0 = norm.ppf(np.sum(distribution < stat) / n)

    # Jackknife resampling
    jackknife_samples = np.array([np.mean(np.delete(distribution, i)) for i in range(n)])
    
    # Jackknife estimate of acceleration (skewness correction)
    mean_jack = np.mean(jackknife_samples)
    numerator = np.sum((mean_jack - jackknife_samples) ** 3)
    denominator = np.sum((mean_jack - jackknife_samples) ** 2) ** 1.5
    # Check if denominator is very small to avoid division by zero
    acc = 0  if denominator==0 else numerator / (6.0 * denominator)

    # Percentiles for confidence interval
    z_alpha = norm.ppf(alpha / 2)  # z-value for alpha/2
    z_1_alpha = norm.ppf(1 - alpha / 2)  # z-value for 1 - alpha/2

    # Adjusted percentiles
    pct_lower = norm.cdf(z0 + (z0 + z_alpha) / (1 - acc * (z0 + z_alpha)))
    pct_upper = norm.cdf(z0 + (z0 + z_1_alpha) / (1 - acc * (z0 + z_1_alpha)))

    # Map percentiles to bootstrap distribution percentiles
    # If pct < 0 or pct > 1 fall back to quantile bootstrapping
    if pct_lower>0 and pct_lower<1 and pct_upper>0 and pct_upper<1:
        ci_lower = np.percentile(distribution, pct_lower * 100)
        ci_upper = np.percentile(distribution, pct_upper * 100)
    else:
        ci_lower = np.percentile(distribution, 100 * (alpha / 2))
        ci_upper = np.percentile(distribution, 100* (1 - alpha/2))
        

    return ci_lower, ci_upper


# Global function to create the Markov chains

def mc_from_data_with_ci(data, individual_col, state_col, states, n_bootstrap=500, bootstrap_method= 'bca', tracker=False, alpha = 0.05):
    ind_data = [list(data.loc[data[individual_col]==c, state_col]) for c in data[individual_col].unique()]
    ind_matrices = [my_mc_from_data(ind_d, states) for ind_d in ind_data]
    observed_matrix = add_mc_matrices(ind_matrices, states)
    observed_p_matrix = mc_freq_to_prob(observed_matrix)
    
    # bootstrap
    p_matrices=[]
    for b in range(n_bootstrap):
        b_data = random.choices(ind_data, k=len(ind_data))
        b_matrices = [my_mc_from_data(b_d, states) for b_d in b_data]
        o_matrix = add_mc_matrices(b_matrices, states)
        p_matrices.append(mc_freq_to_prob(o_matrix))
        if tracker:
            if b%100==0:
                print(f'{b} bootstrap samples completed.')
        
    # results
    if bootstrap_method == 'quantile':
        results = pd.concat(p_matrices).groupby(level=0)
        means= results.mean()
        medians = results.median()
        low_cis = results.quantile(alpha/2)
        high_cis = results.quantile(1 - alpha/2)
        observed_ci = round(observed_p_matrix,4).astype(str) +" ["+ round(low_cis,4).astype(str)+ " - "+ round(high_cis,4).astype(str)+"]"
        
        return observed_p_matrix, observed_ci, p_matrices
    
    elif bootstrap_method == 'bca':
        low_bca = pd.DataFrame(index=p_matrices[0].index, columns=p_matrices[0].columns)
        high_bca = pd.DataFrame(index=p_matrices[0].index, columns=p_matrices[0].columns)

        # Iterate over each cell in the DataFrame
        for row in p_matrices[0].index:
            for col in p_matrices[0].columns:
                # For each cell, collect all values from the corresponding cell in each DataFrame in the list
                b_distr = [df.loc[row, col] for df in p_matrices]
                statistic = observed_p_matrix.loc[row, col]
                low_bca.loc[row, col], high_bca.loc[row, col] = bca_confidence_interval(distribution=b_distr, alpha=alpha, stat=statistic)
        
        stat_bca = observed_p_matrix.round(4).astype(str) +" ["+ low_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+ " - "+ high_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+"]"
        
        return observed_p_matrix, stat_bca , p_matrices
    
# Global function to compare transition probabilities

def prob_diffs_ratios_btstrp_stat(observed, matrices, comparison_lists: list, bootstrap_method= 'bca', alpha= 0.05):
    # fix the input of comparisons
    comparison_lists = comparison_lists if type(comparison_lists[0])==list else [comparison_lists]
    
    comp_diffs_dfs = [[] for i in range(len(comparison_lists))]
    comp_ratios_dfs = [[] for i in range(len(comparison_lists))]
    
    # for observed matrix
    observed_diffs = []
    observed_ratios = []
    for i,l in enumerate(comparison_lists):
            diff_df = pd.DataFrame(index=l)
            ratio_df = pd.DataFrame(index=l)
            
            for r1 in l:
                one = observed.loc[int(r1[0]), int(r1[1])]
                diffs = []
                ratios = []
                
                for r2 in l:
                    two = observed.loc[int(r2[0]), int(r2[1])]
                    diff = one - two
                    diffs.append(diff)
                    ratio = one/max(two, 0.0001) 
                    ratios.append(ratio)
                
                diff_df[r1] = diffs
                ratio_df[r1] = ratios
                
            observed_diffs.append(diff_df)
            observed_ratios.append(ratio_df)
    
    # for bootstrap matrices
    for m in matrices:
        for i,l in enumerate(comparison_lists):
            diff_df = pd.DataFrame(index=l)
            ratio_df = pd.DataFrame(index=l)
            
            for r1 in l:
                one = m.loc[int(r1[0]), int(r1[1])]
                diffs = []
                ratios = []
                
                for r2 in l:
                    two = m.loc[int(r2[0]), int(r2[1])]
                    diff = one - two
                    diffs.append(diff)
                    ratio = one/max(two, 0.0001) 
                    ratios.append(ratio)
                
                diff_df[r1] = diffs
                ratio_df[r1] = ratios
                
            comp_diffs_dfs[i].append(diff_df)
            comp_ratios_dfs[i].append(ratio_df)
            
    results_diffs = [pd.DataFrame() for i in range(len(comparison_lists))]
    p_values_diffs = [pd.DataFrame() for i in range(len(comparison_lists))]
    results_ratios = [pd.DataFrame() for i in range(len(comparison_lists))]
    p_values_ratios = [pd.DataFrame() for i in range(len(comparison_lists))]

    f_diff = lambda x: stats.ttest_1samp(a= x, popmean=0)[1]
    f_ratio = lambda x: stats.ttest_1samp(a= x, popmean=1)[1]


    if bootstrap_method == 'quantile':
        for i,r in enumerate(comp_diffs_dfs):
            results = pd.concat(r).groupby(level=0)
            # means= results.mean()
            # medians = results.median()
            low_cis = results.quantile(alpha/2)
            high_cis = results.quantile(1- alpha/2)
            observed_ci = round(observed_diffs[i],4).astype(str) +" ["+ round(low_cis,4).astype(str)+ " - "+ round(high_cis,4).astype(str)+"]"
            
            results_diffs[i] = observed_ci
            
            # find th p_value of each difference(cell in the dataframe) by applying the lambda function above
            p_values_diffs[i] = pd.DataFrame(
                                                [[
                                                    [df.loc[row, col] for df in r]  # List of values from each df at (row, col)
                                                    for col in r[0].columns
                                                ] for row in r[0].index],
                                                index= r[0].index,
                                                columns= r[0].columns
                                            ).map(f_diff)
            
        for i,r in enumerate(comp_ratios_dfs):
            results = pd.concat(r).groupby(level=0)
            # means= results.mean()
            # medians = results.median()
            low_cis = results.quantile(0.025)
            high_cis = results.quantile(0.975)
            observed_ci = round(observed_ratios[i],4).astype(str) +" ["+ round(low_cis,4).astype(str)+ " - "+ round(high_cis,4).astype(str)+"]"
            
            results_ratios[i] = observed_ci
            
            # find th p_value of each ratio(cell in the dataframe) by applying the lambda function above
            p_values_ratios[i] = pd.DataFrame(
                                                [[
                                                    [df.loc[row, col] for df in r]  # List of values from each df at (row, col)
                                                    for col in r[0].columns
                                                ] for row in r[0].index],
                                                index= r[0].index,
                                                columns= r[0].columns
                                            ).map(f_ratio)
            
    elif bootstrap_method == 'bca':
        for i,r in enumerate(comp_diffs_dfs):
            low_bca = pd.DataFrame(index=r[0].index, columns=r[0].columns)
            high_bca = pd.DataFrame(index=r[0].index, columns=r[0].columns)

            # Iterate over each cell in the DataFrame
            for row in r[0].index:
                for col in r[0].columns:
                    # For each cell, collect all values from the corresponding cell in each DataFrame in the list
                    b_distr = [df.loc[row, col] for df in r]
                    statistic = observed_diffs[i].loc[row, col]
                    low_bca.loc[row, col], high_bca.loc[row, col] = bca_confidence_interval(distribution=b_distr, alpha=alpha, stat=statistic)
            
            results_diffs[i] = observed_diffs[i].round(4).astype(str) +" ["+ low_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+ " - "+ high_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+"]"
            p_values_diffs[i] = pd.DataFrame(
                                                [[
                                                    [df.loc[row, col] for df in r]  # List of values from each df at (row, col)
                                                    for col in r[0].columns
                                                ] for row in r[0].index],
                                                index= r[0].index,
                                                columns= r[0].columns
                                            ).map(f_diff)
            
            
        for i,r in enumerate(comp_ratios_dfs):
            low_bca = pd.DataFrame(index=r[0].index, columns=r[0].columns)
            high_bca = pd.DataFrame(index=r[0].index, columns=r[0].columns)

            # Iterate over each cell in the DataFrame
            for row in r[0].index:
                for col in r[0].columns:
                    # For each cell, collect all values from the corresponding cell in each DataFrame in the list
                    b_distr = [df.loc[row, col] for df in r]
                    statistic = observed_ratios[i].loc[row, col]
                    low_bca.loc[row, col], high_bca.loc[row, col] = bca_confidence_interval(distribution=b_distr, alpha=0.05, stat=statistic)
            
            results_ratios[i] = observed_ratios[i].round(4).astype(str) +" ["+ low_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+ " - "+ high_bca.apply(lambda col: col.map(lambda x: f'{x:.4f}'))+"]"
            p_values_ratios[i] = pd.DataFrame(
                                                [[
                                                    [df.loc[row, col] for df in r]  # List of values from each df at (row, col)
                                                    for col in r[0].columns
                                                ] for row in r[0].index],
                                                index= r[0].index,
                                                columns= r[0].columns
                                            ).map(f_ratio)    
    
    
    return results_diffs, p_values_diffs, results_ratios, p_values_ratios            
            