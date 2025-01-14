import numpy as np
import pandas as pd
import tqdm
from sklearn.calibration import calibration_curve



def calculate_percentile_bins(data, column, num_bins=100):
    """
    Calculate boundaries for dividing the data into equal percentile bins.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): The column name to calculate percentiles for.
        num_bins (int): Number of percentile bins (default: 100).
    
    Returns:
        np.ndarray: Array of boundaries dividing the column into bins.
    """
    percentiles = np.linspace(0, 1, num_bins + 1)
    boundaries = data[column].quantile(percentiles).values
    return boundaries



def assign_to_bins(data, column, boundaries):
    """
    Assign values in the column to percentile bins based on given boundaries.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the new data.
        column (str): The column name to classify.
        boundaries (np.ndarray): Array of bin boundaries.
    
    Returns:
        pd.Series: Bin indices for each value in the column.
    """
    bins = np.digitize(data[column], bins=boundaries, right=True) - 1
    num_bins = len(boundaries) - 1
    assert num_bins == 1000, 'number of bins is not 1000'

    bins = bins/10 # rescale such that percentiles are 0-100
    bins = bins + 0.1
    return bins

def apply_model(data, coeff_dict, cutpoints_dict):

    df = data.copy()
    
    # apply percentile transformations
    for x, boundaries in cutpoints_dict.items():
        df[f'{x}_pct'] = assign_to_bins(df, column=x, boundaries=boundaries)


    # apply model to calculate pd1, pd2, pd3, pd4, pd5
    for y_var in ['default_1', 'default_2', 'default_3', 'default_4', 'default_5']:

        output_var = f'pd_{y_var[-1]}'
        df[output_var] = 0
        for var, coeff in coeff_dict[y_var].items():
            df[output_var] += df[var] * coeff

        df[output_var] = np.exp(df[output_var]) / (1 + np.exp(df[output_var]))

    # calculate cumulative pd
    df['cumulative_pd_1'] = df['pd_1']
    df['cumulative_pd_2'] = df['cumulative_pd_1'] + (1 - df['cumulative_pd_1']) * df['pd_2']
    df['cumulative_pd_3'] = df['cumulative_pd_2'] + (1 - df['cumulative_pd_2']) * df['pd_3']
    df['cumulative_pd_4'] = df['cumulative_pd_3'] + (1 - df['cumulative_pd_3']) * df['pd_4']
    df['cumulative_pd_5'] = df['cumulative_pd_4'] + (1 - df['cumulative_pd_4']) * df['pd_5']

    return df

def build_calibration_curve(data, n_bins=500):

    collection = []
    for t in [1,2,3,4,5]:

        y_var = f'default_{t}'
        pd_var = f'pd_{t}'

        temp = data[[y_var, pd_var]].copy()
        temp = temp[temp[y_var] != -1]

        prop_true_uniform, prop_pred_uniform = calibration_curve(temp[y_var], temp[pd_var], strategy='uniform', n_bins=n_bins)
        df1 = pd.DataFrame({'prop_true': prop_true_uniform, 'prop_pred': prop_pred_uniform})
        df1['t'] = t
        df1['bin_type'] = 'uniform'
        collection.append(df1)

        prop_true_quantile, prop_pred_quantile = calibration_curve(temp[y_var], temp[pd_var], strategy='quantile', n_bins=n_bins)
        df2 = pd.DataFrame({'prop_true': prop_true_quantile, 'prop_pred': prop_pred_quantile})
        df2['t'] = t
        df2['bin_type'] = 'quantile'
        collection.append(df2)

    collection = pd.concat(collection, axis=0)
    collection['n_bins'] = n_bins

    # calculate correlation between prop_true and prop_pred
    collection1 = []
    for t in [1,2,3,4,5]:
        for bin_type in ['uniform', 'quantile']:
            
            temp1 = collection[(collection['t'] == t)] 
            temp1 = temp1[(temp1['bin_type'] == bin_type)]

            corr_pearson = temp1[['prop_true', 'prop_pred']].corr(method='pearson').iloc[0][1]
            corr_spearman = temp1[['prop_true', 'prop_pred']].corr(method='spearman').iloc[0][1]
            collection1.append([t, bin_type, corr_pearson, corr_spearman])

    collection1 = pd.DataFrame(collection1, columns=['t', 'bin_type', 'corr_pearson', 'corr_spearman'])
    collection = collection.merge(collection1, on=['t', 'bin_type'], how='left')
            

    return collection


def confusion_matrix_metrics(data):

    df = data[['default_1', 'default_2', 'default_3', 'default_4', 'default_5',
               'pd_1', 'pd_2', 'pd_3', 'pd_4', 'pd_5',
               'cumulative_pd_1', 'cumulative_pd_2', 'cumulative_pd_3', 'cumulative_pd_4', 'cumulative_pd_5']].copy()
    
    # mark if defaulted at any point up to horizon i 
    for i in [1,2,3,4,5]:
        df[f'cumulative_default_{i}'] = 0
        
        for j in range(1,i+1):
            mask = df[f'default_{j}'] == 1
            df.loc[mask, f'cumulative_default_{i}'] = 1

    # generate confusion matrix metrics for pd 1-5
    collection = []
    for t in [1,2,3,4,5]:

        y_var = f'default_{t}'
        pd_var = f'pd_{t}'

        temp = df[[y_var, pd_var]].copy()
        temp = temp[temp[y_var] != -1]        

        for thresh in tqdm.tqdm(np.linspace(0,.10, 100)):

            temp['y_pred_class'] = 0
            mask = temp[pd_var] >= thresh
            temp.loc[mask, 'y_pred_class'] = 1

            true_pos = ((temp[y_var] == 1) & (temp['y_pred_class'] == 1)).sum()
            false_pos = ((temp[y_var] == 0) & (temp['y_pred_class'] == 1)).sum()
            true_neg = ((temp[y_var] == 0) & (temp['y_pred_class'] == 0)).sum()
            false_neg = ((temp[y_var] == 1) & (temp['y_pred_class'] == 0)).sum()

            sensitivity = true_pos / (true_pos + false_neg)
            specificity = true_neg / (true_neg + false_pos)
            precision = true_pos / (true_pos + false_pos)
            negative_predictive_value = true_neg / (true_neg + false_neg)

            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

            tpr = sensitivity
            fpr = 1 - specificity

            collection.append([str(t),thresh, sensitivity, specificity, precision, negative_predictive_value, f1, tpr, fpr ])

    # generate confusion matrix metrics for cumulative default 1-5
    for t in [1,2,3,4,5]:

        y_var = f'cumulative_default_{t}'
        pd_var = f'cumulative_pd_{t}'

        temp = df[[y_var, pd_var]].copy()
        temp = temp[temp[y_var] != -1]        

        for thresh in tqdm.tqdm(np.linspace(0,.10, 100)):

            temp['y_pred_class'] = 0
            mask = temp[pd_var] >= thresh
            temp.loc[mask, 'y_pred_class'] = 1

            true_pos = ((temp[y_var] == 1) & (temp['y_pred_class'] == 1)).sum()
            false_pos = ((temp[y_var] == 0) & (temp['y_pred_class'] == 1)).sum()
            true_neg = ((temp[y_var] == 0) & (temp['y_pred_class'] == 0)).sum()
            false_neg = ((temp[y_var] == 1) & (temp['y_pred_class'] == 0)).sum()

            sensitivity = true_pos / (true_pos + false_neg)
            specificity = true_neg / (true_neg + false_pos)
            precision = true_pos / (true_pos + false_pos)
            negative_predictive_value = true_neg / (true_neg + false_neg)

            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

            tpr = sensitivity
            fpr = 1 - specificity

            collection.append([f'c{t}',thresh, sensitivity, specificity, precision, negative_predictive_value, f1, tpr, fpr ])        
    

    collection = pd.DataFrame(collection, columns=['t', 'thresh', 'sensitivity', 'specificity', 'precision', 'negative_predictive_value', 'f1', 'tpr', 'fpr'])
    return collection

def calculate_variable_correlations(data, x_vars):

    x_vars_trim = x_vars.copy()
    if 'constant' in x_vars_trim:
        x_vars_trim.remove('constant')
        
    # calculate pearson correlations
    df1 = data[x_vars_trim].corr(method='pearson')
    df1 = df1.reset_index(drop=False)
    df1['method'] = 'pearson'

    # calculate spearman correlations
    df2 = data[x_vars_trim].corr(method='spearman')
    df2 = df2.reset_index(drop=False)   
    df2['method'] = 'spearman'

    df = pd.concat([df1, df2], axis=0)
    df = df.rename(columns={'index': 'x'})

    return df

    