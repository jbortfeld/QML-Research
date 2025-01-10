
import pandas as pd
import numpy as np
import time
import tqdm
import scipy.stats
import datetime


def quantile_analysis_by_default_class(data:pd.DataFrame, metric:str, sector_groupby:str):

    '''
    This function calculates the median, count, mad, 25th, and 75th percentiles of a given metric by default class to be used for box plots. 
    '''

    df = data[data[metric].notnull()].copy()
    df = df[df[metric] != np.inf]
    df = df[df[metric] != -np.inf]
    collection = []
    
    for this_calc in ['median', 'count' , 'mad', '25', '75']:
        
        # aggregate by sector and default
        temp = pd.DataFrame()
        for i in [1,2,3,4,5]:
            if this_calc in ['median', 'count']:
                temp[f't_{i}'] = df.groupby([sector_groupby, f'default_{i}'])[metric].apply(this_calc)
            elif this_calc == '25':
                temp[f't_{i}'] = df.groupby([sector_groupby, f'default_{i}'])[metric].apply(lambda x: x.quantile(0.25))
            elif this_calc == '75':
                temp[f't_{i}'] = df.groupby([sector_groupby, f'default_{i}'])[metric].apply(lambda x: x.quantile(0.75))
            else:
                temp[f't_{i}'] = df.groupby([sector_groupby, f'default_{i}'])[metric].apply(lambda x: scipy.stats.median_abs_deviation(x))

        temp = temp.reset_index()
        temp = temp.rename(columns={'default_1': 'default'})
        temp['ratio'] = metric
        temp['calc'] = this_calc
        collection.append(temp)

        # aggregate by default (include all sectors)
        temp = pd.DataFrame()
        for i in [1,2,3,4,5]:
            if this_calc in ['median', 'count']:
                temp[f't_{i}'] = df.groupby(f'default_{i}')[metric].apply(this_calc)
            elif this_calc == '25':
                temp[f't_{i}'] = df.groupby(f'default_{i}')[metric].apply(lambda x: x.quantile(0.25))
            elif this_calc == '75':
                temp[f't_{i}'] = df.groupby(f'default_{i}')[metric].apply(lambda x: x.quantile(0.75))
            else:
                temp[f't_{i}'] = df.groupby(f'default_{i}')[metric].apply(lambda x: scipy.stats.median_abs_deviation(x))

        temp = temp.reset_index()
        temp = temp.rename(columns={'default_1': 'default'})
        temp[sector_groupby] = 'All'
        temp['ratio'] = metric
        temp['calc'] = this_calc
        collection.append(temp)

    df2 = pd.concat(collection, axis=0)

    # reshape the dataframe for easier plotting
    # -- we want columns: ratio, factset_econ_sector, default, quantile, 1,2,3,4,5
    df2 = df2[['ratio', 'factset_econ_sector', 'default', 'calc', 't_1', 't_2', 't_3', 't_4', 't_5']]

    collection = []
    for m in ['25', 'median', '75']:
        temp = df2[df2['calc'] == m]
        temp = temp[['ratio', 'factset_econ_sector', 'default', 'calc', 't_1', 't_2', 't_3', 't_4', 't_5']]
        temp['quantile'] = m
        collection.append(temp)

    df3 = pd.concat(collection, axis=0)
    df3['quantile'] = df3['quantile'].map(lambda x: '50' if x == 'median' else x)
    df3['quantile'] = df3['quantile'].astype(float) / 100
    df3 = df3.sort_values(by=['ratio', 'factset_econ_sector', 'default', 'quantile'])

    return df3

def default_rate_by_ratio_decile(data:pd.DataFrame, metric:str, groupby:str):

    '''
    This function calculates the default rate by decile of a given metric. 
    '''

    df = data[['fsym_id', 'fiscal_end_date', groupby, metric, 'default_1', 'default_2', 'default_3', 'default_4', 'default_5']].copy()
    df = df[df[metric].notnull()]
    df = df[df[metric] != np.inf]
    df = df[df[metric] != -np.inf]

    # generate sector list
    group_list = list(df[groupby].unique())
    group_list = ['All'] + group_list
    collection = []

    # iterate over sectors
    for this_group in group_list:

        # subset for this sector
        if this_group == 'All':
            temp1 = df.copy()
        else:
            temp1 = df[df[groupby] == this_group].copy()

        # generate decile buckets based on the metric
        temp1['decile'] = pd.qcut(temp1[metric], q=10, labels=False, duplicates='drop')

        # iterate over default horizons
        for t in [1,2,3,4,5]:

            # exclude defaults that are -1 (too close to the default date)
            temp2 = temp1[temp1[f'default_{t}'] != -1].copy()

            # calculate the default rate by decile
            temp3= temp2.groupby(by='decile', as_index=False)[f'default_{t}'].mean()

            # save analysis parameters
            temp3['groupby'] = groupby
            temp3['group'] = this_group
            temp3['metric'] = metric
            temp3['default_horizon'] = t
            temp3 = temp3.rename(columns={f'default_{t}': 'default_rate'})

            # calculate spearman rho
            # if there is only one unique default rate (which is the case when there are no defaults across all deciles), then the spearman rho is not defined
            if temp3['default_rate'].nunique() > 1:
                rho, p = scipy.stats.spearmanr(temp3['decile'], temp3['default_rate'])
                temp3['spearman_rho'] = rho
                temp3['spearman_p'] = p
            else:
                temp3['spearman_rho'] = np.NaN
                temp3['spearman_p'] = np.NaN

            # add to collection
            collection.append(temp3)

    df = pd.concat(collection, axis=0)
    df = df[['metric', 'groupby', 'group', 'default_horizon', 'decile', 'default_rate', 'spearman_rho', 'spearman_p']]
    return df

# construct percentiles

def quantile_analysis(data:pd.DataFrame, 
                      metric:str='total_debt_to_ebitda',
                      quantile_list:list=[0, 0.01, .02, .03, .04, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,0.96,0.97,0.98, 0.99, 1],
                      groupby:str=None
                      ):
    
    # 0. generate quantiles across the entire dataset
    df= data[metric].quantile(quantile_list)
    df = pd.DataFrame(df).reset_index()
    df.columns = ['quantile', 'All']

    # generate observation counts
    count_dict = {'All': data[metric].count()}

    # 1. generate quantiles across the groupby variable
    if groupby is not None:
        
        dff = data.groupby(groupby)[metric].quantile(quantile_list)
        dff = dff.reset_index()
        dff.columns = [groupby, 'quantile', metric]
        dff = dff.pivot(index='quantile', columns=groupby, values=metric)
        df = df.merge(dff, on='quantile', how='left')

        # generate observation counts
        temp = data.groupby(groupby)[metric].count()
        for i in temp.index:
            count_dict[i] = temp.loc[i]

    # add observation counts to the dataframe
    df2 = pd.DataFrame({'quantile': ['n']})
    df = pd.concat([df2, df], ignore_index=True)

    for c in count_dict.keys():
        df.loc[0, c] = count_dict[c]
        

    # 2. calculate median and median absolute deviation (MAD)
    temp = data[data[metric].notnull()]

    values_dict = {}
    for this_group in temp[groupby].unique():
        values_dict[this_group] = temp[temp[groupby] == this_group][metric].values
    values_dict['All'] = temp[metric].values

    mad_dict = {}
    for this_group in values_dict.keys():
        mad_dict[this_group] = scipy.stats.median_abs_deviation(values_dict[this_group])
    mad_dict['All'] = scipy.stats.median_abs_deviation(values_dict['All'])
    
    df2 = pd.DataFrame({'group': mad_dict.keys()})
    df2['mad'] = df2['group'].apply(lambda x: mad_dict[x])
    df2 = df2.set_index('group').T
    df2 = df2.reset_index()
    df2 = df2.rename(columns={'index': 'quantile'})

    df = pd.concat([df, df2], ignore_index=True)

    # timestamp
    df.loc[0, 'timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    return df

def generate_histogram_data(data:pd.DataFrame, metric:str, quantiles:tuple, groupby:str='factset_econ_sector'):

    collection = []
    group_list = list(data[groupby].unique())


    for this_quantiles in [(.01, .99), (.02, .98), (.03, .97), (.04, .96), (.05, .95), (.1, .9)]:
        
        # 0. generate histogram data across the entire dataset
        # use the full dataset (all sectors)
        # remove nulls and infs
        temp1 = data.copy()
        temp1 = temp1[temp1[metric].notnull()].copy()
        temp1 = temp1[temp1[metric] != np.inf]
        temp1 = temp1[temp1[metric] != -np.inf]

        # winsorize the data
        lower, upper = temp1[metric].quantile(this_quantiles)
        temp1[metric] = temp1[metric].clip(lower=lower, upper=upper)

        # calculate histogram bins using the full dataset
        bin_edges = np.histogram_bin_edges(temp1[metric], bins=50)
        counts_all, _ = np.histogram(temp1[metric], bins=bin_edges, density=True
                                    )
        # save the bin edges and bin counts to a datafame
        temp = pd.DataFrame({'count': [np.NaN] + list(counts_all), 'bin': bin_edges})
        temp['metric'] = metric
        temp['sector'] = 'All'
        temp['lower_clip'] = this_quantiles[0]
        temp['upper_clip'] = this_quantiles[1]
        temp = temp[['metric', 'sector', 'lower_clip', 'upper_clip', 'bin', 'count']]
        collection.append(temp)

        # 1. iterate over each sector grouping
        for this_group in group_list:

            # subset for a given sector
            temp2 = temp1[temp1[groupby] == this_group].copy()
            if this_group != 'All':
                temp2 = temp2[temp2[groupby] == this_group]

            # calculate histogram bins
            this_counts, _ = np.histogram(temp2[metric], bins=bin_edges, density=True)

            # save the bin edges and bin counts to a datafame
            temp = pd.DataFrame({'count': [np.NaN] + list(this_counts), 'bin': bin_edges})
            temp['metric'] = metric
            temp['sector'] = this_group
            temp['lower_clip'] = this_quantiles[0]
            temp['upper_clip'] = this_quantiles[1]
            temp = temp[['metric', 'sector', 'lower_clip', 'upper_clip', 'bin', 'count']]

            collection.append(temp)

    df_histogram = pd.concat(collection, axis=0)
    df_histogram = df_histogram.reset_index(drop=True)

    return df_histogram


def build_default_diagnostics(data:pd.DataFrame):

    # get a list of all bankrupt fsym_ids
    temp = data[data['bankruptcy_date'].notnull()].copy()
    bankrupt_fsym_list = temp['fsym_id'].unique()

    all_collection = []
    for this_fsym in tqdm.tqdm(bankrupt_fsym_list):
    
        df1 = data[data['fsym_id'] == this_fsym]

        collection = df1[['fsym_id', 'name1', 'name2', 'factset_econ_sector', 'bankruptcy_date']].head(1).copy()
        for i in [1,2,3,4,5]:

            # keep any cases where there is a default t years forward
            mask = df1[f'default_{i}'] == 1
            df2 = df1[mask].copy()


            if df2.shape[0] > 0:
                # keep the observation closest to 1/2/3/4/5 years prior to the default
                df2['diff'] = df2['bankruptcy_date'] - df2['fiscal_end_date']
                df2['diff'] = df2['diff'].dt.days
                df2['diff'] = abs(df2['diff'] - 365* i)
                df2 = df2.sort_values(by='diff', ascending=True)
                df2 = df2.head(1)[['fsym_id', 'fiscal_end_date', 'ff_assets', 'ff_ebitda_oper_ltm', 'ff_oper_cf_ltm']]

                # count if income statement, balance sheet, cash flow statement variables are available
                df2[f'fund_count_{i}'] = df2[['ff_assets', 'ff_ebitda_oper_ltm', 'ff_oper_cf_ltm']].count(axis=1)
                df2[f'fund_{i}'] = df2[['ff_assets', 'ff_ebitda_oper_ltm', 'ff_oper_cf_ltm']].apply(lambda x: (x.iloc[0], x.iloc[1], x.iloc[2]), axis=1)
                df2[f'fiscal_end_date_{i}'] = df2['fiscal_end_date']

                df2 = df2[['fsym_id', f'fiscal_end_date_{i}', f'fund_count_{i}', f'fund_{i}']]
                df2 = df2.set_index('fsym_id')
                collection = collection.merge(df2, on='fsym_id', how='outer')
        
        all_collection.append(collection)

    df = pd.concat(all_collection, axis=0)


    # flag cases where none of the three sample variables are available across all time horizons
    df['has_fund'] = 0
    for i in [1,2,3,4,5]:
        mask = df[f'fund_count_{i}'] > 0
        df.loc[mask, 'has_fund'] = 1
    df['missing_fund'] = 0
    mask = df['has_fund'] == 0
    df.loc[mask, 'missing_fund'] = 1

    df = df[['fsym_id', 'name1', 'name2', 'factset_econ_sector', 'bankruptcy_date', 'missing_fund',
            'fund_count_1', 'fund_count_2', 'fund_count_3', 'fund_count_4', 'fund_count_5',
            'fiscal_end_date_1', 'fiscal_end_date_2', 'fiscal_end_date_3', 'fiscal_end_date_4', 'fiscal_end_date_5',
            'fund_1', 'fund_2', 'fund_3', 'fund_4', 'fund_5']]
        

    return df