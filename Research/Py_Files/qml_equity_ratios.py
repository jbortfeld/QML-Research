

import pandas as pd
import numpy as np
import ast
import tqdm
import os
from Py_Files import credentials
from Py_Files import factset_api
import matplotlib.pyplot as plt
from Py_Files import financial_modeling_prep as fmp
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import fsolve

def build_benchmark_data():

    '''
    Build a dataframe of benchmark data for the SP500, STOXX and NIKKEI using the FMP API.
    '''
    # get benchmark data
    df_sp500 = fmp.download_stock_returns(ticker='SP500', start_date='1990-01-01')
    df_sp500['date'] = pd.to_datetime(df_sp500['date'])
    df_sp500 = df_sp500[['date', 'total_return', 'adjClose']]
    df_sp500.columns = ['date', 'benchmark_return', 'benchmark_price']
    df_sp500['benchmark'] = 'SP500'

    df_stoxx = fmp.download_stock_returns(ticker='STOXX', start_date='1990-01-01')
    df_stoxx['date'] = pd.to_datetime(df_stoxx['date'])
    df_stoxx = df_stoxx[['date', 'total_return', 'adjClose']]
    df_stoxx.columns = ['date', 'benchmark_return', 'benchmark_price']
    df_stoxx['benchmark'] = 'STOXX'

    df_nikkei = fmp.download_stock_returns(ticker='NIKKEI', start_date='1990-01-01')
    df_nikkei['date'] = pd.to_datetime(df_nikkei['date'])
    df_nikkei = df_nikkei[['date', 'total_return', 'adjClose']]
    df_nikkei.columns = ['date', 'benchmark_return', 'benchmark_price']
    df_nikkei['benchmark'] = 'NIKKEI'

    df_benchmarks = pd.concat([df_sp500, df_stoxx, df_nikkei])

    for bench in df_benchmarks['benchmark'].unique():

        if bench=='STOXX':
            df_benchmarks[df_benchmarks['benchmark'] == bench].set_index('date')['benchmark_price'].plot(secondary_y=True)
        else:
            df_benchmarks[df_benchmarks['benchmark'] == bench].set_index('date')['benchmark_price'].plot()

    return df_benchmarks


def merge_equity_data(fsym_id:str='MH33D6-R',
                      input_dir:str='/Users/joeybortfeld/Documents/QML Solutions Data/factset_data/factset_equity/',
                      ):
    
    '''
    Combine multiple datasets into a single dataframe for a single fsym_id.

    - pre 2006 history downloaded via Factset excel add-in containing daily price, market cap and dividend history
    - post 2006 history downloaded via Factset API containing daily price SPLIT
    - post 2006 history downloaded via Factset API containing daily price UNSPLIT
    - post 2006 history downloaded via Factset API containing daily total returns
    - post 2006 history downloaded via Factset API containing shares outstanding


    '''

    # 0. get the excel add-in download

    # price data
    if os.path.exists(input_dir + f'excel_addin_download/{fsym_id}_equity_history.csv'):

        temp = pd.read_csv(input_dir + f'excel_addin_download/{fsym_id}_equity_history.csv', skiprows=16, names=['date', 'price', 'volume', 'market_cap', 'dividend_date', 'dividend_amount'])
        temp1 = temp[['date', 'price', 'volume', 'market_cap']].copy()

        try:
            temp1['date'] = pd.to_datetime(temp1['date'], format='%m/%d/%Y')
        except:
            temp1['date'] = pd.to_datetime(temp1['date'], format='%d/%m/%Y')

        # dividend data
        temp2 = temp[['dividend_date', 'dividend_amount']].copy()
        try:
            temp2['date'] = pd.to_datetime(temp2['dividend_date'], format='%m/%d/%Y')
        except:
            temp2['date'] = pd.to_datetime(temp2['dividend_date'], format='%d/%m/%Y')
        temp2 = temp2[['date', 'dividend_amount']]


        df1 = temp1.merge(temp2, on='date', how='left')
        df1 = df1[df1['date'].notnull()]
        df1['dividend_amount'] = df1['dividend_amount'].fillna(0)
    else:
        df1 = pd.DataFrame({'date': [], 'price': [], 'volume': [], 'market_cap': [], 'dividend_amount': []})

    # 1. get the split price download data
    if os.path.exists(input_dir + f'prices SPLIT/{fsym_id}.csv'):
        df2 = pd.read_csv(input_dir + f'prices SPLIT/{fsym_id}.csv')
        df2['date'] = pd.to_datetime(df2['date'])
        df2 = df2.rename(columns={'volume': 'volume_split', 'price': 'price_split', 
                                'currency': 'currency_split', 'tradeCount': 'trade_count_split'})
        df2 = df2[['date', 'price_split', 'volume_split', 'currency_split', 'trade_count_split']]
    else:
        df2 = pd.DataFrame({'date': [], 'price_split': [], 'volume_split': [], 'currency_split': [], 'trade_count_split': []})

    df = df1.merge(df2, on='date', how='outer')

    # 2. get the unsplit price download data
    if os.path.exists(input_dir + f'prices UNSPLIT/{fsym_id}.csv'):
        df3 = pd.read_csv(input_dir + f'prices UNSPLIT/{fsym_id}.csv')
        df3['date'] = pd.to_datetime(df3['date'])
        df3 = df3.rename(columns={'volume': 'volume_unsplit', 'price': 'price_unsplit', 
                                'currency': 'currency_unsplit', 'tradeCount': 'trade_count_unsplit'})
        df3 = df3[['date', 'price_unsplit', 'volume_unsplit', 'currency_unsplit', 'trade_count_unsplit']]
    else:
        df3 = pd.DataFrame({'date': [], 'price_unsplit': [], 'volume_unsplit': [], 'currency_unsplit': [], 'trade_count_unsplit': []})

    df = df.merge(df3, on='date', how='outer')

    # 3. get return data
    if os.path.exists(input_dir + f'returns/{fsym_id}.csv'):
        df4 = pd.read_csv(input_dir + f'returns/{fsym_id}.csv')
        df4['date'] = pd.to_datetime(df4['date'])
        df4 = df4[['date', 'totalReturn', 'currency']]
        df4 = df4.rename(columns={'totalReturn': 'total_return', 'currency': 'currency_return'})
        df4['total_return'] /= 100
    else:
        df4 = pd.DataFrame({'date': [], 'total_return': [], 'currency_return': []})

    df = df.merge(df4, on='date', how='outer')

    # get shares outstanding
    if os.path.exists(input_dir + f'shares/{fsym_id}.csv'):
        df5 = pd.read_csv(input_dir + f'shares/{fsym_id}.csv')
        df5['date'] = pd.to_datetime(df5['date'])
        df5 = df5[['date', 'totalOutstanding']]
        df5 = df5.rename(columns={'totalOutstanding': 'total_outstanding', 'currency': 'currency_outstanding'})
    else:
        df5 = pd.DataFrame({'date': [], 'total_outstanding': [], 'currency_outstanding': []})

    df = df.merge(df5, on='date', how='outer')

    # shares outstanding data was downloaded in monthly frequency, so we need to forward/back fill into the daily data
    df['total_outstanding'] = df['total_outstanding'].ffill(limit=90) # forward fill by at most 90 days
    df['total_outstanding'] = df['total_outstanding'].bfill(limit=90) # backward fill by at most 90 days

    # calculate market cap
    df['market_cap_split'] = df['total_outstanding'] * df['price_split']

    # combine and reconcile different sources
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df['price_change'] = df['price'] / df['price'].shift(1) - 1 # pre 2006 data
    df['total_return'] = df['total_return'].fillna(df['price_change'])
    df['market_cap'] = df['market_cap'].fillna(df['market_cap_split'])
    df['price'] = df['price'].fillna(df['price_split'])
    df['volume'] = df['volume'].fillna(df['volume_split'])


    return df

def combine_benchmark_data(data:pd.DataFrame, benchmark_data:pd.DataFrame, benchmark:str='SP500'):

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)
    
    df1 = benchmark_data[benchmark_data['benchmark'] == benchmark].copy()
    df1['date'] = pd.to_datetime(df1['date'])
    df1 = df1.sort_values(by='date')
    df1 = df1.reset_index(drop=True)

    df = df.merge(df1, on='date', how='outer')

    return df
    
    

def calc_capm(data:pd.DataFrame, 
              trailing_periods_list:list=[182,365], 
              frequency:str='ME', 
              outlier_drops:int=4, 
              downside_only:bool=False,
              exponential_weighting:tuple=(False, 0.99)):

    '''
    Calculate the CAPM beta and alpha for a given trailing period.

    - trailing_periods_list: list of trailing periods to calculate the CAPM for (THIS IS THE NUMBER OF DATES SO 1Y is 365)
    - frequency: frequency of the data (e.g. 'ME' for monthly, 'Q' for quarterly)
    - outlier_drops: number of outliers to drop from the data
    - downside_only: if True, only use observations where the benchmark return is negative
    - exponential_weighting: tuple of (True, decay_factor) to use exponential weighting
    '''

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True) 
    df['volume'] = df['volume'].fillna(df['volume_split'])

    # drop nulls and zero volume
    # - zero volume implies zero return and is problematic when comparing against a liquid benchmark return
    df = df[df['total_return'].notnull()]
    df = df[df['volume'].notnull()]
    df = df[df['volume'] != 0]
    df = df[df['benchmark_return'].notnull()]

    # if downside_only is true, only use observations where the benchmark return is negative
    if downside_only:
        df = df[df['benchmark_return'] < 0]

    if df.shape[0] == 0:
        return pd.DataFrame({'date': []})

    # generate the date range to generate capm data
    first_date = df['date'].min()
    last_date = df['date'].max()

    collection1 = pd.DataFrame({'date': []})
    for this_trailing_period in trailing_periods_list:

        suffix = ''
        if downside_only:
            suffix = '_down'
        suffix = suffix + f'_{this_trailing_period}'

        # determine the dates to iterate over
        date_list = pd.date_range(start=first_date, end=last_date, freq=frequency)
        date_list = [d for d in date_list if (d-first_date).days > this_trailing_period]

        collection2 = []
        for this_date in date_list:

            # print(this_date)

            dff = df[df['date'] <= this_date]
            dff = dff[dff['date'] >= (this_date - pd.Timedelta(days=this_trailing_period+outlier_drops))]
            # drop outliers
            dff['total_return_abs'] = dff['total_return'].map(lambda x: abs(x))
            dff = dff.sort_values(by='total_return_abs', ascending=False)
            dff = dff.iloc[outlier_drops:]
            dff = dff.sort_values(by='date')
            dff = dff.reset_index(drop=True)

            # skip this date if there is no data (eg no volume so no returns) and proceed to the next date
            # also skip if only one observation because you need at least two obs to calculate a slope and intercept term
            if dff.shape[0] < 2:
                continue

            # calculate the capm
            dff[['fsym_id', 'date', 'total_return', 'benchmark_return']].to_csv(f'/Users/joeybortfeld/Downloads/temp.csv', index=False)
            y = dff['total_return']
            x = dff['benchmark_return']
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            # print('--', model.params)

            dff[f'capm_beta{suffix}'] = model.params.iloc[1]
            dff[f'capm_alpha{suffix}'] = model.params.iloc[0] * 252
            dff[f'fitted_return'] = model.predict(x)
            dff[f'residual'] = dff['total_return'] - dff['fitted_return']
            dff[f'capm_return_vol{suffix}'] = dff['total_return'].std() * np.sqrt(252)
            dff[f'capm_count{suffix}'] = dff.shape[0]
            dff[f'capm_idio_vol{suffix}'] = dff['residual'].std() * np.sqrt(252)

            dff[f'capm_max_date{suffix}'] = dff['date'].max()
            dff[f'capm_min_date{suffix}'] = dff['date'].min()
            dff[f'date'] = this_date


            # exponential weighting
            decay_factor = exponential_weighting[1]
            if not (0 < decay_factor < 1):
                raise ValueError("Decay factor must be between 0 and 1")

            weights = np.array([(decay_factor**i) for i in range(len(y))])
            weights = weights[::-1]  # Reverse to apply highest weight to recent data
            weights /= weights.sum()  # Normalize weights
            
            dff['weights'] = weights
            dff['weighted_residual'] = dff['residual'] * dff['weights']

            dff[f'capm_idio_vol_weighted{suffix}'] = dff['weighted_residual'].std() * np.sqrt(252)
            dff[f'capm_idio_vol_decay{suffix}'] = decay_factor
            dff[f'capm_benchmark{suffix}'] = dff['benchmark']

            dff = dff[['date', f'capm_min_date{suffix}', f'capm_max_date{suffix}', f'capm_benchmark{suffix}', f'capm_count{suffix}', 
                    f'capm_beta{suffix}', f'capm_alpha{suffix}', f'capm_idio_vol{suffix}', f'capm_return_vol{suffix}',
                    f'capm_idio_vol_weighted{suffix}', f'capm_idio_vol_decay{suffix}']]

            dff = dff.tail(1)

            collection2.append(dff)

        # if any output was generated, concatenate and merge, else don't do anything
        if len(collection2) > 0:
            collection2 = pd.concat(collection2, axis=0)
            collection1 = collection1.merge(collection2, on='date', how='outer')
    

    return collection1

def calc_rolling_returns(data:pd.DataFrame):

    '''
    Calculate the rolling returns for a given trailing 1,2,..., 12 month period
    '''

    df = data[['date', 'price', 'price_split']].copy()

    df['price'] = df['price'].fillna(df['price_split'])
    df = df[df['price'].notnull()]

    for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        df[f'return_{i}'] = df['price'].pct_change(22*i, fill_method=None)
        df['date_i'] = df['date'].shift(22*i)
        df['diff'] = (df['date'] - df['date_i']).dt.days

        mask1 = df['diff'] > (30*i+30)
        mask2 = df['diff'] < (30*i-30)
        df.loc[mask1 | mask2, f'return_{i}'] = np.nan

    df = df.drop(columns=['date_i', 'diff'])

    df['return_smoothed'] = df[['return_1', 'return_2', 'return_3', 'return_4', 'return_5', 'return_6', 
                                'return_7', 'return_8', 'return_9', 'return_10', 'return_11', 'return_12']].mean(axis=1)
    
    df = df.rename(columns={'price': 'return_price'})
    df = df[['date', 'return_price', 'return_1', 'return_2', 'return_3', 'return_6', 'return_12', 'return_smoothed']]

    return df

def calc_drawdown(data:pd.DataFrame):

    # calculate rolling drawdown - difference between current price and the max price from the trailing period

    df = data[['date', 'price', 'price_split']].copy()
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)

    for trailing_obs in [128, 252]:
        rolling_max = df['price'].rolling(window=trailing_obs, min_periods=trailing_obs).max()
        df[f'drawdown_{trailing_obs}'] = (df['price'] - rolling_max) / rolling_max


    return df[['date', 'drawdown_128', 'drawdown_252']]

def calc_downside_volatility(data:pd.DataFrame):

    df = data[['date', 'total_return', 'benchmark_return']].copy()
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)
    df = df[df['total_return'].notnull()]

    for trailing_obs in [128, 252]:
        df[f'downside_vol_{trailing_obs}'] = df['total_return'].rolling(window=trailing_obs, min_periods=trailing_obs).apply(lambda x: x[x < 0].std() * np.sqrt(252))

    return df[['date', 'downside_vol_128', 'downside_vol_252']]


def calc_ulcer_index(data:pd.DataFrame):

    df = data[['date', 'price']].copy()
    df = df[df['price'].notnull()]



    for trailing_obs in [128, 252]:
        rolling_max = df.rolling(trailing_obs, min_periods=trailing_obs)['price'].max()
        drawdown = (df['price'] / rolling_max) - 1
        df[f'ulcer_index_{trailing_obs}'] = np.sqrt((drawdown**2).rolling(trailing_obs, min_periods=trailing_obs).mean())

    return df[['date', 'ulcer_index_128', 'ulcer_index_252']]

def merton_distance_to_default(market_cap, debt, equity_vol, risk_free_rate=0.03, time_horizon=1):
    """ Computes Distance to Default (DD) using Merton's structural model. """
    
    # Initial guess: Assume asset value is close to market cap
    asset_value = market_cap
    asset_vol = equity_vol  # Approximate initial asset volatility

    def equations(vars):
        A, sigma_A = vars
        d1 = (np.log(A / debt) + (risk_free_rate + 0.5 * sigma_A ** 2) * time_horizon) / (sigma_A * np.sqrt(time_horizon))
        d2 = d1 - sigma_A * np.sqrt(time_horizon)

        eq1 = market_cap - (A * norm.cdf(d1) - np.exp(-risk_free_rate * time_horizon) * debt * norm.cdf(d2))
        eq2 = equity_vol * market_cap - norm.cdf(d1) * A * sigma_A

        return [eq1, eq2]

    # Solve for asset value (A) and asset volatility (sigma_A)
    A, sigma_A = fsolve(equations, [asset_value, asset_vol])

    # Compute Distance to Default
    d1 = (np.log(A / debt) + (risk_free_rate + 0.5 * sigma_A ** 2) * time_horizon) / (sigma_A * np.sqrt(time_horizon))
    d2 = d1 - sigma_A * np.sqrt(time_horizon)
    
    distance_to_default = d2
    probability_of_default = norm.cdf(-distance_to_default)

    return distance_to_default, probability_of_default

    # iterate over all applicable fsym_ids


# build a dictionary with mappings from fsym_id to exchange country
def build_fsym_to_country_dict(mapping_file:str='/Users/joeybortfeld/Documents/QML Solutions Data/universe_and_traits/qml_universe_ids.csv',
                               country_to_region_dict:dict=None):
    temp = pd.read_csv(mapping_file)

    fsym_to_country_dict = temp.set_index('fsym_id')['exchange_country'].to_dict()
    return fsym_to_country_dict

# build a dictionary with mappings from exchange country to region
def build_country_to_region_dict(mapping_file:str='/Users/joeybortfeld/Documents/QML Solutions Data/universe_and_traits/country_to_region_mapping.csv'):
    temp = pd.read_csv(mapping_file)
    country_to_region_dict = temp.set_index('exchange_country')['region'].to_dict()
    return country_to_region_dict

def build_region_to_benchmark_dict():

    region_to_benchmark_dict = {
        'North America': 'SP500',
        'Europe': 'STOXX',
        'Japan': 'NIKKEI',
        'Asia ex-Japan': 'None',
        'Africa': 'None', 
        'Middle East': 'None',
        'South America': 'None',
        '@NA': 'None',
        np.nan: 'None'
    }

    return region_to_benchmark_dict

def build_equity_ratios(fsym_list:list,
                        data_benchmarks:pd.DataFrame,
                        fsym_to_country_dict:dict,
                        country_to_region_dict:dict,
                        region_to_benchmark_dict:dict,
                        output_dir:str='/Users/joeybortfeld/Documents/QML Solutions Data/factset_data/factset_equity/processed/'):

    for this_fsym in tqdm.tqdm(fsym_list):

        # collect the equity data and consolidate from various sources
        temp = merge_equity_data(fsym_id=this_fsym)

        temp['fsym_id'] = this_fsym
        this_country = fsym_to_country_dict[this_fsym]
        this_region = country_to_region_dict[this_country]
        this_benchmark = region_to_benchmark_dict[this_region]
        
        # skip if fsym_id is from a region that we do not map to a benchmark
        if this_benchmark == 'None':
            continue

        # combine the equity data with the benchmark data
        temp = combine_benchmark_data(temp, data_benchmarks, benchmark=this_benchmark)

        # calculate equity metrics
        temp1a = calc_capm(temp, trailing_periods_list=[182,365], frequency='ME', outlier_drops=4, downside_only=False, exponential_weighting=(False, 0.99))
        temp1b = calc_capm(temp, trailing_periods_list=[182,365], frequency='ME', outlier_drops=4, downside_only=True, exponential_weighting=(False, 0.99))
        temp2 = calc_rolling_returns(temp)
        temp3 = calc_drawdown(temp)
        temp4 = calc_downside_volatility(temp)
        temp5 = calc_ulcer_index(temp)

        # combine the equity metrics
        df = temp[['fsym_id', 'date', 'market_cap', 'price', 'volume', ]].merge(temp1a, on=['date'], how='outer')
        df = df.merge(temp1b, on=['date'], how='outer')
        df = df.merge(temp2, on=['date',], how='outer')
        df = df.merge(temp3, on=['date',], how='outer')
        df = df.merge(temp4, on=['date',], how='outer')
        df = df.merge(temp5, on=['date',], how='outer')

        # fillin missing 
        # - for the monthly calculations (CAPM, etc) we assume exact month end dates (10/31)
        # - but for the daily data (price, market cap, returns, etc) the last trading date may not be true month end (10/29)
        for c in ['market_cap', 'price', 'volume', 'return_1', 'return_2', 'return_3', 'return_6', 'return_12']:
            df[c] = df[c].ffill(limit=10)
        df['fsym_id'] = this_fsym

        # drop obs out of bounds of the available data date range
        min_date = df[df['price'].notnull()]['date'].min()
        max_date = df[df['price'].notnull()]['date'].max()

        # reduce to monthly
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['max_date'] = df.groupby(['year', 'month'])['date'].transform('max')
        df = df[df['date'] == df['max_date']]
        df = df.drop(columns=['year', 'month', 'max_date'])

        df = df[df['date'] >= min_date]
        df = df[df['date'] <= max_date]

        df.to_csv(output_dir + f'{this_fsym}.csv', index=False)
        collection.append(df)

    return None
    





