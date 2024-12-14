import credentials

import pandas as pd
import numpy as np
import time
import datetime
import requests
import requests
from bs4 import BeautifulSoup
import tqdm
import statsmodels.api as sm
import os
import sys

# import your credentials for financialmodelingprep API
API_KEY = credentials.fmp_api_key

def get_company_profile(ticker:str):
    url = f'https://financialmodelingprep.com/api/v3/company/profile/{ticker}'
    params = {
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def get_financial_statement(ticker:str, statement='income', period:str='annual'):

    '''
    Function to get income statement, balance sheet or cash-flow data for a given company
    '''

    assert period in ['annual', 'quarter'], 'period must be either annual or quarter'
    assert statement in ['income', 'balance-sheet', 'cash-flow'], 'statement must be either income, balance-sheet, or cash-flow'

    url = f'https://financialmodelingprep.com/api/v3/{statement}-statement/{ticker}?period={period}'
    params = {
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)

    # add a timestamp for the date of data download
    df['data_download_timestamp'] = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    return df

def get_sp500_universe():
    url = "https://financialmodelingprep.com/api/v3/sp500_constituent"
    params = {
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)

    # add a timestamp for the date of data download
    df['data_download_timestamp'] = pd.to_datetime('today').strftime('%Y-%m-%d')

    df = df.sort_values(by='name')
    df = df.reset_index(drop=True)

    return df



def build_company_financials(ticker:str, period:str='annual'):
    '''
    Function to build a a combined dataframe of income statement, balance sheet, 
    and cash-flow data for a given company
    '''

    # init an empty dataframe to store the data
    df = pd.DataFrame({'date': [], 'symbol':[], 'reportedCurrency':[],
                       'cik':[]
                       })
    key_vars = ['date', 'symbol', 'reportedCurrency', 'cik']

    # iterate over the three financial statements
    column_dict = {}
    for this_statement in ['income', 'balance-sheet', 'cash-flow']:
        
        dff = get_financial_statement(ticker=ticker, statement=this_statement, period=period)

        # rename columns to avoid duplication
        rename_cols = ['calendarYear', 'period', 'fillingDate', 'acceptedDate','data_download_timestamp']
        dff = dff.rename(columns={c: f'{c}_{this_statement}' for c in rename_cols})
        if this_statement == 'cash-flow':
            dff = dff.rename(columns={'inventory': 'inventory_cash_flow',
                                      'netIncome': 'netIncome_cash_flow',
                                      'depreciationAndAmortization': 'depreciationAndAmortization_cash_flow',})
        dff = dff.drop(['link', 'finalLink'], axis=1)

        column_dict[this_statement] = dff.columns.tolist()

        # merge the data
        df = df.merge(dff, how='outer', on=key_vars)

    df = df.sort_values(by='date', ascending=False)
    df = df.reset_index(drop=True)

    return df, column_dict

def bulk_build_company_financials(tickers:list, period:str='annual', verbose:bool=False):

    '''
    Function to build financial statement data (income, balance-sheet, cash-flow) for a list of tickers
    '''

    if verbose:
        print('start bulk_build_company_financials for {} tickers'.format(len(tickers)))

    collection = []
    start = time.time()
    counter  = 0
    error_list = []
    for this_ticker in tqdm.tqdm(tickers):
        
        counter += 1
        # pause so we don't get blocked by the API
        if counter % 50 == 0:
            time.sleep(2)
        
        try:
            df, _ = build_company_financials(ticker=this_ticker, period=period)
            collection.append(df)
        except:
            print(f'error in build_company_financials with {this_ticker}')
            error_list.append(this_ticker)
            continue

    df = pd.concat(collection, axis=0)

    if verbose:
        print('--done all in {:.2f}s'.format(time.time() - start))

    return df, error_list

def format_quarterly_data(data:pd.DataFrame, ltm_vars:list):
    '''
    '''

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])

    key_vars = ['date', 'symbol']

    # 0. rename columns to indicate that they are from the quarterly filing'
    df = df.rename(columns={c: c+'_quarterly' for c in df.columns if c not in key_vars})

    # 1. drop ratio columns - these aren't needed as we will construct our own ratios
    drop_cols = [c for c in df.columns if 'Ratio' in c]
    df = df.drop(drop_cols, axis=1)

    drop_cols = [c for c in df.columns if 'ratio' in c]
    df = df.drop(drop_cols, axis=1)

    # 2. calculate ltm values
    df = df.sort_values(by=['symbol', 'date'])
    df = df.reset_index(drop=True)

    # validate that the preceding 3 rows are appropriate dates
    # within roughly N quarters of the given date
    date_masks = {}
    for i in [1,2,3]:
        df['date_diff'] = df['date'] - df['date'].shift(i)
        df['date_diff'] = df['date_diff'].dt.days
        date_masks[i] = (df['date_diff'] < (35 * 3 * i)) & (df['date_diff'] > (25 * 3 * (i-1)))
    valid_dates_mask = date_masks[1] & date_masks[2] & date_masks[3]

    # apply ltm transformation which assumes quarterly data
    ltm_vars2 = [c+'_quarterly' for c in ltm_vars]
    for c in ltm_vars2:
        df[c+'_ltm'] = df.groupby('symbol')[c].transform(lambda x: x.rolling(window=4).sum())

    # null out ltm values where the dates are not valid
    for c in ltm_vars2:
        df.loc[~valid_dates_mask, f'{c}_ltm'] = np.nan

    
    return df

def format_annual_data(data:pd.DataFrame):
    
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])

    key_vars = ['date', 'symbol']

    # 0. rename columns to indicate that they are from the quarterly filing'
    df = df.rename(columns={c: c+'_annual' for c in df.columns if c not in key_vars})

    # 1. drop ratio columns - these aren't needed as we will construct our own ratios
    drop_cols = [c for c in df.columns if 'ratio' in c]
    df = df.drop(drop_cols, axis=1)

    return df

def merge_quarterly_annual(quarterly:pd.DataFrame, annual:pd.DataFrame, 
                           ltm_vars: list, stock_vars:list,
                           cleanup:bool=True) -> pd.DataFrame:
    """
    Merge quarterly and annual financial data.
    """
    # 0. merge quarterly and annual data
    merge_keys = ['symbol', 'date']
    df = pd.merge(quarterly, annual, on=merge_keys, how='outer')
    
    # sort by ticker and symbol
    df = df.sort_values(['symbol', 'date'])
    
    # 1. diagnostics checks after merging quarterly and annual data
    # currency matches
    mask1 = df['reportedCurrency_quarterly'] != df['reportedCurrency_annual']
    mask2 = df['cik_quarterly'].notnull()
    mask3 = df['cik_annual'].notnull()
    mask = mask1 & mask2 & mask3
    if mask.sum() > 0:
        print('error: reported currencies for quarterly and annual data do not match')
        temp = df.loc[mask, :]
        print(temp.symbol.unique())
        print(temp[['symbol', 'date', 'reportedCurrency_quarterly', 'reportedCurrency_annual', 'cik_quarterly', 'cik_annual']])
        print()
        
        df = df[-mask]


    # no duplicate company-date rows
    mask = df.duplicated(subset=['symbol', 'date'], keep=False)
    if mask.sum() > 0:
        print('error: duplicate symbol-date rows')
        temp = df.loc[mask, :]
        print(temp.symbol.unique())
        print(temp[['symbol', 'date', 'reportedCurrency_quarterly', 'reportedCurrency_annual', 'cik_quarterly', 'cik_annual']])
        print()
        df = df[-mask]

    # 2. reconcile quarterly and annual data
    # - default to use quarterly data but if missing use annual data
    for c in ltm_vars:
        df[f'{c}_ltm'] = df[f'{c}_quarterly_ltm'].fillna(df[f'{c}_annual'])

    for c in stock_vars:
        df[c] = df[f'{c}_quarterly'].fillna(df[f'{c}_annual'])

    # 3. cleanup, if applicable
    if cleanup:
        for c in ltm_vars:
            df = df.drop(columns=[f'{c}_quarterly_ltm', f'{c}_quarterly', f'{c}_annual'], axis=1)
        for c in stock_vars:
            df = df.drop(columns=[f'{c}_quarterly', f'{c}_annual'], axis=1)

    return df

def download_stock_returns(ticker:str='SP500', start_date:str='1990-01-01'):
    """
    Download the daily total returns of a ticker as a time series using the FinancialModelingPrep API.
    
    Returns:
    pandas.DataFrame: DataFrame containing the daily total returns of the S&P 500 with columns 'date' and 'index_total_return'
    """
    # API endpoint URL
    if ticker == 'SP500':
        url = 'https://financialmodelingprep.com/api/v3/historical-price-full/index/%5EGSPC'
    else:
          
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}'
    # API parameters
    params = {
        "apikey": API_KEY,
        "from": start_date  # Set the start date to 1990-01-01
    }

    # Send GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to JSON format
        data = response.json()
        symbol = data['symbol']

        historical_prices = data['historical']
        
        # Create a DataFrame from the historical prices
        df = pd.DataFrame(historical_prices)
        
        # Select the 'date' and 'adjClose' columns
        df['date'] = pd.to_datetime(df['date'])
        df['ticker'] = symbol
        df = df[['ticker', 'date', 'adjClose', 'volume']]
        df = df.sort_values(by='date')
        df = df.reset_index(drop=True)
        df['total_return'] = df['adjClose'].pct_change()
        return df

    
    else:
        print('Failed to download data. Status code:', response.status_code)
        return None

def capm_regression(data:pd.DataFrame, rolling_period:int=126, exponential_weighting:tuple=(False, 0.99)):
    """
    Perform rolling CAPM regression on a dataframe with total_return and index_total_return columns and return
    the beta, alpha, and idiosyncratic volatility as a new dataframe. 
    
    Args:
    df (pandas.DataFrame): DataFrame containing 'total_return' and 'index_total_return' columns
    rolling_period (int): Number of periods to use in each regression

    Returns:
    pandas.DataFrame: DataFrame with 'date', 'beta', 'alpha', and 'idio_vol' columns

    """

    df = data.copy()

    # drop observations where missing company return or index return
    df = df.dropna(subset=['total_return', 'index_total_return'])

    # drop observations where no volume
    df = df[df['volume'] > 0]

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.reset_index(drop=True)

    df['beta'] = np.nan
    df['alpha'] = np.nan
    df['vol'] = np.nan
    df['idio_vol'] = np.nan
    df['idio_vol_count'] = np.nan

    for i in range(rolling_period, df.shape[0]):
        temp = df.iloc[i-rolling_period:i, :]

        min_date = temp['date'].min()
        y = temp['total_return']
        X = temp['index_total_return']
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        df.loc[i, 'beta'] = model.params['index_total_return']
        df.loc[i, 'alpha'] = model.params['const'] * 252

        # calculate volatility using ewma
        if exponential_weighting[0] == True:
            decay_factor = exponential_weighting[1]
            if not (0 < decay_factor < 1):
                raise ValueError("Decay factor must be between 0 and 1")

            weights = np.array([(decay_factor**i) for i in range(len(y))])
            weights = weights[::-1]  # Reverse to apply highest weight to recent data
            weights /= weights.sum()  # Normalize weights
            
            weighted_y = y * weights
            weighted_resid = model.resid * weights

            df.loc[i, 'vol'] = np.std(weighted_y) * np.sqrt(252)
            df.loc[i, 'idio_vol'] = np.std(weighted_resid) * np.sqrt(252)

        else:
            df.loc[i, 'vol'] = np.std(y) * np.sqrt(252)
            df.loc[i, 'idio_vol'] = np.std(model.resid) * np.sqrt(252)
        df.loc[i, 'idio_vol_count'] = model.resid.shape[0]
        df.loc[i, 'min_date'] = min_date

    # null cases where not enough observations
    # - this happens when the min_date is not N rolling periods from the current date
    df['date_diff'] = (df['date'] - df['min_date']).dt.days
    mask = df['date_diff'] < (rolling_period*0.90)
    df['drop_capm'] = 0
    df.loc[mask, 'drop_capm'] = 1


    return df[['date', 'beta', 'alpha', 'vol', 'idio_vol', 'idio_vol_count', 'min_date', 'drop_capm']]


    
def check_recent_filings(start_date:str=None,
                         filter_list:list=None):

    '''
    Get a list of all recent filings from the SEC.
    '''

    if start_date is None:
        start_date = pd.to_datetime('today') - pd.DateOffset(days=10)
        start_date = start_date.strftime('%Y-%m-%d')
    tomorrow = pd.to_datetime('today') + pd.DateOffset(days=1)

    url = f'https://financialmodelingprep.com/api/v4/rss_feed?limit=100&type=10&from=2021-03-10&to=2021-05-04&isDone=true'
    url = f'https://financialmodelingprep.com/api/v4/rss_feed?type=10&from={start_date}&to={tomorrow}&isDone=true'
    params = {
        "apikey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)

    if filter_list is not None:
        df = df[df['ticker'].isin(filter_list)]

    df = df.reset_index(drop=True)

    return df
