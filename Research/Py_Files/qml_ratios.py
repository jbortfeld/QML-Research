import pandas as pd
import numpy as np
import tqdm
import os
import datetime
import re


def consolidate_local_data(folder_path:str):

    print('start consolidate_local_data()')
    print('-- folder path:', folder_path)

    collection = []
    error_list = []
    for file in tqdm.tqdm(os.listdir(folder_path)):
        
        try:
            df = pd.read_csv(folder_path + file)
            collection.append(df)
        except:
            print(f'error reading {file}')
            error_list.append(file)
    df = pd.concat(collection)
    return df, error_list

def consolidate_selected_files(fsym_list:list, folder_path:str):
    collection = []
    error_list = []

    for fsym in tqdm.tqdm(fsym_list):

        try:
            df = pd.read_csv(folder_path + f'{fsym}.csv')
            collection.append(df)
        except:
            error_list.append(fsym)

    df = pd.concat(collection)
    return df, error_list

def consolidate_s3_data(folder_path:str):

    print('start consolidate_s3_data()')

    # TODO

    return NotImplemented


def preprocess_factset_fundamentals(data, verbose=False):


    '''
    Preprocess Factset fundamentals data
    '''

    if verbose:
        print('start preprocess_factset_fundamentals()')

    df = data.copy()

    # drop rows with missing data
    df = df[df['fsymId'] != '@NA']
    df = df[df['fiscalYear'] != 0]
    df = df[df['fiscalYear'] != 1]
    df = df[df['fiscalYear'].notnull()]
    df = df[df['value'].notnull()]

    df = df.drop_duplicates(subset=['fsymId', 'fiscalYear', 'fiscalPeriod', 'metric', 'epsReportDate', 'value'], keep='first')

    # date conversions
    for d in ['fiscalEndDate', 'epsReportDate']:
        df[d] = pd.to_datetime(df[d])

    # substitution for missing epsReportDate 
    # use 3 months after fiscal period end date 
    mask = df['epsReportDate'].isnull()
    df.loc[mask, 'epsReportDate'] = df.loc[mask, 'fiscalEndDate'].map(lambda x: x + datetime.timedelta(days=30*3))

    ########################################################
    # data validation 

    # check for duplicate fsym_id-metric-date
    validation = df.duplicated(subset=['fsymId', 'metric', 'fiscalYear', 'fiscalPeriod']).sum()
    if validation > 0:
        print(validation)
        print('duplicate fsym_id-metric-date found')
        mask = df.duplicated(subset=['fsymId', 'metric', 'fiscalYear', 'fiscalPeriod'], keep=False)
        df[mask].to_csv('/Users/joeybortfeld/Documents/CreditGradients Data/temp.csv', index=False)
        

        # we expect WC6L93-R to have duplicate fiscalYear, fiscalPeriod due to a change in reporting period
        duplicate_fsyms = df[mask]['fsymId'].unique()
        print('duplicate fsyms:', duplicate_fsyms)
        

        # check for duplicate fsym_id-metric-date
        df = df.sort_values(by=['fsymId', 'metric', 'fiscalYear', 'fiscalPeriod', 'fiscalEndDate'])
        df = df.drop_duplicates(subset=['fsymId', 'metric', 'fiscalYear', 'fiscalPeriod'], keep='last')


    # check for multiple publication date
    df['epsReportDate_min'] = df.groupby(['fsymId', 'fiscalEndDate'])['epsReportDate'].transform(lambda x: x.min())
    validation = df['epsReportDate'] != df['epsReportDate_min']
    if validation.sum() > 0:
        print('multiple publication dates found')
        dfasdfasdfasd

    validation = df['fiscalEndDate'].isnull()
    if validation.sum() > 0:
        print('fiscalEndDate is null')
        dfasdfasdfasd
    ########################################################

    # reshape the dataframe
    # - columns are metrics
    # - rows are fsym_id-date
    df = df.drop(columns=['requestId', 'fiscalPeriodLength', 'reportDate', 'epsReportDate_min'])
    index_cols = ['fsymId', 'fiscalYear', 'fiscalPeriod', 'periodicity', 'fiscalEndDate', 'epsReportDate', 'updateType', 'currency']
    df = df.pivot(index=index_cols, columns='metric', values='value')
    df = df.reset_index()

    validation = df.duplicated(subset=['fsymId', 'fiscalEndDate']).sum()
    if validation > 0:
        print('duplicate fsymId-fiscalEndDate found after pivot reshape')
        dfasdfasdfasd

    # rename columns to lowercase, underscore

    # convert camelcase to lowercase underscore
    def camel_to_snake(name):
        """Convert camelCase to snake_case"""
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    df.columns = [camel_to_snake(col) for col in df.columns]
    # df.columns = df.columns.str.lower().str.replace(' ', '_')

    if verbose:
        print(df.columns)
        print(df.shape)
        print('-- fsym_id count:', df['fsym_id'].nunique())
        print('-- min date:', df['fiscal_end_date'].min())
        print('-- max date:', df['fiscal_end_date'].max())

    return df


def format_annual_data(data:pd.DataFrame, flow_vars:list, stock_vars:list, verbose:bool=False):

    '''Rename variables to indicate annual data'''

    if verbose:
        print('start format_annual_data()')

    df = data.copy()
    df['fiscal_end_date'] = pd.to_datetime(df['fiscal_end_date'])
    for v in flow_vars:
        df = df.rename(columns={v:f'{v}_af'})
    for v in stock_vars:
        df = df.rename(columns={v:f'{v}_af'})


    return df   

def format_quarterly_data(data:pd.DataFrame, flow_vars:list, stock_vars:list, verbose:bool=False):

    '''
    Rename variables to indicate quarterly data and apply ltm transformation
    '''

    if verbose:
        print('start format_quarterly_data()')

    df = data.copy()
    df['fiscal_end_date'] = pd.to_datetime(df['fiscal_end_date'])
    for v in flow_vars:
        df = df.rename(columns={v:f'{v}_qf'})
    for v in stock_vars:
        df = df.rename(columns={v:f'{v}_qf'})

    # apply ltm summation
    df = df.sort_values(by=['fsym_id', 'fiscal_end_date'])

    # validate that the preceding 3 rows are appropriate dates
    # within roughly N quarters of the given date
    date_masks = {}
    for i in [1,2,3]:
        df['date_diff'] = df['fiscal_end_date'] - df['fiscal_end_date'].shift(i)
        df['date_diff'] = df['date_diff'].dt.days
        date_masks[i] = (df['date_diff'] < (35 * 3 * i)) & (df['date_diff'] > (25 * 3 * (i-1)))
    valid_dates_mask = date_masks[1] & date_masks[2] & date_masks[3]

    # apply ltm transformation which assumes quarterly data
    ltm_vars2 = [c+'_qf' for c in flow_vars]
    for c in ltm_vars2:
        df[c+'_ltm'] = df.groupby('fsym_id')[c].transform(lambda x: x.rolling(window=4).sum())

    # null out ltm values where the dates are not valid
    for c in ltm_vars2:
        df.loc[~valid_dates_mask, f'{c}_ltm'] = np.nan


    return df

def format_semi_annual_data(data:pd.DataFrame, flow_vars:list, stock_vars:list, verbose:bool=False):

    if verbose:
        print('start format_semi_annual_data()')

    df = data.copy()
    df['fiscal_end_date'] = pd.to_datetime(df['fiscal_end_date'])
    for v in flow_vars:
        df = df.rename(columns={v:f'{v}_saf'})
    for v in stock_vars:
        df = df.rename(columns={v:f'{v}_saf'})

    # apply ltm summation
    df = df.sort_values(by=['fsym_id', 'fiscal_end_date'])

    # validate that the preceding 1 rows are appropriate dates
    # within roughly N semi-annual period of the given date
    date_masks = {}
    for i in [1]:
        df['date_diff'] = df['fiscal_end_date'] - df['fiscal_end_date'].shift(i)
        df['date_diff'] = df['date_diff'].dt.days
        date_masks[i] = (df['date_diff'] < (35 * 3 * i)) & (df['date_diff'] > (25 * 3 * (i-1)))
    valid_dates_mask = date_masks[1]

    # apply ltm transformation which assumes quarterly data
    ltm_vars2 = [c+'_saf' for c in flow_vars]
    for c in ltm_vars2:
        df[c+'_ltm'] = df.groupby('fsym_id')[c].transform(lambda x: x.rolling(window=2).sum())

    # null out ltm values where the dates are not valid
    for c in ltm_vars2:
        df.loc[~valid_dates_mask, f'{c}_ltm'] = np.nan

    return df

def format_assets_in_usd_data(data_annual:pd.DataFrame, data_semi_annual:pd.DataFrame, data_quarterly:pd.DataFrame, cleanup:bool=True):

    mask = data_annual['currency'] != 'USD'
    assert mask.sum() == 0, 'annual data contains non-USD currencies'
    df1 = data_annual[['fsym_id', 'fiscal_end_date', 'ff_assets']]
    df1 = df1.rename(columns={'ff_assets': 'ff_assets_in_usd_af'})
   
    mask = data_semi_annual['currency'] != 'USD'
    assert mask.sum() == 0, 'semi-annual data contains non-USD currencies'
    df2 = data_semi_annual[['fsym_id', 'fiscal_end_date', 'ff_assets']]
    df2 = df2.rename(columns={'ff_assets': 'ff_assets_in_usd_saf'})

    mask = data_quarterly['currency'] != 'USD'
    assert mask.sum() == 0, 'quarterly data contains non-USD currencies'
    df3 = data_quarterly[['fsym_id', 'fiscal_end_date', 'ff_assets']]
    df3 = df3.rename(columns={'ff_assets': 'ff_assets_in_usd_qf'})

    df = df1.merge(df2, on=['fsym_id', 'fiscal_end_date'], how='outer')
    df = df.merge(df3, on=['fsym_id', 'fiscal_end_date'], how='outer')
    df['fiscal_end_date'] = pd.to_datetime(df['fiscal_end_date'])
    df['ff_assets_in_usd'] = df['ff_assets_in_usd_af'].fillna(df['ff_assets_in_usd_saf']).fillna(df['ff_assets_in_usd_qf'])

    if cleanup:
        df = df.drop(columns=['ff_assets_in_usd_af', 'ff_assets_in_usd_saf', 'ff_assets_in_usd_qf'], axis=1)

    return df



def merge_quarterly_semi_and_annual(quarterly:pd.DataFrame, semi_annual:pd.DataFrame, annual:pd.DataFrame, 
                           flow_vars: list, stock_vars:list,
                           cleanup:bool=True) -> pd.DataFrame:
    """
    Merge quarterly and annual financial data.
    """
    # 0. merge quarterly and annual data
    merge_keys = ['fsym_id', 'fiscal_end_date', 'currency']
    df = pd.merge(quarterly, annual, on=merge_keys, how='outer', suffixes=('_qf', '_af'))

    # 2. merge semi-annual
    temp = semi_annual.copy()
    temp = temp.rename(columns={'fiscal_year':'fiscal_year_saf',
                                'fiscal_period': 'fiscal_period_saf',
                                'periodicity': 'periodicity_saf',
                                'eps_report_date': 'eps_report_date_saf',
                                'update_type': 'update_type_saf'})
    df = pd.merge(df, temp, on=merge_keys, how='outer')

    # sort by ticker and symbol
    df = df.sort_values(merge_keys)
    
    # 2. validation: no duplicate company-date rows
    mask = df.duplicated(subset=['fsym_id', 'fiscal_end_date'], keep=False)
    if mask.sum() > 0:
        print('error: duplicate fsym_id-fiscal_end_date rows')
        temp = df.loc[mask, :]
        print(temp.fsym_id.unique())
        print(temp[merge_keys])
        print()
        df = df[-mask]

    # 3. reconcile quarterly, semi_annual and annual data
    # - default to use quarterly data but if missing use annual data
    for c in flow_vars:
        df[f'{c}_ltm'] = df[f'{c}_qf_ltm'].fillna(df[f'{c}_saf_ltm'].fillna(df[f'{c}_af']))

    for c in stock_vars:
        if c != 'ff_pens_liabs_unfunded':
            df[c] = df[f'{c}_qf'].fillna(df[f'{c}_saf'].fillna(df[f'{c}_af']))
        else:
            df[c] = df[f'{c}_qf'].fillna(df[f'{c}_af'])


    # 3. cleanup, if applicable
    if cleanup:
        for c in flow_vars:
            df = df.drop(columns=[f'{c}_qf_ltm', f'{c}_qf', f'{c}_saf_ltm', f'{c}_saf', f'{c}_af'], axis=1)
        for c in stock_vars:
            if c != 'ff_pens_liabs_unfunded':
                df = df.drop(columns=[f'{c}_qf', f'{c}_saf', f'{c}_af'], axis=1)
            else:
                df = df.drop(columns=[f'{c}_qf', f'{c}_af'], axis=1)

    return df




def build_qml_model_ratios(data:pd.DataFrame, verbose:bool=False):

    '''
    Build QML ratios for a given dataframe from factset variables
    '''

    if verbose:
        print('start build_qml_model_ratios()')

    df = data.copy()

    # variable preprocessing
    df['total_debt'] = df['ff_debt_st'].fillna(0) + df['ff_debt_lt'].fillna(0)
    df['net_debt'] = (df['total_debt'] - df['ff_cash_st']).clip(lower=0)
    df['net_debt_plus_unfunded_pension'] = (df['total_debt'] - df['ff_cash_st'] + df['ff_pens_liabs_unfunded'].fillna(0)).clip(lower=0) 

    df['total_equity'] = df['ff_com_eq'].fillna(0) + df['ff_pfd_stk'].fillna(0)
    mask1 = df['ff_com_eq'].isnull()
    mask2 = df['ff_pfd_stk'].isnull()
    df.loc[mask1 & mask2, 'total_equity'] = np.NaN
    
    df['total_equity_plus_minority'] = df['total_equity'] + df['ff_min_int_accum'].fillna(0)

    df['debt_plus_equity'] = df['total_debt'] + df['ff_com_eq'].fillna(0) + df['ff_min_int_accum'].fillna(0)
    df['debt_plus_common_equity'] = df['total_debt'] + df['ff_com_eq'].fillna(0)

    df['ebitda_minus_capex'] = df['ff_ebitda_oper_ltm'] - df['ff_capex_ltm'].fillna(0)
    df['ebitda_minus_capex_interest'] = df['ebitda_minus_capex'] - df['ff_int_exp_net_ltm'].fillna(0)

    df['sales_minus_cogs'] = df['ff_sales_ltm'] - df['ff_cogs_ltm']
    
    # many of these variables can be zero in the data and are used in ratio denominators 
    # so we set a floor to avoid division by zero
    df['interest_expense_floor'] = df['ff_int_exp_net_ltm'].clip(lower=0.01)
    df['dividends_ltm_floor'] = df['ff_div_com_cf_ltm'].clip(lower=0.01)
    df['total_debt_floor'] = df['total_debt'].clip(lower=0.01)
    df['st_debt_floor'] = df['ff_debt_st'].clip(lower=0.01)
    df['current_liabilities_floor'] = df['ff_liabs_curr'].clip(lower=0.01) 
    df['assets_floor'] = df['ff_assets'].clip(lower=0.01)
    df['sales_floor'] = df['ff_sales_ltm'].clip(lower=0.01)

    # these can plausibly be negative in the data so we don't set a floor
    # but we do a replacement to prevent division by zero in ratios
    df['ebitda_ltm_floor'] = df['ff_ebitda_oper_ltm'].replace([0], .0001)
    df['net_debt_floor'] = df['net_debt'].replace([0], .0001)

    # 0. size ratios
    # convert assets to USD
    df['exchange_rate_to_usd'] = 1.0
    df['exchange_rate_to_eur'] = 1.0
    df['ff_assets_in_usd'] = df['ff_assets'] * df['exchange_rate_to_usd']
    df['ff_assets_in_eur'] = df['ff_assets'] * df['exchange_rate_to_eur']
    
    # 1. leverage ratios
    tops = [('net_debt', 'net_debt'),
            ('total_debt', 'total_debt'),
            ('net_debt_plus_unfunded_pension', 'net_debt_plus_unfunded_pension'),
            ('total_equity', 'total_equity')]
    bottoms = [('ebitda', 'ebitda_ltm_floor'),
               ('debt_plus_equity', 'debt_plus_equity'),
               ('debt_plus_common_equity', 'debt_plus_common_equity'),
               ('assets', 'assets_floor')]

    leverage_ratios_list = []
    print('leverage ratios')
    for top in tops:
        for bottom in bottoms:
            this_name = f'{top[0]}_to_{bottom[0]}'
            df[this_name] = df[top[1]] / df[bottom[1]]
            print('--', this_name)
            leverage_ratios_list.append(this_name)
    print()

    # 2. coverage ratios
    tops = [('ebitda', 'ebitda_ltm_floor'),
            ('ffo', 'ff_funds_oper_gross_ltm'),
            ('free_cash_flow', 'ff_free_cf_ltm'),
            ('operating_cash_flow', 'ff_oper_cf_ltm'),
            ('ebitda_minus_capex', 'ebitda_minus_capex'),
            ('ebitda_minus_capex_interest', 'ebitda_minus_capex_interest')]
    bottoms = [('interest_expense', 'interest_expense_floor'),
               ('dividends_ltm_floor', 'dividends_ltm_floor'),
               ('total_debt', 'total_debt_floor'),
               ('net_debt', 'net_debt_floor')]
    coverage_ratios_list = []
    print('coverage ratios')
    for top in tops:
        for bottom in bottoms:
            this_name = f'{top[0]}_to_{bottom[0]}'
            df[this_name] = df[top[1]] / df[bottom[1]]
            print('--', this_name)
            leverage_ratios_list.append(this_name)
    print()

    # 3. profitability ratios
    tops = [('ebit', 'ff_ebit_oper_ltm'),
            ('ebitda', 'ff_ebitda_oper_ltm'),
            ('sales_minus_cogs', 'sales_minus_cogs'),
            ('net_income', 'ff_net_inc_ltm'),
            ('ebitda_minus_capex', 'ebitda_minus_capex'),
            ('ebitda_minus_capex_interest', 'ebitda_minus_capex_interest')]
    bottoms = [('sales', 'sales_floor'),
               ('total_equity', 'total_equity'),
               ('assets', 'assets_floor')]

    profitability_ratios_list = []
    print('profitability ratios')
    for top in tops:
        for bottom in bottoms:
            this_name = f'{top[0]}_to_{bottom[0]}'
            df[this_name] = df[top[1]] / df[bottom[1]]
            print('--',this_name)
            profitability_ratios_list.append(this_name)
    print()

    # 4. earnings volatility / earnings growth
    # pass

    # 5. liquidity ratios
    tops = [('current_assets', 'ff_assets_curr'),
            ('cash', 'ff_cash_st'),
            ('st_debt', 'ff_debt_st')]
    bottoms = [('current_liabilities', 'current_liabilities_floor'),
               ('st_debt', 'st_debt_floor'),
               ('total_debt', 'total_debt_floor')]

    liquidity_ratios_list = []
    print('liquidity ratios')
    for top in tops:
        for bottom in bottoms:
            this_name = f'{top[0]}_to_{bottom[0]}'
            df[this_name] = df[top[1]] / df[bottom[1]]
            print('--',this_name)
            liquidity_ratios_list.append(this_name)
    print()


    return df

def calculate_earnings_volatility(data:pd.DataFrame, freq:str='qf'):

    '''
    Calculate earnings volatility from a dataframe with earnings and date columns
    '''

    df = data.copy()
    df = df.sort_values(by=['fsym_id', 'fiscal_end_date'])
    df = df.reset_index(drop=True)
    df = df[['fsym_id', 'fiscal_end_date', 
             f'ff_net_inc_{freq}_ltm', f'ff_ebitda_oper_{freq}_ltm', 
             f'ff_ebit_oper_{freq}_ltm', f'ff_sales_{freq}_ltm']]

    # iterate over earnings metrics
    for m in [('net_income', f'ff_net_inc_{freq}_ltm'),
              ('ebitda', f'ff_ebitda_oper_{freq}_ltm'),
              ('ebit', f'ff_ebit_oper_{freq}_ltm'),
              ('sales', f'ff_sales_{freq}_ltm')]:
        
        # calculate perior-over-period changes
        df['earnings_change'] = df.groupby(by='fsym_id')[m[1]].transform(lambda x: x.pct_change())

        # apply a cap/floor to period-over-period changes in earnings
        # in order to smooth outliers
        df['earnings_change'] = df['earnings_change'].clip(lower=-2.0, upper=2.0)

        # calculate rolling standard deviation of earnings changes by ticker
        freq_to_window = {'qf':12, 'saf': 6}
        annualization_factor = {'qf':4, 'saf':2}
        window = freq_to_window[freq]
        this_vol = df.groupby(by='fsym_id', as_index=False)['earnings_change'].rolling(window=window, min_periods=window).std()
        df[f'{m[0]}_vol_{freq}'] = this_vol['earnings_change']
        df[f'{m[0]}_vol_{freq}'] = df[f'{m[0]}_vol_{freq}'] * np.sqrt(annualization_factor[freq])

    df = df[['fsym_id', 'fiscal_end_date', 
             f'net_income_vol_{freq}', f'ebitda_vol_{freq}', 
             f'ebit_vol_{freq}', f'sales_vol_{freq}']]

    return df

def build_lseg_ratios(data:pd.DataFrame):

    df = data.copy()

    # missing columns
    df['unfundedPensionAndPostretirementBenefit'] = 0
    df['unrealizedLossOnAvailableForSaleSecurities'] = 0


    # leverage
    df['equity_to_assets'] = df['totalEquity'] / df['totalAssets']
    df['net_debt_to_assets'] = (df['totalDebt'] - df['cashAndCashEquivalents']) / df['totalAssets']
    df['unfunded_pension_to_equity'] = df['unfundedPensionAndPostretirementBenefit'] / df['totalEquity']
    df['intangibles_to_assets'] = df['intangibleAssets'] / df['totalAssets']

    # profitability
    df['return_on_tangible_capital'] = df['operatingIncome_ltm'] / (df['totalAssets'] - df['intangibleAssets'] -df['goodwill'])
    df['net_profit_margin'] = df['netIncome_ltm'] / df['revenue_ltm']
    df['unrealized_losses_to_tangible_capital'] = df['unrealizedLossOnAvailableForSaleSecurities'] / (df['totalAssets'] - df['intangibleAssets'] -df['goodwill'])
    df['unrealized_losses_to_revenues'] = df['unrealizedLossOnAvailableForSaleSecurities'] / df['revenue_ltm']

    # coverage
    df['ebit_to_interest_expense'] = (df['ebitda_ltm'] - df['depreciationAndAmortization_ltm']) / df['interestExpense_ltm']
    df['ebitda_to_interest_expense'] = df['ebitda_ltm'] / df['interestExpense_ltm']
    df['free_cash_flow_to_debt'] = (df['freeCashFlow_ltm']) / df['totalDebt']

    # liquidity
    df['cash_to_debt'] = df['cashAndCashEquivalents'] / df['totalDebt']
    df['st_debt_to_debt'] = df['shortTermDebt'] / df['totalDebt']
    df['quick_ratio'] = (df['cashAndCashEquivalents']) / df['totalCurrentLiabilities']


    # growth and stability

    return df
