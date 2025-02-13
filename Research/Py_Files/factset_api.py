
import json
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import tqdm
import time

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

flow_var_list = ['ff_ebitda_oper', 'ff_ebit_oper', 'ff_net_inc', 
               'ff_sales', 'ff_cogs',
               'ff_int_exp_net', 'ff_capex',
               'ff_funds_oper_gross', 'ff_free_cf', 
               'ff_oper_cf', 'ff_div_com_cf']

stock_var_list = ['ff_assets', 'ff_cash_st', 'ff_assets_curr', 'ff_intang', 'ff_gw', 'ff_ppe_gross',
                  'ff_debt_lt', 'ff_debt_st', 'ff_pens_liabs_unfunded', 'ff_liabs_curr', 'ff_liabs',
                  'ff_com_eq', 'ff_pfd_stk', 'ff_min_int_accum']

def batch_a_list(a_list, batch_size):

    ''' Convert a list into a list of lists, each of size batch_size '''

    return [a_list[i:i+batch_size] for i in range(0, len(a_list), batch_size)]


def load_universe_dict(data:pd.DataFrame):

    ''' Load set of all fsym_ids into a dictionary using the master fsym_id universe file '''

    print('start load_universe_dict()')

    universe_dict = {}

    # save the full universe fsym_ids as a list

    # drop entities with no fsym_id
    df = data[data['fsym_id'] != '@NA']

    # load a list of all fsym_ids
    full_list = list(df['fsym_id'].unique())
    universe_dict['full'] = full_list

    # save fsym_id universe at various asset thresholds
    for this_threshold in [('$10B', 10_000), ('$5B', 5_000), ('$1B', 1_000), ('$500M', 500), ('$250M', 250), ('$100M', 100)]:
        temp = df[df['max_assets_in_usd'] > this_threshold[1]].copy()
        temp = temp[temp['fsym_id'] != '@NA']
        temp = temp['fsym_id'].unique()
        universe_dict[this_threshold[0]] = temp

    # special $100M universe for US and non-financial
    for this_threshold in [('us_nonfin_1b', 1_000), ('us_nonfin_500m', 500), ('us_nonfin_100m', 100)]:
        temp = df[df['max_assets_in_usd'] > this_threshold[1]].copy()
        temp = temp[temp['exchange_country'].isin(['UNITED STATES'])]
        mask1 = temp['factset_econ_sector'] == 'Financials'
        mask2 = temp['factset_industry'] != 'Real Estate Development'
        temp = temp[~(mask1 & mask2)]
        temp = temp[temp['fsym_id'] != '@NA']
        temp = temp['fsym_id'].unique()
        universe_dict[this_threshold[0]] = temp


    print('-- universe counts:')
    for k,v in universe_dict.items():
        print('-- ',k,':', len(v))

    return universe_dict

def download_fundamentals(id_list:list=['MH33D6-R',],
                          field_list:list=['ff_assets', 'ff_liabs'],
                          periodicity:str='ANN',
                          start_date:str='1990-01-01',
                          end_date:str='2024-12-31',
                          currency:str='LOCAL',
                          update_type:str='RP',
                            verbose:bool=False,
                            authorization=None):

    fundamentals_endpoint = 'https://api.factset.com/content/factset-fundamentals/v2/fundamentals'
    fundamentals_request={
        "data": {
        "ids": id_list,
        "periodicity": periodicity,
        "fiscalPeriod": {
        "start": start_date,
        "end": end_date
        },
        "metrics": field_list,
        "currency": currency,
        "updateType": update_type
    }
    }
    headers = {'Accept': 'application/json','Content-Type': 'application/json'}

    #create a post request
    fundamentals_post = json.dumps(fundamentals_request)
    if verbose:
        print('POST request:')
        print(fundamentals_endpoint)
        print(fundamentals_post)
        print()

    fundamentals_response = requests.post(url = fundamentals_endpoint, 
                                            data=fundamentals_post, 
                                            auth = authorization, 
                                            headers = headers, 
                                            verify= False)
    if verbose:
        print('HTTP Status: {}'.format(fundamentals_response.status_code))
        print(fundamentals_response.text)

    # create a dataframe from POST request, show dataframe properties
    fundamentals_data = json.loads(fundamentals_response.text)

    if fundamentals_response.status_code == 200:
        return [fundamentals_response.status_code, pd.DataFrame(fundamentals_data['data'])]
    else:
        return [fundamentals_response.status_code, None]
    

def batch_fundamental_download(fsym_list:list,
                               field_list:list=['FF_ASSETS'],
                               currency:str='LOCAL',
                               periodicity_list:list=['annual', 'quarterly', 'semi_annual'],
                               start_date:str='1990-01-01',
                               end_date='2024-12-31',
                               skip_if_done:bool=True,
                               output_folder:str='/Users/joeybortfeld/Documents/CreditGradients Data/Factset Data/factset_fundamentals/',
                               factset_api_authorization:str=None):
    

    '''
    Download the fundamentals data for a list of fsym_ids using the Factset Fundamentals API and write results to a separate csv file for each fsym_id.
    '''

    print('start batch_fundamental_download()')

    download_type_dict = {'annual': ['ANN', 20], 
                      'quarterly': ['QTR', 10],
                      'semi_annual': ['SEMI', 15]}

    for this_periodicity in periodicity_list:

        print('working on periodicity:', this_periodicity)

        # 0. create the output folder if it doesn't exist
        try:
            os.mkdir(output_folder + this_periodicity)
            print('-- created output folder:', output_folder + this_periodicity)
        except:
            print('-- output folder already exists:', output_folder + this_periodicity)
            pass

        # 1. screen the list of fsym_ids to see if they are already in the output folder
        company_set = fsym_list
        print('-- original fsym_id count:', len(fsym_list))


        # prescreen to see if the fsym_id is already in the output folder
        if skip_if_done:
            print('--skipping files that are already done')
            output_files = os.listdir(output_folder + this_periodicity + '/')
            print('--file count in output folder:', len(output_files))
            output_files = [e.replace('.csv', '') for e in output_files]    
            company_set = [e for e in company_set if e not in output_files]

            print('--new company set size when skipping:', len(company_set))

        
        # 2. loop through fsym_ids and download data
        error_list = []
        start_time = time.time()
        print('-- start download')
        for this_fsym in tqdm.tqdm(company_set):

            try:
                result = download_fundamentals(id_list=[this_fsym],
                                            field_list=field_list,
                                            periodicity=download_type_dict[this_periodicity][0],
                                            start_date=start_date,
                                            end_date=end_date,
                                            currency=currency,
                                            verbose=False,
                                            authorization=factset_api_authorization)
                response_code, fundamentals_df = result

                if response_code != 200:
                    error_list.append(this_fysm)
                else:

                    fundamentals_df.to_csv(output_folder + this_periodicity + '/' + f'{this_fsym}.csv', index=False)

            except:
                error_list.append(this_fsym)

        print('-- done first pass in {}m'.format((time.time() - start_time) / 60))

        if len(error_list) > 0:
            print('-- errors in download:', len(error_list))

            # try downloading the error list
            error_list2 = []
            print('-- retrying failed downloads')
            for this_fsym in tqdm.tqdm(error_list):

                try:
                    result = download_fundamentals(id_list=[this_fsym],
                                                field_list=field_list,
                                                periodicity=download_type_dict[this_periodicity][0],
                                                start_date=start_date,
                                                end_date=end_date,
                                                currency=currency,
                                                verbose=False,
                                                authorization=factset_api_authorization)
                    response_code, fundamentals_df = result

                    if response_code != 200:
                        error_list2.append(this_fysm)
                    else:

                        fundamentals_df.to_csv(output_folder + this_periodicity + '/' + f'{this_fsym}.csv', index=False)

                except:
                    error_list2.append(this_fsym)

                print('-- error count after retrying:', len(error_list2))
                print ('-- done')
                


        else:
            print('--  No errors after second pass')
            print('-- done')


        print()


                


    
def download_company_profile(id_list:list=['AAPL-US',],
                            authorization=None,
                            verbose:bool=False):
    
    endpoint = 'https://api.factset.com/content/factset-fundamentals/v2/company-reports/profile?ids=' + ','.join(id_list)
    if verbose:
        print(endpoint)
    headers = {'Accept': 'application/json','Content-Type': 'application/json'}
    response = requests.get(url = endpoint, auth = authorization, headers = headers, verify= False )
    if verbose:
        print(response.status_code)
    if response.status_code == 200:
        temp = pd.DataFrame(json.loads(response.text)['data'])
        return [200, temp]
    else:
        return [response.status_code, None]


def get_stock_prices(id_list:str=['MH33D6-R'], 
                     field_list:list=["price", "volume", "tradeCount"], 
                     start_date:str='2006-01-03', 
                     end_date:str='2024-12-31', 
                     frequency:str='D',
                     split:str='SPLIT',
                     verbose:bool=False,
                     authorization=None):

    '''
    Get stock prices for a given ticker.

    Split is either SPLIT, SPLIT_SPINOFF', UNSPLIT. use SPLIT to be consistent with the shares outstanding API which is split adjusted. But if
    you are using financial statement data then those share counts would be UNSPLIT. 
    '''

    prices_endpoint = 'https://api.factset.com/content/factset-global-prices/v1/prices'

    prices_request ={
    "ids": id_list,
        "fields": field_list,
        "startDate":start_date,
        "endDate":end_date,
        "frequency":frequency,
        "adjust":split,


    }




    headers = {'Accept': 'application/json','Content-Type': 'application/json'}

    #create a post request
    prices_post = json.dumps(prices_request)

    if verbose:
        print('post request:')
        print(prices_endpoint)
        print(prices_post)
        print()

    prices_response = requests.post(url = prices_endpoint, data=prices_post, auth = authorization, headers = headers, verify= False )

    if verbose:
        print('HTTP Status: {}'.format(prices_response.status_code))
        print(prices_response.text)

    if prices_response.status_code != 200:
        if verbose:
            print('error: failed to get stock prices')
        return [prices_response.status_code,None]
    else:
        prices_data = json.loads(prices_response.text)
        prices_df = pd.DataFrame(prices_data['data'])
        return [prices_response.status_code, prices_df]

def get_stock_returns(id_list:str=['MH33D6-R'], 
                     start_date:str='2006-01-03', 
                     end_date:str='2024-12-31', 
                     frequency:str='D',
                     verbose:bool=False,
                     authorization=None):

    '''
    Get stock returns.

    '''

    returns_endpoint = 'https://api.factset.com/content/factset-global-prices/v1/returns'

    returns_request ={
    "ids": id_list,
        "startDate":start_date,
        "endDate":end_date,
        "frequency":"D",
        "dividendAdjust": "EXDATE"
    }

    headers = {'Accept': 'application/json','Content-Type': 'application/json'}

    #create a post request
    returns_post = json.dumps(returns_request)

    if verbose:
        print('post request:')
        print(returns_endpoint)
        print(returns_post)
        print()

    returns_response = requests.post(url = returns_endpoint, data=returns_post, auth = authorization, headers = headers, verify= False )

    if verbose:
        print('HTTP Status: {}'.format(returns_response.status_code))
        print(returns_response.text)

    if returns_response.status_code != 200:
        if verbose:
            print('error: failed to get stock prices')
        return [returns_response.status_code,None]
    else:
        returns_data = json.loads(returns_response.text)
        returns_df = pd.DataFrame(returns_data['data'])
        return [returns_response.status_code, returns_df]

def batch_get_stock_data(metric:str='prices', 
                        fsym_list:list=['MH33D6-R'], 
                        start_date:str='2006-01-03', 
                        end_date:str='2024-12-31', 
                        starts_dict:dict=None,
                        frequency:str='D',
                        verbose:bool=False,
                        skip_if_done:bool=True,
                        output_folder:str='/Users/joeybortfeld/Documents/QML Solutions Data/factset_data/factset_prices/prices/',
                        authorization=None):
    
    assert metric in ['prices', 'returns'], 'error: metric must be either prices or returns'

    # 1. screen the list of fsym_ids to see if they are already in the output folder
    company_set = fsym_list
    print('-- original fsym_id count:', len(fsym_list))


    # prescreen to see if the fsym_id is already in the output folder
    if skip_if_done:
        print('--skipping files that are already done')
        output_files = os.listdir(output_folder)
        print('--file count in output folder:', len(output_files))
        output_files = [e.replace('.csv', '') for e in output_files]    
        company_set = [e for e in company_set if e not in output_files]

        print('--new company set size when skipping:', len(company_set))

    # 2. loop through fsym_ids and download data
    error_list = []
    start_time = time.time()
    print('-- start download')

    m_dict = {'prices': 'price', 'returns': 'totalReturn', 'shares': 'totalOutstanding'}    
    m = m_dict[metric]

    for this_fsym in tqdm.tqdm(company_set):

        if metric == 'prices':
            status, df = get_stock_prices(id_list=[this_fsym], 
                                        field_list=["price", "volume", "tradeCount"], 
                                        start_date=start_date, 
                                        end_date=end_date, 
                                        frequency=frequency,
                                        split='SPLIT',
                                        verbose=False,
                                        authorization=authorization)
            
            
        elif metric == 'returns':
            status, df = get_stock_returns(id_list=[this_fsym], 
                                        start_date=start_date, 
                                        end_date=end_date, 
                                        frequency=frequency,
                                        verbose=False,
                                        authorization=authorization)
            
        try:
            # if successful status response, check if data is present
            if m in df.columns:

                if status == 200:
                    if df[m].count() > 0:
                        df.to_csv(output_folder + f'{this_fsym}.csv', index=False)
                    else:
                        error_list.append(this_fsym)
                else:
                    error_list.append(this_fsym)

            # if unsuccessful status response, add to error list
            else:
                error_list.append(this_fsym)
        except:
            error_list.append(this_fsym)
        
        
    # retry failed downloads
    if len(error_list) > 0:
        print('-- retrying failed downloads:', len(error_list))
        error_list2 = []
        for this_fsym in tqdm.tqdm(error_list):
            if metric == 'prices':
                status, df = get_stock_prices(id_list=[this_fsym], 
                                            field_list=["price", "volume", "tradeCount"], 
                                            start_date=start_date, 
                                            end_date=end_date, 
                                            frequency=frequency,
                                            split='UNSPLIT',
                                            verbose=False,
                                            authorization=authorization)
            
            
            elif metric == 'returns':
                status, df = get_stock_returns(id_list=[this_fsym], 
                                            start_date=start_date, 
                                            end_date=end_date, 
                                            frequency=frequency,
                                            verbose=False,
                                            authorization=authorization)
            try:
                if m in df.columns:
                    if status == 200:
                        if df[m].count() > 0:
                            df.to_csv(output_folder + f'{this_fsym}.csv', index=False)
                        else:
                            error_list2.append(this_fsym)
                    else:
                        error_list2.append(this_fsym)
                else:
                    error_list2.append(this_fsym)
            except:
                error_list2.append(this_fsym)

        print('-- done second download attempts')
        print('-- error count:', len(error_list2))
        print('-- done in {}m'.format((time.time() - start_time) / 60))

        return error_list2

    print('-- done first download attempts')
    print('-- error count:', len(error_list))
    print('-- done in {}m'.format((time.time() - start_time) / 60))
    return error_list

            


    print('-- done')
    print('-- error count:', len(error_list))
    print('-- done in {}m'.format((time.time() - start_time) / 60))
    


def get_shares_outanding(id_list:str=['MH33D6-R'], 
                     start_date:str='2006-01-03', 
                     end_date:str='2024-12-31', 
                     frequency:str='D',
                     verbose:bool=False,
                     authorization=None):

    '''
    Get shares outstanding for a given ID
    '''

    shares_endpoint = 'https://api.factset.com/content/factset-global-prices/v1/security-shares'

    shares_request ={
        'data': {
            "ids": id_list,
            "startDate":start_date,
            "endDate":end_date,
            "frequency":frequency,
            "calendar": 'FIVEDAY',
            "batch": "N"
        }
    }

    headers = {'Accept': 'application/json','Content-Type': 'application/json'}

    #create a post request
    shares_post = json.dumps(shares_request)

    if verbose:
        print('post request:')
        print(shares_endpoint)
        print(shares_post)
        print()

    shares_response = requests.post(url=shares_endpoint, 
                                    data=shares_post, 
                                    auth=authorization, 
                                    headers=headers, 
                                    verify=False )

    if verbose:
        print('HTTP Status: {}'.format(shares_response.status_code))
        print(shares_response.text)

    if shares_response.status_code != 200:
        if verbose:
            print('error: failed to get shares outstanding')
        return [shares_response.status_code,None]
    else:
        shares_data = json.loads(shares_response.text)
        shares_df = pd.DataFrame(shares_data['data'])
        return [shares_response.status_code, shares_df]

def batch_get_shares_outanding(fsym_list:list=['MH33D6-R'], 
                              start_date_dict:dict=None, 
                              end_date:str='2024-12-31', 
                              frequency:str='M',
                              verbose:bool=False,
                              skip_if_done:bool=True,
                              output_folder:str='/Users/joeybortfeld/Documents/QML Solutions Data/factset_data/factset_equity/shares/',
                              authorization=None):
    

    assert start_date_dict is not None, 'error: start_date_dict must be provided'

    company_set = fsym_list
    print('-- original fsym_id count:', len(fsym_list))

    # prescreen to see if the fsym_id is already in the output folder
    if skip_if_done:
        print('--skipping files that are already done')
        output_files = os.listdir(output_folder)
        print('--file count in output folder:', len(output_files))
        output_files = [e.replace('.csv', '') for e in output_files]    
        company_set = [e for e in company_set if e not in output_files]

        print('--new company set size when skipping:', len(company_set))

    error_list = []
    for fsym in tqdm.tqdm(company_set):
        status, df = get_shares_outanding(id_list=[fsym], 
                                        start_date=start_date_dict[fsym], 
                                        end_date=end_date, 
                                        frequency=frequency,
                                        verbose=False,
                                        authorization=authorization)
        # if the request response indicates success, proceed to check the data
        if status==200:
            
            # check to see if share data is present
            if 'totalOutstanding' in df.columns:
                # if share data is present, save to csv
                df.to_csv(output_folder + f'{fsym}.csv', index=False)
            else:

               
                # try to increment the start date forward by one month at a time
                # (a start date prior to the first available data will return a dataframe with no column for totalOutstanding)
                found = False
                for i in range(12):

                    # increment the start date by n months
                    _start_date = pd.to_datetime(start_date_dict[fsym]) + pd.DateOffset(months=i)
                    _start_date = _start_date.to_period('M').to_timestamp('M')
                    _start_date = _start_date.strftime('%Y-%m-%d')

                    status, df = get_shares_outanding(id_list=[fsym], 
                                                    start_date=_start_date, 
                                                    end_date=end_date, 
                                                    frequency=frequency,
                                                    verbose=False,
                                                    authorization=authorization)
                    
                    if status == 200:
                        if 'totalOutstanding' in df.columns:
                            df.to_csv(output_folder + f'{fsym}.csv', index=False)
                            found = True
                            break

                if found is False:
                    error_list.append([fsym, 999])
            

        # if the request response indicates failure, add the fsym to the error list
        else:
            error_list.append([fsym, status])

    print('done')
    print('error count:', len(error_list))
    return error_list

ticker_list =  [
      "AAPL-US",
      "MSFT-US",
      "XOM-US",
      "SBUX-US",
      "T-US",
      "NKE-US",
      "CE-US",
      "TOL-US",
      "DOW-US",
      "LLY-US",
    ]

default_set = [
    'P0XT7P-R', # Lehman Brothers
    'LGHWYC-R', # Cloud Peak Energy
    'F87563-R', # Enron
    'C5CPSB-R', # PG&E
    'Q9MX71-R', # Worldcom
    'GYKB0G-R', # Hertz
    'H2ZS9L-R', # Evergrand
    'H1PKGQ-R', # AMR Corporation (American Airlines)
    'XQ6H9X-R', # Eastman Kodak
    'DG71WP-R', # Frontier Communications,
    'P0SXV1-R', # Peabody Energy
    'HN0LVT-R', # Sears Holding Corp
    'H8JXH0-R', # Chesapeake Energy Corporation

]

fundamentals_var_list =  [
    # 'FF_FISCAL_DATE',
    # assets
    "FF_ASSETS",
    'FF_CASH_ST', 
    'FF_ASSETS_CURR', 
    'FF_INTANG',
    'FF_GW',
    'FF_PPE_GROSS',

    # liabilities
    'FF_DEBT_LT', 
    'FF_DEBT_ST', 
    'FF_PENS_LIABS_UNFUNDED',
    'FF_LIABS_CURR', 
    'FF_LIABS',

    # equity
    'FF_COM_EQ', 
    'FF_PFD_STK', 
    'FF_MIN_INT_ACCUM',
    'FF_EQ_TOT',

    # income
    'FF_SALES',
    'FF_COGS',
    'FF_NET_INC',
    'FF_INT_EXP_NET',
    'FF_EBITDA_OPER',
    'FF_EBIT_OPER',


    # cash flow
    'FF_FUNDS_OPER_GROSS',
    'FF_CAPEX',
    'FF_DIV_COM_CF',
    'FF_FREE_CF',
    'FF_OPER_CF',
]


def build_market_cap(fsym_id:str, market_cap_type:str='monthly', factset_dir:str='/Users/joeybortfeld/Documents/CreditGradients Data/Factset Data/'):

    # CALCULATE MARKET CAP
    # use month end price data and combine with shares outstanding data

    # market cap parameters
    ffill_limit_dict = {'monthly': 16, 'daily': 375}
    assert market_cap_type in ['monthly', 'daily'], 'error: market_cap_type must be either monthly or daily'



    # build the set of fsym_ids with semi annual share data
    semi_annual_share_dir = f'{factset_dir}/factset_api_fundamentals_shares_outstanding_semi_annual/'
    semi_annual_share_list = os.listdir(semi_annual_share_dir)
    semi_annual_share_list = [f for f in semi_annual_share_list if f.endswith('.csv')]
    semi_annual_share_list = [f.replace('.csv', '') for f in semi_annual_share_list ]


    # 0. get price data
    if market_cap_type == 'monthly':
        df1 = pd.read_csv(f'{factset_dir}/factset_api_stock_prices_split_month_end/{fsym_id}.csv')
        df1['date'] = pd.to_datetime(df1['date'])
        df1 = df1[['fsymId', 'date', 'price', 'currency']]
    else:
        df1 = pd.read_csv(f'{factset_dir}/factset_api_stock_prices_split/{fsym_id}.csv')
        df1['date'] = pd.to_datetime(df1['date'])
        df1 = df1[['fsymId', 'date', 'price', 'currency']]

    # 1. get shares outstanding
    df2 = pd.read_csv(f'{factset_dir}/factset_api_fundamentals_shares_outstanding_quarterly/{fsym_id}.csv')
    df2['fiscalEndDate'] = pd.to_datetime(df2['fiscalEndDate'])
    df2['epsReportDate'] = pd.to_datetime(df2['epsReportDate'])
    df2['epsReportDate'] = df2['epsReportDate'].fillna(df2['fiscalEndDate'] + pd.Timedelta(days=90))
    df2 = df2[['fsymId', 'epsReportDate', 'value']]
    df2.columns=['fsymId', 'date', 'shares_outstanding_quarterly']

    df3 = pd.read_csv(f'{factset_dir}/factset_api_fundamentals_shares_outstanding_annual/{fsym_id}.csv')
    df3['fiscalEndDate'] = pd.to_datetime(df3['fiscalEndDate'])
    df3['epsReportDate'] = pd.to_datetime(df3['epsReportDate'])
    df3['epsReportDate'] = df3['epsReportDate'].fillna(df3['fiscalEndDate'] + pd.Timedelta(days=90))
    df3 = df3[['fsymId', 'epsReportDate', 'value']]
    df3.columns=['fsymId', 'date', 'shares_outstanding_annual']

    # combine all shares outstanding data
    df4 = df2.merge(df3, how='outer', on=['fsymId', 'date'])

    # check if semi annual data exists
    if fsym_id in semi_annual_share_list:
        df5 = pd.read_csv(f'{factset_dir}/factset_api_fundamentals_shares_outstanding_semi_annual/{fsym_id}.csv')
        df5['fiscalEndDate'] = pd.to_datetime(df5['fiscalEndDate'])
        df5['epsReportDate'] = pd.to_datetime(df5['epsReportDate'])
        df5['epsReportDate'] = df5['epsReportDate'].fillna(df5['fiscalEndDate'] + pd.Timedelta(days=90))
        df5 = df5[['fsymId', 'epsReportDate', 'value']]
        df5.columns=['fsymId', 'date', 'shares_outstanding_semi_annual']

        df4 = df4.merge(df5, how='outer', on=['fsymId', 'date'])

    # 2. merge shares outstanding with monthly prices
    df = df1.merge(df4, how='outer', on=['fsymId', 'date'])

    # 3. data cleaning
    # get the date of the first observation with non null price
    first_date = df[df['price'].notnull()]['date'].min()
    df = df[df['date'] >= first_date]

    # fill forward shares outstanding
    df = df.sort_values(by=['date'])
    df['shares_outstanding'] = df['shares_outstanding_quarterly'].fillna(df['shares_outstanding_annual'])
    df['shares_outstanding'] = df['shares_outstanding'].ffill(limit=ffill_limit_dict[market_cap_type])   # allow 12 months of fill forward (12 row) plut 4 more rows for quarterly financial filings

    # 4. market cap calculation (millions)
    df['market_cap'] = df['price'] * df['shares_outstanding']

    # 5. cleanup
    df = df[df['price'].notnull()]
    df = df[['fsymId', 'date', 'market_cap', 'price', 'shares_outstanding', 'currency']]
    df.columns = ['fsym_id', 'market_cap_date', 'market_cap', 'price', 'shares_outstanding', 'market_cap_currency']
    df = df.sort_values(by=['fsym_id', 'market_cap_date'])
    df = df.reset_index(drop=True)

    df['year'] = df['market_cap_date'].dt.year
    df['month'] = df['market_cap_date'].dt.month

    return df