  

display_name_dict = {'ff_sales_ltm': {'name': 'Sales', 'type': 'numeric', 'category': 'income_statement'},
                     'ff_cogs_ltm': {'name': 'COGS', 'type': 'numeric', 'category': 'income_statement'},
                     'ff_int_exp_net_ltm': {'name': 'Int Expense', 'type': 'numeric', 'category': 'income_statement'},
                     'ff_ebitda_oper_ltm': {'name': 'EBITDA', 'type': 'numeric', 'category': 'income_statement'},
                     'ff_net_inc_ltm': {'name': 'Net Inc', 'type': 'numeric', 'category': 'income_statement'},

                     'ff_assets': {'name': 'Assets', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_cash_st': {'name': 'Cash', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_assets_curr': {'name': 'Current Assets', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_intang': {'name': 'Intangibles', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_gw': {'name': 'Goodwill', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_ppe_gross': {'name': 'PPE', 'type': 'numeric', 'category': 'balance_sheet'},

                     'ff_liabs': {'name': 'Liabilities', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_liabs_curr': {'name': 'Current Liabilities', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_debt_st': {'name': 'ST Debt', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_debt_lt': {'name': 'LT Debt', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_pens_liabs_unfunded': {'name': 'Unfunded Pensions', 'type': 'numeric', 'category': 'balance_sheet'},

                     'ff_com_eq': {'name': 'Common Equity', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_pfd_stk': {'name': 'Pref Stock', 'type': 'numeric', 'category': 'balance_sheet'},
                     'ff_min_int_accum': {'name': 'Minority Interest', 'type': 'numeric', 'category': 'balance_sheet'},

                     'ff_oper_cf_ltm': {'name': 'Oper Cash Flow', 'type': 'numeric', 'category': 'cash_flow_statement'},
                     'ff_free_cf_ltm': {'name': 'Free Cash Flow', 'type': 'numeric', 'category': 'cash_flow_statement'},
                     'ff_funds_oper_gross_ltm': {'name': 'Funds from Op', 'type': 'numeric', 'category': 'cash_flow_statement'},
                     'ff_capex_ltm': {'name': 'CapEx', 'type': 'numeric', 'category': 'cash_flow_statement'},
                     'ff_div_com_cf_ltm': {'name': 'Comm Dividends', 'type': 'numeric', 'category': 'cash_flow_statement'},

                     'fiscal_end_date': {'name': 'Date', 'type': 'date', 'category': 'date'},

                    #  'is_placeholder': {'name': 'Income Statement', 'type': 'boolean', 'category': 'placeholder'},
                    #  'assets_placeholder': {'name': 'Balance Sheet (Assets)', 'type': 'boolean', 'category': 'placeholder'},
                    #  'liabs_placeholder': {'name': 'Balance Sheet (Liabilities)', 'type': 'boolean', 'category': 'placeholder'},
                    #  'cf_placeholder': {'name': 'Cash Flow Statement', 'type': 'boolean', 'category': 'placeholder'},
                    #  'blank': {'name': '', 'type': 'boolean', 'category': 'blank'},
                     
   
                    # SIZE
                    'ff_assets_in_usd': {'name': 'Assets in USD', 'type': 'numeric', 'category': 'size'},

                    # LEVERAGE
                     'net_debt_to_ebitda': {'name': 'Net Debt to EBITDA', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_to_debt_plus_equity': {'name': 'Net Debt to Debt plus Equity', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_to_debt_plus_common_equity': {'name': 'Net Debt to Debt plus Common Equity', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_to_assets': {'name': 'Net Debt to Assets', 'type': 'numeric', 'category': 'leverage'},
                     
                     'total_debt_to_ebitda': {'name': 'Total Debt to EBITDA', 'type': 'numeric', 'category': 'leverage'},
                     'total_debt_to_debt_plus_equity': {'name': 'Total Debt to Debt plus Equity', 'type': 'numeric', 'category': 'leverage'},
                     'total_debt_to_debt_plus_common_equity': {'name': 'Total Debt to Debt plus Common Equity', 'type': 'numeric', 'category': 'leverage'},
                     'total_debt_to_assets': {'name': 'Total Debt to Assets', 'type': 'numeric', 'category': 'leverage'},
                     
                     'net_debt_plus_unfunded_pension_to_ebitda': {'name': 'Net Debt plus Unfunded Pensions to EBITDA', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_plus_unfunded_pension_to_debt_plus_equity': {'name': 'Net Debt plus Unfunded Pensions to Debt plus Equity', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_plus_unfunded_pension_to_debt_plus_common_equity': {'name': 'Net Debt plus Unfunded Pensions to Debt plus Common Equity', 'type': 'numeric', 'category': 'leverage'},
                     'net_debt_plus_unfunded_pension_to_assets': {'name': 'Net Debt plus Unfunded Pensions to Assets', 'type': 'numeric', 'category': 'leverage'},
                     
                     'total_equity_to_ebitda': {'name': 'Total Equity to EBITDA', 'type': 'numeric', 'category': 'leverage'},
                     'total_equity_to_debt_plus_equity': {'name': 'Total Equity to Debt plus Equity', 'type': 'numeric', 'category': 'leverage'},
                     'total_equity_to_debt_plus_common_equity': {'name': 'Total Equity to Debt plus Common Equity', 'type': 'numeric', 'category': 'leverage'},
                     'total_equity_to_assets': {'name': 'Total Equity to Assets', 'type': 'numeric', 'category': 'leverage'},
                     
                     #'COVERAGE': 'Coverage',
                     'ebitda_to_interest_expense': {'name': 'EBITDA to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_to_dividends_ltm_floor': {'name': 'EBITDA to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_to_total_debt': {'name': 'EBITDA to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_to_net_debt': {'name': 'EBITDA to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     'ffo_to_interest_expense': {'name': 'FFO to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'ffo_to_dividends_ltm_floor': {'name': 'FFO to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'ffo_to_total_debt': {'name': 'FFO to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'ffo_to_net_debt': {'name': 'FFO to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     'free_cash_flow_to_interest_expense': {'name': 'Free Cash Flow to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'free_cash_flow_to_dividends_ltm_floor': {'name': 'Free Cash Flow to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'free_cash_flow_to_total_debt': {'name': 'Free Cash Flow to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'free_cash_flow_to_net_debt': {'name': 'Free Cash Flow to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     'operating_cash_flow_to_interest_expense': {'name': 'Operating Cash Flow to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'operating_cash_flow_to_dividends_ltm_floor': {'name': 'Operating Cash Flow to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'operating_cash_flow_to_total_debt': {'name': 'Operating Cash Flow to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'operating_cash_flow_to_net_debt': {'name': 'Operating Cash Flow to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     'ebitda_minus_capex_to_interest_expense': {'name': 'EBITDA minus CapEx to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_to_dividends_ltm_floor': {'name': 'EBITDA minus CapEx to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_to_total_debt': {'name': 'EBITDA minus CapEx to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_to_net_debt': {'name': 'EBITDA minus CapEx to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     'ebitda_minus_capex_interest_to_interest_expense': {'name': 'EBITDA minus CapEx to Interest Expense', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_interest_to_dividends_ltm_floor': {'name': 'EBITDA minus CapEx to Dividends LTM Floor', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_interest_to_total_debt': {'name': 'EBITDA minus CapEx to Total Debt', 'type': 'numeric', 'category': 'coverage'},
                     'ebitda_minus_capex_interest_to_net_debt': {'name': 'EBITDA minus CapEx to Net Debt', 'type': 'numeric', 'category': 'coverage'},
                     
                     #'PROFITABILITY': 'Profitability',
                     'ebit_to_sales': {'name': 'EBIT to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'ebit_to_total_equity': {'name': 'EBIT to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'ebit_to_assets': {'name': 'EBIT to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     'ebitda_to_sales': {'name': 'EBITDA to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_to_total_equity': {'name': 'EBITDA to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_to_assets': {'name': 'EBITDA to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     'sales_minus_cogs_to_sales': {'name': 'Sales minus COGS to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'sales_minus_cogs_to_total_equity': {'name': 'Sales minus COGS to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'sales_minus_cogs_to_assets': {'name': 'Sales minus COGS to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     'net_income_to_sales': {'name': 'Net Income to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'net_income_to_total_equity': {'name': 'Net Income to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'net_income_to_assets': {'name': 'Net Income to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     'ebitda_minus_capex_to_sales': {'name': 'EBITDA minus CapEx to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_minus_capex_to_total_equity': {'name': 'EBITDA minus CapEx to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_minus_capex_to_assets': {'name': 'EBITDA minus CapEx to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     'ebitda_minus_capex_interest_to_sales': {'name': 'EBITDA minus CapEx to Sales', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_minus_capex_interest_to_total_equity': {'name': 'EBITDA minus CapEx to Total Equity', 'type': 'numeric', 'category': 'profitability'},
                     'ebitda_minus_capex_interest_to_assets': {'name': 'EBITDA minus CapEx to Assets', 'type': 'numeric', 'category': 'profitability'},
                     
                     #'LIQUIDITY': 'Liquidity',
                     'current_assets_to_current_liabilities': {'name': 'Current Assets to Current Liabilities', 'type': 'numeric', 'category': 'liquidity'},
                     'current_assets_to_st_debt': {'name': 'Current Assets to ST Debt', 'type': 'numeric', 'category': 'liquidity'},
                     'current_assets_to_total_debt': {'name': 'Current Assets to Total Debt', 'type': 'numeric', 'category': 'liquidity'},
                     
                     'cash_to_current_liabilities': {'name': 'Cash to Current Liabilities', 'type': 'numeric', 'category': 'liquidity'},
                     'cash_to_st_debt': {'name': 'Cash to ST Debt', 'type': 'numeric', 'category': 'liquidity'},
                     'cash_to_total_debt': {'name': 'Cash to Total Debt', 'type': 'numeric', 'category': 'liquidity'},
                     
                     'st_debt_to_current_liabilities': {'name': 'ST Debt to Current Liabilities', 'type': 'numeric', 'category': 'liquidity'},
                     'st_debt_to_st_debt': {'name': 'ST Debt to ST Debt', 'type': 'numeric', 'category': 'liquidity'},
                     'st_debt_to_total_debt': {'name': 'ST Debt to Total Debt', 'type': 'numeric', 'category': 'liquidity'},

                     # VOLATILITY
                     'net_income_vol': {'name': 'Net Income Volatility', 'type': 'numeric', 'category': 'volatility'},
                     'sales_vol': {'name': 'Sales Volatility', 'type': 'numeric', 'category': 'volatility'},
                     'ebitda_vol': {'name': 'EBITDA Volatility', 'type': 'numeric', 'category': 'volatility'},
                     'ebit_vol': {'name': 'EBIT Volatility', 'type': 'numeric', 'category': 'volatility'},

                     # INTERACTION TERMS
                    #  'size_x_leverage': {'name': 'Size x Leverage', 'type': 'numeric', 'category': 'interaction'},
                    #  'size_x_profitability': {'name': 'Size x Profitability', 'type': 'numeric', 'category': 'interaction'},

                     
                     }

fin_metrics_list = ['ff_net_inc_ltm', 'ff_ebitda_oper_ltm', 'ff_sales_ltm', 'ff_cogs_ltm', 'ff_int_exp_net_ltm', 
                    'ff_assets', 'ff_cash_st', 'ff_assets_curr', 'ff_intang', 'ff_gw', 'ff_ppe_gross',
                    'ff_liabs', 'ff_liabs_curr', 'ff_debt_st', 'ff_debt_lt', 'ff_pens_liabs_unfunded', 
                    'ff_com_eq', 'ff_pfd_stk', 'ff_min_int_accum']

bs_metrics_list = ['ff_assets', 'ff_cash_st', 'ff_assets_curr', 'ff_intang', 'ff_gw', 'ff_ppe_gross',
                   'ff_liabs', 'ff_liabs_curr', 'ff_debt_st', 'ff_debt_lt', 'ff_pens_liabs_unfunded',
                   'ff_com_eq', 'ff_pfd_stk', 'ff_min_int_accum']

main_ratios_list = ['net_debt_to_ebitda', 'total_debt_to_ebitda', 'total_equity_to_assets',
                    'ebitda_to_interest_expense', 
                    'ebitda_to_sales', 
                    'current_assets_to_current_liabilities',]

# build lists of each metric type
bs_metrics_list = []
size_ratio_list = []
leverage_ratio_list = []
coverage_ratio_list = []
profitability_ratio_list = []
liquidity_ratio_list = []
volatility_ratio_list = []
interaction_ratio_list = []
ratio_dict = {c: [] for c in ['size', 'leverage', 'coverage', 'profitability', 'liquidity', 'volatility', 'interaction']}

for m in display_name_dict.keys():

    if display_name_dict[m]['category'] == 'size':
        size_ratio_list.append(m)
        ratio_dict['size'].append(m)

    if display_name_dict[m]['category'] == 'balance_sheet':
        bs_metrics_list.append(m)

    if display_name_dict[m]['category'] == 'leverage':
        leverage_ratio_list.append(m)
        ratio_dict['leverage'].append(m)

    if display_name_dict[m]['category'] == 'coverage':
        coverage_ratio_list.append(m)
        ratio_dict['coverage'].append(m)

    if display_name_dict[m]['category'] == 'profitability':
        profitability_ratio_list.append(m)
        ratio_dict['profitability'].append(m)

    if display_name_dict[m]['category'] == 'liquidity':
        liquidity_ratio_list.append(m)
        ratio_dict['liquidity'].append(m)

    if display_name_dict[m]['category'] == 'volatility':
        volatility_ratio_list.append(m)
        ratio_dict['volatility'].append(m)

    if display_name_dict[m]['category'] == 'interaction':
        interaction_ratio_list.append(m)
        ratio_dict['interaction'].append(m)

all_ratios_list = size_ratio_list + leverage_ratio_list + coverage_ratio_list + profitability_ratio_list + liquidity_ratio_list + volatility_ratio_list + interaction_ratio_list
