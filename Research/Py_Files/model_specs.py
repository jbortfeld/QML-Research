model_list = [



 {'model_number': 0, 'model_name': 'sector dummies + variable ratios + test',
 'x1_specs': [('ff_assets_in_usd', 'pct'),('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x2_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x3_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x4_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x5_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'incl_sector_dummies': True,
 'sector_dummies_var': 'factset_econ_sector_broad',
 'sector_var': 'factset_econ_sector_broad', # used for intra-sector AUROC calculations, cross-sector rank correlation
 'test_split_date_list': ['2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01'],
  'write_company_outputs': False,
'write_high_pd_no_default': False,
'boost_defaults': False,
 },

 {'model_number': 1, 'model_name': 'boosted defaults',
 'x1_specs': [('ff_assets_in_usd', 'pct'),('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x2_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x3_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x4_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'x5_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('constant', 'level')],
 'incl_sector_dummies': True,
 'sector_dummies_var': 'factset_econ_sector_broad',
 'sector_var': 'factset_econ_sector_broad', # used for intra-sector AUROC calculations, cross-sector rank correlation
 'test_split_date_list': ['2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01'],
  'write_company_outputs': False,
'write_high_pd_no_default': False,
'boost_defaults': True,
 },


#   {'model_number': 2, 'model_name': 'equity vars',
#  'x1_specs': [('ff_assets_in_usd', 'pct'),('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x2_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x3_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x4_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x5_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'incl_sector_dummies': True,
#  'sector_dummies_var': 'factset_econ_sector_broad',
#  'sector_var': 'factset_econ_sector_broad', # used for intra-sector AUROC calculations, cross-sector rank correlation
#  'test_split_date_list': ['2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01'],
#  'write_company_outputs': False,
#  'write_high_pd_no_default': True,
#  'boost_defaults': False,
#  },

#    {'model_number': 3, 'model_name': 'equity vars + no test',
#  'x1_specs': [('ff_assets_in_usd', 'pct'),('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x2_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x3_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_interest_expense', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x4_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'x5_specs': [('ff_assets_in_usd', 'pct'), ('total_equity_to_assets', 'pct'), ('net_income_to_sales', 'pct'), ('ebitda_to_net_debt', 'pct'), ('cash_to_total_debt', 'pct'), ('ebitda_vol', 'pct'), ('market_leverage', 'pct'), ('capm_idio_vol_182', 'pct'), ('constant', 'level')],
#  'incl_sector_dummies': True,
#  'sector_dummies_var': 'factset_econ_sector_broad',
#  'sector_var': 'factset_econ_sector_broad', # used for intra-sector AUROC calculations, cross-sector rank correlation
#  'test_split_date_list': None,
#  'write_company_outputs': False,
#  'write_high_pd_no_default': True,
#  'boost_defaults': False,
#  },


]