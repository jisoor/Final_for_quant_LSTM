import investpy

asset_class_list = ['stocks', 'funds', 'etfs', 'indices', 'currency_crosses', 'bonds', 'commodities', 'certificates', 'cryptos', 'news', 'technical']
# futures_list = ['stocks', 'funds', 'etfs', 'indices', 'currency_crosses', 'bonds', 'commodities', 'certificates', 'cryptos', 'news', 'technical']
# get_futures_list = ['get_commodities_list()', 'get_funds_list()', 'get_etfs_list()', 'get_indices_list()', 'get_currency_crosses_list()', 'get_bonds_list()', 'get_commodities_list()',
#                     'get_certificates_list()', 'get_crypto_list()', 'get_news_list()', 'get_technical_list()']
func_list = [ investpy.indices.get_indices_list ]
# investpy.indices.get_indices_list,  investpy.commodities.get_commodities_list,
#              investpy.currency_crosses.get_currency_crosses_list,
# E-mini S&P 500
# E-mini NASDAQ 100
# Mini DowJones
# E-mini S&P MidCap 400
# Nikkei 225 Dollar-Based
# Nikkei 225 Yen-Based
# E-mini Russell 2000
# print(investpy.get_historical_data('Gold'))
# exit()
for num, func in enumerate(func_list):
    print(func()[:])
    print('.')
    print('.')
    print('.')
    print(num+1, len(func()))
#
# print(investpy.indices.get_index_countries())
# print(len(investpy.indices.get_index_countries()))

# currencies = ['GBP/USD','CAD/USD','USD/JPY','USD/CHF','EUR/USD','AUD/USD','USD/MXN','NZD/USD','USD/ZAR','USD/BRL']
# for currency in currencies:
#     if currency in investpy.currency_crosses.get_currency_crosses_list():
#         print('yes')
#     else:
#         print('no')
#
# if 'Mini Hang Seng Index' in investpy.currency_crosses.get_currency_crosses_list():
#     print('yes')
# else:
#     print('no')

# if 'Micro Gold' in investpy.commodities.get_commodities_list():
#     print('yes')
# else:
#     print('no')







