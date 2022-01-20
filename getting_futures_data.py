import investpy

asset_class_list = ['stocks', 'funds', 'etfs', 'indices', 'currency_crosses', 'bonds', 'commodities', 'certificates', 'crypto', 'news', 'technical']
# futures_list = ['stocks', 'funds', 'etfs', 'indices', 'currency_crosses', 'bonds', 'commodities', 'certificates', 'crypto', 'news', 'technical']
# get_futures_list = ['get_commodities_list()', 'get_funds_list()', 'get_etfs_list()', 'get_indices_list()', 'get_currency_crosses_list()', 'get_bonds_list()', 'get_commodities_list()',
#                     'get_certificates_list()', 'get_crypto_list()', 'get_news_list()', 'get_technical_list()']
futures_list = [('stocks','get_stocks_list()'), ('funds', 'get_funds_list()')]
func_list = [investpy.stocks.get_stocks_list, investpy.funds.get_funds_list]
# for i, j in futures_list:
#     print(investpy.i.j)
#     print(len(investpy.i.j)
#
for func in func_list:
    print(func())




