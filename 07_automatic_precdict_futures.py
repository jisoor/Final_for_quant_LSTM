import investpy
import pandas as pd
import FinanceDataReader as fdr   # pip install -U finance-datareader
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'nanumyeongjo'
plt.rcParams['figure.figsize'] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
import datetime
import pickle
# print(fdr.__version__) # 0.9.31

# 선물 데이터 갖고오기
ticker = ''     # ticker은 종목이름을 ticker로 바꾸는 코드 따로 만들기. (유저가 티커만 입력하면 찾아올 수 있게)
name = 'Gold'

# print(investpy.indices.get_index_countries())

commodity_list = investpy.commodities.get_commodities_list()
print(type(commodity_list))  # list
currency_list = []   #직접 작성
etf_list = investpy.etfs.get_etfs_list()
indices_list = investpy.indices.get_indices_list()



# if name in commodity_list:
#     print(investpy.commodities.get_commodity_historical_data(name))
# elif name in currency_list:
#     print(investpy.currency_crosses.get_currency_crosses_historical_data(name))
# elif name in etf_list:
#     print(investpy.etfs.get_etf_historical_data(name))
# else:
#     print(investpy.indices.get_index_historical_data(name)) # indices 는 나라 다 구분해야 하므로 일단 보류.

# yyyy-mm-dd 를 dd/mm/yyyy로 바꿔주기

# 1. 오늘을 어제로 만들어주기 (datetime형태)
Today = datetime.date.today()  # 오늘 날짜 2022-01-21 00:00:00
one_day = datetime.timedelta(days=1) # 1일을 datetime형태로 변환
Yesterday = Today - one_day # 어제

######### datetime을 str으로 바꾸는 절차
dt_stamp = datetime.datetime(Yesterday.year, Yesterday.month, Yesterday.day)  #2022, 01, 21
print(type(dt_stamp))  #datetime.datetime
dt_stamp_str = str(dt_stamp)  # string으로
print(dt_stamp_str)
Yesterday_v = datetime.datetime.strptime(dt_stamp_str.split()[0], '%Y-%m-%d')  # 앞에것만
print(Yesterday_v)
Yesterday = dt_stamp.strftime('%d/%m/%Y')   # 드디어 변환
print(Yesterday)

if name in commodity_list:
   historical_data = investpy.commodities.get_commodity_historical_data(name, '01/04/2008', Yesterday)
elif name in currency_list:
   historical_data = investpy.currency_crosses.get_currency_crosses_historical_data(name, '01/04/2008', Yesterday)
else:
   historical_data = investpy.etfs.get_etf_historical_data(name, '01/04/2008', Yesterday)

historical_data = historical_data[['Close']]
historical_data.rename(columns={'Close':'Price'}, inplace=True)
print(historical_data) # 01/04/2008 부터 어제까지의 데이터 가져옴

# price 데이터 그려보기
# plt.plot(historical_data.index , historical_data['Price'])
# plt.show()

# 전처리 시작
historical_data.info()
print('null:', historical_data.isnull().sum())
print(historical_data.isnull().sum()[0])

if historical_data.isnull().sum()[0] != 0:
   historical_data.fillna(method='ffill', inplace=True) # Nan값처리: 이전값으로 채우기(investing.com에는 결측률이 거의 희박하긴 함.)

print(type(historical_data['Date']))
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
historical_data = historical_data.sort_values('Date')
