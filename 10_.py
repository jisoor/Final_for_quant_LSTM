import pandas as pd
import yfinance as yf
import datetime
import time
world_indices = [('^GSPC','S&P 500'), ('^DJI','Dow Jones Industrial Average'),('^IXIC','NASDAQ Composite'), ('^NYA','NYSE COMPOSITE (DJ)')
    ,('^XAX','NYSE AMEX COMPOSITE INDEX'),('^BUK100P', 'Cboe UK 100'),('^RUT','Russell 2000'),('^VIX','Vix'),('^FTSE','FTSE 100'),('^GDAXI','DAX PERFORMANCE-INDEX'),
    ('^FCHI', 'CAC 40'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^BFX', 'BEL 20'),('IMOEX.ME','MOEX Russia Index	'),('^N225', 'Nikkei 225')
    ,('^HSI','HANG SENG INDEX'),('000001.SS', 'SSE Composite Index'),('399001.SZ', 'Shenzhen Component'),('^STI','STI Index')
    ,('^AORD', 'ALL ORDINARIES'),('^BSESN', 'S&P BSE SENSEX'),('^JKSE', 'Jakarta Composite Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
    ('^KS11','KOSPI Composite Index'),('^TWII','TSEC weighted index'),('^BVSP','IBOVESPA'),('^MXX','IPC MEXICO'),('^MERV','MERVAL'),('^TA125.TA', 'TA-125')]
# 일단 S&P500 으로 해봐야지..

# last_30 데이터 먼저 가져오기
col_list = ['High', 'Low', 'Adj Close', 'Change']

last_30_High = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(world_indices[1][1], 'High'))
print(last_30_High.tail())
last_30_Low = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(world_indices[0][1], 'Low'))
print(last_30_Low.tail())
last_30_Close = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(world_indices[0][1], 'Adj Close'))
print(last_30_Close.tail())
last_30_Change = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(world_indices[0][1], 'Change'))
print(last_30_Change.tail())

# user가 예측하고 싶은 날짜가 예를 들어 2022-01-26 ~ 2022-01-28로 3일이라면, 28까지의 데이터를 가져옴
from_date = '2022-01-26'
from_date = pd.to_datetime(from_date)  #timestamps
print(type(from_date))
to_date = '2022-01-28'
to_date = pd.to_datetime(to_date)

now = datetime.datetime.now()
nowTime = now.strftime('%H:%M:%S')
print(nowTime)
print(type(nowTime))
nowTime = datetime.datetime.strptime(nowTime, '%H:%M:%S').date()
print(nowTime)
print(type(nowTime))


# 지금시간
print(now.hour,' 시', now.minute, ' 분')
# if 06:00:00 <= now_time <= 23:59:00:


last_date_from_previous_df = pd.to_datetime(last_30_Change.index[-1])
one_day = datetime.timedelta(days=1)
two_days = datetime.timedelta(days=2)
new_data = yf.download(world_indices[0][0], start=last_date_from_previous_df+two_days ,end=to_date+one_day) # 1/26, 1/27, 1/28 데이터를 가져옴
# 1/28일 데이터가 만약 없다면 안가져옴, 있다면 가져옴
print(new_data)
# new_data = new_data[['High', 'Low', 'Adj Close']]
new_High = new_data[['High']]
updated_High = pd.concat([last_30_High, new_High])
new_Low = new_data[['Low']]
updated_Low = pd.concat([last_30_Low, new_Low])
new_Close = new_data[['Adj Close']]
updated_Close = pd.concat([last_30_Close, new_Close])
print(updated_Close)
# new_Change = new_Close.pct_change(periods=1)
# print(new_Change)
# new_data['Change'] =

# A. 내일의 종가 예측
# - 08:00:00 – 23:59:00 이라면 하루전까지의 데이터
# - 24:00:00 – 05:59:00 이라면 이틀전까지의 데이터(마지막 종가 이전 것 까지)
# - 06:00:00 - 07:59:00 이라면 ' 8am 부터 다시 시도하시오 '



# B. 정해진 기간범위의 정답률 예측