# B. 정해진 기간범위의 정답률 예측
import pandas as pd
import yfinance as yf
import datetime
import pickle
import time
from tensorflow.keras.models import load_model
import numpy as np
import pickle

world_indices =  [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
 ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('IMOEX.ME','MOEX Russia Index'),('^MXX','IPC MEXICO'),
 ('^JKSE', 'Jakarta Composite Index'),('^KS11','KOSPI Composite Index'),('^MERV','MERVAL'),('^IXIC','NASDAQ Composite'),
 ('^N225', 'Nikkei 225'),('^XAX','NYSE AMEX COMPOSITE INDEX'),('^NYA','NYSE COMPOSITE (DJ)'),('^RUT','Russell 2000'),('^GSPC','S&P 500'),
 ('^BSESN', 'S&P BSE SENSEX'), ('399001.SZ', 'Shenzhen Component'),('000001.SS', 'SSE Composite Index'),('^STI','STI Index'),
 ('^TA125.TA', 'TA-125'),('^TWII','TSEC weighted index'),('^VIX','Vix')]
currencies = []
futures = []

# 일단 나는 world_indicies로만
# 실행 시 ,3개 섹터(world_indices, currencies, futures) 전 84개 종목의 기간별 예측값/실제값/정답 알려줌(과거 예측 데이터만 가능)
# 아하 컬럼별 pickle data last_30 -> last_60으로 바꾸기....!! 2021/12/29 1달전 데이타 부터 원하는 기간 예측 가능함.

col_lists = ['High', 'Low', 'Adj Close', 'Change']
for ticker, name in world_indices:
    for col in col_lists:
        df = pd.read_pickle('./pickles/{}_{}_last60_data.pickle'.format(name, col))
        print(df.head(3))


# 종목 입력시 이름 찾아준다.
ticker = '^AORD'

for world_indice in world_indices:
    if world_indice[0] == ticker:
        name = world_indice[1]
        break

# user가 정답률 표를 알고싶은 날짜의범위를 입력
# ex) 2022-01-20 ~ 2022-01-28로 7일이라면, 7개의 데이터를 가져옴.

# from_date = '2022-01-26'
# from_date = pd.to_datetime(from_date)  #timestamps
# print(type(from_date))
# to_date = '2022-01-28'
# to_date = pd.to_datetime(to_date)
# now = datetime.datetime.now().time()
# nowTime = now.strftime('%H:%M:%S')
# print(nowTime)
# print(type(nowTime))