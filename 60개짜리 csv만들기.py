import yfinance as yf
# import FinanceDataReader as fdr
import pandas as pd
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
import numpy as np

world_indices = [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
 ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('IMOEX.ME','MOEX Russia Index'),('^MXX','IPC MEXICO'),
 ('^JKSE', 'Jakarta Composite Index'),('^KS11','KOSPI Composite Index'),('^MERV','MERVAL'),('^IXIC','NASDAQ Composite'),
 ('^N225', 'Nikkei 225'),('^XAX','NYSE AMEX COMPOSITE INDEX'),('^NYA','NYSE COMPOSITE (DJ)'),('^RUT','Russell 2000'),('^GSPC','S&P 500'),
 ('^BSESN', 'S&P BSE SENSEX'), ('399001.SZ', 'Shenzhen Component'),('000001.SS', 'SSE Composite Index'),('^STI','STI Index'),
 ('^TA125.TA', 'TA-125'),('^TWII','TSEC weighted index'),('^VIX','Vix')]
USA = []
Asia = []
HK = []
CHINA = []

## 'ESTX 50 PR.EUR' 부터 'MOEX Russia Index	'까지 , 'STI Index'부터 끝까지
print(len(world_indices))
print(world_indices[0][1])
# 업무분담 2(모델링)
world_indices_paths = glob.glob('./datasets/world_indices/*.csv')
print(world_indices_paths)
df = pd.DataFrame()
for num, world_indices_path in enumerate(world_indices_paths):
    print(num) # 0 부터
    print(type(num)) # int
    df = pd.read_csv(world_indices_path, index_col=0)
    print(df.head(3))
    # df_high = df[['High']]
    # df_low = df[['Low']]
    # df_close = df[['Adj Close']]
    # df_change = df[['Change']]
    df_lists = [('df_high','High'), ('df_low','Low'), ('df_close','Adj Close'), ('df_change','Change')] # ['df_high',' df_low', 'df_close', 'df_change']
    # print(type(df_low))
    # print(df_low.info())
    for df_each, colname in df_lists:
        df_each = df[[colname]]
        last_60_df = df_each[-60:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
        print(last_60_df.tail())
        df_each = df_each[:-30]  # 마지막 30개 빼고 모델링
        print(type(df_each)) # DataFrame
        last_60_df.to_csv('./updated/{}_{}_updated.csv'.format(world_indices[num][1], colname)) # 기존의 30개자리 pickle을 60개짜리 csv로 만들어서
        # updated  폴더에저장


        print (world_indices[num][1], colname, ' 모델링및 저장 까지 완료 ')