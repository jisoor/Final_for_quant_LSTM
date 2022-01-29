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

pd.set_option('display.max_columns', None) # 열이 전부다 나오게

# krx = fdr.StockListing('KRX')
# print(len(krx))
# print(krx)
# print(krx.columns)
# data = yf.download("ES=F", start="2008-04-01", end="2022-01-25")
# print(data.head())
# # print(data.info())
# data = yf.download('^GSPC', start="2007-01-01", end="2022-01-27")
# df = data[['Open', 'High', 'Adj Close']]
# df.info()
# print(df.tail(5))
# print(df.index[0])
# print(str(df.index[0]))
# print(str(df.index[0]).split()[0])
# exit()
# nul = df.isnull().sum()
# if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):       # 'Open'에 난값이 존재하면
#     df.fillna(method='ffill', inplace=True)
# # Nan값 분포에 대한 plot도 작성
# # 기본적으로 (현재행 - 이전행) / 이전행 / -1이면 반대로
# change = pd.DataFrame({'Change' : df['Adj Close'].pct_change(periods=1)})
# print(type(change))
# print('Change 등락의 백분율: ', change)
#
# change.iloc[0] = 0   # 첫행은 Nan이므로 0으로 통일 시켜줌
# print(change.head())
# df = pd.concat([df, change], axis=1)
# print('콘캣된: ', df)

# 업무분담 1 (전처리)
world_indices = [('^GSPC','S&P 500'), ('^DJI','Dow Jones Industrial Average'),('^IXIC','NASDAQ Composite'), ('^NYA','NYSE COMPOSITE (DJ)')
    ,('^XAX','NYSE AMEX COMPOSITE INDEX'),('^BUK100P', 'Cboe UK 100'),('^RUT','Russell 2000'),('^VIX','Vix'),('^FTSE','FTSE 100'),('^GDAXI','DAX PERFORMANCE-INDEX'),
    ('^FCHI', 'CAC 40'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^BFX', 'BEL 20'),('IMOEX.ME','MOEX Russia Index	'),('^N225', 'Nikkei 225')
    ,('^HSI','HANG SENG INDEX'),('000001.SS', 'SSE Composite Index'),('399001.SZ', 'Shenzhen Component'),('^STI','STI Index'),('^AXJO','S&P/ASX 200')
    ,('^AORD', 'ALL ORDINARIES'),('^BSESN', 'S&P BSE SENSEX'),('^JKSE', 'Jakarta Composite Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
    ('^NZ50','S&P/NZX 50 INDEX GROSS' ),('^KS11','KOSPI Composite Index'),('^TWII','TSEC weighted index'),('^GSPTSE', 'S&P/TSX Composite index')
    ,('^BVSP','IBOVESPA'),('^MXX','IPC MEXICO'),('^IPSA', 'S&P/CLX IPSA'),('^MERV','MERVAL'),('^TA125.TA', 'TA-125')]

# no data : 21 번째 ^AXJO / 26 번째 ^NZ50 / 29 번째 ^GSPTSE / 32 번째 ^IPSA / 33번쨰  ^CASE30/ 34번째는 데이터 수 적음(천몇개)
print(len(world_indices)) # world_indices : 36개->30개

# count = 0
# for world_indice, name in world_indices:
#     try:
#         count += 1
#         data = yf.download(world_indice, start="2007-01-01", end="2022-01-26") # 26일까지로 해야 25일까지의 데이터가 나옴
#         df = data[['High', 'Low' , 'Adj Close']]
#         # print(world_indice , df.head())
#         print(count , '번째' , world_indice, ':', df.info())
#         nul = df.isnull().sum()
#         if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):  # 'Open'에 난값이 존재하면
#             df.fillna(method='ffill', inplace=True)
#         # 등락율 백분율 전환
#         change = pd.DataFrame({'Change': df['Adj Close'].pct_change(periods=1)}) # (현재행 - 이전행) / 이전행 <-period=1일 경우.
#         change.iloc[0] = 0  # 첫행은 Nan이므로 0으로 통일 시켜줌
#         change['Change'] = (change['Change'] * 100).round(2)
#         # print(change[:5])
#         df = pd.concat([df, change], axis=1)
#         # 시작날짜, 마지막 날짜
#         first_date = str(df.index[0]).split()[0]
#         last_date = str(df.index[-1]).split()[0]
#         print('날짜' , first_date, last_date)
#         df.to_csv('./datasets/world_indices/{}_{}-{}.csv'.format(name, first_date, last_date))
#     except:
#         print('No data')

# 결론 : ^CASE30은 데이터 없어서 제외시킴
# ^JNOU.JO는 데이터가 1085개로 다른 것들에 비해 1/3이므로 제외시킴
# 총 리스트는 : ['^GSPC', '^DJI', '^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX','^FTSE','^GDAXI',
#     '^FCHI','^STOXX50E','^N100','^BFX','IMOEX.ME','^N225','^HSI','000001.SS','399001.SZ','^STI','^AXJO','^AORD','^BSESN',
#     '^JKSE','^KLSE','^NZ50','^KS11','^TWII','^GSPTSE','^BVSP','^MXX','^IPSA','^MERV','^TA125.TA']
# index를 path 꺼내오는 순서대로 재정렬함
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
        last_60_df.to_pickle('./pickles/{}_{}_last60_data.pickle'.format(world_indices[num][1], colname))
        # minmaxscaler = MinMaxScaler()
        # scaled_data = minmaxscaler.fit_transform(df_each)  # 스케일링해주기
        # with open('./minmaxscaler/{}_{}_minmaxscaler.pickle'.format(world_indices[num][1], colname), 'wb') as f:
        #     pickle.dump(minmaxscaler, f)
        # sequence_X = []
        # sequence_Y = []
        # for i in range(len(scaled_data) - 30):
        #     _x = scaled_data[i:i + 30]  # 총 30개
        #     _y = scaled_data[i + 30]  # 31번째를 예측
        #     sequence_X.append(_x)
        #     sequence_Y.append(_y)
        # sequence_X = np.array(sequence_X)
        # sequence_Y = np.array(sequence_Y)
        # X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
        # xy = X_train, X_test, Y_train, Y_test
        # np.save('./train_test_split/{}_{}_train_test.npy'.format(world_indices[num][1], colname), xy)  # 저장하고
        #
        # model = Sequential()
        # model.add(LSTM(512, input_shape=(30, 1), activation='tanh', return_sequences=2))
        # model.add(Flatten())
        # model.add(Dropout(0.2))
        # model.add(Dense(128))
        # model.add(Dropout(0.2))
        # model.add(Dense(1))
        # model.compile(loss='mse', optimizer='adam')
        # fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
        # model.save('./models/{}_{}_model.h5'.format(world_indices[num][1], colname))  # 모델 저장하기
        print (world_indices[num][1], colname, ' 모델링및 저장 까지 완료 ')