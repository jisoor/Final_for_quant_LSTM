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

#### 재 모델링 ####
# index를 path 꺼내오는 순서대로 재정렬함
world_indices = [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
 ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('IMOEX.ME','MOEX Russia Index'),('^MXX','IPC MEXICO'),
 ('^JKSE', 'Jakarta Composite Index'),('^KS11','KOSPI Composite Index'),('^MERV','MERVAL'),('^IXIC','NASDAQ Composite'),
 ('^N225', 'Nikkei 225'),('^XAX','NYSE AMEX COMPOSITE INDEX'),('^NYA','NYSE COMPOSITE (DJ)'),('^RUT','Russell 2000'),('^GSPC','S&P 500'),
 ('^BSESN', 'S&P BSE SENSEX'), ('399001.SZ', 'Shenzhen Component'),('000001.SS', 'SSE Composite Index'),('^STI','STI Index'),
 ('^TA125.TA', 'TA-125'),('^TWII','TSEC weighted index'),('^VIX','Vix')]
print(len(world_indices))

count = 0
for world_indice, name in world_indices:
    try:
        count += 1
        data = yf.download(world_indice, start="2007-01-01", end="2022-01-26") # 26일까지로 해야 25일까지의 데이터가 나옴
        df = data[['High', 'Low' , 'Adj Close']]
        # print(world_indice , df.head())
        print(count , '번째' , world_indice, ':', df.info())
        nul = df.isnull().sum()
        if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):  # 'Open'에 난값이 존재하면
            df.fillna(method='ffill', inplace=True)
        # 등락율 백분율 전환
        change = pd.DataFrame({'Change': df['Adj Close'].pct_change(periods=1)}) # (현재행 - 이전행) / 이전행 <-period=1일 경우.
        change.iloc[0] = 0  # 첫행은 Nan이므로 0으로 통일 시켜줌
        change['Change'] = (change['Change'] * 100).round(2)
        # print(change[:5])
        df = pd.concat([df, change], axis=1)
        # 시작날짜, 마지막 날짜
        first_date = str(df.index[0]).split()[0]
        last_date = str(df.index[-1]).split()[0]
        print('날짜' , first_date, last_date)
        df.to_csv('./datasets/world_indices/{}_{}-{}.csv'.format(name, first_date, last_date))
    except:
        print('No data')

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
        minmaxscaler = MinMaxScaler()
        scaled_data = minmaxscaler.fit_transform(df_each)  # 스케일링해주기
        with open('./minmaxscaler/{}_{}_minmaxscaler.pickle'.format(world_indices[num][1], colname), 'wb') as f:
            pickle.dump(minmaxscaler, f)
        sequence_X = []
        sequence_Y = []
        for i in range(len(scaled_data) - 30):
            _x = scaled_data[i:i + 30]  # 총 30개
            _y = scaled_data[i + 30]  # 31번째를 예측
            sequence_X.append(_x)
            sequence_Y.append(_y)
        sequence_X = np.array(sequence_X)
        sequence_Y = np.array(sequence_Y)
        X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
        xy = X_train, X_test, Y_train, Y_test
        np.save('./train_test_split/{}_{}_train_test.npy'.format(world_indices[num][1], colname), xy)  # 저장하고

        model = Sequential()
        model.add(LSTM(512, input_shape=(30, 1), activation='tanh', return_sequences=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
        model.save('./models/{}_{}_model.h5'.format(world_indices[num][1], colname))  # 모델 저장하기
        print (world_indices[num][1], colname, ' 모델링및 저장 까지 완료 ')