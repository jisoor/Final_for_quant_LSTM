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
from tensorflow.keras.callbacks import EarlyStopping  # early_stopping 걸기
# from matplotlib.backends.backend_pdf import PdfPages
early_stopping=EarlyStopping(monitor='val_loss',patience=5) # early_stopping 걸기
pd.set_option('display.max_columns', None) # 열이 전부다 나오게

futures = [('BZ=F','BRENT_OIL' ),('CC=F','COCOA'),('KC=F', 'Coffee'),('HG=F','COPPER'),('ZC=F','CORN'),('CT=F','COTTON'),
    ('CL=F','CRUDE_OIL' ),('YM=F', 'DOW'),('GF=F','FEEDER_CATTLE'),('GC=F','GOLD'),('HE=F','LEAN_HOGS'),('LE=F','LIVE_CATTLE'),
    ('LBS=F','LUMBER'),('NQ=F', 'NASDAQ'),('NG=F','NATURAL_GAS'),('ZO=F','OAT'),('PA=F','PALLADIUM' ),('PL=F','PLATINUM'),('ZR=F','ROUGH_RICE'),
    ('RTY=F', 'RUSSEL2000'),('SI=F','SILVER'),('ZS=F','SOYBEAN'),('ZM=F','SOYBEAN_MEAL'),('ZL=F','SOYBEAN_OIL'),
    ('ES=F', 'SPX'),('SB=F','SUGAR'),('ZT=F','US2YT'),('ZF=F', 'US5YT'),( 'ZN=F', 'US10YT'), ('ZB=F','US30YT' ), ('KE=F','WHEAT')]
print(len(futures)) # 31개

futures_paths = glob.glob('./futures/futures_data/*.csv')
print(len(futures_paths))
# 인덱스와 컬럼만 지정한 빈 DF 만들기
# 30개 자산클래스의 이름
class_name = []
for i in range(31):
    class_name.append(futures[i][1])
print(class_name)

mse = ['High', 'Low', 'Adj Close', 'Change', 'Average'] # 인덱스
df_loss = pd.DataFrame(columns=class_name) # 31개 자산클래스 column
df_loss = pd.DataFrame({'mse':mse})     # 'mse' 5개 값을 가진 column을 만듬

for ticker,name in futures:   # 31개 클래스 이름이 모두 columns로 들어옴
    df_loss[name] = np.nan           # nan 값으로 채워서 빈 데이터프레임 만들기
df_loss.set_index('mse', inplace=True)  # mse 안 5개 값을 인덱스로 나열해줌(추후 transpose(T)해서 인덱스-컬럼 바꿀것임).


plt.figure(figsize=(8, 18))
for num, futures_path in enumerate(futures_paths):
    df = pd.read_csv(futures_path, index_col=0)
    df.rename(columns={'Adj_Close':'Adj Close'}, inplace=True)
    df_lists = [('df_high','High'), ('df_low','Low'), ('df_close','Adj Close'), ('df_change','Change')] # ['df_high',' df_low', 'df_close', 'df_change']
    plot_num = 0
    for df_each, colname in df_lists:
        plot_num += 1   # 1, 2, 3, 4
        df_each = df[[colname]]  # 오류 "None of [Index(['Adj Close'], dtype='object')] are in the [columns]"
        last_60_df = df_each[-60:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
        print(last_60_df.tail())
        df_each = df_each[:-30]  # 마지막 30개 빼고 모델링
        print(type(df_each)) # DataFrame
        last_60_df.to_csv('./futures/updated/{}_{}_updated.csv'.format(futures[num][1], colname))
        minmaxscaler = MinMaxScaler()
        scaled_data = minmaxscaler.fit_transform(df_each)  # 스케일링해주기
        with open('./futures/minmaxscaler/{}_{}_minmaxscaler.pickle'.format(futures[num][1], colname), 'wb') as f:
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
        np.save('./futures/train_test_split/{}_{}_train_test.npy'.format(futures[num][1], colname), xy)  # 저장하고

        model = Sequential()
        model.add(LSTM(512, input_shape=(30, 1), activation='tanh', return_sequences=1))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        fit_hist = model.fit(X_train, Y_train, epochs=100, callbacks=[early_stopping],  shuffle=False, validation_data=(X_test, Y_test))

        # 플롯 차트 #
        plt.plot(fit_hist.history['loss'][-30:], label='loss')
        plt.plot(fit_hist.history['val_loss'][-30:], label='val_loss')
        mse = fit_hist.history['val_loss'][-1]
        print('val_loss값은?? :', mse)
        plt.subplot(4, 1, plot_num)
        plt.title(futures[num][1])
        plt.ylabel(colname)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        # 위에서 만든 DataFrame에 Key값들 채우기.
        if colname == 'Change':
            df_loss.loc[colname][futures[num][1]] = mse
            df_loss.loc['Average'][futures[num][1]] = (df_loss.loc['High'][futures[num][1]] + df_loss.loc['Low'][futures[num][1]]
                     + df_loss.loc['Adj Close'][futures[num][1]] + df_loss.loc['Change'][futures[num][1]]) / 4
        else:
            df_loss.loc[colname][futures[num][1]] = mse # '클래스이름'행 - 열에 mse 값이 들어가게 하기.

        model.save('./models/{}_{}_model.h5'.format(futures[num][1], colname))  # 모델 저장하기
        print (futures[num][1], colname, ' 모델링및 저장 까지 완료 ')

    # 한 클래스당 4개의 컬럼에 대한 mse(val_loss)의 추이에 대한 그래프를 저장.
    plt.savefig('./futures/mse_plot/{}_mse_plot.png'.format(futures[num][1]))
    plt.show(block=False)
    plt.pause(1) # 1초후 자동으로 창 닫음
    plt.close()

df_loss = df_loss.T  # 행-열 전환 transpose.
df_loss.to_csv('./futures/mse_plot/futures_mse.csv', index=True)


