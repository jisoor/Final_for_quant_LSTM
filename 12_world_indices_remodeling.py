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


# 업무분담 2(모델링)
world_indices_paths = glob.glob('./datasets/world_indices/*.csv')
print(world_indices_paths)
# 인덱스와 컬럼만 지정한 빈 DF 만들기
# 인덱스를 리스트로 만들기
class_name = []
for i in range(30):
    class_name.append(world_indices[i][1])
print(class_name) # world_indicies에 있는 모든 종목이름을 리스트로 만듬
print(len(class_name))  # 30개

mse =  ['High', 'Low', 'Adj Close', 'Change', 'Average'] # 인덱스로 설정

df_loss = pd.DataFrame(columns=class_name)
df_loss = pd.DataFrame({'mse':mse})
print(df_loss)
for ticker, name in world_indices:   # 30개 클래스 이름이 모두 columns로 들어옴
    df_loss[name] = np.nan          #nan값으로 채워서 빈 데이터프레임 만들기
df_loss.set_index('mse', inplace=True)
print(df_loss)
print(df_loss.columns)
print(len(df_loss.columns))

for num, world_indices_path in enumerate(world_indices_paths):
    print(num) # 0 부터
    df = pd.read_csv(world_indices_path, index_col=0)
    print(df.head(3))
    df_lists = [('df_high','High'), ('df_low','Low'), ('df_close','Adj Close'), ('df_change','Change')] # ['df_high',' df_low', 'df_close', 'df_change']
    plot_num = 0
    plt.figure(figsize=(8,18))

    for df_each, colname in df_lists:
        plot_num += 1
        df_each = df[[colname]]
        last_60_df = df_each[-60:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
        print(last_60_df.tail())
        df_each = df_each[:-30]  # 마지막 30개 빼고 모델링
        print(type(df_each)) # DataFrame
        last_60_df.to_csv('./updated/{}_{}_updated.csv'.format(world_indices[num][1], colname))
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
        fit_hist = model.fit(X_train, Y_train, epochs=1, callbacks=[early_stopping],  shuffle=False, validation_data=(X_test, Y_test))

        print(fit_hist)
        plt.plot(fit_hist.history['loss'][:], label='loss')
        plt.plot(fit_hist.history['val_loss'][:], label='val_loss')
        mse = fit_hist.history['val_loss'][-1]
        print('val_loss값은?? :', fit_hist.history['val_loss'][-1])
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('LSTM val loss:', score)
        plt.subplot(4, 1, plot_num)
        plt.title(world_indices[num][1])
        plt.ylabel(colname)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        # plt.savefig('./datasets/{}_{}_mse_plot'.format(colname, plot_num)) # './datasets/{}_world_indices_mse.png'.format(world_indices[num][1])
        # plt.show()
        # pp.close()
        if colname == 'Change':
            df_loss.loc[colname][world_indices[num][1]] = mse
            df_loss.loc['Average'][world_indices[num][1]] = (df_loss.loc['High'][world_indices[num][1]] + df_loss.loc['Low'][world_indices[num][1]]
                     + df_loss.loc['Adj Close'][world_indices[num][1]] + df_loss.loc['Change'][world_indices[num][1]]) / 4
        else:
            df_loss.loc[colname][world_indices[num][1]] = mse # '클래스이름'행 - 열에 mse 값이 들어가게 하기.

        print(df_loss)
        model.save('./models/{}_{}_model.h5'.format(world_indices[num][1], colname))  # 모델 저장하기

        print (world_indices[num][1], colname, ' 모델링및 저장 까지 완료 ')
    plt.savefig('./datasets/{}_mse_plot'.format(world_indices[num][1]))  # './datasets/{}_world_indices_mse.png'.format(world_indices[num][1])
    plt.show()
    plt.close()

df_loss = df_loss.T
df_loss.to_csv('./datasets/world_indices_mse.csv', index =True)


# <데이터 프레임 만드는 법>
# for문 밖에서
# class_name = []
# for i in range(30):
#     class_name.append(world_indices[i][1])
# print(class_name) # world_indicies에 있는 모든 종목이름을 리스트로 만듬
# print(len(class_name))  # 30개
#
# mse =  ['High', 'Low', 'Adj Close', 'Change', 'Average'] # 인덱스로 설정
#
# df_loss = pd.DataFrame(columns=class_name)
# df_loss = pd.DataFrame({'mse':mse})
# print(df_loss)
# for ticker, name in world_indices:   # 30개 클래스 이름이 모두 columns로 들어옴
#     df_loss[name] = np.nan          #nan값으로 채워서 빈 데이터프레임 만들기
# df_loss.set_index('mse', inplace=True)
# print(df_loss)
# print(df_loss.columns)
# print(len(df_loss.columns))
# # for문 안에서
# mse = fit_hist.history['val_loss'][-1]
# if colname == 'Change':
#     df_loss.loc[colname][world_indices[num][1]] = mse
#     df_loss.loc['Average'][world_indices[num][1]] = (df_loss.loc['High'][world_indices[num][1]] + df_loss.loc['Low'][
#         world_indices[num][1]]
#                                                      + df_loss.loc['Adj Close'][world_indices[num][1]] +
#                                                      df_loss.loc['Change'][world_indices[num][1]]) / 4
# else:
#     df_loss.loc[colname][world_indices[num][1]] = mse  # '클래스이름'행 - 열에 mse 값이 들어가게 하기.
#
# print(df_loss)
# # for문 끝나고
# df_loss = df_loss.T
# df_loss.to_csv('./datasets/world_indices_mse.csv', index =True)
