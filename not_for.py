import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option('display.max_columns', None) # 열이 전부다 나오게
currencies_lists = [('EURUSD=X', 'EUR-USD'), ('JPY=X', 'USD-JPY'), ('GBPUSD=X', 'GBP-USD'), ('AUDUSD=X', 'AUD-USD'), ('NZDUSD=X', 'NZD-USD'),
                    ('EURJPY=X', 'EUR-JPY'), ('GBPJPY=X', 'GBP-JPY'), ('EURGBP=X', 'EUR-GBP'), ('EURCAD=X', 'EUR-CAD'), ('EURSEK=X', 'EUR-SEK'),
                    ('EURCHF=X', 'EUR-CHF'), ('EURHUF=X', 'EUR-HUF'), ('CNY=X', 'USD-CNY'), ('HKD=X', 'USD-HKD'), ('SGD=X', 'USD-SGD'),
                    ('INR=X', 'USD-INR'), ('MXN=X', 'USD-MXN'), ('PHP=X', 'USD-PHP'), ('IDR=X', 'USD-IDR'), ('THB=X', 'USD-THB'),
                    ('MYR=X', 'USD-MYR'), ('ZAR=X', 'USD-ZAR'), ('RUB=X', 'USD-RUB')]
file_name = 'RUB=X' #MXN=X까지완료 PHP부터 시작
print(len(currencies_lists))
print(currencies_lists[0][1])
# 업무분담 2(모델링)
currencies_lists_paths = ['./preprocessing_currencies/currencies_list/round2_{}_2007-01-01-2022-01-25.csv'.format(file_name)]
print(currencies_lists_paths)
# df = pd.DataFrame()
for num, currencies_lists_path in enumerate(currencies_lists_paths):
    print(num) # 0 부터
    print(type(num)) # int
    df = pd.read_csv(currencies_lists_path, index_col=0)
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
        df_each = df_each[:-30]  # 마지막 30개 빼고 모델링
        print(type(df_each)) # DataFrame
        #last_30_df = df_each[-30:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
        last_60_df = df_each[-60:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
        #last_30_df.to_pickle('./pickles/{}_{}_last30_data.pickle'.format(currencies_lists[num][1], colname))
        last_60_df.to_csv('./updated/{}_{}_updated.csv'.format(currencies_lists[num][1], colname))
        minmaxscaler = MinMaxScaler()
        scaled_data = minmaxscaler.fit_transform(df_each)  # 스케일링해주기
        with open('./minmaxscaler/{}_{}_minmaxscaler.pickle'.format(currencies_lists[num][1], colname), 'wb') as f:
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
        np.save('./train_test_split/{}_{}_train_test.npy'.format(currencies_lists[num][1], colname), xy)  # 저장하고

        model = Sequential()
        model.add(LSTM(512, input_shape=(30, 1), activation='tanh', return_sequences=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
        model.save('./models/{}_{}_model.h5'.format(currencies_lists[num][1], colname))  # 모델 저장하기
        print(currencies_lists[num][1], colname, ' 모델링및 저장 까지 완료 ')