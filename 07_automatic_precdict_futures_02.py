import investpy
import pandas as pd
import FinanceDataReader as fdr   # pip install -U finance-datareader
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
import numpy as np

# 기존 07번 파일에서 안쓰는 코드 깔끔하게 제거

# user가 선물 종목 입력
ticker = ''     # ticker은 종목이름을 ticker로 바꾸는 코드 따로 만들기. (유저가 티커만 입력하면 찾아올 수 있게)
name = 'Gold'


# 4개 섹터의 선물 리스트
commodity_list = investpy.commodities.get_commodities_list()
currency_list = ['GBP/USD','CAD/USD','USD/JPY','USD/CHF','EUR/USD','AUD/USD','USD/MXN','NZD/USD','USD/ZAR','USD/BRL']
etf_list = investpy.etfs.get_etfs_list()
indices_list = investpy.indices.get_indices_list()


# yyyy-mm-dd 를 dd/mm/yyyy로 바꿔주기
# 1. 오늘을 어제로 만들어주기 (datetime형태)
Today = datetime.date.today()  # 오늘 날짜 2022-01-21 00:00:00
one_day = datetime.timedelta(days=1) # 1일을 datetime형태로 변환
Yesterday = Today - one_day # 어제


# datetime을 str으로 바꾸는 절차
dt_stamp = datetime.datetime(Yesterday.year, Yesterday.month, Yesterday.day)  #2022, 01, 21
dt_stamp_str = str(dt_stamp)  # datetime을 string으로
Yesterday_v = datetime.datetime.strptime(dt_stamp_str.split()[0], '%Y-%m-%d')  # 앞에것만 '2022-01-20'
Yesterday = dt_stamp.strftime('%d/%m/%Y')   # 드디어 변환


# data 다운로드 받는 코드
if name in commodity_list:
   historical_data = investpy.commodities.get_commodity_historical_data(n , Yesterday)
   asset_class = 'commodity'
   asset_func = investpy.commodities.get_commodity_recent_data
elif name in currency_list:
   historical_data = investpy.currency_crosses.get_currency_cross_historical_data(name, '01/04/2008', Yesterday)
   asset_class = 'currency'
   asset_func = investpy.currency_crosses.get_currency_cross_recent_data
elif name in etf_list:
    historical_data = investpy.etfs.get_etf_historical_data(name, '01/04/2008', Yesterday)
    asset_class = 'etf'
    asset_func = investpy.etfs.get_etf_recent_data
else:
    print('정확한 이름인지 다시 확인해주세요 / 존재하지 않는 종목입니다.')


# 변수를 피클 담구기..
variables = (name, asset_class,asset_func) #피클 담글 변수 , 값을 변하지 못하게 튜플로 묶는다.


# 다운로드 받은 데이터를 'Price'이름의 하나의 열로만 구성된 DatafFrame으로 바꾸는 코드 /  01/04/2008 부터 어제까지의 데이터 가져옴
historical_data = historical_data[['Close']]
historical_data.rename(columns={'Close':'Price'}, inplace=True)



##  전처리 시작  ##

# Nan값 제거
if historical_data.isnull().sum()[0] != 0:
   historical_data.fillna(method='ffill', inplace=True) # Nan값처리: 이전값으로 채우기(investing.com에는 결측률이 거의 희박하긴 함.)


# 예측을 위한 마지막 30개 데이터로 구성된 DataFrame만들기
last_30_data = historical_data[-30:]  # df형태


# minmaxscaler 객체 생성 및 적용, pickle 로 저장
minmaxscaler = MinMaxScaler()  # 0과 1사이의 값으로 변환 하는 인스턴스 생성
scaled_data = minmaxscaler.fit_transform(historical_data) # historical data를 scaling시킨것


# x, y값의 리스트
sequence_X = []
sequence_Y = []
for i in range(len(scaled_data)-30):
    _x = scaled_data[i:i+30] #  총 30개
    _y = scaled_data[i+30] # 31번째를 예측
    sequence_X.append(_x)
    sequence_Y.append(_y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)

# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
xy = X_train, X_test, Y_train, Y_test

# 모델링
model = Sequential()
model.add(LSTM(128, input_shape=(30, 1), activation='tanh', return_sequences=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)


# 각종 저장 파일 (필요시 주석처리하기 편하게 모아놓음)
# 튜플 형태로 만든 변수 ->  (name, asset_class ,asset_func)
# with open('./datasets/variable_names.pickle', 'wb') as f:
#     pickle.dump(variables, f)
# # 마지막 30개 행 [-30:] -> DataFrame을 pickle 담그기(minmaxscaling 하기 전)
# last_30_data.to_pickle('./datasets/last_{}_data.pickle'.format(name))
# # minmaxscaler 인스턴스
# with open('./models/minmaxscaler_{}.pickle'.format(name), 'wb') as f:
#     pickle.dump(minmaxscaler, f)
# # train_test_split을 npy로
# np.save('./models/{}_train_test.npy'.format(name), xy)
# # 모델.h5
# model.save('./models/model_{}.h5'.format(name)) # 모델링 돌릴때마다 숫자가 매겨지게 흠,,,
