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
# from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)

plt.rcParams['font.family'] = 'nanumyeongjo'
plt.rcParams['figure.figsize'] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

############ 티커명을 입력하면 데이터를 다운받아 전처리하고 모델링까지 해주는 코드 ###################
############ 모델은 여러개 만들 수 있다. #################
# 해결못한 부분 1. ticker 명을 name으로 자동전환해주는 것
# 2. indices 종목 데이터 다운로드받기(country를 하나하나 찾아 자동 지정해주는 코드)
# 3. 모델을 여러개 만들 수 있게 하는데, 그럴려면 모델 저장번호를 자동으로 _1, _2 등등 이름 바꿔주는 코드가 필요

# user가 선물 종목 입력
ticker = ''     # ticker은 종목이름을 ticker로 바꾸는 코드 따로 만들기. (유저가 티커만 입력하면 찾아올 수 있게)
name = 'Gold'

# 4개 섹터의 선물 리스트
commodity_list = investpy.commodities.get_commodities_list()
print(type(commodity_list))  # list
currency_list = ['GBP/USD','CAD/USD','USD/JPY','USD/CHF','EUR/USD','AUD/USD','USD/MXN','NZD/USD','USD/ZAR','USD/BRL']   #직접 작성
etf_list = investpy.etfs.get_etfs_list()
indices_list = investpy.indices.get_indices_list()


##### yyyy-mm-dd 를 dd/mm/yyyy로 바꿔주기

# 1. 오늘을 어제로 만들어주기 (datetime형태)
Today = datetime.date.today()  # 오늘 날짜 2022-01-21 00:00:00
one_day = datetime.timedelta(days=1) # 1일을 datetime형태로 변환
Yesterday = Today - one_day # 어제

######### datetime을 str으로 바꾸는 절차
dt_stamp = datetime.datetime(Yesterday.year, Yesterday.month, Yesterday.day)  #2022, 01, 21
print(type(dt_stamp))  #datetime.datetime
print(dt_stamp)        #2022-01-20 00:00:00
dt_stamp_str = str(dt_stamp)  # datetime을 string으로
print(dt_stamp_str)
Yesterday_v = datetime.datetime.strptime(dt_stamp_str.split()[0], '%Y-%m-%d')  # 앞에것만 '2022-01-20'
print(Yesterday_v)
Yesterday = dt_stamp.strftime('%d/%m/%Y')   # 드디어 변환
print(Yesterday)

if name in commodity_list:
   historical_data = investpy.commodities.get_commodity_historical_data(name, '01/04/2008', Yesterday)
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
with open('./datasets/variable_names.pickle', 'wb') as f:
    pickle.dump(variables, f)

# 다운로드 받은 데이터를 'Price'이름의 하나의 열로만 구성된 DatafFrame으로 바꾸는 코드
historical_data = historical_data[['Close']]
historical_data.rename(columns={'Close':'Price'}, inplace=True)
print(historical_data) # 01/04/2008 부터 어제까지의 데이터 가져옴

# 전처리 시작
# historical_data.info()
# print('null:', historical_data.isnull().sum())
# print(historical_data.isnull().sum()[0])

# Nan값 제거
if historical_data.isnull().sum()[0] != 0:
   historical_data.fillna(method='ffill', inplace=True) # Nan값처리: 이전값으로 채우기(investing.com에는 결측률이 거의 희박하긴 함.)
# print(type(historical_data.index))

# 예측을 위한 마지막 30개 데이터로 구성된 DataFrame만들기
last_30_data = historical_data[-30:]  # df형태
print(type(last_30_data)) # DataFrame
print(len(last_30_data))  # 30
last_30_data.to_pickle('./datasets/last_{}_data.pickle'.format(name)) # 마지막 30개 데이터 스케일링 전

# minmaxscaler 객체 생성 및 적용, pickle 로 저장
minmaxscaler = MinMaxScaler()  # 0과 1사이의 값으로 변환 하는 인스턴스 생성
scaled_data = minmaxscaler.fit_transform(historical_data) # historical data를 scaling시킨것
print(scaled_data[:5])
with open('./models/minmaxscaler_{}.pickle'.format(name), 'wb') as f:
    pickle.dump(minmaxscaler, f)

# x, y값의 리스트
sequence_X = []
sequence_Y = []
for i in range(len(scaled_data)-30):
    _x = scaled_data[i:i+30] #  총 30개
    _y = scaled_data[i+30] # 31번째를 예측
    sequence_X.append(_x)
    sequence_Y.append(_y)

sequence_X = np.array(sequence_X) # train_test_split 하기 위해 np.array해주어야 함.
sequence_Y = np.array(sequence_Y)

X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
xy = X_train, X_test, Y_train, Y_test
np.save('./models/{}_train_test.npy'.format(name), xy)

model = Sequential()
model.add(LSTM(128, input_shape=(30, 1), activation='tanh', return_sequences=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
model.save('./models/model_{}.h5'.format(name)) # 모델링 돌릴때마다 숫자가 매겨지게 흠,,,

