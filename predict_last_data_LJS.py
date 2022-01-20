import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle

X_train, X_test, Y_train, Y_test = np.load('./models/crude_oil_scaled_data.npy', allow_pickle=True)
model = load_model('./models/crude_oil_model_1.h5')

# 마지막 30개 데이터 예측 시키기 ,  30개의 데이터가 필요
raw_data = pd.read_csv('./datasets/Clear_Crude_Oil_Data.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date']) # Date열을 datetime으로 바꾼다.
raw_data.set_index('Date', inplace=True) # 날짜대로 슬라이싱 하고싶으면 index 만 가능 column은 안됨
raw_data.info()
print('날짜 정렬된 df: ', raw_data.head(10))


last_test_data = raw_data[:60][['Price']]  # 해도 됨.
last_test_data.info()
last_test_data.to_pickle("last_60_내림차순.pkl")
# minmaxscaler 어떻게 파일 가져오지??
with open('./models/minmaxscaler_oil.pickle', 'rb') as f:
  minmaxscaler = pickle.load(f)

scaled_last_test_data = minmaxscaler.transform(last_test_data)
print('shape : ',scaled_last_test_data.shape)  # (60, 1)
print('scaled_data 정보:', scaled_last_test_data)
#ndarray 저장하는 코드
# np.save('./datasets/last60_data.npy', scaled_last_test_data)

sequence_test_X = []
sequence_test_Y = []
for i in range(len(scaled_last_test_data)-31):# 30개 데이타
  x = scaled_last_test_data[i:i+30]
  y = scaled_last_test_data[i+30]
  sequence_test_X.append(x)
  sequence_test_Y.append(y)
  if i == 0:
    print(x, '->', y) #전의 연소된 30개 가지고 다음 31번째 값을 예측

sequence_test_X = np.array(sequence_test_X)
sequence_test_Y = np.array(sequence_test_Y)
# print(sequence_test_Y)

predict_last = model.predict(sequence_test_X)
plt.plot(sequence_test_Y, label='actual')
plt.plot(predict_last, label='predict')
plt.legend()
plt.show()

######### 내일 값을 예측 시켜보자
tomorrow_predict = model.predict(scaled_last_test_data[-30:].reshape(1, 30, 1)) #마지막 30개 리셰잎 (30,1)짜리가 1개 있다.
print(tomorrow_predict) #minmaxscale 된값으로 나온담.

tomorrow_predicted_value = minmaxscaler.inverse_transform(tomorrow_predict) # inverse_transform하면 minmaxscaling한 값 다시 원래 값으로 복원시켜줌
print('$ %2f '%tomorrow_predicted_value[0][0])

tmr_scaled_last_test_data = np.append(scaled_last_test_data, [tomorrow_predict])
print(len(tmr_scaled_last_test_data))
# 내일 모레 예측
dat_predict = model.predict(tmr_scaled_last_test_data[-30:].reshape(1, 30, 1))
print(dat_predict)
dat_predicted_value = minmaxscaler.inverse_transform(dat_predict)
print('dat_value : $ %2f '%dat_predicted_value[0][0])



