import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

raw_data = pd.read_csv('../Crude_Oil_LSTM/datasets/Clear_Crude_Oil_Data.csv')
print(raw_data.info())

minmaxscaler = MinMaxScaler()  # 0과 1사이의 값으로 변환 하는 인스턴스 생성
data = raw_data[:-30][['Price']] #마지막 30개는 벡테스팅용, 'Price'열만 가져와서
print(data.head(5))
scaled_data = minmaxscaler.fit_transform(data)
print(scaled_data[:5])
sequence_X = []
sequence_Y = []
#minmaxscaler 를 pickle 담구기.
with open('./models/minmaxscaler_oil.pickle', 'wb') as f:
    pickle.dump(minmaxscaler, f)


for i in range(len(scaled_data)-30): #3515번 돌아감 : 3575 - 30 - 30 + 1
    _x = scaled_data[i:i+30] # i, i+1 ... i+ 29 까지 총 30개
    _y = scaled_data[i+30] # i + 30번째를 예측
    sequence_X.append(_x)
    sequence_Y.append(_y)
    if i is 0: # 0~29번째를 토대로 30번째 예측
        print(_x, '->', _y)

print('length:', len(sequence_X), len(sequence_Y))
sequence_X = np.array(sequence_X) # train_test_split 하기 위해 np.array해주어야 함.
sequence_Y = np.array(sequence_Y)

# X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2) # sequence_X는 총3515 * 30개 ,sequence_Y는 총 3515개
# scaled_data = X_train, X_test, Y_train, Y_test

# np.save('./models/crude_oil_scaled_data.npy', scaled_data)



