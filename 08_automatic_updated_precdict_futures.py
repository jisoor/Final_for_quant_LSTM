import investpy
import pandas as pd
import numpy as np
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle

# 내일 Price 예측 시키는 파일


# 튜플화된 3개의 변수 name, asset_class,asset_func 를 pickle 담가서 불러오기
# 예를들어 name = 'Gold'였으면 , asset_class='commodity', asset_func=investpy.commodities.get_commodity_recent_data 불러옴.
# asset_func는 investpy.commodities.get_commodity_recent_data 이렇게 최근데이터 다운 받는 것.
with open('./datasets/variable_names.pickle', 'rb') as f:
    name, asset_class,asset_func = pickle.load(f)
print(asset_func(name))



# last_data 데이터프레임 가져와서 예측 시키기(맨처음 만든 30개던, 업그레이드 된 DF든 불러옴)
last_data = pd.read_pickle('./datasets/last_{}_data.pickle'.format(name))
print(last_data)

# 만약 df의 마지막행의 날짜가 Yesterday가 아니면, 데이터를 새로 추가하여 concat할 것이다.
Today = datetime.date.today()
one_day = datetime.timedelta(days=1)
Yesterday = Today - one_day
print(type(Yesterday))
len_index = len(last_data.index)
start_day = last_data.index[len_index-1] + one_day
print('start', start_day)

# 만약 last_data에 어제까지의 데이터가 없다면, 어제까지의 데이터를 불러와서 concat 해라
if last_data.iloc[-1][0] != Yesterday:  # 마지막 데이터, 0번째 열('Price')
    added_data = asset_func(name) # 업데이트된 데이터
    added_data = added_data[['Close']] # 'Close' 하나의 열로 이루어진 DataFrame으로 변환
    added_data.rename(columns={'Close':'Price'}, inplace=True) # 'Close' -> 'Price'
    added_data = added_data[start_day:Yesterday] # 날짜는 업뎃된 날짜로부터 어제까지 인덱싱.
else:
    added_data = []

last_data = pd.concat([last_data, added_data])
last_data = last_data.sort_index(ascending=True)  # concat 될때 혹시라도 날짜가 뒤죽박죽 되는 것을 방지, 인덱스를 오름차순 정렬
last_data.to_pickle('./datasets/last_{}_data.pickle'.format(name))  # 업데이트된 DataFrame 저장시키기(나중에 다시 불러올 데이터)
print('last data', last_data)

# 30개의 데이터를 스케일링 시키기.
last_data = last_data[-30:]  # 30개 뽑아서
with open('./models/minmaxscaler_{}.pickle'.format(name), 'rb') as f:
    minmaxscaler = pickle.load(f)
last_scaled_data = minmaxscaler.transform(last_data)
print('30개의 스케일링된 데이터 :', last_scaled_data)

#30개로 내일 예측
model = load_model('./models/model_{}.h5'.format(name))
X_train, X_test, Y_train, Y_test = np.load('./models/{}_train_test.npy'.format(name), allow_pickle=True)
tomorrow_predict = model.predict(last_scaled_data.reshape(1, 30, 1))
print(tomorrow_predict) #minmaxscale 된값으로 나온담.
tomorrow_predicted_value = minmaxscaler.inverse_transform(tomorrow_predict)
print('내일예측값$ %2f '%tomorrow_predicted_value[0][0])









