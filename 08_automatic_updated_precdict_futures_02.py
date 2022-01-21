# < 내일 Price 예측 시키는 파일 > #
import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle


###  불러올 파일  ###
# 튜플화된 3개의 변수 name, asset_class,asset_func 를 pickle 담가서 불러오기
with open('./datasets/variable_names.pickle', 'rb') as f:
    name, asset_class,asset_func = pickle.load(f)
# last_data 데이터프레임 가져와서 예측 시키기(맨처음 만든 30개던, 업그레이드 된 DF든 불러옴)
last_data = pd.read_pickle('./datasets/last_{}_data.pickle'.format(name))
# minmaxscaler
with open('./models/minmaxscaler_{}.pickle'.format(name), 'rb') as f:
    minmaxscaler = pickle.load(f)
# model 및 train_test_set
model = load_model('./models/model_{}.h5'.format(name))
X_train, X_test, Y_train, Y_test = np.load('./models/{}_train_test.npy'.format(name), allow_pickle=True)
###  ###   #### ##### ####


# 만약 df의 마지막행의 날짜가 Yesterday가 아니면, 데이터를 새로 추가하여 concat할 것이다.
Today = datetime.date.today()
one_day = datetime.timedelta(days=1)
Yesterday = Today - one_day
len_index = len(last_data.index)
start_day = last_data.index[len_index-1] + one_day


# 만약 last_data에 어제까지의 데이터가 없다면, 어제까지의 데이터를 불러와서 concat 해라
if last_data.iloc[-1][0] != Yesterday:  # 마지막 데이터, 0번째 열('Price')
    added_data = asset_func(name) # 업데이트된 데이터
    added_data = added_data[['Close']] # 'Close' 하나의 열로 이루어진 DataFrame으로 변환
    added_data.rename(columns={'Close':'Price'}, inplace=True) # 'Close' -> 'Price'
    added_data = added_data[start_day:Yesterday] # 날짜는 업뎃된 날짜로부터 어제까지 인덱싱.
else:
    added_data = []
last_data = pd.concat([last_data, added_data])
last_data = last_data.sort_index(ascending=True) # concat 될때 혹시라도 날짜가 뒤죽박죽 되는 것을 방지, 인덱스를 오름차순 정렬


# 30개의 데이터를 스케일링 시키기.
last_data = last_data[-30:]  # 30개 뽑아서
last_scaled_data = minmaxscaler.transform(last_data)


# 30개로 내일 예측
tomorrow_predict = model.predict(last_scaled_data.reshape(1, 30, 1))
print(tomorrow_predict) # minmaxscale 된값으로 나온담.
tomorrow_predicted_value = minmaxscaler.inverse_transform(tomorrow_predict)
print('내일예측값$ %2f '%tomorrow_predicted_value[0][0])


# 저장할 파일
last_data.to_pickle('./datasets/last_{}_data.pickle'.format(name))  # 업데이트된 DataFrame 저장시키기(나중에 다시 불러올 데이터)