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
# 헉....
with open('./datasets/variable_names.pickle', 'rb') as f:
    name, asset_class,asset_func = pickle.load(f)
print(asset_func(name))

exit()


# 마지막 30개 df 가져와서 예측 시키기
last_data = pd.read_pickle('./datasets/last_{}_data.pickle'.format(name))

# 만약 df의 마지막행의 날짜가 Yesterday가 아니면, 데이터를 새로 추가하여 concat할 것이다.

Today = datetime.date.today()
one_day = datetime.timedelta(days=1)
Yesterday = Today - one_day
len_index = len(last_data.index)
start_day = last_data.index[len_index-1] + one_day


if last_data[-1] != Yesterday:
    added_data = pd.DataFrame(asset_func(name))
    added_data = added_data[start_day:Yesterday][['Close']]
    added_data.rename(columns={'Close': 'Price'}, inplace=True)

updated_df = pd.concat([last_data, added_data])
updated_df.to_pickle('./datasets/last_{}_data.pickle'.format(name))  # 업데이트된 DataFrame 저장시키기(나중에 다시 불러올 데이터)

# 30개의 데이터를 스케일링 시키기.
with open('./models/minmaxscaler_{}.pickle', 'rb') as f:
    minmaxscaler = pickle.load(f)
last_scaled_data = minmaxscaler.transform(last_data)

#30개로 내일 예측
model = load_model('./models/model_{}.h5'.format(name))
X_train, X_test, Y_train, Y_test = np.load('./models/{}_train_test.npy'.format(name), allow_pickle=True)
tomorrow_predict = model.predict(last_scaled_data[:].reshape(1, 30, 1))
print(tomorrow_predict) #minmaxscale 된값으로 나온담.
tomorrow_predicted_value = minmaxscaler.inverse_transform(tomorrow_predict)
print('내일예측값$ %2f '%tomorrow_predicted_value[0][0])

last_data.to_pickle('./datasets/updated_last_data.pickle') # scaled 되기 전의 df를 저장함.







