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


# 거래시간(한국시간)	(월~토) 10:00~08:00 (썸머타임 시, 1시간 앞당겨짐)

# 마지막 60개 짜리 데이터프레임   2021-10-27 ~ 2022-01-18 피클 담가서 가져오기
# updated_df = pd.read_pickle('./last_60.pkl')    # 얘는 이제 주석 처리
updated_df = pd.read_pickle('./datasets/updated_df.pickle')  # datasets폴더 안에 updated_df 데이터프레임 저장해서 늘 최신 데이터 불러오게 하기
print(updated_df.tail())
updated_df.info()

len_index = len(updated_df.index)
print(updated_df.index[len_index-1])  # 기존 데이터 프레임에서 가장 마지막 날짜 가져오기 2022-01-18
# 가장마지막 날짜에 1을 더한다
one_day = datetime.timedelta(days=1)
start_day = updated_df.index[len_index-1] + one_day
print(start_day)

# weekday 오늘 요일 뽑기.
# t = ['월', '화', '수', '목', '금', '토', '일']
# Today_weekday = datetime.datetime.today().weekday()
# print('오늘 몇요일 : ', t[Today_weekday])

Today = datetime.date.today()
print('오늘: ', Today)
Yesterday = Today - one_day
print('어제 : ', Yesterday)
# # investing.com에서 Crude Oil WTI 선물 종가 따오기
df_WTI = pd.DataFrame(investpy.commodities.get_commodity_recent_data('Crude Oil WTI'))

df_WTI = df_WTI[start_day:Yesterday][['Close']] # 01-19부터 어제꺼까지. 기존 행의 마지막 날짜로 부터 하루 뒤부터 어제 까지 데이터를 인베스팅 닷컴엣서 끌어옴.
df_WTI.rename(columns={'Close':'Price'}, inplace=True)

print(df_WTI.head())
print(df_WTI.tail())

updated_df = pd.concat([updated_df, df_WTI])
print('업데이트된 df :', updated_df.tail(10))
updated_df.to_pickle('./datasets/updated_df.pickle')  # 업데이트된 DataFrame 저장시키기(나중에 다시 불러올 데이터)

############### 여기까지, 기존 프레임에 새로운날짜를 업데이트 하는 거 완료 ######################################
last_30_data = updated_df[-30:][['Price']]
print('업데이트 마지막 30개 ', last_30_data.tail(10))

X_train, X_test, Y_train, Y_test = np.load('./models/crude_oil_scaled_data.npy', allow_pickle=True)
model = load_model('./models/crude_oil_model_1.h5')

with open('./models/minmaxscaler_oil.pickle', 'rb') as f:
  minmaxscaler = pickle.load(f)

scaled_last_30_data = minmaxscaler.transform(last_30_data)
print('shape : ',scaled_last_30_data.shape)
print('scaled_data 정보:', scaled_last_30_data) # 마지막 30 개 데이타당

######### 내일 값을 예측 시켜보자
today_predict = model.predict(scaled_last_30_data[:].reshape(1, 30, 1)) #마지막 30개 리셰잎 (30,1)짜리가 1개 있다.
print(today_predict) #minmaxscale 된값으로 나온담.

tomorrow_predicted_value = minmaxscaler.inverse_transform(today_predict) # inverse_transform하면 minmaxscaling한 값 다시 원래 값으로 복원시켜줌
print('$ %2f '%tomorrow_predicted_value[0][0])


# 일주일 간 프로그램 안쓰다가 어느날 오늘 정제오일 종가 예측하려고 '버튼'을 누른다.
# 버튼을 누르면 마지막으로 업데이트된 날짜(1주일) 이후부터 어제까지의 종가데이터가 긁어와져 한 데이터프레임으로 concat 된다
# 그 업데이트 된 DataFrame에서 마지막 30개를 꺼내 오늘의 정제오일 종가를 예측한 값을 화면에 보여준다.
# 업데이트 데이터 프레임이 자동으로 저장되고 불러와져야함.



