import investpy
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.2f}'.format

# bn=investpy.indices.get_indices(country='japan' )
# print(bn)
df=investpy.indices.get_index_historical_data('Nikkei 225' , 'japan' , from_date='03/01/2007' , to_date='25/01/2022')
print(df.info())

# df.to_csv('Cleaned_nikkei.csv', index=False)

df.drop('Currency', axis=1, inplace=True)
df.drop('Volume', axis=1, inplace=True)

df['Close'].plot()
plt.show()
print(df.info())

# # ##Investing.com에서 받은 자료는 시간 역순으로 뒤바꿔줘야 한다!!
# df = df.iloc[::-1]

data=df[:-30][['Close']]  ##날짜기준 슬라이싱********최근 한달간은 제외하고 모델 학습
print(data.info())

# print(df.head())
print(data.head())

from sklearn.preprocessing import MinMaxScaler
minmaxscaler=MinMaxScaler()
scaled_data=minmaxscaler.fit_transform(data)
#
#
#
sequence_X=[]
sequence_Y=[]
with open('./models/minmaxscaler_nikkei_30.pickle', 'wb') as f:
    pickle.dump(minmaxscaler, f)
# with open('./models/minmaxscaler_nikkei_30.pickle', 'rb') as f:
#     minmaxscaler = pickle.load(f)

for i in range(len(scaled_data)-31):
    x=scaled_data[i:i+30]
    y=scaled_data[i+30]
    sequence_X.append(x)
    sequence_Y.append(y)

    if i is 0:
        print(x, '->',y)
        print(len(x))

sequence_X=np.array(sequence_X)
sequence_Y=np.array(sequence_Y)
print(sequence_X[0])
print(sequence_Y[0])
print(sequence_X.shape)
print(sequence_Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(
sequence_X, sequence_Y, test_size=0.1)
XY = X_train, X_test, Y_train, Y_test
#
np.save('./models/nikkei_XY.npy', XY)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#
# #
# #
# X_train, X_test, Y_train, Y_test = np.load('./models/nikkei_XY.npy', allow_pickle=True)
# model = load_model('./models/nikkei_lstm_512_30.h5')
######################################################

model=Sequential()
model.add(LSTM(512,input_shape=(30,1), activation='tanh', return_sequences=True)) #리턴이 없으면 맨 마지막 값만 보게된다.
model.add(Dropout(0.2))
#
#
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam') ##값을 예측하는 모델에서는 마지막 layer에서 sigmoid activation을 쓰면 안됨
model.summary()
##LSTM이 여러층 쌓여있을 경우 마지막은 없고 마지막 전의 레이어들엔 무조건 return sequences가 있어야 한다
#

fit_hist=model.fit(X_train, Y_train, epochs=100,
                  validation_data=(X_test, Y_test),shuffle=False) #셔플=False의 의미==한 에폭안에서 섞어넣지 말고 시간순으로 입력
####예측값이기 때문에 loss만 본다
######################################################


model.save('./models/nikkei_lstm_512_30.h5')

plt.plot(fit_hist.history['loss'][5:])
plt.plot(fit_hist.history['val_loss'][5:])
plt.show()
#
#
#
# # #h5 파일 로딩
# model = load_model('./models/nikkei_lstm_512_30.h5')
predict=model.predict(X_test)

plt.figure(figsize=(20, 10))
plt.plot(Y_test[0:110], label='actual') ##날짜순이 아닌 랜덤하게
plt.plot(predict[0:110], label='predict')
plt.legend()
plt.show()
#
#
# last_test_data=pd.concat([last_30,recent_data])
# print(last_test_data.head())
# print(last_test_data.tail())
# print(last_test_data.info())
last_test_data = df[-60:][['Close']]
last_test_data.to_pickle("./models/last_60_nikkei.pkl")

# with open('./models/minmaxscaler_nikkei_30.pickle', 'rb') as f:
#     minmaxscaler = pickle.load(f)
#
scaled_last_test_data=minmaxscaler.transform(last_test_data)
print(scaled_last_test_data.shape)
print('scaled_data 정보:', scaled_last_test_data)
#
# #ndarray 저장하는 코드
# np.save('./models/nikkei_last60_data.npy', scaled_last_test_data)
#
predict_last=model.predict(sequence_X)
plt.plot(sequence_Y, label='actual')
plt.plot(predict_last, label='predict')
plt.legend()
plt.show()

tomorrow_predict=model.predict(scaled_last_test_data[-30:].reshape(1,30,1))#모델이 한묶음씩 학습을 하기땜에 번들형식으로 묶음으로 (30,1)을 1개씩
print(tomorrow_predict) ##빼서 학습시킨다
print(tomorrow_predict.shape)


tomorrow_predict_value=minmaxscaler.inverse_transform(tomorrow_predict)[0][0] ##[0][0]은 [][]벗겨주는것
print('$ %2f  ' % tomorrow_predict_value)

# ##내일실제 가격이 나오면 데이터에 추가하여 또 다음날을 예상하는 코드
# dat_scaled_last_test_data=np.append(scaled_last_test_data,tomorrow_predict)
# print(len(dat_scaled_last_test_data))
#
# dat_predict=model.predict(dat_scaled_last_test_data[-30:].reshape(1,30,1))
# print(dat_predict)
#
# dat_predicted_value=minmaxscaler.inverse_transform(dat_predict)
# print('$ %2f' % dat_predicted_value)

