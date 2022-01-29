import pandas as pd
import yfinance as yf
import datetime
import pickle
import time
from tensorflow.keras.models import load_model
import numpy as np

world_indices =  [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
 ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('IMOEX.ME','MOEX Russia Index'),('^MXX','IPC MEXICO'),
 ('^JKSE', 'Jakarta Composite Index'),('^KS11','KOSPI Composite Index'),('^MERV','MERVAL'),('^IXIC','NASDAQ Composite'),
 ('^N225', 'Nikkei 225'),('^XAX','NYSE AMEX COMPOSITE INDEX'),('^NYA','NYSE COMPOSITE (DJ)'),('^RUT','Russell 2000'),('^GSPC','S&P 500'),
 ('^BSESN', 'S&P BSE SENSEX'), ('399001.SZ', 'Shenzhen Component'),('000001.SS', 'SSE Composite Index'),('^STI','STI Index'),
 ('^TA125.TA', 'TA-125'),('^TWII','TSEC weighted index'),('^VIX','Vix')]
# 일단 ALL ORDINARIES 으로 해봐야지..

# last_30 데이터 먼저 가져오기

ticker = '^AORD'    # 종목 입력시 이름 찾아준다.

for world_indice in world_indices:
    if world_indice[0] == ticker:
        name = world_indice[1]
        break
print(name)


last_30_High = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(name, 'High'))
print(last_30_High.tail())
last_30_Low = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(name, 'Low'))
print(last_30_Low.tail())
last_30_Close = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(name, 'Adj Close'))
print(last_30_Close.tail())
last_30_Change = pd.read_pickle('./pickles/{}_{}_last30_data.pickle'.format(name, 'Change'))
print(last_30_Change.tail())





# A. 내일의 종가 예측 (밑에 대로 해야, 주말도 감안할 수 있음) 티커입력해주면 가져와서 4개 column에 관해 전부 다 예측
# - 08:00:00 – 23:59:00 이라면 하루전까지의 데이터
# - 24:00:00 – 05:59:00 이라면 이틀전까지의 데이터(마지막 종가 이전 것 까지)
# - 06:00:00 - 07:59:00 이라면 ' 8am 부터 다시 시도하시오 '

a = datetime.datetime.strptime('08:00:00', '%H:%M:%S').time()
print(type(a))
print(a)
b = datetime.datetime.strptime('23:59:00', '%H:%M:%S').time()
c = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
d = datetime.datetime.strptime('05:59:00', '%H:%M:%S').time()

currentTime = datetime.datetime.now().time() # 현재시간만.
print(currentTime) # 13:29:20.309091
Today = datetime.date.today()
print(Today) # 2022-01-29
last_date_from_previous_df = pd.to_datetime(last_30_Change.index[-1]).date() #01.25일
print(last_date_from_previous_df) # 2022-01-25 00:00:00
one_day = datetime.timedelta(days=1)
two_days = datetime.timedelta(days=2)


col_list = ['High', 'Low', 'Adj Close', 'Change']
# 01/26 - 01/28거까지 추가할거임.
print('start: ' , last_date_from_previous_df + one_day) # 01-26
print('end : ', Today) # 01-29
if a <= currentTime <= b:
    print('하루 전 데이터까지')
    new_data = yf.download(world_indices[0][0], start=last_date_from_previous_df + one_day, end=Today)  # 마지막 날짜부터 어제
    # date만 가져오는 코드 어떡하지?(시간제외)
    print(new_data)  # 01/26(휴무일이라 안나옴) -> 01/27, 01/28 데이터만 나옴
    print(new_data.index)

elif c <= currentTime <= d:
    print('마지막 종가 이전거 까지')
    new_data = yf.download(world_indices[0][0], start=last_date_from_previous_df + one_day, end=Today)  # 마지막 날짜부터 어제-1day 데이터까지
    new_data = new_data[:-1]
    print(new_data)
else:
    print(' 8AM에 이후로 다시 시도하시오.')

print('Low :', new_data[['Low']])
# new_High = new_data[['High']]
# final_High = pd.concat([last_30_High, new_High])
# # updated_High 저장코드
# pd.to_csv('./updated/{}_{}_updated.csv'.format(world_indices[0][1], col_list[0]), index=False) # index 이거 맞나?
# # updated_High로 마지막 30개 예측
# with open('./minmaxscaler/{}_{}_minmaxscaler.pickle'.format(world_indices[0][1], col_list[0]), 'rb') as f:
#     minmaxscaler = pickle.load(f)
# scaled_final_High = minmaxscaler.transform(final_High)
# # model 및 train_test_set
# model = load_model('./models/{}_{}_model.h5'.format(world_indices[0][1], col_list[0]))
# tmr_predict = model.predict(scaled_final_High.reshape(1, 30, 1))
# print(tmr_predict)
# tmr_predicted_value = minmaxscaler.inverse_transfrom(tmr_predict)
# print('내일예측값$ %2f '%tmr_predicted_value[0][0])
# X_train, X_test, Y_train, Y_test = np.load('./models/{}_{}_train_test.npy'.format(world_indices[0][1], col_list[0]), allow_pickle=True)


# # High, Low, Close 의 데이터프레임 각각 만들기
# new_High = new_data[['High']]
# updated_High = pd.concat([last_30_High, new_High])
# print(updated_High.tail())
# new_Low = new_data[['Low']]
# updated_Low = pd.concat([last_30_Low, new_Low])
# print('Low :' , updated_Low)
# new_Close = new_data[['Adj Close']]
# updated_Close = pd.concat([last_30_Close, new_Close])
# print(updated_Close)
# exit()
#
# # Change열 전환하여 추가하는 과정.
# last_30_Close_last_one_row = last_30_Close.iloc[[-1]] # 마지막 종가 행 하나가져오기
# print(last_30_Close_last_one_row)
# updated_Close_before_Change = pd.concat([last_30_Close_last_one_row, new_Close]) # 마지막 종가하나랑 추가된 종가랑 합친 df 만들기
# print(updated_Close_before_Change)
# new_Change = (updated_Close_before_Change.pct_change(periods=1) * 100).round(2)
# print(new_Change)
# new_Change = new_Change[1:]
# print( ' NAn값 제외한 clean한 값만 :', new_Change)
# new_Change.rename(columns={'Adj Close':'Change'}, inplace=True)
# updated_Change = pd.concat([last_30_Change, new_Change])
# print(updated_Change)

# 인덱스의 이름이 '종목'이고 '열'이름이 예측치인 빈 DataFrame을 만든다.
data = { '종목': [name],
         'High' : ['high'] }
predict_4_df = pd.DataFrame(data)
predict_4_df.set_index('종목', inplace=True)
predict_4_df.columns.name = '예측치'
print(predict_4_df.index)

col_list = [(last_30_High, 'High'), (last_30_Low, 'Low'),(last_30_Close, 'Adj Close'), (last_30_Change, 'Change')]
for last_30, col in col_list:
    if col == 'Change':
        # change 열 전환하여 추가하는 과정
        last_30_Close_last_one_row = last_30_Close.iloc[[-1]]  # 마지막 종가 행 하나가져오기
        print(last_30_Close_last_one_row)
        new_Close = new_data[['Adj Close']]
        updated_Close_before_Change = pd.concat([last_30_Close_last_one_row, new_Close])
        print(updated_Close_before_Change)
        new_Change = (updated_Close_before_Change.pct_change(periods=1) * 100).round(2)
        print(new_Change)
        new_Change = new_Change[1:]
        new_Change.rename(columns={'Adj Close': 'Change'}, inplace=True)
        final_df = pd.concat([last_30, new_Change])
        print(final_df)
    else:
        divided_new_data = new_data[[col]]
        final_df = pd.concat([last_30, divided_new_data])


        # updated_High 저장코드
    final_df.to_csv('./updated/{}_{}_updated.csv'.format(name, col), index=True )
    # updated_High로 마지막 30개 예측
    with open('./minmaxscaler/{}_{}_minmaxscaler.pickle'.format(name, col), 'rb') as f:
        minmaxscaler = pickle.load(f)
    final_df = final_df[-30:]
    scaled_final_df = minmaxscaler.transform(final_df)
    model = load_model('./models/{}_{}_model.h5'.format(name, col))
    tmr_predict = model.predict(scaled_final_df.reshape(1, 30, 1))
    print(tmr_predict)
    tmr_predicted_value = minmaxscaler.inverse_transform(tmr_predict)
    print('내일예측값$ %2f '%tmr_predicted_value[0][0])
    print('{}의 내일 예측값_{}'.format(col, tmr_predicted_value))

    # if col == 'High':
    # data = { '종목' : [name],
    #     col : [tmr_predicted_value] }
    predict_4_df[col] =  [tmr_predicted_value]
    # save_df = pd.DataFrame(data, index=['종목'])
    # save_df.set_index('종목')
    # save_df.columns.name = '예측치'
    # print(save_df)
    predict_4_df.to_csv('./save_predict_tmr/{}_의 내일 예측치.csv'.format(name))
    predict_4_df.info()
# 최종적으로 4개의 모든 데이터를 업데이트 하면서 csv로 저장하고, 4개 각각의 예측값을 알려준다.




# B. 정해진 기간범위의 정답률 예측