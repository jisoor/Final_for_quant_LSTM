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

# 종목 입력시 이름 찾아준다.
ticker = '^AORD'

for world_indice in world_indices:
    if world_indice[0] == ticker:
        name = world_indice[1]
        break


# last_30 데이터 먼저 가져오기
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
b = datetime.datetime.strptime('23:59:00', '%H:%M:%S').time()
c = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
d = datetime.datetime.strptime('05:59:00', '%H:%M:%S').time()

currentTime = datetime.datetime.now().time() # 현재시간만.
Today = datetime.date.today()
last_date_from_previous_df = pd.to_datetime(last_30_Change.index[-1]).date() #01.25일
one_day = datetime.timedelta(days=1)


# 현재 시간대에 따라 알아서 적합한 데이터갯수를 추가하여 데이터프레임을 업데이트시켜줌
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


# 인덱스의 이름이 '종목'이고 '열'이름이 예측치인 빈 DataFrame을 만든다.
data = { '종목': [name],
         'High' : ['high'] }
predict_4_df = pd.DataFrame(data)
predict_4_df.set_index('종목', inplace=True)
predict_4_df.columns.name = '예측치'

# 각각의 column을 예측해주는 과정 및 모든 예측결과를 데이터프레임으로 save_predict_tmr 폴더에 저장해준다,
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
    final_df.to_csv('./updated/{}_{}_updated.csv'.format(name, col), index=True )

    # 마지막 30개 예측
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


    predict_4_df[col] =  [tmr_predicted_value]
    predict_4_df.to_csv('./save_predict_tmr/{}_의 내일 예측치.csv'.format(name))
    predict_4_df.info()

# 최종적으로 4개의 모든 데이터를 업데이트 하면서 csv로 저장하고, 4개 각각의 예측값을 알려준다.




