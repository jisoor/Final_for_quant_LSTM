import investpy
import pandas as pd
import numpy as np
import datetime
updated_df = np.load('./datasets/last60_data.npy')
# 거래시간(한국시간)	(월~토) 10:00~08:00 (썸머타임 시, 1시간 앞당겨짐)

# 마지막 60개 짜리 데이터프레임   2021-10-27 ~ 2022-01-18 피클 담가서 가져오기
updated_df = pd.read_pickle('./last_60.pkl')
print(updated_df.tail())
updated_df.info()

len_index = len(updated_df.index)   # 60
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

df_WTI = df_WTI[start_day:Yesterday][['Close']] # 01-19부터 어제꺼까지. 기존 행의 마지막 날짜로 부터 하루 뒤부터 어제 까지 데이터를 인베스팅 닷컴엣서 끌어옴. 만약 일요일이면 +2일 더해라.
df_WTI.rename(columns={'Close':'Price'}, inplace=True)

print(df_WTI.head())
print(df_WTI.tail())

updated_df = pd.concat([updated_df, df_WTI])
print('업데이트된 df :', updated_df.tail(10))

############### 여기까지, 기존 프레임에 새로운날짜를 업데이트 하는 거 완료 ######################################
updated_df[-30:]

