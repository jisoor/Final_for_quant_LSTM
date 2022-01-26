import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd

pd.set_option('display.max_columns', None) # 열이 전부다 나오게

# krx = fdr.StockListing('KRX')
# print(len(krx))
# print(krx)
# print(krx.columns)
# data = yf.download("ES=F", start="2008-04-01", end="2022-01-25")
# print(data.head())
# print(data.info())
data = yf.download('^GSPC', start="2007-01-01", end="2022-01-25")
df = data[['Open', 'High', 'Adj Close']]
df.info()
print(df.head(5))
nul = df.isnull().sum()
if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):       # 'Open'에 난값이 존재하면
    df.fillna(method='ffill', inplace=True)
# Nan값 분포에 대한 plot도 작성
# 기본적으로 (현재행 - 이전행) / 이전행 / -1이면 반대로
change = pd.DataFrame({'Change' : df['Adj Close'].pct_change(periods=1)})
print(type(change))
print('Change 등락의 백분율: ', change)

change.iloc[0] = 0   # 첫행은 Nan이므로 0으로 통일 시켜줌
print(change.head())
df = pd.concat([df, change], axis=1)
print('콘캣된: ', df)
exit()

print(type(nul)) # series
print(nul[0])  # 0
print(nul[1])  # 0
print(type(nul[2]))  # int
print(df.isnull().sum())
# df['Change'] = df['Adj Close'].pct_change(df)

world_indices = ['^GSPC', '^DJI', '^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX','^FTSE','^GDAXI',
    '^FCHI','^STOXX50E','^N100','^BFX','IMOEX.ME','^N225','^HSI','000001.SS','399001.SZ','^STI','^AXJO','^AORD','^BSESN',
    '^JKSE','^KLSE','^NZ50','^KS11','^TWII','^GSPTSE','^BVSP','^MXX','^IPSA','^MERV','^TA125.TA','^CASE30','^JN0U.JO']

# 맡는 파트 -> 갑래: futures, 영준: currencies, 지수:world indices 안의 모든 종목들을
# 다운로드 받아서 Date를 index 로 가지고  ' low', 'High', 'Adj Close' 의 3개 열로로 구성된 데이터프레임 만들기 (for 문 사용)
# 하나의 열을 더 추가할 건데 name = Change, 이고 Adj Close열에다가 pct_change를 쓴다.(전날에 비해 등락율)
# 2007-01-01 ~ 2022-01-25 인지 확인.

# 저장은 world_indices_df.csv 등 의 형식으로
# 완성 되면 코드랑 csv파일을 드라이브에 올려줍니다.


# 각자 맡은 파트 전부 for문 사용해서 modeling 하기
# 모델링 이름은 '티커명'_model

# for world_indice in world_indices:
#     try:
#         data = yf.download(world_indice, start="2007-01-01", end="2022-01-25")
#         df = data[['Open', 'High', 'Adj Close']]
#     except:
#         print('No data')
