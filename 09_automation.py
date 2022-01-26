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
print(data) # Open High Adj Close
df = data[['Open', 'High', 'Adj Close']]
df.info()
print(df.head(5))

world_indices = ['^GSPC', '^DJI', '^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX','^FTSE','^GDAXI',
    '^FCHI','^STOXX50E','^N100','^BFX','IMOEX.ME','^N225','^HSI','000001.SS','399001.SZ','^STI','^AXJO','^AORD','^BSESN',
    '^JKSE','^KLSE','^NZ50','^KS11','^TWII','^GSPTSE','^BVSP','^MXX','^IPSA','^MERV','^TA125.TA','^CASE30','^JN0U.JO']

# 맡는 파트 -> 갑래: futures, 영준: currencies, 지수:world indices 안의 모든 종목들을
# 다운로드 받아서 Date를 index 로 가지고  ' low', 'High', 'Adj Close' 의 3개 열로로 구성된 데이터프레임 만들기 (for 문 사용)
# 하나의 열을 더 추가할 건데 name = Change, 이고 Adj Close열에다가 pct_change를 쓴다.(전날에 비해 등락율)
# 2007-01-01 ~ 2022-01-25 인지 확인.
# Nan 값 처리는 fillna로 이전값을 취함.
# Nan값 분포에 대한 plot도 작성
# 저장은 world_indices_df.csv 등 의 형식으로
# 완성 되면 코드랑 csv파일을 드라이브에 올려줍니다.


# 각자 맡은 파트 전부 for문 사용해서 modeling 하기
# 모델링 이름은 '티커명'_model
#




for world_indice in world_indices:
    try:
        data = yf.download(world_indice, start="2007-01-01", end="2022-01-25")
        df = data[['Open', 'High', 'Adj Close']]
    except:
        print('No data')
