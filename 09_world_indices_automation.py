import yfinance as yf
# import FinanceDataReader as fdr
import pandas as pd

pd.set_option('display.max_columns', None) # 열이 전부다 나오게

# krx = fdr.StockListing('KRX')
# print(len(krx))
# print(krx)
# print(krx.columns)
# data = yf.download("ES=F", start="2008-04-01", end="2022-01-25")
# print(data.head())
# # print(data.info())
# data = yf.download('^GSPC', start="2007-01-01", end="2022-01-27")
# df = data[['Open', 'High', 'Adj Close']]
# df.info()
# print(df.tail(5))
# print(df.index[0])
# print(str(df.index[0]))
# print(str(df.index[0]).split()[0])
# exit()
# nul = df.isnull().sum()
# if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):       # 'Open'에 난값이 존재하면
#     df.fillna(method='ffill', inplace=True)
# # Nan값 분포에 대한 plot도 작성
# # 기본적으로 (현재행 - 이전행) / 이전행 / -1이면 반대로
# change = pd.DataFrame({'Change' : df['Adj Close'].pct_change(periods=1)})
# print(type(change))
# print('Change 등락의 백분율: ', change)
#
# change.iloc[0] = 0   # 첫행은 Nan이므로 0으로 통일 시켜줌
# print(change.head())
# df = pd.concat([df, change], axis=1)
# print('콘캣된: ', df)



world_indices = ['^GSPC', '^DJI', '^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX','^FTSE','^GDAXI',
    '^FCHI','^STOXX50E','^N100','^BFX','IMOEX.ME','^N225','^HSI','000001.SS','399001.SZ','^STI','^AXJO','^AORD','^BSESN',
    '^JKSE','^KLSE','^NZ50','^KS11','^TWII','^GSPTSE','^BVSP','^MXX','^IPSA','^MERV','^TA125.TA']
print(len(world_indices)) # world_indices : 36개->34개

for world_indice in world_indices:
    try:
        data = yf.download(world_indice, start="2007-01-01", end="2022-01-26") # 26일까지로 해야 25일까지의 데이터가 나옴
        df = data[['Open', 'High', 'Adj Close']]
        # print(world_indice , df.head())
        print(world_indice, ':', df.info())
        print(len(df))
        nul = df.isnull().sum()
        if (nul[0] > 0) or (nul[1] > 0) or (nul[2] > 0):  # 'Open'에 난값이 존재하면
            df.fillna(method='ffill', inplace=True)
        # 등락율 백분율 전환
        change = pd.DataFrame({'Change': df['Adj Close'].pct_change(periods=1)}) # (현재행 - 이전행) / 이전행 <-period=1일 경우.
        change.iloc[0] = 0  # 첫행은 Nan이므로 0으로 통일 시켜줌
        df = pd.concat([df, change], axis=1)
        # 시작날짜, 마지막 날짜
        first_date = str(df.index[0]).split()[0]
        last_date = str(df.index[-1]).split()[0]
        print('날짜' , first_date, last_date)
        df.to_csv('./datasets/world_indices/{}_{}-{}.csv'.format(world_indice, first_date, last_date))
    except:
        print('No data')


# 결론 : ^CASE30은 데이터 없어서 제외시킴
# ^JNOU.JO는 데이터가 1085개로 다른 것들에 비해 1/3이므로 제외시킴
# 총 리스트는 : ['^GSPC', '^DJI', '^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX','^FTSE','^GDAXI',
#     '^FCHI','^STOXX50E','^N100','^BFX','IMOEX.ME','^N225','^HSI','000001.SS','399001.SZ','^STI','^AXJO','^AORD','^BSESN',
#     '^JKSE','^KLSE','^NZ50','^KS11','^TWII','^GSPTSE','^BVSP','^MXX','^IPSA','^MERV','^TA125.TA']





