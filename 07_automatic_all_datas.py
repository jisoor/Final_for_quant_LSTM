import investpy
import pandas as pd
import FinanceDataReader as fdr   # pip install -U finance-datareader
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'nanumyeongjo'
plt.rcParams['figure.figsize'] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
# print(fdr.__version__) # 0.9.31

df_nasdaq = fdr.StockListing('NASDAQ') # KRX, KOSPI, KOSDAQ, AMEX, NYSE, SP500 다됨
print('NASDAQ 상장종목: ' ,df_nasdaq.head())
print(len(df_nasdaq)) # NASDAQ상장종목 전체 4481개

# Apple의 데이터 가져와보기 기간은 2008-04-01 부터 2022-01-18
# try:
#     df = fdr.DataReader('AAPL', '2008-04-01', '2022-01-18')[['Close']]
#     print(df.head(10))
# except:
#     df = fdr.DataReader('AAPL')[['Close']]   # 만약 2008년 이후 상장되었다면 최근 데이터 부터 2022-01-18데이터 가져오기
#     print('debug_01')
# df.info()
# df['Close'].plot()
# plt.legend()
# plt.show()

ticker = 'AAPL' #MSFT, AMZN, GOOGL 등 티커명 입력하면 데이터 알아서 불러와주게..
if df_nasdaq['Symbol'] == ticker:
    try:
        df = fdr.DataReader(ticker, '2008-04-01', '2022-01-18')[['Close']]
        print(df.head(10))
    except:
        df = fdr.DataReader(ticker)[['Close']]  # 만약 2008년 이후 상장되었다면 최근 데이터 부터 2022-01-18데이터 가져오기
        print('해당 주식의 상장일이 2008년 이후')
    df.info()
    print(df.isnull().sum())
# 데이터 가공 절차 nan값제거, 날짜 정렬, 앞에서 30개 씩 땡기기

# https://m.blog.naver.com/freed0om/221981429329