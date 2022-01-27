import yfinance as yf
# import FinanceDataReader as fdr
import pandas as pd


pd.set_option('display.max_columns', None) # 열이 전부다 나오게

currencies_lists = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'EURCAD=X', 'EURSEK=X',
                   'EURCHF=X', 'EURHUF=X', 'CNY=X', 'HKD=X', 'SGD=X', 'INR=X', 'MXN=X', 'PHP=X', 'IDR=X', 'THB=X',
                   'MYR=X', 'ZAR=X', 'RUB=X']

print(len(currencies_lists)) # currencies_lists :23개

for currencies_list  in currencies_lists:
    try:
        data = yf.download(currencies_list, start="2007-01-01", end="2022-01-26") # 26일까지로 해야 25일까지의 데이터가 나옴
        df = data[['Open', 'High', 'Adj Close']]
        print(currencies_list, ':', df.info())
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
        # df.to_csv('./datasets/currencies/{}_{}-{}.csv'.format(currencies_list, first_date, last_date))
    except:
        print('No data')