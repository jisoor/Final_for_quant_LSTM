import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
# import pandas_datareader import data as pdr
futures_list=['ES=F','YM=F','NQ=F','RTY=F','ZB=F',
              'ZN=F','ZF=F','ZT=F','GC=F','SI=F',
              'PL=F','HG=F','PA=F','CL=F','NG=F',
             'BZ=F','ZC=F','ZO=F','KE=F','ZR=F',
              'ZM=F','ZL=F','ZS=F','GF=F','HE=F',
             'LE=F','CC=F','KC=F','CT=F','LBS=F',
             'OJ=F','SB=F' ]
pd.options.display.float_format = '{:.2f}'.format

##csv다운로드 받는 코드#################################################
# Today = datetime.date.today()
# print('오늘: ', Today)
# print(today)
# for i in futures_list:
#     df=yf.download({i}, start='2007-01-01', end='2022-01-26')
#     df.to_csv('./yfinance_futures/futures_{}.csv'.format(i))
#############################################################################

###########폴더에서 티커명으로 딕셔너리dfs에 각각의 dataframe으로 저장###################3
dfs = {i: pd.read_csv('./yfinance_futures/futures_{}.csv'.format(i),index_col=0) for i in ['ES=F','YM=F','NQ=F','RTY=F','ZB=F',
              'ZN=F','ZF=F','ZT=F','GC=F','SI=F',
              'PL=F','HG=F','PA=F','CL=F','NG=F',
             'BZ=F','ZC=F','ZO=F','KE=F','ZR=F',
              'ZM=F','ZL=F','ZS=F','GF=F','HE=F',
             'LE=F','CC=F','KC=F','CT=F','LBS=F',
             'SB=F']}
print(type(dfs))

########################티커명을 보기쉽게 자산클래스이름으로 변경#######################
dfs['SPX'] = dfs.pop('ES=F')
dfs['DOW'] = dfs.pop('YM=F')
dfs['NASDAQ'] = dfs.pop('NQ=F')
dfs['RUSSEL2000'] = dfs.pop('RTY=F')
dfs['US30YT'] = dfs.pop('ZB=F')
dfs['US10YT'] = dfs.pop('ZN=F')
dfs['US5YT'] = dfs.pop('ZF=F')
dfs['US2YT'] = dfs.pop('ZT=F')
dfs['GOLD'] = dfs.pop('GC=F')
dfs['SILVER'] = dfs.pop('SI=F')
dfs['PLATINUM'] = dfs.pop('PL=F')
dfs['COPPER'] = dfs.pop('HG=F')
dfs['PALLADIUM'] = dfs.pop('PA=F')
dfs['CRUDE_OIL'] = dfs.pop('CL=F')
# dfs['HEATING_OIL'] = dfs.pop('HO=F')
dfs['NATURAL_GAS'] = dfs.pop('NG=F')
# dfs['GASOLINE'] = dfs.pop('RB=F')
dfs['BRENT_OIL'] = dfs.pop('BZ=F')
dfs['CORN'] = dfs.pop('ZC=F')
dfs['OAT'] = dfs.pop('ZO=F')
dfs['WHEAT'] = dfs.pop('KE=F')
dfs['ROUGH_RICE'] = dfs.pop('ZR=F')
dfs['SOYBEAN_MEAL'] = dfs.pop('ZM=F')
dfs['SOYBEAN_OIL'] = dfs.pop('ZL=F')
dfs['SOYBEAN'] = dfs.pop('ZS=F')
dfs['FEEDER_CATTLE'] = dfs.pop('GF=F')
dfs['LEAN_HOGS'] = dfs.pop('HE=F')
dfs['LIVE_CATTLE'] = dfs.pop('LE=F')
dfs['COCOA'] = dfs.pop('CC=F')
dfs['COTTON'] = dfs.pop('CT=F')
dfs['LUMBER'] = dfs.pop('LBS=F')
dfs['SUGAR'] = dfs.pop('SB=F')
# dfs['ORANGE_JUICE'] = dfs.pop('OJ=F')

############딕셔너리의 각 데이터프레임에 k(자산클래스이름), v(그 자산클래스의 데이터프레임)#################
############# for loop으로 일괄 전처리작업#####################################
for k,v in dfs.items():

    v.index = pd.to_datetime((v.index))
    v=v[['High','Low','Adj Close']]
    v.columns = ['High', 'Low', 'Adj_Close']
    v['Change'] = v.Adj_Close.pct_change()
    v['Change']=v['Change']*100
    v['Change'] = v['Change'].astype("float").round(2)
    v['Change'] = v['Change'].fillna(0)
    print(v.isnull().sum())
    v.fillna(method='ffill')
    print(v.isnull().sum())
    # print(v.tail(20))
    print(k, v)
    v.to_csv('./yfinance_futures/cleaned/futures_{}.csv'.format(k))
    # i.index = pd.to_datetime((i.index))
    # print(i.info())
# # print (dfs['ES=F'])
# spx=dfs['ES=F']
# print(spx.info())
#
