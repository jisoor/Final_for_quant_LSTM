import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# 필요한 패키지와 라이브러리를 가져옴
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fma
# %matplotlib inline
pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 14)

mpl.rcParams['axes.unicode_minus'] = False

# customize matplitlib
plt.rcParams["figure.figsize"] = (20,15)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['font.size'] = 15

df = pd.read_csv("./PCA/cleaned_01.csv")
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date')
df=df.drop(['Date'],axis=1)
# print(df.info())
# print(df.columns)
scaler = MinMaxScaler()
df_scale = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index = df.index)

x = df.drop(['futures_CORN'], axis=1).values # 독립변인들의 value값만 추출
y = df['futures_CORN'].values # 종속변인 추출

x = MinMaxScaler().fit_transform(x) # x객체에 x를 표준화한 데이터를 저장
#
features = ['AUDUSD=X_2007-01-01-2022-01-25', 'BEL 20_2007-01-02-2022-01-25',
       'CAC 40_2007-01-02-2022-01-25', 'CNY=X_2007-01-01-2022-01-25',
       'DAX PERFORMANCE-INDEX_2007-01-02-2022-01-25',
       'Dow Jones Industrial Average_2007-01-03-2022-01-25', 'ESTX 50 PR',
       'EURCAD=X_2007-01-01-2022-01-25', 'EURCHF=X_2007-01-01-2022-01-25',
       'EURGBP=X_2007-01-01-2022-01-25', 'EURHUF=X_2007-01-01-2022-01-25',
       'EURJPY=X_2007-01-01-2022-01-25',
       'Euronext 100 Index_2007-01-02-2022-01-25',
       'EURSEK=X_2007-01-01-2022-01-25', 'EURUSD=X_2007-01-01-2022-01-25',
       'FTSE 100_2007-01-02-2022-01-25',
       'FTSE Bursa Malaysia KLCI_2007-01-03-2022-01-25', 'futures_BRENT_OIL',
       'futures_COCOA', 'futures_Coffee', 'futures_COPPER','KOSPI Composite Index_2007-01-02-2022-01-25' ,
       'futures_COTTON', 'futures_CRUDE_OIL', 'futures_DOW',
       'futures_FEEDER_CATTLE', 'futures_GOLD', 'futures_LEAN_HOGS',
       'futures_LIVE_CATTLE', 'futures_LUMBER', 'futures_NASDAQ',
       'futures_NATURAL_GAS', 'futures_OAT', 'futures_PALLADIUM',
       'futures_PLATINUM', 'futures_ROUGH_RICE', 'futures_SILVER',
       'futures_SOYBEAN', 'futures_SOYBEAN_MEAL', 'futures_SOYBEAN_OIL',
       'futures_SPX', 'futures_SUGAR', 'futures_US10YT', 'futures_US2YT',
       'futures_US30YT', 'futures_US5YT', 'futures_WHEAT',
       'GBPJPY=X_2007-01-01-2022-01-25', 'GBPUSD=X_2007-01-01-2022-01-25',
       'HANG SENG INDEX_2007-01-02-2022-01-25', 'HKD=X_2007-01-01-2022-01-25',
       'IBOVESPA_2007-01-02-2022-01-25', 'IDR=X_2007-01-01-2022-01-25',
       'INR=X_2007-01-01-2022-01-25', 'IPC MEXICO_2007-01-02-2022-01-25',
       'Jakarta Composite Index_2007-01-02-2022-01-25',
       'JPY=X_2007-01-01-2022-01-25',

       'MERVAL_2007-01-02-2022-01-25', 'MXN=X_2007-01-01-2022-01-25',
       'MYR=X_2007-01-01-2022-01-25', 'NASDAQ Composite_2007-01-03-2022-01-25',
        'Nikkei 225_2007-01-04-2022-01-25',
       'NYSE AMEX COMPOSITE INDEX_2007-01-03-2022-01-25',
       'NYSE COMPOSITE (DJ)_2007-01-03-2022-01-25',
       'NZDUSD=X_2007-01-01-2022-01-25', 'PHP=X_2007-01-01-2022-01-25',
       'RUB=X_2007-01-01-2022-01-25', 'Russell 2000_2007-01-03-2022-01-25',
       'SGD=X_2007-01-01-2022-01-25',
       'Shenzhen Component_2007-01-04-2022-01-25',
       'SSE Composite Index_2007-01-04-2022-01-25',
       'STI Index_2007-01-03-2022-01-25', 'S_P 500_2007-01-03-2022-01-25',
       'S_P BSE SENSEX_2007-01-02-2022-01-25', 'THB=X_2007-01-01-2022-01-25',
       'TSEC weighted index_2007-01-02-2022-01-25',
       'Vix_2007-01-03-2022-01-25', 'ZAR=X_2007-01-01-2022-01-25']
# print(pd.DataFrame(x, columns=features).head())

from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(x)
# X_pc = model.transform(x)
principalDf = pd.DataFrame(data=printcipalComponents,columns = ['pc1', 'pc2'])
# n_pcs= pcs.components_.shape[0]
X_rec = pca.inverse_transform(principalDf)
print(X_rec)
# 주성분으로 이루어진 데이터 프레임 구성
# columns = ['principal component1', 'principal component2',
# 'principal component3','principal component4','principal component5','principal component6',
# 'principal component7','principal component8','principal component9','principal component10']
# print(abs(principalDf.corr()['principal component1']))

# print(principalDf.tail())
#
print(sum(pca.explained_variance_ratio_))

# print(principal component1)


