import yfinance as yf
# import FinanceDataReader as fdr
import pandas as pd
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping  # early_stopping 걸기
# from matplotlib.backends.backend_pdf import PdfPages
early_stopping=EarlyStopping(monitor='val_loss',patience=5) # early_stopping 걸기
pd.set_option('display.max_columns', None) # 열이 전부다 나오게

#####################  재모델링 #################
world_indice = ('^VIX','Vix')


# 인덱스와 컬럼만 지정한 빈 DF 만들기

df_loss = pd.DataFrame(columns=['Vix'])
mse = ['High', 'Low', 'Adj Close', 'Change', 'Average'] # 인덱스
df_loss = pd.DataFrame({'mse':mse})     # 'mse' 5개 값을 가진 column을 만듬
df_loss['Vix'] = np.nan           # nan 값으로 채워서 빈 데이터프레임 만들기
df_loss.set_index('mse', inplace=True)  # mse 안 5개 값을 인덱스로 나열해줌(추후 transpose(T)해서 인덱스-컬럼 바꿀것임).
print(df_loss)



plt.figure(figsize=(8, 18))
df = pd.read_csv('./datasets/world_indices/Vix_2007-01-03-2022-01-25.csv', index_col=0)
df_lists = [('df_high','High'), ('df_low','Low'), ('df_close','Adj Close'), ('df_change','Change')] # ['df_high',' df_low', 'df_close', 'df_change']
plot_num = 0
for df_each, colname in df_lists:
    plot_num += 1   # 1, 2, 3, 4
    df_each = df[[colname]]
    last_60_df = df_each[-60:]  # 마지막 30개만 따로 빼놓기(벡테스팅용)
    print(last_60_df.tail())
    df_each = df_each[:-30]  # 마지막 30개 빼고 모델링
    print(type(df_each)) # DataFrame
    last_60_df.to_csv('./updated/Vix_{}_updated.csv'.format(colname))
    minmaxscaler = MinMaxScaler()
    scaled_data = minmaxscaler.fit_transform(df_each)  # 스케일링해주기
    with open('./minmaxscaler/Vix_{}_minmaxscaler.pickle'.format(colname), 'wb') as f:
        pickle.dump(minmaxscaler, f)
    sequence_X = []
    sequence_Y = []
    for i in range(len(scaled_data) - 30):
        _x = scaled_data[i:i + 30]  # 총 30개
        _y = scaled_data[i + 30]  # 31번째를 예측
        sequence_X.append(_x)
        sequence_Y.append(_y)
    sequence_X = np.array(sequence_X)
    sequence_Y = np.array(sequence_Y)
    X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
    xy = X_train, X_test, Y_train, Y_test
    np.save('./train_test_split/Vix_{}_train_test.npy'.format(colname), xy)  # 저장하고

    model = Sequential()
    model.add(LSTM(512, input_shape=(30, 1), activation='tanh', return_sequences=1))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    fit_hist = model.fit(X_train, Y_train, epochs=100, callbacks=[early_stopping],  shuffle=False, validation_data=(X_test, Y_test))

    # 플롯 차트 #
    plt.plot(fit_hist.history['loss'][-30:], label='loss')
    plt.plot(fit_hist.history['val_loss'][-30:], label='val_loss')
    mse = fit_hist.history['val_loss'][-1]
    print('val_loss값은?? :', mse)
    plt.subplot(4, 1, plot_num)
    plt.title('Vix')
    plt.ylabel(colname)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

        # 위에서 만든 DataFrame에 Key값들 채우기.
    if colname == 'Change':
        df_loss.loc[colname]['Vix'] = mse
        df_loss.loc['Average']['Vix'] = (df_loss.loc['High']['Vix'] + df_loss.loc['Low']['Vix']
                 + df_loss.loc['Adj Close']['Vix'] + df_loss.loc['Change']['Vix']) / 4
    else:
        df_loss.loc[colname]['Vix'] = mse # '클래스이름'행 - 열에 mse 값이 들어가게 하기.

model.save('./models/Vix_{}_model.h5'.format(colname))  # 모델 저장하기
print('Vix', colname, ' 모델링및 저장 까지 완료 ')

# 한 클래스당 4개의 컬럼에 대한 mse(val_loss)의 추이에 대한 그래프를 저장.
plt.savefig('./datasets/Vix_mse_plot.png')
plt.show(block=False)
plt.close()

df_loss = df_loss.T  # 행-열 전환 transpose.
df_loss.to_csv('./datasets/Vix_mse.csv', index=True)


