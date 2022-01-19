import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

X_train, X_test, Y_train, Y_test = np.load('./models/crude_oil_scaled_data.npy', allow_pickle=True)

# h5 파일 여는 법 (load_model)
model = load_model('./models/crude_oil_model_1.h5')
predict = model.predict(X_test) # 20프로의 테스트 파일 X값 예측 시키기(709개)
plt.figure(figsize=(20, 10))
plt.plot(Y_test[:100], label='actual')
plt.plot(predict[:100], label='predict')
plt.legend()
plt.show()