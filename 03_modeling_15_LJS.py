import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = np.load('./models/crude_oil_scaled_data_15.npy', allow_pickle=True)

model = Sequential()
model.add(LSTM(128, input_shape=(15, 1), activation='tanh', return_sequences=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()

fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
model.save('./models/crude_oil_model_15.h5')
plt.plot(fit_hist.history['loss'][5:], label='loss')
plt.plot(fit_hist.history['val_loss'][5:], label='val_loss')
plt.legend()
plt.show()
