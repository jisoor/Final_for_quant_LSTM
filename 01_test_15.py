import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('./datasets/Crude Oil WTI Futures Historical Data.csv')
print(raw_data.head())
print(raw_data.tail())
print(raw_data.info())
print(raw_data.isnull().sum())
df = raw_data.isnull() #False = 데이터가 있음. True = 데이터가 없음
print(df)

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data = raw_data.sort_values('Date')
print(raw_data.head())

raw_data.to_csv('./datasets/Clear_Crude_Oil_Data_15.csv', index=False)