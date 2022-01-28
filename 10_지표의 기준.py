import pandas as pd
import pickle

df = pd.read_csv('./Dow Jones Industrial Average_2007-01-03-2022-01-25.csv')
print(df) # 전체 3793개

df_low = df[['Low']]
diff = df_low.iloc[1] - df_low.iloc[0]
diff = abs(diff) # 절대값
print(diff)

result = 0
for i in range(3792): # 0~3791
  diff = (df_low.iloc[i+1] - df_low.iloc[i]) /  df_low.iloc[i] * 100 # 격차의 퍼센티지의 절대값을 전부 다해서 전체갯수만큼 나누어 퍼센티지의 평균값을 구함
  diff = abs(diff)
  result = result + diff
print('격차 절대값의 합 : ' , result)
print('격차 절대값의 평균 : ' , (result/len(df_low)) )

# 절대값의 평균의 절반 + 방향성만 맞춰도 정답이라 치부?? -> 0.702938 / 2 =

result = 0
df_high = df[['High']]
for i in range(3792): # 0~3791
  diff = (df_high.iloc[i+1] - df_high.iloc[i]) / df_high.iloc[i] * 100
  diff = abs(diff)
  result = result + diff
print('격차 절대값의 합 : ' , result)
print('격차 절대값의 평균 : ' , (result/len(df_low)) )

# low
yesterday = 4304.8
today = 4309.5  # 실제값
predicted_low_today = 4315  # 오늘 예측치
diff = (today - yesterday) / yesterday
print(diff)
diff_pct = diff * 100
print(diff_pct)  # 0.10918044973052914 실제값
# 만약 예측값이 똑같이 양수이고 그 차의 절대값(abs)이 0.351469이하이면, 정답으로 치부???

predicted_diff = (predicted_low_today - yesterday) / yesterday
print(predicted_diff)
diff_pct = predicted_diff * 100
print(diff_pct)