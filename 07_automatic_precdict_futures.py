import investpy
import pandas as pd
import FinanceDataReader as fdr   # pip install -U finance-datareader
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'nanumyeongjo'
plt.rcParams['figure.figsize'] = (14,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
# print(fdr.__version__) # 0.9.31
# https://notebook.community/arcyfelix/Courses/17-09-17-Python-for-Financial-Analysis-and-Algorithmic-Trading/11-Advanced-Quantopian-Topics/.ipynb_checkpoints/05-Futures-checkpoint

# 선물 데이터 갖고오기 