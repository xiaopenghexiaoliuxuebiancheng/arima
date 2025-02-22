import pandas as pd
from myArima import *  # 假设你有 ARIMA 模型类
import matplotlib.pyplot as plt

# 1. 读取 CSV 数据
filename = "data_first701.csv"
df = pd.read_csv(filename)

# 2. 确保 DateTime 列为数值（天数表示）
df['DateTime'] = pd.to_numeric(df['DateTime'])

# 3. 将 Global_active_power 转换为数值
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])

# 4. 绘制原始时间序列
plt.figure(figsize=(10, 6))
plt.plot(df['DateTime'], df['Global_active_power'], color='g', label='Global_active_power')
plt.xlabel('Days')
plt.ylabel('Power Consumption')
plt.title('Daily Global Active Power')
plt.legend()
plt.show()

# 5. ARIMA 参数设置
arima_para = {'p': range(2), 'd': range(2), 'q': range(2)}
seasonal_para = 7  # 数据是以天为单位，季节性周期设为 1

# 6. 训练 ARIMA 模型
arima = Arima_Class(arima_para, seasonal_para)
arima.fit(df['Global_active_power'])

# 7. 一步预测
plot_start = 1  # 从第 1 天开始绘图
pred_start = 15  # 第 15 天开始预测
dynamic = False
arima.pred(df['Global_active_power'], plot_start, pred_start, dynamic, 'Global_active_power')

# 8. 动态预测
dynamic = True
arima.pred(df['Global_active_power'], plot_start, pred_start, dynamic, 'Global_active_power')

# 9. 未来 30 天的预测
n_steps = 30
arima.forcast(df['Global_active_power'], n_steps, 'Global_active_power')
