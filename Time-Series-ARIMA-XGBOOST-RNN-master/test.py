import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. 读取 CSV 数据
df = pd.read_csv("data_first701.csv")

# 2. 查看前几行数据，确认读取是否正确
print(df.head())

# 3. 将 DateTime 设置为索引
#    由于 DateTime 是整数（第几天），我们直接使用整数索引即可
df.set_index("DateTime", inplace=True)

# 4. 绘制原始时间序列图
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Global_active_power"], color='g', marker='o', linestyle='-')
plt.xlabel("Day")  # x轴为天数
plt.ylabel("Global_active_power")
plt.title("Daily Global Active Power")
plt.grid(True)
plt.tight_layout()
plt.savefig("daily_power_plot.png", dpi=300)
plt.show()

# 5. 时间序列分解（可选）
#    如果你认为数据存在周期性（例如 7 天），可做季节性分解
#    注意：period=7 代表一周的周期，仅在数据确有周周期的前提下使用
decomposition = sm.tsa.seasonal_decompose(df["Global_active_power"], model='additive', period=7)
fig = decomposition.plot()
plt.tight_layout()
plt.savefig("seasonal_decompose.png", dpi=300)
plt.show()

# 6. 其他可视化或分析（根据需求）
#    - 由于只有一个功率列，无法做相关性热力图
#    - 如果需要预测，可使用 ARIMA / SARIMA 模型
