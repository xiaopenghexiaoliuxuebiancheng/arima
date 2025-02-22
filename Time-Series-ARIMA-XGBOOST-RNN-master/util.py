import pandas as pd
import numpy as np
from matplotlib import dates
import matplotlib.pyplot as plt

def preprocess(N_rows, filename):
    # 1) 计算文件总行数（如果你仍想只读取最后 N_rows 行）
    total_rows = sum(1 for _ in open(filename))

    # 2) 读取前几行获取列名（假设用逗号分隔）
    variable_names = pd.read_csv(
        filename, header=0, sep=',', nrows=5
    )

    # 3) 读取最后 N_rows 行数据，并将 parse_dates 指定的列解析为日期
    #    注意：如果你的 Datetime 在第一列，且想把它当作索引，就用 index_col=0
    df = pd.read_csv(
        filename,
        header=0,
        sep=',',
        names=variable_names.columns,
        parse_dates=[0],  # 让 Pandas 自动识别这一列为日期时间
        index_col=0,            # 若 Datetime 在第 0 列，则此处设为 0
        nrows=N_rows,
        skiprows=total_rows - N_rows
    )

    # 4) 如果没有 “?” 这样的缺失标记，可以省略 replace；若有，可以保留
    df_no_na = df.replace('?', np.nan)
    df_no_na.dropna(inplace=True)

    # 5) 尝试将所有列转为浮点数（若有无法转换的列需要额外处理）
    return df_no_na.astype(float, errors='ignore')


def timeseries_plot(y, color, y_label):
    # y 是以 datetime 为索引的 Series
    days = dates.DayLocator()
    dfmt_minor = dates.DateFormatter('%m-%d')
    weekday = dates.WeekdayLocator(byweekday=(), interval=1)

    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dfmt_minor)
    ax.xaxis.set_major_locator(weekday)
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%a'))

    ax.set_ylabel(y_label)
    ax.plot(y.index, y, color)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(y_label + '.png', dpi=300)
    plt.show()

def bucket_avg(ts, bucket):
    # 对时间序列 ts 进行按 bucket（例如 "30T"）平均
    y = ts.resample(bucket).mean()
    return y

def config_plot():
    plt.style.use('seaborn-v0_8-paper')  # 使用环境中有效的样式
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'figure.figsize': (10, 6)})
    plt.rcParams.update({'legend.fontsize': 20})
    return 1

# 以下为数据预处理辅助函数，可根据需要进行调整

def date_transform(df, encode_cols):
    # 从 datetime 索引中提取特征
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    # weekofyear 已弃用，可使用 isocalendar().week
    df['WeekofYear'] = df.index.isocalendar().week
    df['DayofWeek'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    # 对指定的分类变量进行 one-hot 编码
    for col in encode_cols:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=encode_cols)
    return df

def get_unseen_data(unseen_start, steps, encode_cols, bucket_size):
    index = pd.date_range(unseen_start, periods=steps, freq=bucket_size)
    df = pd.DataFrame(pd.Series(np.zeros(steps), index=index),
                      columns=['Global_active_power'])
    return df

def data_add_timesteps(data, column, lag):
    column_series = data[column]
    step_columns = [column_series.shift(i) for i in range(2, lag + 1, 2)]
    df_steps = pd.concat(step_columns, axis=1)
    # 将原数据与 lag 特征拼接
    df = pd.concat([data, df_steps], axis=1)
    return df
