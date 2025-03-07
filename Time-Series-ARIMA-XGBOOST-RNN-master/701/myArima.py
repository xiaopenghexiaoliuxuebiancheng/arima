import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Arima_Class:
    def __init__(self, arima_para, seasonal_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]

    def fit(self, ts):
        warnings.filterwarnings("ignore")
        results_list = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    print('ARIMA{}x{}seasonal - AIC:{}'.format(param,
                                                               param_seasonal, results.aic))
                    results_list.append([param, param_seasonal, results.aic])
                except Exception as e:
                    print(f"Failed to fit ARIMA{param}x{param_seasonal} seasonal. Error: {e}")
                    continue
        if len(results_list) == 0:
            print("Warning: No ARIMA models were successfully fitted. Check your data or parameters.")
            return

        # 提取所有 AIC 值，并找到最小的索引
        aic_values = [item[2] for item in results_list]
        lowest_AIC = np.argmin(aic_values)

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('ARIMA{}x{}seasonal with lowest_AIC:{}'.format(
            results_list[lowest_AIC][0],
            results_list[lowest_AIC][1],
            results_list[lowest_AIC][2]))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        mod = sm.tsa.statespace.SARIMAX(ts,
                                        order=results_list[lowest_AIC][0],
                                        seasonal_order=results_list[lowest_AIC][1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.final_result = mod.fit()
        print('Final model summary:')
        print(self.final_result.summary().tables[1])
        print('Final model diagnostics:')
        self.final_result.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        plt.savefig('model_diagnostics.png', dpi=300)
        plt.show()

    def pred(self, ts, plot_start, pred_start, dynamic, ts_label):
        # 判断索引是否为 DatetimeIndex
        if isinstance(ts.index, pd.DatetimeIndex):
            start = pd.to_datetime(pred_start)
            plot_start_index = pd.to_datetime(plot_start)
        else:
            start = pred_start  # 直接使用整数索引
            plot_start_index = plot_start

        # 获取预测结果
        pred_dynamic = self.final_result.get_prediction(
            start=start, dynamic=dynamic, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()

        # 绘制观测值图形，注意如果索引为整数直接切片即可
        ax = ts[plot_start_index:].plot(label='observed', figsize=(15, 10))

        # 绘制预测值
        if dynamic == False:
            pred_dynamic.predicted_mean.plot(label='One-step ahead Forecast', ax=ax)
        else:
            pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

        # 填充置信区间
        ax.fill_between(pred_dynamic_ci.index,
                        pred_dynamic_ci.iloc[:, 0],
                        pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        ax.fill_betweenx(ax.get_ylim(), plot_start_index, ts.index[-1],
                         alpha=.1, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.legend()
        plt.tight_layout()
        if dynamic == False:
            plt.savefig(ts_label + '_one_step_pred.png', dpi=300)
        else:
            plt.savefig(ts_label + '_dynamic_pred.png', dpi=300)
        plt.show()

    def forcast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        ax = ts.plot(label='observed', figsize=(15, 10))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast in Future')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.tight_layout()
        plt.savefig(ts_label + '_forcast.png', dpi=300)
        plt.legend()
        plt.show()
