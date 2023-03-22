from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

from scipy import stats
from tqdm import tqdm

from dateutil.relativedelta import relativedelta

from functions_common import define_seasonal_len
from prepare_data import reindex
from data_generate import generate_period
from params import params_for_predict, params_for_dataset, params_for_sarima as params

import constants as cnst


warnings.filterwarnings('ignore')


invboxcox = lambda y, lmbda: np.exp(y) if lmbda == 0 else np.exp(np.log(lmbda*y+1)/lmbda)

Dickey_Fuller_test = lambda series, init: sm.tsa.stattools.adfuller(series[init:]) #[1]
Student_test = lambda series, init: stats.ttest_1samp(series[init:], 0) #[1]


def trend_exist(time_series):
    time_series_rolling = time_series.rolling(window=len(time_series)//100,min_periods=1,center=True,win_type="nuttall").mean()
    return time_series_rolling.mean()/1000 < max(time_series_rolling.values) - min(time_series_rolling.values)


def use_boxcox(data_analysis, column_predict):
    flag_boxcox, lmbda = False, 1
    if trend_exist(data_analysis[column_predict]): 
        try: 
            data_analysis['data_box'], lmbda = stats.boxcox(data_analysis[column_predict])
            flag_boxcox = True
        except Exception as e: pass # По хорошему надо разбираться с ошибкой, здесь опустим
        if lmbda is None: lmbda = 1
    analysis_column = 'data_box' if flag_boxcox else column_predict
    return data_analysis, lmbda, flag_boxcox, analysis_column


def series_differentiation(data:pd.DataFrame):
    data['data_box_diff_0'] = data.data_box
    i, init = 0, 0
    while Dickey_Fuller_test(data[f'data_box_diff_{i}'], init)[1]>10E-7:
        diff = define_seasonal_len(data[f'data_box_diff_{i}'][init:])
        init += diff
        i+=1
        data[f'data_box_diff_{i}'] = data[f'data_box_diff_{i-1}'] - data[f'data_box_diff_{i-1}'].shift(diff)
    return data, init


def trend_for_sarima(time_series):
    if trend_exist(time_series): 
        trend = 'tc' if abs(time_series.mean())>0.01 else 't'
    else: trend = 'n'
    return trend


def study_sarima_model(time_series, param:list, S:int, d=1, D=1):
    return sm.tsa.statespace.SARIMAX(time_series, order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], S)).fit(disp=-1)


def define_best_model(time_series, parameters_list:list, S:int):
    results = list()
    best_aic = float("inf")
    print("Выбор наиболее оптимальной модели:")
    for param in tqdm(parameters_list):
        try: model = study_sarima_model(time_series, param, S)
        except Exception as e: continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    return best_model, best_param, results



def sarima_predict_for_data(file_source_name:str, file_target_name:str, sarima_params:dict):

    params = params_for_dataset(file_source_name) | params_for_predict()

    data, column_predict, date_start, date_end, shops, items, data_item_shop, item_shop_list, date_split, \
        column_date, period_future, freq, season_length, lags, level, metrics, predict_columns = params.values()

    data.Date = pd.to_datetime(data.Date)
    data.asfreq(freq)

    forecast_start_date = date_end + relativedelta(days=1)
    date_list = generate_period(forecast_start_date, forecast_start_date + relativedelta(days=period_future-1)) #[datetime.datetime.strptime(forecast_start_date, "%Y-%m-%d") + relativedelta(days=x) for x in range(0, count)]

    data_for_predict = data.groupby("Date").agg("mean").reset_index(["Date"])

    data_for_predict, lmbda, flag_boxcox, analysis_column = use_boxcox(data_for_predict, column_predict)
    data_for_predict = reindex(data_for_predict, date_start, date_end, "Date")

    S = define_seasonal_len(data_for_predict[analysis_column])

    parameters, parameters_list = sarima_params['parameters'], sarima_params['parameters_list']
    
    ps, qs, Ps, Qs, d, D = parameters

    best_model, best_param, results = define_best_model(data_for_predict[analysis_column], parameters_list, S)

    # Тестируем всем скопом, поскольку времени тестировать индивидуально нет; по хорошему хотя бы разные модели по разным продуктам
    print("Прогнозируем на основе модели ARIMA:")
    start = None
    item_shop_indx = data.groupby(["Item", "Shop"]).agg('mean').index

    if start is None: start, end = item_shop_indx.shape[0], item_shop_indx.shape[0] + period_future - 1

    data_predict = pd.DataFrame()
    for (item, shop) in tqdm(item_shop_indx):
        
        data_for_predict = data_item_shop.loc[(item, shop)]

        data_for_predict, lmbda, flag_boxcox, analysis_column = use_boxcox(data_for_predict, column_predict)
        data_for_predict = reindex(data_for_predict, date_start, date_end, "Date")
        try: model = study_sarima_model(data_for_predict[analysis_column], best_param, S)
        except Exception as e:  print(f"Error sarima_predict: {(item, shop)}", e, sep="\n")

        pred_uc = model.get_forecast(steps=period_future) 
        forecast = pd.DataFrame(invboxcox(pred_uc.predicted_mean, lmbda)) if flag_boxcox else pd.DataFrame(pred_uc.predicted_mean)
        pred_ci = pred_uc.conf_int(alpha=1-level)
        future_ci = invboxcox(pred_ci, lmbda) if flag_boxcox else pred_ci
        forecast = pd.concat([forecast, future_ci], axis=1)
        forecast.columns = predict_columns
        for col in forecast.columns: forecast[col] = forecast[col].apply(lambda x: max(0, x))
        forecast = forecast.astype(int)
        forecast["Date"] = date_list
        forecast["Shop"] = len(forecast)*[shop]
        forecast["Item"] = len(forecast)*[item]
        data_predict = pd.concat([data_predict, forecast], axis=0)
    
    data_predict = data_predict[cnst.column_names + predict_columns]
    data_predict.sort_values(["Shop", "Item"], inplace=True)
    
    if file_target_name is not None: data_predict.to_csv(file_target_name, index=False)

    return data_predict



file_source_name, file_target_name, sarima_params = params().values()

# sarima_predict_for_data(file_source_name, file_target_name, sarima_params)