import numpy as np
import logging
from sklearn.metrics import (mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2, 
                             mean_absolute_percentage_error as mape, median_absolute_error as mdae)

import statsmodels.api as sm
from dateutil.relativedelta import relativedelta


rmse = lambda y_true, y_pred: np.sqrt(mse(y_true, y_pred))


metrics_func = {'mae': mae, 'mse': mse, "rmse": rmse, "r2": r2, "mape":mape, 'mdae': mdae}


def del_el_from_list(lst:list, el):
    """
    Функция, которая удаляет заданный элемент из списка по значению и сразу же возвращает список
    """
    lst.remove(el)
    return lst


def define_seasonal_len(time_series):
    """
    Вычисляет сезонность временного ряда
    На вход подавать уже обработанный ряд
    В некоторых случаях возможна некорректная работа 
    """
    seasonal = list(sm.tsa.seasonal_decompose(time_series).seasonal)
    season_len, flag = len(set(seasonal)), True
    while flag and season_len<len(seasonal):
        flag = False
        for j in range(season_len):
            if seasonal[j]!=seasonal[j+season_len]:
                flag = True
                season_len+=1
    return season_len


def stop_logging():
    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)