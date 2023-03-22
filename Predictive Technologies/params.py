import pandas as pd
import datetime as dt
from itertools import product
from dateutil.relativedelta import relativedelta



"""
По хорошему вообще не привязываться к shop и item и использовать абстрактные параметры.
Но что-то сразу не придумала, какие конкретно они должны быть, и в рамках данной конкретной задачи shop и item достаточно.
При наличии готового кода и хорошей IDE переименовать поля в нужные имена большого труда не составит (если делаешь для самого себя, конечно).
"""


def prepare_data(file_source_name:str):
    """
    Считываение файла и подготовка базовых констант для работы.
    """
    
    data = pd.read_csv(file_source_name)

    data.Date = pd.to_datetime(data.Date)
    date_start = min(data.Date)
    date_end = max(data.Date)

    shops = data.Shop.unique()
    items = data.Item.unique()
    
    data_info = {"date_start": date_start, 
                 "date_end": date_end, 
                 "shops": shops, 
                 "items": items}
    
    return data, data_info


def params_for_dataset(file_source_name:str, column_predict:str="Rolling_Number"):
    
    data, params = prepare_data(file_source_name)

    date_start, date_end, shops, items = params["date_start"], params["date_end"], params["shops"], params["items"]

    data_item_shop = data.groupby(["Item", "Shop", "Date"]).mean().reset_index(["Date"])[["Date", column_predict]] #.rename(columns={"Date": "ds", column_predict: "Y"})

    item_shop_list = data_item_shop.index.unique()
    
    date_split = date_end - relativedelta(months=1)
    
    params = {"data":data, 
              "column_predict":column_predict, 
              "date_start":date_start, 
              "date_end":date_end, 
              "shops":shops, 
              "items":items,
              "data_item_shop":data_item_shop, 
              "item_shop_list":item_shop_list, 
              "date_split":date_split}
    
    return params


def params_for_generate():
    """
        Параметры для генерации данных
    """
    
    file_target_name = 'data.csv'
    file_target_name_assort = 'data_assort.csv'

    random_seed = 71

    shops_count = 10
    items_count = 50
    date_start = dt.date(2020, 1, 1)
    date_end = date_start + relativedelta(years=3) - relativedelta(days=1)
    
    data_columns = ['Shop', 'Item', 'Date', 'Number']
    
    params = {"file_target_name": file_target_name, 
              "file_target_name_assort": file_target_name_assort,
              "shops_count": shops_count, 
              "items_count": items_count, 
              "date_start": date_start,
              "date_end": date_end, 
              "random_seed": random_seed,
              "columns": data_columns}
    
    return params


def params_for_filling():
    """
    Параметры для заполнения пропусков
    """
    name_source_file = "data.csv"
    name_target_file = "data_fill.csv"
    source_column = "Number"
    target_column = "Rolling_Number"
    params = {"name_source_file": name_source_file, 
              "name_target_file": name_target_file,
              "source_column": source_column,
              "target_column": target_column}
    return params


def params_for_predict():

    column_date = "ds"

    period_future, freq = 28, "D"

    season_length = 7
    lags = 32
        
    level = 0.95
    
    """ 
    Дает возможность использовать взвешенные метрики
    Для метрик, где важно больше значение, вроде R2, вес указывать отрицательный
    Returns:
        dict: {метрика: вес}
    """
    metrics = {"mae": 2, "mse": 3, "rmse": 3, "r2": 0, "mdae": 2, "mape": 2} 
    
    predict_columns = ["Predict", "Min", "Max"]

    params = {"column_date": column_date, 
              "period_future": period_future, 
              "freq": freq,
              "season_length": season_length, 
              "lags": lags, 
              "level": level, 
              "metrics": metrics,
              "predict_columns": predict_columns}
    
    return params


def params_for_statsforecast():
    """
    Параметры для statsforecast
    """
    
    file_source_name = 'data_fill.csv'
    file_target_name = 'data_predict_statsforecast.csv'

    model_source_file = 'statsforecast.pkl'
    model_target_file = 'statsforecast.pkl'

    time_for_testing = 1000
    
    params = {"file_source_name": file_source_name, 
              "file_target_name": file_target_name, 
              "model_source_file": model_source_file, 
              "model_target_file": model_target_file, 
              "time_for_testing": time_for_testing}
    
    return params


def params_for_prophet():
    """
    Параметры для prophet
    """
    
    file_source_name = 'data_fill.csv'
    file_target_name = 'data_predict_prophet.csv'

    column_predict = "Number" # "Rolling_Number"
    
    params = {"file_source_name": file_source_name, 
              "file_target_name": file_target_name, 
              "column_predict": column_predict}
    
    return params


def params_for_sarima():
    """
    Параметры для sarima
    """
    
    file_source_name = 'data_fill.csv'
    file_target_name = 'data_predict_sarima.csv'

    ps, qs, Ps, Qs, d, D = range(0, 5), range(0, 5), range(0, 3), range(0, 3), 1, 1
    parameters_list = list(product(ps, qs, Ps, Qs))
    sarima_params = {'parameters': [ps, qs, Ps, Qs, d, D],
                     'parameters_list': parameters_list}

    
    params = {"file_source_name": file_source_name, 
              "file_target_name": file_target_name,
              "sarima_params": sarima_params}
    
    return params


def params_for_predictions():
    """
    Параметры для общих предсказаний
    """
    
    file_source_names = ['data_predict_statsforecast.csv', 
                         'data_predict_prophet.csv', 
                         'data_predict_sarima.csv'] 
    file_target_name = 'data_predictions.csv'

    predict_columns = ["Predict", "Min", "Max"]
    
    save_all_predictions = True
    
    params = {"file_source_names": file_source_names,
              "file_target_name": file_target_name, 
              "predict_columns": predict_columns,
              "save_all_predictions": save_all_predictions}
    
    return params


def params_for_accortment():
    """
    Параметры для учета ассортимента
    """
    
    file_source_name = 'data_predictions.csv'
    file_assort_name = 'data_assort.csv'
    file_target_name = 'data_predict_assort.csv'

    predict_columns = ["Predict", "Min", "Max"]
    
    params = {"file_source_name": file_source_name, 
              "file_assort_name": file_assort_name, 
              "file_target_name": file_target_name, 
              "column_predict": predict_columns}
    
    return params