import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from prepare_data import reindex

from params import prepare_data, params_for_filling
from functions_common import rmse
import constants as cnst



warnings.filterwarnings('ignore')


calculate_target = lambda df, target_col, source_col, win_size, win_type: round(df[source_col] * rolling_relative(df, target_col, source_col, win_size, win_type=win_type))


def rolling_relative(df:pd.DataFrame, target_col:str, source_col:str, period:int, win_type):
    """
    Считает отношение соответствующих скользящих окон
    """
    return df[target_col].rolling(period,min_periods=1,center=True,win_type=win_type).mean()/\
        df[source_col].rolling(period,min_periods=1,center=True,win_type=win_type).mean()


def define_window_params(df:pd.DataFrame, target_col:str, source_col:str, best_counts:int=5):
    """
    Тестирует различные параметры и возвращает список с результатами best_counts 'лучших' моделей
    """
    win_sizes = [7, 10, 14, 30, 56]
    win_types = ["boxcar", "triang", "blackman", "hamming", "bartlett", "parzen", "bohman", \
                 "blackmanharris", "nuttall", "barthann"]
    metrics = {"mse": mse, "rmse": rmse}
    indx = df[~df.Number.isna()].index
    df_target_col_index = df[target_col].loc[indx]
    param_list = {key: list() for key in metrics.keys()}
    for win_size in win_sizes:
        for win_type in win_types:
            df[f"calculate"] = calculate_target(df, target_col, source_col, win_size, win_type)
            for name_crit, crit in metrics.items():
                param_list[name_crit].append((win_size, win_type, crit(df_target_col_index, df["calculate"].loc[indx])))
    for key in param_list.keys(): param_list[key] = sorted(param_list[key], key=lambda x: x[-1])[:best_counts]
    return param_list


def shop_item_filling_gaps(data_item_data:pd.DataFrame, data_item_shop_data:pd.DataFrame, data_info:dict, params:dict, item:str, shop:str) -> pd.DataFrame:
    """ 
    Заполняет пропуски для конкретного (shop, item)
    """
        
    date_start, date_end, shops, items = data_info.values()
    _, _, source_column, target_column = params.values()

    data_item_data = data_item_data.loc[item]

    if (item, shop) not in data_item_shop_data.index: return None
    df = data_item_shop_data.loc[(item, shop)]

    df = reindex(df, date_start, date_end, "Date")

    df[f"data_item_{source_column}"] = data_item_data[source_column]
    
    win_sizes, win_type, i = [7, 14, 30, 90, 180, 365], "nuttall", 0
    df.loc[df[~df[source_column].isna()].index, target_column] = df.loc[df[~df[source_column].isna()].index, source_column]
    while sum(df[target_column].isna())>0 and i<len(win_sizes): 
        df[f"{target_column}_{i}"] = calculate_target(df, target_col=source_column, source_col=f"data_item_{source_column}", win_size=win_sizes[i], win_type=win_type)
        df.loc[df[df[target_column].isna()].index, target_column] = df.loc[df[df[target_column].isna()].index, f"{target_column}_{i}"]
        i+=1
    df.loc[df[df[target_column].isna()].index, target_column] = df[target_column].mean() # Закрывает баг, который нет отлавливать времени, что иногда все равно остаются пустые значения
        
    return pd.concat([df[source_column], df[target_column]], axis=1)


def filling_gaps(params:dict):
    """
    Основная функция заполнения пропусков в данных
    """

    file_name, file_name_target, source_column, target_column = params.values()
    
    data, data_info = prepare_data(file_name)

    date_start, date_end, shops, items = data_info.values()

    data_item_data = data.groupby(["Item", "Date"]).mean()
    data_item_shop_date = data.groupby(["Item", "Shop", "Date"]).mean()

    data_fill, data_item_shop = pd.DataFrame(), pd.DataFrame()
    print("Обработка пропусков в данных")
    for item in tqdm(items):
        for shop in shops:
            data_item_shop = shop_item_filling_gaps(data_item_data, data_item_shop_date, data_info=data_info, params=params, item=item, shop=shop)
            if data_item_shop is not None:
                data_item_shop["Shop"] = len(data_item_shop)*[shop]
                data_item_shop["Item"] = len(data_item_shop)*[item]
                data_fill = pd.concat([data_fill, data_item_shop], axis=0)

    data_fill.Date = pd.to_datetime(data.Date).dt.date
    data_fill.astype({target_column: int}).reset_index()[cnst.column_names + [source_column, target_column]].to_csv(file_name_target, index=False)


# filling_gaps(params_for_filling())