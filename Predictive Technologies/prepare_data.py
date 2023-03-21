import pandas as pd
import numpy as np


def reindex(df, date_start, date_end, field):
    """
        Изменение индекса в датасете df на полный от date_start до date_end. 
        Отсутствующие значения заполняются np.nan.
    """
    if field not in df.index.names: df = df.set_index(field)
    return df.reindex(pd.date_range(start=date_start, end=date_end)).fillna(np.nan).rename_axis(field)  #.reset_index()


def prepare_data(file:str):
    """
    Считываение файла и подготовка базовых констант для работы.
    """
    
    data = pd.read_csv(file)

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


def split_df(df:pd.DataFrame, column_date:str, date_split):
    """
    Делим датасет df на обучающую и тестовую выборки по столбцу column_date и дате date_split.
    """
    Y_train_df = df[df[column_date]<=date_split]
    Y_test_df = df[df[column_date]>date_split]
    return Y_train_df, Y_test_df