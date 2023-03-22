import pandas as pd
from tqdm import tqdm
import warnings

import os.path

from params import params_for_accortment
from functions_common import del_el_from_list
import constants as cnst


warnings.filterwarnings('ignore')


def assortment_for_predict(data:pd.DataFrame=None):
    """
    Обработка полученных предсказаний имеющимся ассортиментом.
    """

    file_source_name, file_assort_name, file_target_name, predict_columns = params_for_accortment().values()
    if data is None: 
        if os.path.isfile(file_source_name): data = pd.read_csv(file_source_name)
        else: 
            print(f"Error assortment_for_predict: Остутствует файл {file_source_name} с предсказаниями!")
            return None
    if os.path.isfile(file_assort_name): data_assortment = pd.read_csv(file_assort_name)
    else: 
        print(f"Error assortment_for_predict: Остутствует файл {file_assort_name} с планируемым ассортиментом!")
        return None
    data_shop_item = data.groupby(cnst.column_names).mean().reset_index(["Date"])[["Date"] + predict_columns].sort_index()
    data_assortment = data_assortment.set_index(del_el_from_list(cnst.column_names,"Date"))

    data_assortment_predict = pd.DataFrame()
    data_shop_item_indx = data_shop_item.index.unique()
    data_assortment_indx = data_assortment.index.unique()
    for index in data_assortment_indx & data_shop_item_indx:
        data_predict_shop_item = data_shop_item.loc[index]
        data_assortment_shop_item = data_assortment.loc[index]
        query =  " or ".join(f"('{str(d1)}' <= Date <= '{str(d2)}')" for d1, d2 in zip(data_assortment_shop_item.Date_1, data_assortment_shop_item.Date_2))
        data_query = data_predict_shop_item.query(query).reset_index()
        data_assortment_predict = pd.concat([data_assortment_predict, data_query], axis=0)
    
    data_assortment_predict = data_assortment_predict.astype({value: int for value in predict_columns})
    data_assortment_predict.to_csv(file_target_name, index=False)
    print(data_assortment_predict)
    return data_assortment_predict


if __name__ == '__main__':
    assortment_for_predict()