import pandas as pd
from prophet import Prophet
from tqdm import tqdm
import warnings

from prepare_data import reindex
from functions_common import metrics_func, stop_logging
from params import params_for_predict, params_for_dataset, params_for_prophet as params
import constants as cnst


warnings.filterwarnings('ignore')


def prophet_params():
    """
    Параметры для предиктивной модели prophet
    """

    params = dict()
    params['weekly_seasonality'] = True
    
    return params


def prophet_predict(df:pd.DataFrame, params:dict, period_future:int=28, freq:str="D", level:int=0.95, include_history:bool = False):
    """
    Построение предиктивной модели prophet
    """

    model = Prophet(
        weekly_seasonality=params['weekly_seasonality'],
        interval_width=level,
        )
    model.fit(df)

    future = model.make_future_dataframe(periods=period_future, freq=freq, include_history=include_history)
    forecast = model.predict(future)
    
    return forecast


def prophet_test_model(Y_train_df:pd.DataFrame, Y_test_df:pd.DataFrame, periods, freq, level:int, metrics:list):
    """
    Тестирование модели prophet
    """
    Y_predict_df = prophet_predict(Y_train_df, period_future=periods, freq=freq, level=level, include_history = False)
    metric_test_dict = dict()
    for metric in metrics: metric_test_dict[metric] = metrics_func[metric](Y_test_df, Y_predict_df)
    return metric_test_dict


def prophet_predict_for_data(file_source_name:str, file_target_name:str=None, column_predict:str="Number"):
    """
    Прогнозирование для данных на основании модели prophet
    """

    params = params_for_dataset(file_source_name) | params_for_predict()

    data, column_predict, date_start, date_end, shops, items, data_item_shop, item_shop_list, date_split, \
        column_date, period_future, freq, season_length, lags, level, metrics, predict_columns = params.values()
    
    data_item_shop = data_item_shop.rename(columns={"Date": "ds", column_predict: "y"})
    predict_columns = {"ds": "Date", "yhat": predict_columns[0], 'yhat_lower': predict_columns[1], 'yhat_upper': predict_columns[2]}
    predict_column_types = {value: int for value in list(predict_columns.values())[1:]}
    
    
    stop_logging()

    params = prophet_params()
    
    data_predict = pd.DataFrame()
    print(f"Прогнозируем на основании модели Prophet:")
    item_shop_indx = data.groupby(["Item", "Shop"]).agg('mean').index
    for (item, shop) in tqdm(item_shop_indx):
        data_for_predict = data_item_shop.loc[(item, shop)]
        if column_predict=="Number": 
            data_for_predict = reindex(data_for_predict, date_start, date_end, "ds").reset_index()
        try: forecast_df = prophet_predict(data_for_predict, params, period_future, freq, level=level)
        except Exception as e: print(f"Error prophet_predict: {(item, shop)}", e, sep="\n")
        forecast_df = forecast_df[forecast_df["ds"]>date_end][predict_columns.keys()]
        forecast_df["Shop"] = len(forecast_df)*[shop]
        forecast_df["Item"] = len(forecast_df)*[item]
        data_predict = pd.concat([data_predict, forecast_df], axis=0)
    
    for col in list(predict_columns)[1:]: data_predict[col] = data_predict[col].apply(lambda x: max(0, x))
    
    data_predict.rename(columns=predict_columns, inplace=True)
    data_predict = data_predict[cnst.column_names+list(predict_columns.values())[1:]].astype(predict_column_types)
    data_predict.sort_values(["Shop", "Item"], inplace=True)

    if file_target_name is not None: data_predict.to_csv(file_target_name, index=False)
    
    return data_predict


file_source_name, file_target_name, column_predict = params().values()

if __name__ == '__main__':
    prophet_predict_for_data(file_source_name, file_target_name, column_predict)