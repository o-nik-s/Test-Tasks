import pandas as pd
import time
import random
import pickle
import warnings

from progress.bar import IncrementalBar
from tqdm.autonotebook import tqdm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoRegressive
from statsforecast.models import ARIMA, ARCH, GARCH, HoltWinters, CrostonClassic, DynamicOptimizedTheta, SeasonalNaive

from prepare_data import split_df
from functions_common import metrics_func, define_seasonal_len
from params import params_for_predict, params_for_dataset, params_for_statsforecast as params
import constants as cnst


warnings.filterwarnings('ignore')


def statsforecast_params(season_length:int, lags:int):

    dict_models = \
        {"AutoARIMA": AutoARIMA(season_length=season_length),
        "CES": AutoCES(season_length=season_length),
        "AutoETS": AutoETS(season_length=season_length),
        "AutoRegressive": AutoRegressive(lags=lags),
        "ARIMA": ARIMA(season_length=season_length),
        "ARCH": ARCH(),
        "GARCH": GARCH(), 
        "HoltWinters": HoltWinters(season_length=season_length), 
        "CrostonClassic": CrostonClassic(), 
        "DynamicOptimizedTheta": DynamicOptimizedTheta(season_length=season_length),
        "SeasonalNaive": SeasonalNaive(season_length=season_length)}

    params = {"dict_models": dict_models}
    
    return params


def statsforecast_predict(Y_train_df, models, freq, horizon, level=None):
    sf = StatsForecast(
        df=Y_train_df,
        models=models,
        freq=freq,
        n_jobs=-1
    )
    level = [level] if level is not None else None
    Y_predict_df = sf.forecast(horizon, level=level)
    return Y_predict_df


def statsforecast_testing(df, column_date, date_split, models, freq, level, metrics):
    Y_train_df, Y_test_df = split_df(df, column_date, date_split)
    horizon = len(Y_test_df)
    Y_predict_df = statsforecast_predict(Y_train_df, models=models, freq=freq, horizon=horizon) 
    result_dict = {metric: list() for metric in metrics}
    for column in Y_predict_df.columns:
        if column == column_date: continue
        for metric in metrics.items():
            result_dict[metric[0]].append((column, metrics_func[metric[0]](Y_test_df.Y, Y_predict_df[column])))
    for value in result_dict.values(): value.sort(key=lambda x: x[1])
    return result_dict


def statsforecast_test_models(params:dict, time_for_testing:int, best_count:int = 5):
    
    data_item_shop, item_shop_list, column_predict, column_date, date_split, models, freq, metrics, level = \
        params["data_item_shop"], params["item_shop_list"], params["column_predict"], params["column_date"], params["date_split"], params["dict_models"].values(), \
        params["freq"], params["metrics"], params["level"]
    data_item_shop["unique_id"] = 1
    data_item_shop = data_item_shop.rename(columns={"Date": "ds", column_predict: "Y"})

    time_start = time.time()
    ln = len(item_shop_list)-1
    dct_result = dict()
    bar = IncrementalBar(max = time_for_testing)
    print("Statsforecast: идет тестирование моделей")
    curr_time = 0
    while time.time() - time_start < time_for_testing:
        item_shop = item_shop_list[random.randint(0, ln-1)]
        df = data_item_shop.loc[item_shop]
        try:
            metrics_forecast_dict = statsforecast_testing(df, column_date, date_split, models, freq, level, metrics)
            for metric, metrics_forecast_list in metrics_forecast_dict.items():
                for i, model in enumerate(metrics_forecast_list[:best_count][::-1], 1):
                    dct_result[model[0]] = dct_result.get(model[0], 0) + i * metrics[metric]
        except Exception as e: print(f"Eroror statsforecast_testing: {item_shop}", e, sep="\n")
        for _ in range(int(time.time() - time_start - curr_time)):
            bar.next()
            curr_time += 1
    bar.finish()
    print("Результаты тестирования моделей: ", dct_result, sep=" ")
    return dct_result


def statsforecast_predict_for_data(file_source_name:str, time_for_testing:int=300, model_source_file:str=None, model_target_file:str=None, file_target_name:str=None):
    
    params = params_for_dataset(file_source_name) | params_for_predict()

    data, column_predict, date_start, date_end, shops, items, data_item_shop, item_shop_list, date_split, \
        column_date, period_future, freq, season_length, lags, level, metrics, predict_columns = params.values()
    level *= 100

    data_item_shop = data_item_shop.rename(columns={"Date": "ds", column_predict: "Y"})

    param_statsforecast = statsforecast_params(season_length, lags)
    dict_models = param_statsforecast["dict_models"]

    if model_source_file is not None:
        with open(model_source_file, 'rb') as pkl: open_model = pickle.load(pkl)
        best_model_name, best_model = str(open_model), open_model
    else:
        '''
        За отсутствием времени тестируем сразу на всех продуктах и всех магазинах скопом. 
        Ясно, что по хорошему в разных ситуациях следует использовать разные модели.
        '''
        dct_result = statsforecast_test_models(params | param_statsforecast, time_for_testing=time_for_testing)
        best_model_name = sorted(dct_result.items(), key=lambda x: x[1], reverse=True)[0][0]
        best_model = dict_models[best_model_name.split('-')[0]]
        if model_target_file is not None: 
            with open(model_target_file, 'wb') as pkl: pickle.dump(best_model, pkl)

    if "unique_id" not in data_item_shop.columns: data_item_shop["unique_id"] = 1

    predict_columns = {"ds": "Date", best_model_name: predict_columns[0], f'{best_model_name}-lo-{level}': predict_columns[1], f'{best_model_name}-hi-{level}': predict_columns[2]}
    predict_column_types = {value: int for value in list(predict_columns.values())[1:]}

    season_in_model = hasattr(best_model, "season_length")
    
    data_predict = pd.DataFrame()
    print(f"Прогнозируем на основании выбранной модели: {best_model_name}")
    item_shop_indx = data.groupby(["Item", "Shop"]).agg('mean').index
    for (item, shop) in tqdm(item_shop_indx):
        data_for_predict = data_item_shop.loc[(item, shop)]
        if season_in_model:
            season = define_seasonal_len(data_for_predict.set_index("ds").Y)
            best_model = statsforecast_params(season_length=season, lags=lags)["dict_models"][best_model_name]
        forecast_df = statsforecast_predict(data_for_predict, [best_model], freq, horizon=period_future, level=level)
        forecast_df["Shop"] = len(forecast_df)*[shop]
        forecast_df["Item"] = len(forecast_df)*[item]
        data_predict = pd.concat([data_predict, forecast_df], axis=0)
    data_predict.reset_index(inplace=True)
    if 'unique_id' in data_predict.columns: data_predict.drop('unique_id', axis=1, inplace=True)
    
    data_predict.rename(columns=predict_columns, inplace=True)
    slicer = slice(1, None)
    if len(data_predict.columns)>4: 
        for col in list(predict_columns.values())[1:]: data_predict[col] = data_predict[col].apply(lambda x: max(0, x))
    else: slicer = slice(1, 2)
    data_predict = data_predict[cnst.column_names + list(predict_columns.values())[slicer]]
    data_predict = data_predict.astype({col: typ for col, typ in predict_column_types.items() if col in data_predict.columns})
    data_predict.sort_values(["Shop", "Item"], inplace=True)
    
    if file_target_name is not None: data_predict.to_csv(file_target_name, index=False)

    return data_predict


file_source_name, file_target_name, model_source_file, model_target_file, time_for_testing = params().values()

# statsforecast_predict_for_data(file_source_name=file_source_name, file_target_name=file_target_name, time_for_testing=30)