import pandas as pd
import numpy as np

from params import params_for_predictions

import constants as cnst


def unioin_predictions(predict_models:dict, predict_columns:str):
    """
    Объединение предсказаний разных моделей
    """
    predict_df = list(predict_models.values())[0].rename({predict_columns[0]: list(predict_models.keys())[0]}, axis='columns')
    for model, pred_df in list(predict_models.items())[1:]: 
        predict_df[model] = pred_df[predict_columns[0]].values
        if len(predict_columns)>1: 
            predict_df[predict_columns[1]] = np.minimum(np.array(predict_df[predict_columns[1]]), np.array(pred_df[predict_columns[1]]))
            predict_df[predict_columns[2]] = np.maximum(np.array(predict_df[predict_columns[2]]), np.array(pred_df[predict_columns[2]]))
    predict = [0 for _ in range(len(predict_df))]
    for col in predict_models.keys(): predict += predict_df[col].values
    predict = predict/len(predict_models.values())
    predict_df[predict_columns[0]] = list(map(round, predict))
    predict_df = predict_df[[col for col in predict_df.columns if col not in predict_columns] + predict_columns]
    return predict_df


def predictions_for_data(predictions:dict=None, predict_columns:list=None, save_all_predictions:bool=None):
    """
    Обрабатывает предсказания разных моделей и сохраняет в файл
    """
    file_source_names, file_target_name, predict_columns_default, save_all_predictions_default = params_for_predictions().values()
    if save_all_predictions is None: save_all_predictions = save_all_predictions_default
    if predict_columns is None: predict_columns = predict_columns_default
    column_names = cnst.column_names + predict_columns
    if predictions is None:
        predictions = dict()
        for file in file_source_names:
            data_predict = pd.read_csv(file)
            predictions[file.split('.')[0].split('_')[-1].capitalize()] = pd.DataFrame(data_predict.values, columns=column_names).sort_values(column_names[:2])
    predictions = unioin_predictions(predictions, predict_columns)
    if not save_all_predictions: predictions = predictions[column_names]
    print(predictions)
    if file_target_name is not None: predictions.to_csv(file_target_name)
    return predictions


if __name__ == '__main__':
    predictions_for_data(save_all_predictions=True)
