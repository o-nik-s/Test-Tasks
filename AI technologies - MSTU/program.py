import pandas as pd

import constant as cnst
import dataset_open as dst

from data_processing import prepare_X_y, preprocessing
from classification import test_classification_models


def program(target_field="10", fit_another_model=None):


    dst.dataset_prepare()
    data = dst.dataset_open()
    data.fillna('-1', inplace=True)

    X, y = prepare_X_y(data, target_field)

    print("target_column = ", target_field)

    X_prepr, y_prepr = preprocessing(X, y)
    results = test_classification_models(X_prepr, y_prepr, metric='roc_auc')
    results = sorted(results, key=lambda x:-x[2])[:7]
    print(results)

    header = ["name", "model", "res.mean", "metric", "time"]
    with open(cnst.testing_result_file, "w") as file:
        file.write(', '.join(header)+'\n')
        file.write('\n'.join([', '.join(map(str,res)) for res in results]))

    if fit_another_model:
        use_model(results[0][1], target_field='1') 
    #     print("Обучаем модель")
    #     y_pred = use_model(results[0][1]) 
    #     # Здесь в качестве тестового нужно подавать набор из другого файла - На это времени не хватило
    #     print(y_pred)


def use_model(model, target_field='1'):
    data = dst.dataset_open(cnst.data_file)
    unite_data = data
    if target_field=='1':
        file_another_name = cnst.data_another_file
        dst.dataset_prepare(file_another_name)
        df = dst.dataset_open(file_another_name)
        unite_data = pd.concat([data, df], axis=0)
    X, y = prepare_X_y(unite_data, target_field=target_field)
    X, y = preprocessing(X, y)
    X_data, y_data = X[:len(data)], y[:len(data)]
    y_data_pred = model.predict(X_data)
    print(y_data_pred)
    pd.DataFrame(y_data_pred).to_csv(cnst.prediction_file, header=False, index=False)
    if target_field=='1':
        X_df, y_df = X[len(data):], y[len(data):]
        y_df_pred = model.predict(X_df)
        print(y_df_pred)
        pd.DataFrame(y_df_pred).to_csv(cnst.prediction_another_file, header=False, index=False)
    return True



if __name__ == '__main__':
    program()