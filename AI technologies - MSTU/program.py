import constant as cnst
import dataset_open as dst

from data_processing import prepare_X_y, preprocessing
from classification import test_classification_models


def program(target_column="10", fit_model=None):


    dst.dataset_prepare()
    data = dst.dataset_open()
    data.fillna('-1', inplace=True)

    X, y = prepare_X_y(data, target_column)

    print("target_column = ", target_column)

    X_prepr, y_prepr = preprocessing(X, y)
    results = test_classification_models(X_prepr, y_prepr, metric='roc_auc')
    results = sorted(results, key=lambda x:-x[2])[:7]
    print(results)

    header = ["name", "model", "res.mean", "metric", "time"]
    with open(cnst.testing_result_file, "w") as file:
        file.write(', '.join(header)+'\n')
        file.write('\n'.join([', '.join(map(str,res)) for res in results]))

    if fit_model:
        print("Обучаем модель")
        y_pred = use_model(X_prepr, y_prepr, results[0][1], X_prepr) 
        # Здесь в качестве тестового нужно подавать набор из другого файла - На это времени не хватило
        print(y_pred)


def use_model(X, y, model, X_pred):
    model.fit(X, y)
    y_pred = model.predict(X_pred)
    return y_pred



if __name__ == '__main__':
    program()