import os.path

from generate_data import params_for_generate, generate_data
from gaps_filling import params_for_filling, filling_gaps
from params import params_for_statsforecast, params_for_prophet, params_for_sarima
from predict_statsforecast import statsforecast_predict_for_data
from predict_prophet import prophet_predict_for_data
from predict_sarima import sarima_predict_for_data
from predict_models import predictions_for_data
from assortment_use import assortment_for_predict
from params import params_for_accortment as params_for_acc


def user_dialog():
    """
    Пользовательский диалог.
    Вообще как другой вариант можно запрашивать выбор моделей одновременно, а затем уже считать.
    """
    answers = dict()
    answers["generate_data"] = input("Требуется сгенерировать данные? Y/N ")
    if answers["generate_data"] in ["Y", "y"]:
        generate_data(params_for_generate())
    answers["filling_data"] = input("Требуется обработать данные? Y/N ") if answers["generate_data"] not in ["Y", "y"] else "Y"
    if answers["filling_data"] in ["Y", "y"]:
        filling_gaps(params_for_filling())
    predictions = dict()
    answers["statsforecast"] = input("Использовать пакет Statsforecast? Y/N ")
    if answers["statsforecast"] in ["Y", "y"]:
        file_source_name, file_target_name, model_source_file, model_target_file, time_for_testing = params_for_statsforecast().values()
        if os.path.isfile(model_target_file):
            predictions["Statsforecast"] = statsforecast_predict_for_data(file_source_name=file_source_name, model_source_file=model_source_file, file_target_name=file_target_name)
        else:
            answers["statsforecast_testing_time"] = int(input("Сколько времени готовы потратить на тестирование моделей (в секундах)?: "))
            predictions["Statsforecast"] = statsforecast_predict_for_data(file_source_name=file_source_name, time_for_testing=answers["statsforecast_testing_time"], model_target_file=model_target_file, file_target_name=file_target_name)
    answers["prophet"] = input("Использовать пакет Prophet? Y/N ")
    if answers["prophet"] in ["Y", "y"]:
        file_source_name, file_target_name, column_predict = params_for_prophet().values()
        predictions["Prophet"] = prophet_predict_for_data(file_source_name, file_target_name, column_predict)
    answers["sarima"] = input("Использовать пакет SARIMA? Y/N ") if len(predictions)!=0 else "Y"
    if answers["sarima"] in ["Y", "y"]:
        file_source_name, file_target_name, column_predict = params_for_sarima().values()
        predictions["SARIMA"] = sarima_predict_for_data(file_source_name, file_target_name, column_predict)
    answers["predictions"] = "Y" # input("Сохранить предсказания всех моделей? Y/N ") if len(predictions)>1 else "N"
    predicitions = predictions_for_data(predictions=predictions, save_all_predictions=answers["predictions"] in ["Y", "y"])
    data_assortment_predict = assortment_for_predict(predicitions)
    if data_assortment_predict is not None:
        print(f"Обработка данных выполнена. Результат смотрите в файле {params_for_acc()['file_target_name']} .")