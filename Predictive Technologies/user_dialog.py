import os.path

from data_generate import params_for_generate, generate_data
from gaps_filling import params_for_filling, filling_gaps
from params import params_for_statsforecast, params_for_prophet, params_for_sarima
from predict_statsforecast import predict_statsforecast_for_data
from predict_prophet import predict_prophet_for_data
from predict_sarima import predict_sarima_for_data
from predict_models import predictions_for_data
from assortment_use import assortment_for_predict
from params import params_for_accortment as params_for_acc


def user_dialog():
    """
    Пользовательский диалог.
    """
    
    
    answers = dict()
    answers["generate_data"] = input("Требуется сгенерировать данные? Y/N ")
    answers["filling_data"] = input("Требуется обработать данные? Y/N ") if answers["generate_data"] not in ["Y", "y"] else "Y"
    answers["statsforecast"] = input("Использовать пакет Statsforecast? Y/N ")
    if answers["statsforecast"] in ["Y", "y"]:
        file_source_name, file_target_name, model_source_file, model_target_file, time_for_testing = params_for_statsforecast().values()
        if not os.path.isfile(model_target_file):
            answers["statsforecast_testing_time"] = int(input("Сколько времени готовы потратить на тестирование моделей (в секундах)?: "))
    answers["prophet"] = input("Использовать пакет Prophet? Y/N ")
    answers["sarima"] = input("Использовать пакет SARIMA? Y/N ")
    answers["predictions"] = "Y" # input("Сохранить предсказания всех моделей? Y/N ") if len(predictions)>1 else "N"

    
    if answers["generate_data"] in ["Y", "y"]:
        generate_data(params_for_generate())
    if answers["filling_data"] in ["Y", "y"]:
        filling_gaps(params_for_filling())
    predictions = dict()
    if answers["statsforecast"] in ["Y", "y"]:
        print(f"Построение модели с помощью statsforecast")
        file_source_name, file_target_name, model_source_file, model_target_file, time_for_testing = params_for_statsforecast().values()
        if os.path.isfile(model_target_file):
            predictions["Statsforecast"] = predict_statsforecast_for_data(file_source_name=file_source_name, model_source_file=model_source_file, file_target_name=file_target_name)
        else:
            answers["statsforecast_testing_time"] = int(input("Сколько времени готовы потратить на тестирование моделей (в секундах)?: "))
            predictions["Statsforecast"] = predict_statsforecast_for_data(file_source_name=file_source_name, time_for_testing=answers["statsforecast_testing_time"], model_target_file=model_target_file, file_target_name=file_target_name)
    if answers["prophet"] in ["Y", "y"]:
        print(f"Построение модели с помощью prophet")
        file_source_name, file_target_name, column_predict = params_for_prophet().values()
        predictions["Prophet"] = predict_prophet_for_data(file_source_name, file_target_name, column_predict)
    if answers["sarima"] in ["Y", "y"]:
        print(f"Построение модели с помощью sarima")
        file_source_name, file_target_name, column_predict = params_for_sarima().values()
        predictions["SARIMA"] = predict_sarima_for_data(file_source_name, file_target_name, column_predict)
    if len(predictions)>1: print(f"Объединение полученных предсказаний и учет ассортимента")
    predicitions = predictions_for_data(predictions=predictions, save_all_predictions=answers["predictions"] in ["Y", "y"])
    data_assortment_predict = assortment_for_predict(predicitions)
    if data_assortment_predict is not None:
        print(f"Обработка данных выполнена. Результат смотрите в файле {params_for_acc()['file_target_name']} .")


if __name__ == '__main__':    
    user_dialog()