import math
import random
import time
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import lognorm
from dateutil.relativedelta import relativedelta

from params import params_for_generate



"""
    Генерация тестовых данных.
    
    Вообще можно было бы попробовать сгенерировать с помощью какой-то из готовых библиотек. 
    Например: tf.keras.utils.timeseries_dataset_from_array, keras.preprocessing.sequence.TimeseriesGenerator
    
    Плюс при самостоятельной генерации можно учесть больше требуемых факторов.
"""


def generate_period(date_start, date_end):
    """
    Генерация периода от start_date до end_date
    Возможно лучше использовать pd.date_range(start_date, end_date)
    """
    dt_list = list()
    for i in range((date_end-date_start).days + 1):
        dt_list.append(date_start + dt.timedelta(i))
    return dt_list


def generate_count(start_count: int, days: int, change_days: list, total_trend=1.001, rnd=0.005) -> list:
    """
    Генерируем количество товаров на days дней
    """
    count_list = list()
    count = start_count
    for d in range(days+1):
        if d in change_days: trend = total_trend*random.uniform(1-rnd, 1+rnd)
        count *= trend
        count_list.append(count*(1+(2*random.random()-1)/10))
    return count_list


def season_correct(season=None):
    """
    Сезонная корректировка
    """
    if season == None:
        season = random.choice(['exist', 'nothing'])
    # if season == 'exist': season = random.choice(['week', 'month', 'quarter', 'half-year', 'year'])
    if season == 'exist': season = random.choice(['week', 'month'])
    season_change = {'nothing': lambda x, date, k: x,
                    'week': lambda x, date, k: int(x*(1+date.isoweekday()/k)),
                    'month': lambda x, date, k: int(x*(1+date.day/k))}
    return season_change[season]


def generate_counts_for_item(dt_list, start_count, days, change_days, total_trend=1.001):
    """
    Генерация с сезонной корректировкой
    """
    counts = generate_count(start_count, days, change_days, total_trend)
    season_adjust = season_correct()
    counts = [season_adjust(counts[i], d, 100) for i, d in enumerate(dt_list)]
    return counts


def generate_assortment_periods(start_date, finish_date, days_between=15, days_assortment=60):
    """
    Генерирует периоды для ассортимента
    """
    assort_date_list = list()
    curr_date = start_date - dt.timedelta(days=2)
    while curr_date < finish_date:
        start_date = curr_date + dt.timedelta(days=random.randint(2, days_between))
        curr_date = start_date + dt.timedelta(days=random.randint(0, days_assortment))
        assort_date_list.append([start_date, curr_date])
    if assort_date_list[-1][0] > finish_date: assort_date_list.pop()
    assort_date_list[-1][1] = min(assort_date_list[-1][1], finish_date)
    return assort_date_list


def assort_shop_item_count(shop_item_counts, assortment_data, shop, item):
    """
    Ассортимент для конкретного магазина и продукта
    """
    if (shop, item) not in shop_item_counts.keys(): return list()
    shop_item_cnt = shop_item_counts[(shop, item)]
    assort_data = assortment_data[(shop, item)]
    i, assort_list = 0, list()
    for d, cnt in shop_item_cnt:
        if assort_data[i][0] <= d <= assort_data[i][1]: assort_list.append((d, cnt))
        if d == assort_data[i][1]: i += 1
    return assort_list


def shuffle(days): 
    """
    shuffle = периодическое случайное изменение поведения (тренда) в другую сторону / под другим углом
    """
    return sorted([0] + [int(random.random()*days) for _ in range(days//30)])



def save_data(data_list:list, file_name:str):
    """
    Сохранение данных в файл
    """

    time_start = time.time()
    data = pd.DataFrame(data_list).rename(columns={0:'Shop', 1:'Item', 2:'Date', 3:'Number'})
    data.to_csv(file_name)
    print(6, time.time() - time_start)
 
    # with open(file_name, 'w') as file:
    #     file.write(','.join(['Shop', 'Item', 'Date', 'Number'])+'\n')
    #     file.write('\n'.join(map(lambda x: ','.join(map(str,x)), data_list)))
    # print(7, time.time() - time_start)


def generate_data(params:dict):
    """
    Основной код генерации данных. 
    Задается общий тренд для магазина (пусть коэффициент магазина будет),
    общий тренд для товара, индивидуальные отклонения.
    Возможно стоит разбить на отдельные функции.
    """
    
    print("Генерация данных")

    file_target_name, file_target_name_assort, shops_count, items_count, date_start, date_end, random_seed, data_columns = params.values()

    random.seed(random_seed)
    np.random.seed(random_seed)

    time_start = time.time()

    shops = [f'shop_{i}' for i in range(shops_count)]
    items = [f'item_{i}' for i in range(items_count)]

    dt_list = generate_period(date_start, date_end)
    days = (dt_list[-1] - dt_list[0]).days + 1

    start_counts = list(map(int, lognorm.rvs(s=1, scale=math.exp(5), size=shops_count*items_count)))


    print(1, time.time() - time_start)
    time_start = time.time()

    item_counts = {item: generate_counts_for_item(dt_list=dt_list,
        start_count=start_counts[i], days=days, change_days=shuffle(days),
        total_trend=1) for i, item in enumerate(items)}


    print(2, time.time() - time_start)
    time_start = time.time()

    shop_k = {s: round(3*random.random(), 2) for s in shops}
    item_k = {item: round(3*random.random(), 2) for item in items}

    shop_item_counts = {(shop, item): list(zip(dt_list, list(map(lambda x: int(x*shop_k[shop]*item_k[item]), item_counts[item])))) \
        for shop in shops for item in items if random.randint(1,10)<9}


    print(3, time.time() - time_start)


    forecast_date = date_end + relativedelta(weeks=+4)
    assortment_data = {(s, i): generate_assortment_periods(date_start, forecast_date) for i in items for s in shops}


    print(4, time.time() - time_start)
    time_start = time.time()

    data_list = [(shop, item, d, cnt) for shop in shops for item in items for d, cnt in assort_shop_item_count(shop_item_counts, assortment_data, shop, item) if cnt>0]


    print(5, time.time() - time_start)
    time_start = time.time()
    
    save_data(data_list=data_list, file_name=file_target_name)
    
    assort_future_list = [(shop, item, max(date_end+dt.timedelta(days=1), date_1), date_2) for shop in shops for item in items for date_1, date_2 in assortment_data[(shop, item)] if date_2>date_end]
    pd.DataFrame(assort_future_list).rename(columns={0: data_columns[0], 1: data_columns[1], 2:'Date_1', 3:'Date_2'}).to_csv(file_target_name_assort, index=False)



# generate_data(params_for_generate())