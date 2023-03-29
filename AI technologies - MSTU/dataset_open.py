import pandas as pd

import task_read as tsk



def dataset_instance(file_text):
    text = list()
    for line in file_text:
        if line[0]=='=': number = line.split()[-2][:-1]
        else: text.append(' '.join(line.split() + [number]))
    return text


def classif(number:int, class_dict):
    for key, value in class_dict.items():
        if number in value: return key


def dataset_class(data):
    dataset_dscrb = tsk.dataset_describe(prnt=False)
    classes = dataset_dscrb.text.split("\n\n")[-1].split("\n")[:-1]
    class_dict = dict()
    for cl in classes:
        cl_splt = cl.split()
        if ":" in cl:
            if '+' not in cl:
                cls = int(cl_splt[1][:-1])
                class_dict[cls] = list(map(int, cl_splt[2:]))
            else:
                cls = (int(cl_splt[1]), int(cl_splt[4][:-1]))
                class_dict[cls] = list(map(int, cl_splt[5:]))
        else: class_dict[cls].extend(list(map(int, cl_splt)))
    data[10] = data[9].apply(lambda x: classif(x, class_dict))
    if len(data[data[10].isna()][9])>0: print("Отсутствие класса:", set(data[data[10].isna()][9]), len(data[data[10].isna()][9]))
    return data


def dir_to_list(data, col=7):
    dataset_dscrb_text = tsk.dataset_describe(prnt=False).text
    dir_list = dataset_dscrb_text[dataset_dscrb_text.index('{')+1:dataset_dscrb_text.index('}')].split(',\n')
    dir_list = {dr[:2]: list(map(lambda x: x.strip(), dr[4:-1].split(','))) for dr in dir_list}
    sdir = list()
    for dr in dir_list.values(): sdir.extend(dr)
    sdir = set(sdir)
    data[f'categ_{col}'] = data[col].map(dir_list).fillna(data[col]).values
    for i in range(3):
        data[f'categ_{col}_{i+1}'] = data[f'categ_{col}'].apply(lambda x: x[i] if list(x)==x else x)
    return data


def dataset_open(file_name = 'mechanical-analysis.data.txt', sep=',', header='infer'):
    return pd.read_csv(file_name, header=header, sep=sep)


def dataset_prepare(file_name = 'mechanical-analysis.data.txt'):
    with open(f"Mechanical DataSet/{file_name}", "r") as file:
        file_text = list()
        for line in file: file_text.append(line.strip())
    file_text = dataset_instance(file_text)
    pd.DataFrame(file_text).to_csv(file_name, header=None, index=False)
    data = dataset_class(dataset_open(file_name, sep=' ', header=None))
    data = dir_to_list(data)
    data.to_csv(file_name, index=False)