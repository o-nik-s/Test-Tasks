import time
import tqdm

import pandas as pd
import numpy as np
from collections import Counter

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, cross_validate, KFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

from data_processing import preprocessing


def define_models_classifier(class_weight = None, rnd=71):
    class_weight = 'balanced'
    models = {}
    models['LogisticRegression'] = LogisticRegression(random_state=rnd, class_weight=class_weight)
    # models['BernoulliNB'] = BernoulliNB()
    models['DecisionTreeClassifier'] = DecisionTreeClassifier(random_state=rnd, class_weight=class_weight)
    models['LinearSVC'] = LinearSVC(random_state=rnd, class_weight=class_weight)
    # models['SVC'] = SVC(random_state=rnd, class_weight=class_weight)
    models['KNeighborsClassifier'] = KNeighborsClassifier()
    models['RandomForestClassifier'] = RandomForestClassifier(random_state=rnd, class_weight=class_weight)
    models['SGDClassifier'] = SGDClassifier(random_state=rnd, class_weight=class_weight)
    models['RidgeClassifier'] = RidgeClassifier(random_state=rnd, class_weight=class_weight)
    # models['GradientBoostingClassifier'] = GradientBoostingClassifier(random_state=rnd)
    models['AdaBoostClassifier'] = AdaBoostClassifier(random_state=rnd)
    # models['CatBoostClassifier'] = CatBoostClassifier(random_state=rnd, verbose=False)
    # models['XGBClassifier'] = XGBClassifier(random_state=rnd)
    models['LGBMClassifier'] = LGBMClassifier(is_unbalance=True)
    # models['MLPClassifier'] = MLPClassifier(
    #     hidden_layer_sizes=(20,10),
    #     validation_fraction=0.1,
    #     batch_size=1000,
    #     max_iter=1000,
    #     early_stopping=True,
    #     n_iter_no_change= 100,
    #     verbose=False)
    return models


def get_classification_report(name_report, model, X_true, y_true, y_pred):
    print(name_report, metrics.classification_report(y_true, y_pred), sep='\n', end='\n')
    print('precision_score', metrics.precision_score(y_true, y_pred))
    print('recall_score', metrics.recall_score(y_true, y_pred))
    print('accuracy_score', metrics.accuracy_score(y_true, y_pred))
    print('balanced_accuracy_score', metrics.balanced_accuracy_score(y_true, y_pred))
    print('f1_score', metrics.f1_score(y_true, y_pred))
    print('roc_auc_score', metrics.roc_auc_score(y_true, y_pred))
    print('cohen_kappa_score', metrics.cohen_kappa_score(y_true, y_pred), end='\n\n')
    print('Confusion matrix')
    print(pd.crosstab(y_true, y_pred), end='\n')
    metrics.plot_roc_curve(model, X_true, y_true)


dict_metrics = {"accuracy": metrics.accuracy_score,
                "roc_auc": metrics.roc_auc_score
                }


def call_model(X, y, name, model, test_size=0.25, folds=5, metric='accuracy', random_state=71, prnt=False):
    start_time = time.time()
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
        res = cross_val_score(model, X, y, cv=folds, scoring=metric, error_score="raise", n_jobs=-1)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        metr = dict_metrics[metric](y_pred, y_test)
        result = (name, model, res.mean(), metr, time.time() - start_time)
    except Exception as e: 
        print(e)
        result = (name, model, None, None, time.time() - start_time)
    if prnt: print(result)
    return result



def test_classification_models(X, y, metric="accuracy", test_size=0.25, folds=5):
    # metrics = ['accuracy', 'f1', 'roc_auc']
    models = define_models_classifier()
    results = list()
    # X_prepr, y_prepr = preprocessing(X, y)
    for name in models:
        # try: 
        result = call_model(X, y, name, models[name], test_size=test_size, folds=folds, metric=metric)
        if result[2] != None: 
            results.append(result)
            # print(results[-1])
        # except: pass
    return results #sorted(results, key=lambda x: -x[2])
    print(results[-1])