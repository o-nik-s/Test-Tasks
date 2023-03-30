import math

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

import warnings
# warnings.filterwarnings('ignore')


dict_resample = {"SMOTE": SMOTE(),
                 "ADASYN": ADASYN(n_neighbors=5)
                 }


def dummies_fields(data, categ_fields):
    data = pd.get_dummies(data, drop_first=True, columns=categ_fields)
    categ_fields = list(data.columns[list(data.columns).index(f'{categ_fields[0]}_1'):])
    return data, categ_fields


def generate_features_for_number(X, fields):
    for col in fields:
        X[f'{col}_log'] = X[col].apply(lambda x: math.log10(x) if x>0 else -100)
        X[f'{col}_square'] = X[col].apply(lambda x: x**2)
        X[f'{col}_root'] = X[col].apply(lambda x: x**(1/2))
    return X


def prepare_y(data, target_field='10'):
    y = data[target_field].copy()
    if target_field=='10': y[y!='6'], y[y=='6'] = 0, 1
    elif target_field=='1': y[y!=6], y[y==6] = 0, 1
    return y


def prepare_X_y(data, target_field='10'):
    null_field = ['0']
    number_fields = ['4', '5', '6', '8']
    categ_fields = ['2', '3', '7', '9'] + ['categ_7_1', 'categ_7_2', 'categ_7_3']
    data, dummies_categ_fields = dummies_fields(data, categ_fields)
    categ_fields = dummies_categ_fields
    X, y = data[number_fields + categ_fields], prepare_y(data, target_field)
    X = generate_features_for_number(X, number_fields)
    return X, y


def resample(X, y, method):
    return dict_resample[method].fit_resample(X, y)


def rfc_feature(X, y, count):
    n_estimators = min(len(X.columns), 3*count)
    model = RandomForestClassifier(n_estimators = n_estimators)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_df = pd.DataFrame({"Features" : pd.DataFrame(X).columns, "Importances" : importances})
    feature_col = feature_df.sort_values('Importances', ascending=False)[:count].sort_values("Features").Features
    return X[feature_col.values]


def preprocessing(X, y, border = 0.9, max_count=25):
    X_scaled = StandardScaler().fit_transform(X)
    X_minmax_scaled = MinMaxScaler().fit_transform(X_scaled)
    y_label = LabelEncoder().fit_transform(y)
    pca_cmsm_vr = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
    X_feat = PCA(n_components = pca_cmsm_vr.shape[0] - pca_cmsm_vr[pca_cmsm_vr>border].shape[0]).fit_transform(X_minmax_scaled)
    if X_feat[0].shape[0] > max_count: X_feat = PCA(n_components=max_count).fit_transform(X_minmax_scaled)
    X_resample, y_resample = resample(X_feat, y_label, "SMOTE")
    return X_resample, y_resample