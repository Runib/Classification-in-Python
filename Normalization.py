import pandas as pd
from sklearn import preprocessing

def standarizeMinMax(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def standarizeStandard(data):
    standard_scaler = preprocessing.StandardScaler()
    data_scaled = standard_scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def standarizeRoboust(data):
    standard_scaler = preprocessing.RobustScaler()
    data_scaled = standard_scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def normalize(data):
    scaler = preprocessing.Normalizer()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

