import pandas as pd
from sklearn import preprocessing
import ReadAndClean as rd
import Plot

class Normalizer:

    def compareStandarizationMethods(self, data):
        # standarize data
        dataScaledStandard = self.standarizeStandard(data)
        dataNormalized = self.normalize(data)
        dataScaledMinMax = self.standarizeMinMax(data)
        dataScaledRoboust = self.standarizeRoboust(data)

        Plot.makeComparsionChart(['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                              'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                              'Median', 'Variance', 'Tendency'],
                              [data, dataScaledStandard, dataScaledMinMax,
                               dataScaledRoboust, dataNormalized])

    def standarizeMinMax(self, data):
        attributes = rd.extractAttributes(data)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(attributes)
        data_scaled = pd.DataFrame(data_scaled, columns=attributes.columns)
        return data_scaled.join(how='left', other=data[['CLASS', 'NSP']])

    def standarizeStandard(self, data):
        attributes = rd.extractAttributes(data)
        standard_scaler = preprocessing.StandardScaler()
        data_scaled = standard_scaler.fit_transform(attributes)
        data_scaled = pd.DataFrame(data_scaled, columns=attributes.columns)
        return data_scaled.join(how='left', other=data[['CLASS', 'NSP']])

    def standarizeRoboust(self, data):
        attributes = rd.extractAttributes(data)
        standard_scaler = preprocessing.RobustScaler()
        data_scaled = standard_scaler.fit_transform(attributes)
        data_scaled = pd.DataFrame(data_scaled, columns=attributes.columns)
        return data_scaled.join(how='left', other=data[['CLASS', 'NSP']])

    def normalize(self, data):
        attributes = rd.extractAttributes(data)
        scaler = preprocessing.Normalizer()
        data_scaled = scaler.fit_transform(attributes)
        data_scaled = pd.DataFrame(data_scaled, columns=attributes.columns)
        return data_scaled.join(how='left', other=data[['CLASS', 'NSP']])

