import pandas as pd
from sklearn import preprocessing
import ReadAndClean as rd
import Plot

class Normalizer:

    def compareStandarizationMethods(self, data):
        # extract attributes from dataset
        attributes = rd.extractAttributes(data)

        # standarize data
        dataScaledStandard = self.standarizeStandard(attributes)
        dataNormalized = self.normalize(attributes)
        dataScaledMinMax = self.standarizeMinMax(attributes)
        dataScaledRoboust = self.standarizeRoboust(attributes)

        Plot.makeComparsionChart(['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                              'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                              'Median', 'Variance', 'Tendency'],
                              [data, dataScaledStandard, dataScaledMinMax,
                               dataScaledRoboust, dataNormalized])

    def standarizeMinMax(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
        return data_scaled

    def standarizeStandard(self, data):
        standard_scaler = preprocessing.StandardScaler()
        data_scaled = standard_scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
        return data_scaled

    def standarizeRoboust(self, data):
        standard_scaler = preprocessing.RobustScaler()
        data_scaled = standard_scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
        return data_scaled

    def normalize(self, data):
        scaler = preprocessing.Normalizer()
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
        return data_scaled

