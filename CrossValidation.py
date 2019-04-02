from sklearn.model_selection import StratifiedKFold
import HistogramOfClasses
import matplotlib.pyplot as plt

def crossValidation(data):
    target = data['NSP']
    data_x = data.drop(['NSP'],axis=1)

    skf = StratifiedKFold(n_splits=10)
    crossValidationData = []

    for train_index, test_index in skf.split(data, target):
        data_train_x = data_x.iloc[train_index]
        data_train_y = target.iloc[train_index]
        data_train_x.loc[:, 'NSP'] = data_train_y

        data_test_x = data_x.iloc[test_index]
        data_test_y = target.iloc[test_index]
        data_test_x.loc[:, 'NSP'] = data_test_y

        crossValidationData.append([data_train_x, data_test_x])

    return crossValidationData