from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


def randomForest(dataValTrain, dataValTest, nameOfClass):
    clf = RandomForestClassifier(n_estimators=100)
    y_pred = cross_val_predict(clf, dataValTrain.values, dataValTest[nameOfClass].values, cv=10)
    y_score = cross_val_score(clf, dataValTrain.values, dataValTest[nameOfClass].values, cv=10)
    return y_pred, y_score.mean()