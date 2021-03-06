from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

def kNNClass(dataValTrain, dataValTest, nameOfClass):
    knn = KNeighborsClassifier(n_neighbors=7)
    y_pred = cross_val_predict(knn, dataValTrain.values, dataValTest[nameOfClass].values, cv=10)
    y_score = cross_val_score(knn, dataValTrain.values, dataValTest[nameOfClass].values, cv=10)
    return y_pred, y_score.mean()

