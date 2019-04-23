import Normalization
import kNNClassification as knn
import ReadAndClean as rd
import HistogramOfClasses as hoc
import CrossValidation
import svmClassification as mysvm
import matplotlib.pyplot as plt
import pandas as pd
from LDA import *
from PCA import *
from Normalization import Normalizer
import Plot
from sklearn.metrics import f1_score


def main():
    nameOfNSP = ['1', '2', '3']
    nameOfCLASS = ['A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP']
    ourData = rd.ReadData()
    ourData = rd.CleanData(ourData)
    normalizer = Normalizer()

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False)

    #Histogramy dwa
    #hoc.CreateHistNSP(ourData['NSP'])
    #hoc.CreateHistClass(ourData['CLASS'])

    #Wyciagniecie dwoch kolumn po ktorych klasyfikujemy
    classification_columns = ourData[['NSP', 'CLASS']]
    #ourData = ourData.drop(labels=['NSP', 'CLASS'], axis=1)


    #Tu odbywa sie normalizacja/standaryzacja itd + wykresy/ narazie zakomentowane
    # Plot a graph comparing scaled data
    #normalizer.compareStandarizationMethods(ourData)

    # normalize data
    ourDataNormalized = normalizer.normalize(ourData)

    # transform data with PCA and plot results
    dataTransformedPCA = transformPCA(normalizer.standarizeStandard(ourData))
    #Plot.scatterPlotNSP(dataTransformedPCA, ourData['NSP'])
    #Plot.scatterPlotCLASS(dataTransformedPCA, ourData['CLASS'])

    # transform data with LDA and plot results
    #ourData = transformLDA(ourData, 'NSP')
    #Plot.scatterPlotNSP(dataTransformedLDA, ourData['NSP'])
    #dataTransformedLDA = transformLDA(ourData, 'CLASS')
    #Plot.scatterPlotCLASS(dataTransformedLDA, ourData['CLASS'])

    # plot heatmap - correlation between data and
    #Plot.heatmap(ourDataNormalized)


    ourDataNormalized = ourDataNormalized.drop(labels=['CLASS','NSP'],axis=1)
    #crossValidationDataNSP = CrossValidation.crossValidationNSP(ourDataNormalized)
    #crossValidationDataCLASS = CrossValidation.crossValidationCLASS(ourDataNormalized)
    #hoc.CreateHistAfterValidationNSP(crossValidationDataNSP, ourData['NSP'])
    #hoc.CreateHistAfterValidationCLASS(crossValidationDataCLASS, ourData['CLASS'])


    y_pred_nps = mysvm.svmClassificationSimple(ourData,classification_columns,'NSP')
    mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_nps,classes=nameOfNSP, normalize=True)
    score_nps = mysvm.svmClassificationSimpleScore(ourData,classification_columns, 'NSP')
    print(str(score_nps*100) + "% skutecznosci svm nsp")
    print("f1-score svm nsp")
    print(f1_score(classification_columns['NSP'].values, y_pred_nps, average=None))

    y_pred_class = mysvm.svmClassificationSimple(ourData,classification_columns,'CLASS')
    mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_class,classes=nameOfCLASS, normalize=True)
    score_class = mysvm.svmClassificationSimpleScore(ourData,classification_columns, 'CLASS')
    print(str(score_class*100) + "% skutecznosci svm class")
    print("f1-score svm class")
    print(f1_score(classification_columns['CLASS'].values, y_pred_class, average=None))

    # y_pred_nps = mysvm.svmClassificationSimple(ourDataNormalized, classification_columns, 'NSP')
    # y_pred_class = mysvm.svmClassificationSimple(ourDataNormalized, classification_columns, 'CLASS')
    # mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_nps, classes=nameOfNSP)
    # mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_class, classes=nameOfCLASS)


    y_pred_knn_nsp = knn.kNNClass(ourData,classification_columns,'NSP')
    mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_knn_nsp,classes=nameOfNSP, normalize=True)
    plt.show()
    score_knn_nps = knn.kNNClassScore(ourData,classification_columns, 'NSP')
    print(str(score_knn_nps*100) + "% skutecznosci knn nsp")
    print("f1-score knn nsp")
    print(f1_score(classification_columns['NSP'].values, y_pred_knn_nsp, average=None))

    y_pred_knn_class = knn.kNNClass(ourData,classification_columns,'CLASS')
    mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_knn_class,classes=nameOfCLASS, normalize=True)
    plt.show()
    score_knn_class = knn.kNNClassScore(ourData,classification_columns, 'CLASS')
    print(str(score_knn_class*100) + "% skutecznosci knn class")
    print("f1-score knn class")
    print(f1_score(classification_columns['CLASS'].values, y_pred_knn_class, average=None))

#wywolanie funkcji main
if __name__ == '__main__':
    main()
