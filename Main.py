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
import RandomForestClassification as rfc


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
    Plot.scatterPlotNSP(dataTransformedPCA, ourData['NSP'])
    Plot.scatterPlotCLASS(dataTransformedPCA, ourData['CLASS'])

    # transform data with LDA and plot results
    dataTransformedLDA = transformLDA(ourData, 'NSP')
    Plot.scatterPlotNSP(dataTransformedLDA, ourData['NSP'])
    dataTransformedLDA = transformLDA(ourData, 'CLASS')
    Plot.scatterPlotCLASS(dataTransformedLDA, ourData['CLASS'])

    # plot heatmap - correlation between data and
    Plot.heatmap(ourDataNormalized)


    ourDataNormalized = ourDataNormalized.drop(labels=['CLASS','NSP'],axis=1)
    #crossValidationDataNSP = CrossValidation.crossValidationNSP(ourDataNormalized)
    #crossValidationDataCLASS = CrossValidation.crossValidationCLASS(ourDataNormalized)
    #hoc.CreateHistAfterValidationNSP(crossValidationDataNSP, ourData['NSP'])
    #hoc.CreateHistAfterValidationCLASS(crossValidationDataCLASS, ourData['CLASS'])

    #Data without LDA and normalization
    allClassification(ourData, classification_columns, nameOfNSP, nameOfCLASS)

    #Data with LDA and normalization


def allClassification(ourData, classification_columns, nameOfNSP, nameOfCLASS):
    # SVM
    y_pred_nps, y_score_nps = mysvm.svmClassificationSimple(ourData, classification_columns, 'NSP')
    y_pred_class, y_score_class = mysvm.svmClassificationSimple(ourData, classification_columns, 'CLASS')
    print(y_score_nps, y_score_class)

    f1_svm_nsp = mysvm.getInfoMatrix(classification_columns, y_pred_nps, 'NSP')
    f1_svm_class = mysvm.getInfoMatrix(classification_columns, y_pred_class, 'CLASS')

    mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_nps, classes=nameOfNSP,
                                normalize=True)
    mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_class, classes=nameOfCLASS,
                                normalize=True)


    # KNN
    y_pred_knn_nsp, y_score_knn_nsp = knn.kNNClass(ourData, classification_columns, 'NSP')
    y_pred_knn_class, y_score_knn_class = knn.kNNClass(ourData, classification_columns, 'CLASS')
    print(y_score_knn_nsp, y_score_knn_class)

    f1_knn_nsp = mysvm.getInfoMatrix(classification_columns, y_pred_knn_nsp, 'NSP')
    f1_knn_class = mysvm.getInfoMatrix(classification_columns, y_pred_knn_class, 'CLASS')

    mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_knn_nsp, classes=nameOfNSP,
                                normalize=True)
    mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_knn_class,
                                classes=nameOfCLASS, normalize=True)

    # Random Forest
    y_pred_rf_nsp, y_score_rf_nsp = rfc.randomForest(ourData, classification_columns, 'NSP')
    y_pred_rf_class, y_score_rf_class = rfc.randomForest(ourData, classification_columns, 'CLASS')
    print(y_score_rf_nsp, y_score_rf_class)

    f1_rf_nsp = mysvm.getInfoMatrix(classification_columns, y_pred_rf_nsp, 'NSP')
    f1_rf_class = mysvm.getInfoMatrix(classification_columns, y_pred_rf_class, 'CLASS')

    mysvm.plot_confusion_matrix(y_true=classification_columns['NSP'].values, y_pred=y_pred_rf_nsp, classes=nameOfNSP,
                                normalize=True)
    mysvm.plot_confusion_matrix(y_true=classification_columns['CLASS'].values, y_pred=y_pred_rf_class,
                                classes=nameOfCLASS, normalize=True)

    print("F1 dla NSP/SVM: \n" + str(f1_svm_nsp))
    print("F1 dla CLASS/SVM: \n" + str(f1_svm_class))
    print("F1 dla NSP/KNN: \n" + str(f1_knn_nsp))
    print("F1 dla CLASS/KNN: \n" + str(f1_knn_class))
    print("F1 dla NSP/RF: \n" + str(f1_rf_nsp))
    print("F1 dla CLASS/RF: \n" + str(f1_rf_class))
    plt.show()

#wywolanie funkcji main
if __name__ == '__main__':
    main()
