from Normalization import Normalizer
import Plot
import ReadAndClean as rd
import HistogramOfClasses as hoc
import CrossValidation
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def main():
    ourData = rd.ReadData()
    ourData = rd.CleanData(ourData)

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False).to_frame()


    #Histogramy dwa
    hoc.CreateHistNSP(ourData['NSP'])
    hoc.CreateHistClass(ourData['CLASS'])

    #Wyciagniecie dwoch kolumn po ktorych klasyfikujemy
    classification_columns = ourData[['NSP', 'CLASS']]
    #ourData = ourData.drop(labels=['NSP', 'CLASS'], axis=1)

    normalizer = Normalizer()


    #Tu odbywa sie normalizacja/standaryzacja itd + wykresy/ narazie zakomentowane
    # Plot a graph comparing scaled data
    normalizer.compareStandarizationMethods(ourData)

    # normalize data
    ourDataNormalized = normalizer.normalize(rd.extractAttributes(ourData))

    # extract attributes from dataset
    attributes = rd.extractAttributes(ourDataNormalized)
    X_norm = attributes.loc[:, :]

    # transform data with PCA and plot results
    pca = sklearnPCA(n_components=2)
    dataTransformedPCA = pd.DataFrame(pca.fit_transform(X_norm))
    Plot.scatterPlotNSP(dataTransformedPCA, ourData['NSP'])
    Plot.scatterPlotCLASS(dataTransformedPCA, ourData['CLASS'])

    # transform data with LDA and plot results
    lda = LDA(n_components=2)  # 2-dimensional LDA
    dataTransformedLDA = pd.DataFrame(lda.fit_transform(X_norm, ourData['NSP']))
    Plot.scatterPlotNSP(dataTransformedLDA, ourData['NSP'])
    Plot.scatterPlotCLASS(dataTransformedLDA, ourData['CLASS'])

    # plot heatmap - correlation between data and
    Plot.heatmap(ourDataNormalized.join(how='left', other=ourData[['CLASS', 'NSP']]))

    ourDataNormalized[['NSP','CLASS']] = pd.DataFrame(classification_columns[['NSP','CLASS']])
    crossValidationData = CrossValidation.crossValidation(ourDataNormalized)

    hoc.CreateHistAfterValidation(crossValidationData)

#wywolanie funkcji main
if __name__ == '__main__':
    main()