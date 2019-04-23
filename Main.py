from Normalization import Normalizer
import Plot
import ReadAndClean as rd
import HistogramOfClasses as hoc
import CrossValidation
import pandas as pd
from LDA import *
from PCA import *


def main():
    ourData = rd.ReadData()
    ourData = rd.CleanData(ourData)
    normalizer = Normalizer()

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False).to_frame()

    #Histogramy dwa
    hoc.CreateHistNSP(ourData['NSP'])
    hoc.CreateHistClass(ourData['CLASS'])

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


    ourDataNormalized[['NSP','CLASS']] = pd.DataFrame(classification_columns[['NSP','CLASS']])
    crossValidationData = CrossValidation.crossValidation(ourDataNormalized)

    hoc.CreateHistAfterValidation(crossValidationData)

#wywolanie funkcji main
if __name__ == '__main__':
    main()