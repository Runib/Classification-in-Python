from Normalization import Normalizer
import Plot
import ReadAndClean as rd
import HistogramOfClasses as hoc
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def main():
    ourData = rd.ReadData()
    ourData = rd.CleanData(ourData)

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False).to_frame()
    hoc.CreateHistNSP(ourData['NSP'])
    hoc.CreateHistClass(ourData['CLASS'])

    normalizer = Normalizer()
    # Plot a graph comparing scaled data
    normalizer.compareStandarizationMethods(ourData)

    # standarize data
    ourDataScaledStandard = normalizer.standarizeStandard(rd.extractAttributes(ourData))

    # extract attributes from dataset
    attributes = rd.extractAttributes(ourDataScaledStandard)
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
    Plot.heatmap(ourDataScaledStandard.join(how='left', other=ourData[['CLASS', 'NSP']]))

#wywolanie funkcji main
if __name__ == '__main__':
    main()