import Normalization
import Plot
import ReadAndClean as rd
import HistogramOfClasses as hoc
import CrossValidation
import pandas as pd

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

    #Tu odbywa sie normalizacja/standaryzacja itd + wykresy/ narazie zakomentowane
    #standarize data
    #ourDataScaledStandard = Normalization.standarizeStandard(ourData)
    #ourDataScaledMinMax = Normalization.standarizeMinMax(ourData)
    #ourDataScaledRoboust = Normalization.standarizeRoboust(ourData)
    ourDataNormalized = Normalization.normalize(ourData)
    #Plot a graph comparing scaled data
    #Plot.makeComparsionChart(["ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP"],
                             #[ourData, ourDataScaledStandard, ourDataScaledMinMax,
                              #ourDataScaledRoboust, ourDataNormalized])


    ourDataNormalized[['NSP','CLASS']] = pd.DataFrame(classification_columns[['NSP','CLASS']])
    crossvalidationData = CrossValidation.crossValidation(ourDataNormalized)
    print("c")

#wywolanie funkcji main
if __name__ == '__main__':
    main()