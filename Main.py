import Normalization
import Plot
import ReadAndClean as rd
import HistogramOfClasses as hoc


def main():
    ourData = rd.ReadData()
    ourData = rd.CleanData(ourData)

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False).to_frame()
    hoc.CreateHistNSP(ourData['NSP'])
    hoc.CreateHistClass(ourData['CLASS'])

    # standarize data
    ourDataScaledStandard = Normalization.standarizeStandard(ourData)
    ourDataScaledMinMax = Normalization.standarizeMinMax(ourData)
    ourDataScaledRoboust = Normalization.standarizeRoboust(ourData)
    ourDataNormalized = Normalization.normalize(ourData)
    # Plot a graph comparing scaled data
    Plot.makeComparsionChart(["ASTV", "MSTV", "ALTV", "MLTV", "DL", "DS", "DP"],
                             [ourData, ourDataScaledStandard, ourDataScaledMinMax,
                              ourDataScaledRoboust, ourDataNormalized])


#wywolanie funkcji main
if __name__ == '__main__':
    main()