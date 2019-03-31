import pandas as pd
from openpyxl import Workbook
import Normalization
import Plot
import xlrd

def main():
    #Podanie sciezki, r powoduje ze sciezka moze byc smialo uzyto w innych funkcjach
    file = r'D:\Studia Magistreskie PK\Sieci neuronowe\Dane 2\CTG.xls'

    #Wczytanie danych z naszego pliku i z konkretnego arkusza
    ourData = pd.read_excel(file, sheet_name='Raw Data')

    #usuniecie odpowiednich kolumn z wartosciami nic nie znaczacymi
    ourData = ourData.drop(labels=['FileName', 'Date', 'SegFile'], axis=1)

    #usuniecie pierwszego wiersza, nie zawiera on zadnych wartosci
    ourData = ourData.drop([0])

    #usuniecie w petli trzech ostatnich wierszy w ktorych znajduja sie dane, ktorych nie da sie przeanalizowac
    for i in range(3):
        ourData = ourData.drop(ourData.__len__())

    #zapis danych do pliku dataCleaned
    ourData.to_excel("dataCleaned.xlsx", sheet_name="Data1", index=False)

    #Te linijki to zgrupowanie według klas i zliczenie ile elementow ma kazda klasa, dwa rodzaje
    NSPCardiotography = ourData['NSP'].sort_values().value_counts(sort=False)
    print(NSPCardiotography)
    ClassCardiotography = ourData['CLASS'].sort_values().value_counts(sort=False).to_frame()
    print(ClassCardiotography)

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