import pandas as pd


def ReadData():
    # Podanie sciezki, r powoduje ze sciezka moze byc smialo uzyto w innych funkcjach
    file = r'D:\Studia Magistreskie PK\Sieci neuronowe\Dane 2\CTG.xls'

    # Wczytanie danych z naszego pliku i z konkretnego arkusza
    ourData = pd.read_excel(file, sheet_name='Raw Data')

    return ourData


def CleanData(data):
    # usuniecie odpowiednich kolumn z wartosciami nic nie znaczacymi
    ourData = data.drop(labels=['FileName', 'Date', 'SegFile', 'b', 'e', 'LBE', 'DR', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'], axis=1)

    # usuniecie pierwszego wiersza, nie zawiera on zadnych wartosci
    ourData = ourData.drop([0])

    # usuniecie w petli trzech ostatnich wierszy w ktorych znajduja sie dane, ktorych nie da sie przeanalizowac
    for i in range(3):
        ourData = ourData.drop(ourData.__len__())

    # reindex data to fit indexes from 0..
    # indexes has changed after drop and this may cause problems
    # when operating on dataframe
    ourData = ourData.reset_index(drop=True)

    # zapis danych do pliku dataCleaned
    SaveData(ourData, "dataCleanded.xlsx")

    return ourData

def SaveData(data, name):
    data.to_excel(name, sheet_name="Data1", index=False)