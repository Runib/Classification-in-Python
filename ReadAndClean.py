import pandas as pd
from openpyxl import Workbook
import xlrd


def ReadData():
    # Podanie sciezki, r powoduje ze sciezka moze byc smialo uzyto w innych funkcjach
    file = r'D:\Studia Magistreskie PK\Sieci neuronowe\Dane 2\CTG.xls'

    # Wczytanie danych z naszego pliku i z konkretnego arkusza
    ourData = pd.read_excel(file, sheet_name='Raw Data')

    return ourData


def CleanData(data):
    # usuniecie odpowiednich kolumn z wartosciami nic nie znaczacymi
    ourData = data.drop(labels=['FileName', 'Date', 'SegFile'], axis=1)

    # usuniecie pierwszego wiersza, nie zawiera on zadnych wartosci
    ourData = ourData.drop([0])

    # usuniecie w petli trzech ostatnich wierszy w ktorych znajduja sie dane, ktorych nie da sie przeanalizowac
    for i in range(3):
        ourData = ourData.drop(ourData.__len__())

    # zapis danych do pliku dataCleaned
    SaveData(ourData, "dataCleanded.xlsx")

    return ourData

def SaveData(data, name):
    data.to_excel(name, sheet_name="Data1", index=False)