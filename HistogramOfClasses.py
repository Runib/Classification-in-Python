import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def CreateHistNSP(data):
    fig = plt.figure(figsize=(10, 7))
    plt.subplot(1,2,1)
    n, bins, patches = plt.hist(data, color = 'green',  rwidth=0.9)
    plt.xlabel('NSPClass')
    plt.ylabel('Number of Elements')
    plt.title('Wykres NSPClasses')
    plt.grid(True)
    plt.xticks(np.arange(3)+1, ['Normal', 'Suspect', 'Pathologic'])

    for i in patches:
        if i.get_height() > 0:
            plt.text(i.get_x()-.03, i.get_height()+.9, \
                 str(i.get_height()), fontsize=9, color = 'dimgrey')

def CreateHistClass(dataSeries):
    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(dataSeries, color = 'green',  rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Number of Elements')
    plt.title('Wykres Class')
    plt.grid(True)
    plt.xticks(np.arange(10)+1, ['A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP'])

    for i in patches:
        if i.get_height() > 0:
            plt.text(i.get_x()-.03, i.get_height()+.9, \
                 str(i.get_height()), fontsize=9, color = 'dimgrey')
    plt.show()

def CreateHistAfterValidationNSP(dataValidation, originalData):

    i = 1
    fig = []
    N = 3
    ind = np.arange(N)
    width = 0.27

    for dataVal in dataValidation:
        fig.append(plt.figure(i))
        value1 = list(dataVal[0].NSP.sort_values().value_counts(sort=False))
        value2 = list(dataVal[1 ].NSP.sort_values().value_counts(sort=False))
        value3 = list(originalData.sort_values().value_counts(sort=False))
        patches1 = plt.bar(ind, value3, width, label="original")
        patches2 = plt.bar(ind + width, value1, width, label = "train")
        patches3 = plt.bar(ind + width * 2, value2, width, label="test")

        plt.ylabel('Elements')
        plt.title('Group' + str(i))
        plt.grid(True)
        plt.xticks(ind + width / 2, ['Normal', 'Suspect', 'Pathologic'])
        plt.legend(loc='best')
        autolabel(patches1)
        autolabel(patches2)
        autolabel(patches3)
        i = i +1

    plt.show()


def CreateHistAfterValidationCLASS(dataValidation, originalData):

    i = 1
    fig = []
    N = 10
    ind = np.arange(N)
    width = 0.27

    for dataVal in dataValidation:
        fig.append(plt.figure(i, figsize=(10,10)))
        value1 = list(dataVal[0].CLASS.sort_values().value_counts(sort=False))
        value2 = list(dataVal[1 ].CLASS.sort_values().value_counts(sort=False))
        value3 = list(originalData.sort_values().value_counts(sort=False))
        patches1 = plt.bar(ind, value3, width, label="original")
        patches2 = plt.bar(ind + width, value1, width, label = "train")
        patches3 = plt.bar(ind + width * 2, value2, width, label="test")

        plt.ylabel('Elements')
        plt.title('Group' + str(i))
        plt.grid(True)
        plt.xticks(ind + width / 2, ['A', 'B', 'C', 'D', 'SH', 'AD', 'DE', 'LD', 'FS', 'SUSP'])
        plt.legend(loc='best')
        autolabel(patches1)
        autolabel(patches2)
        autolabel(patches3)
        i = i +1

    plt.show()



def autolabel(patches):
    for patch in patches:
        if patch.get_height() > 0:
            plt.text(patch.get_x() + .05, patch.get_height() + 4, \
                     str(patch.get_height()), fontsize = 10, color ='dimgrey')

