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
    plt.xticks(np.arange(3)+1, ['1', '2', '3'])

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
    plt.xticks(np.arange(10)+1, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    for i in patches:
        if i.get_height() > 0:
            plt.text(i.get_x()-.03, i.get_height()+.9, \
                 str(i.get_height()), fontsize=9, color = 'dimgrey')
    plt.show()

def CreateHistAfterValidation(dataValidation):
    fig = plt.figure(figsize=(15, 15))
    i = 1

    for dataVal in dataValidation:
        plt.subplot(2, 5, i)
        n, bins, patches = plt.hist(dataVal[0]['NSP'], color='green', rwidth=0.9)
        plt.xlabel('NSPClass')
        plt.ylabel('Elements')
        plt.title('Group' + str(i))
        i = i + 1
        plt.grid(True)
        plt.xticks(np.arange(3) + 1, ['1', '2', '3'])
        for j in patches:
            if j.get_height() > 0:
                plt.text(j.get_x() - .03, j.get_height() + .9, \
                         str(j.get_height()), fontsize=8, color='dimgrey')

    plt.show()

    fig = plt.figure(figsize=(15, 15))
    i = 1
    for dataVal in dataValidation:
        plt.subplot(2, 5, i)
        n, bins, patches = plt.hist(dataVal[1]['NSP'], color='green', rwidth=0.9)
        plt.xlabel('NSPClass')
        plt.ylabel('Elements')
        plt.title('Group' + str(i))
        i = i + 1
        plt.grid(True)
        plt.xticks(np.arange(3) + 1, ['1', '2', '3'])
        for j in patches:
            if j.get_height() > 0:
                plt.text(j.get_x() - .03, j.get_height() + .9, \
                         str(j.get_height()), fontsize=8, color='dimgrey')
    plt.show()

