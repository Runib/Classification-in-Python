import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def CreateHistNSP(dataSeries):
    fig = plt.figure(figsize=(10, 7))
    plt.subplot(1,2,1)
    n, bins, patches = plt.hist(dataSeries, color = 'green',  rwidth=0.9)
    plt.xlabel('NSPClass')
    plt.ylabel('Number of Elements')
    plt.title('Wykres NSPClasses')
    plt.grid(True)
    plt.xticks(np.arange(3)+1, ['1', '2', '3'])

def CreateHistClass(dataSeries):
    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(dataSeries, color = 'green',  rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Number of Elements')
    plt.title('Wykres Class')
    plt.grid(True)
    plt.xticks(np.arange(10)+1, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.show()


