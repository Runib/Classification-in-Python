import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

colors = ['red', 'blue', 'lightgreen', 'gold', 'black', 'darkgreen', 'fuchsia', 'yellowgreen',
          'orangered', 'sienna', 'coral', 'gray', 'darkviolet', 'olive', 'royalblue', 'maroon',
          'teal', 'orange', 'skyblue', 'darkslategray']

def heatmap(data):
    correlations = data.corr()
    f, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(round(correlations, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    plt.show()

def scatterPlotNSP(data, y):
    labels = ['Normal', 'Suspect', 'Pathological']
    colors = ['red', 'blue', 'lightgreen']

    dataSplittedToSeries = []

    i = 1.0
    while i <= 3.0:
        dataSplittedToSeries.append(data[y == i])
        i += 1.0

    scatterPlot(dataSplittedToSeries, labels, colors)

def scatterPlotCLASS(data, y):
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    colors = ['red', 'blue', 'lightgreen', 'gold', 'black', 'darkgreen', 'fuchsia', 'yellowgreen',
              'orangered', 'sienna']

    dataSplittedToSeries = []

    i = 1.0
    while i <= 10.0:
        dataSplittedToSeries.append(data[y == i])
        i += 1.0

    scatterPlot(dataSplittedToSeries, labels, colors)


def scatterPlot(data, labels, colors):
    for dataSeries, label, color in zip(data, labels, colors):
        plt.scatter(dataSeries[0], dataSeries[1], label=label, c=color)

    plt.legend()
    plt.show()

def makeComparsionChart(columns, data):
    """ Creates plots that compares different data sets.
        columns - array of column names of data to be plotted
        data - array with data sets to be compared
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, wspace = 0.2, hspace=0.2, right=0.96, left=0.04)
    ax1 = plt.subplot(gs[0, 0:1], label="")
    ax2 = plt.subplot(gs[0, 1:2], label="" )
    ax3 = plt.subplot(gs[0, 2:3], label="" )
    ax4 = plt.subplot(gs[1, 0:1], label="" )
    ax5 = plt.subplot(gs[1, 1:2], label="" )
    ax1.set_title('Before Scaling')
    ax2.set_title('After Standard Scaler')
    ax3.set_title('After Min-Max Scaler')
    ax4.set_title('After Roboust Scaler')
    ax5.set_title('After Normalization')

    for column in columns:
        sns.kdeplot(data[0][column], ax=ax1, legend=False)
        sns.kdeplot(data[1][column], ax=ax2, legend=False)
        sns.kdeplot(data[2][column], ax=ax3, legend=False)
        sns.kdeplot(data[3][column], ax=ax4, legend=False)
        sns.kdeplot(data[4][column], ax=ax5, legend=False)

    plt.show()