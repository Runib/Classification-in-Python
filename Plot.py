import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


matplotlib.style.use('ggplot')

def makeComparsionChart(columns, data):
    """ Creates plots that compares different data sets.
        columns - array of column names of data to be plotted
        data - array with data sets to be compared
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, wspace = 0.2, hspace=0.2, right=0.96, left=0.04)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(14, 5))
    ax1 = plt.subplot(gs[0, 0:1], )
    ax2 = plt.subplot(gs[0, 1:2])
    ax3 = plt.subplot(gs[0, 2:3])
    ax4 = plt.subplot(gs[1, 0:1])
    ax5 = plt.subplot(gs[1, 1:2])
    ax1.set_title('Before Scaling')
    ax2.set_title('After Standard Scaler')
    ax3.set_title('After Min-Max Scaler')
    ax4.set_title('After Roboust Scaler')
    ax5.set_title('After Normalization')


    for column in columns:
        sns.kdeplot(data[0][column], ax=ax1)
        sns.kdeplot(data[1][column], ax=ax2)
        sns.kdeplot(data[2][column], ax=ax3)
        sns.kdeplot(data[3][column], ax=ax4)
        sns.kdeplot(data[4][column], ax=ax5)

    plt.show()