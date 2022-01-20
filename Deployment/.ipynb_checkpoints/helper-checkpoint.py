import matplotlib.pyplot as plt
import numpy as np

def normalised_hist(df_1, df_2, column: str, feature: str, xlimits: list):
    #Erasing ages below 16 years
    df_1 = df_1[df_1['EDAD'] > 16]
    df_2 = df_2[df_2['EDAD'] > 16]
    #Histogram
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = 'col', sharey = 'row', figsize = (9, 6), dpi = 80)
    bins = np.linspace(xlimits[0], xlimits[1], 40)
    df_1[column].hist(bins = bins, ax = ax, ec = 'black', histtype = 'bar', density = True)
    df_2[column].hist(bins = bins, ax = ax, alpha = 0.5, ec = 'black', histtype = 'bar', density = True)
    ax.set_title(f'Histogram for: {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequencies')
    plt.legend(['Non-retired', 'Retired'], loc = 'best')
    return fig