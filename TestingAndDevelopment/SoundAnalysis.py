#######################################################################################################################
#                                       Sham code for testing purposes
#######################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

class DistractorBeltSoundAnalysis():
    def __init__(self):
        #self.plot = plot
        self.cmap = plt.cm.get_cmap('tab20c', 20)


    def choose_plot(self):
        # select function based on user input ???
        return

    def runbackXdB_scatter_allMice(self, path, rbtype, lastrun=False):
        # path: path to file where experiment logs are
        # rbtype: can be 'rb' or 'RB

        # path is "Z:\Holly\Data\Behaviour\Dual-belt_APAs\analysis\DLC_DualBelt\Logs\Experiment_logs\Aug22_APACharacterise_log_raw.csv"

        raw_df = pd.read_csv(path, index_col = ['Date', 'MouseID', 'Phase', 'Belt1speed', 'Belt2speed', 'RunNo'])

        miceList = list(raw_df.index.get_level_values(level='MouseID').unique())
        dayList = list(raw_df.index.get_level_values(level='Date').unique())

        if lastrun == False:
            df = raw_df
        else:
            df = raw_df.copy(deep=True)
            for d, day in enumerate(dayList):
                for m, mouse in enumerate(miceList):
                    df.loc(axis=0)[day, mouse].loc(axis=1)['dB'] = df.loc(axis=0)[day, mouse].loc(axis=1)['dB'].shift(1)

        # Plot showing the dB readings measured per number of run backs
        fig, ax = plt.subplots(figsize=[10, 12])

        DATA = []
        for i in range(0,max(df.loc(axis=1)[rbtype].fillna(0).astype(int)) + 1):
            data = df.loc(axis=1)['dB'].values[df.loc(axis=1)[rbtype].fillna(0) == i]
            filtered_data = data[~np.isnan(data)]
            DATA.append(filtered_data)

        # Plot box plot summarising data
        ax.boxplot(DATA, positions=range(0, max(df.loc(axis=1)[rbtype].fillna(0).astype(int)) + 1), patch_artist=True,
                   meanline=True, showmeans=True,
                   boxprops=dict(facecolor=self.cmap(3), color=self.cmap(3)),
                   capprops=dict(color=self.cmap(2), linewidth=2),
                   whiskerprops=dict(color=self.cmap(2), linewidth=2),
                   flierprops=dict(color=self.cmap(3), markeredgecolor=self.cmap(0)),
                   medianprops=dict(color=self.cmap(1), linewidth=0),
                   meanprops=dict(color=self.cmap(1), linewidth=2),
                   zorder=0
                   )

        # Create jitter in the x-axis of scatter plot using a kernal density estimate
        kde = gaussian_kde(df.loc(axis=1)['dB'].fillna(0))
        density = kde(df.loc(axis=1)['dB'].fillna(0)) # estimate the local density at each data point
        jitter = np.random.rand(*df.loc(axis=1)['dB'].fillna(0).shape) - 0.5 # generate some random jitter between 0 and 1
        xvals = df.loc(axis=1)[rbtype].fillna(0) + (density * jitter * 4) # scale the jitter by the KDE estimate and add it to the centre x-coordinate

        # Plot scatter plot on top of box plot
        ax.scatter(xvals, df.loc(axis=1)['dB'].values, alpha=0.2, zorder=1, c=self.cmap(0), s=10)

        #ax.scatter(df.loc(axis=1)[rbtype].fillna(0),df.loc(axis=1)['dB'].values, alpha=0.2, zorder=1, c=cmap(0), s=10)

        # set basic properties
        ax.set_xlabel('Number of run backs (rb)')
        ax.set_ylabel('Sound level during run (dB)')
        ax.set_xticks(list(range(min(df.loc(axis=1)[rbtype].fillna(0).astype(int)), max(df.loc(axis=1)[rbtype].fillna(0).astype(int)) + 1, 1)))
        ax.spines['right'].set_color((.8,.8,.8))
        ax.spines['top'].set_color((.8,.8,.8))

        # Format axis labels
        xlab = ax.xaxis.get_label()
        ylab = ax.yaxis.get_label()
        xlab.set_size(10)
        ylab.set_size(10)

    def runbackXdb_scatter_perMouse(self, path, rbtype):
        df = pd.read_csv(path + ".csv", index_col=['Date', 'MouseID', 'Phase', 'Belt1speed', 'Belt2speed', 'RunNo'])
        miceList = list(df.index.get_level_values(level='MouseID').unique())
        dayList = list(df.index.get_level_values(level='Date').unique())
