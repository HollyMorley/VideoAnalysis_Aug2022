#######################################################################################################################
#                                       Sham code for testing purposes
#######################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

cmap = plt.cm.get_cmap('tab20c', 20)

dir = r"Z:\Holly\Data\Behaviour\Dual-belt_APAs\analysis\DLC_DualBelt\Logs\Experiment_logs"
filename = "Aug22_APACharacterise_log_raw"

df = pd.read_csv(dir + "\\" + filename + ".csv", index_col = ['Date', 'MouseID', 'Phase', 'Belt1speed', 'Belt2speed', 'RunNo'])

# Plot showing the dB readings measured per number of rbs
fig_rb, ax = plt.subplots(figsize=[10, 12])

DATA = []
for i in range(0,max(df.loc(axis=1)['rb'].fillna(0).astype(int)) + 1):
    data = df.loc(axis=1)['dB'].values[df.loc(axis=1)['rb'].fillna(0) == i]
    filtered_data = data[~np.isnan(data)]
    DATA.append(filtered_data)

# Plot box plot summarising data
ax.boxplot(DATA, positions=range(0, max(df.loc(axis=1)['rb'].fillna(0).astype(int)) + 1), patch_artist=True,
           meanline=True, showmeans=True,
           boxprops=dict(facecolor=cmap(3), color=cmap(3)),
           capprops=dict(color=cmap(2), linewidth=2),
           whiskerprops=dict(color=cmap(2), linewidth=2),
           flierprops=dict(color=cmap(3), markeredgecolor=cmap(0)),
           medianprops=dict(color=cmap(1), linewidth=0),
           meanprops=dict(color=cmap(1), linewidth=2),
           zorder=0
           )

# Create jitter in the x-axis of scatter plot using a kernal density estimate
kde = gaussian_kde(df.loc(axis=1)['dB'].fillna(0))
density = kde(df.loc(axis=1)['dB'].fillna(0)) # estimate the local density at each data point
jitter = np.random.rand(*df.loc(axis=1)['dB'].fillna(0).shape) - 0.5 # generate some random jitter between 0 and 1
xvals = df.loc(axis=1)['rb'].fillna(0) + (density * jitter * 4) # scale the jitter by the KDE estimate and add it to the centre x-coordinate

# Plot scatter plot on top of box plot
ax.scatter(xvals, df.loc(axis=1)['dB'].values, alpha=0.2, zorder=1, c=cmap(0), s=10)

#ax.scatter(df.loc(axis=1)['rb'].fillna(0),df.loc(axis=1)['dB'].values, alpha=0.2, zorder=1, c=cmap(0), s=10)

# set basic properties
ax.set_xlabel('Number of run backs (rb)')
ax.set_ylabel('Sound level during run (dB)')
ax.set_xticks(list(range(min(df.loc(axis=1)['rb'].fillna(0).astype(int)), max(df.loc(axis=1)['rb'].fillna(0).astype(int)) + 1, 1)))
ax.spines['right'].set_color((.8,.8,.8))
ax.spines['top'].set_color((.8,.8,.8))

# Format axis labels
xlab = ax.xaxis.get_label()
ylab = ax.yaxis.get_label()

xlab.set_size(10)
ylab.set_size(10)


