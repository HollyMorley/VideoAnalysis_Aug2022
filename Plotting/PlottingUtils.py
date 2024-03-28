from Helpers.utils import *
import pandas as pd
import re
import matplotlib.patches as patches
import seaborn as sns


def load_measures_files(conditions, measure_type):
    measures = dict.fromkeys(conditions)
    for con in conditions:
        print('Loading measures dataframes for condition: %s' % con)
        segments = con.split('_')
        filename = 'MEASURES_%s.h5' % measure_type
        if 'Day' not in con and 'Wash' not in con:
            conname, speed = segments[0:2]
            measure_filepath = r'%s\%s_%s\%s' % (paths['filtereddata_folder'], conname, speed, filename)
        else:
            conname, speed, repeat, wash, day = segments
            measure_filepath = r'%s\%s_%s\%s\%s\%s\%s' % (
            paths['filtereddata_folder'], conname, speed, repeat, wash, day, filename)
        measures[con] = pd.read_hdf(measure_filepath)
    return measures

def remove_prepruns(data, con):
    con_split = con.split('_')
    exptype = con_split[0]
    speed_b1 = re.findall('[A-Z][^A-Z]*', con_split[1])[0]
    if exptype == 'APAChar':
        prep = 2 if speed_b1 == 'Low' else 3
        # drop first 2 runs
        data[con] = data[con].drop(data[con].index[data[con].index.get_level_values(level='Run') < prep])
        # reindex run level by subtracting 2 from the run number
        data[con].index = data[con].index.set_levels(data[con].index.levels[1] - (prep - 1), level='Run')
    else:
        raise ValueError("Experiment type not recognized, haven't configured to remove prepruns for this type")
    return data[con]

def set_colormap(comparison):
    if comparison == 'Day':
        cmap = "mako_r"
    elif comparison == 'ExpPhase':
        cmap = "hls"
    elif comparison == 'Stride':
        cmap = "plasma"
    return cmap

def remove_vowel(string):
    return re.sub(r'[aeiou]', '', string)

########################################################################################################################
# Plotting functions
########################################################################################################################
class MkFigs():
    def __init__(self, axes_label_fs=14, title_fs=16, legend_fs=12, tick_fs=12, labelpad=10):
        self.axes_label_fs, self.title_fs, self.legend_fs, self.tick_fs, self.labelpad = axes_label_fs, title_fs, legend_fs, tick_fs, labelpad

    def MkFig_linearSpeed(self, xlabel, ylabel, speed_min, speed_max):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_xlim(speed_min, speed_max)
        ax.set_xlabel('Speed (mm/s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax

    def MkFig_TSRuns_with_ExpPhase_blocks(self, ax, xlabel='Run', ylabel='', run_num=40, xlabels_visible=True):
        #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_xlim(0, run_num + 1)
        # ax.set_xticklabels([])
        # set the xticks to start at 0 and go up to 39 in 5s, eg 0,4,9,14,19,24,29,34,39
        ax.set_xticks(range(1, run_num + 1, 1), minor=True)
        ax.set_xticks(range(1, run_num + 1, 3))
        if xlabels_visible:
            fc_box = '0.9'
            lw_box = 0.1
            lw_arrow = 1
            ax.annotate('Baseline', xytext=(0.135, -0.23), xy=(0.135, -0.15), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
            ax.annotate('APA', xytext=(0.5, -0.23), xy=(0.5, -0.15), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=13, lengthB=1', lw=lw_arrow))
            ax.annotate('Baseline', xytext=(0.865, -0.23), xy=(0.865, -0.15), xycoords='axes fraction', ha='center', va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
            ax.set_xlabel(xlabel, fontsize=self.axes_label_fs)
        ax.axvline(10.5, ls='--', color='k', linewidth=0.5)
        ax.axvline(30.5, ls='--', color='k', linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=self.axes_label_fs)
        plt.subplots_adjust(bottom=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def MkFig_TSStride(self, ax, xlabel='Stride (%)', ylabel='', buffer=0.25):
        #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        lower = int(0-100*buffer)
        upper = int(100+100*buffer)
        ax.set_xlim(lower,upper)
        ax.set_xticks(range(lower,upper+1, 5), minor=True)
        ax.set_xticks(range(0,101, 50))
        ax.set_xlabel(xlabel, fontsize=self.axes_label_fs)
        ax.set_ylabel(ylabel, fontsize=self.axes_label_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def MkFig_byStride(self, ax, xlabel='Stride', ylabel=''):
        ax.set_xlim(-3.5, 1.5)
        ax.set_xticks(range(-3, 2, 1))
        # ax.set_xticks(range(0, 5, 1))
        ax.set_xticklabels(['-3', '-2', '-1', '0', '1'], fontsize=self.tick_fs)
        ax.axvline(0, ls='--', color='k', linewidth=3, alpha=0.2)
        ax.set_xlabel(xlabel, fontsize=self.axes_label_fs)
        ax.set_ylabel(ylabel, fontsize=self.axes_label_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def MkFig_bySpeed(self, ax, xlabel='Speed (mm/s)', ylabel=''):
        ax.set_xlabel(xlabel, fontsize=self.axes_label_fs)
        ax.set_ylabel(ylabel, fontsize=self.axes_label_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def MkFig_BackShape(self, ax, xlabel='Back marker', ylabel='z (mm)', xlabels_visible=False):
        #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_xlim(0,13)
        xticks = range(1,13,1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks[::-1], fontsize=self.tick_fs)
        ax.set_xlabel(xlabel, fontsize=self.axes_label_fs)
        ax.set_ylabel(ylabel, fontsize=self.axes_label_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.subplots_adjust(bottom=0.15)
        if xlabels_visible:
            ax.annotate('Nose', xytext=(0.85, -0.12), xy=(1, -0.12), xycoords='axes fraction', ha='center', va='center',
                        fontstyle='italic', arrowprops=dict(arrowstyle='-|>'))
            ax.annotate('Tail', xytext=(0.15, -0.12), xy=(0, -0.12), xycoords='axes fraction', ha='center', va='center',
                        fontstyle='italic', arrowprops=dict(arrowstyle='-|>'))
        return ax

    def MkFig_PawPref(self, ax):
        #fig, ax = plt.subplots(1, 1, figsize=(5, 8))
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Left', 'Right'])
        # ax.set_xlabel('Paw preference')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #
        return ax

    def MkFig_PolarbyTime(self, ax):
        #fig, ax = plt.subplots(1, 1, figsize=(10,8), subplot_kw=dict(polar=True))
        # set ax as a polar plot
        #ax.plt.subplot(1, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        return ax

    # def MkFig_HeatMap(correlation_matrix, labels):
    #     fig, ax = plt.subplots(1, 1, figsize=(10,8))
    #     sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5)
    #     return fig, ax

