from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem


class GetData():
    def __init__(self, conditions):
        self.conditions = conditions

    def load_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'single_kinematics_runXstride')
        for con in self.conditions:
            data[con] = PlottingUtils.remove_prepruns(data, con)
        return data

    def get_measure(self, measure):
        data = self.load_data()
        data_frames = []
        for condition, df in data.items():
            df['Condition'] = condition  # Adding a new column for condition info
            data_frames.append(df)
        result = pd.concat(data_frames)
        result.reset_index(inplace=True)
        result.set_index(['Condition', 'MouseID', 'Run', 'Stride'], inplace=True)
        return result.loc(axis=1)[measure]


class PlotKinematics(GetData):
    def __init__(self, conditions):
        super().__init__(conditions)

    def plot_WalkingSpeed_counts(self, plot_type, bodypart='Tail1', speed_correct=True, colormap='viridis'):
        param_name = "bodypart:%s, speed_correct:%s" %(bodypart, speed_correct)
        data = self.get_measure('walking_speed').loc(axis=1)[param_name]
        # plot a histogram of all the walking speeds
        cmap = plt.get_cmap(colormap)

        if plot_type == 'AcrossDays_byStride':
            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            for i, con in enumerate(self.conditions):
                colors = cmap(np.linspace(0, 1, len(data.loc(axis=0)[con].index.get_level_values(level='Stride').unique())))
                grouped = data.loc(axis=0)[con].groupby('Stride')
                count = dict.fromkeys([-3, -2, -1, 0, 1])
                for stride, group in grouped:
                    ax[i].hist(group.values, bins=20, alpha=0.5, label=f'Stride {stride}', color=colors[stride-1])
                    ax[i].set_ylim(0, 60)
                    ax[i].set_xlim(-50, 1100)
                    ax[i].set_ylabel('Count')
                    ax[i].set_xlabel('Walking Speed (mm/s)')
                    ax[i].spines['top'].set_visible(False)
                    ax[i].spines['right'].set_visible(False)
                    count[stride] = len(group)
                ax[i].set_title('%s\n---------------\nTotal stride counts:\n'
                                '-3: %s\n-2: %s\n-1: %s\n0: %s\n1: %s' % (
                                con.split('_')[-1], [-3], count[-2], count[-1], count[0], count[1]), pad=-10, y=0.9, x=0.1, loc='left',
                                fontsize=10)
            ax[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
            fig.subplots_adjust(right=0.87)
            fig.subplots_adjust(left=0.08)

            #check if the folder exists, if not create it
            if not os.path.exists(r"%s\Kinematics_single\WalkingSpeed" % paths['plotting_destfolder']):
                os.makedirs(r"%s\Kinematics_single\WalkingSpeedCounts" % paths['plotting_destfolder'])
            plt.savefig(r"%s\Kinematics_single\WalkingSpeed\WalkingSpeed_%s_%s.png" % (paths['plotting_destfolder'], self.conditions, plot_type), format='png')
            plt.close(fig)

        elif plot_type == 'AllDays_byMouse':
            num_mice = len(data.index.get_level_values(level='MouseID').unique())
            fig, ax = plt.subplots(num_mice, 1, figsize=(7, 15))
            for midx, mouse in enumerate(data.index.get_level_values(level='MouseID').unique()):
                mdata = data.xs(mouse, level='MouseID')
                colors = cmap(np.linspace(0, 1, len(mdata.index.get_level_values(level='Stride').unique())))
                grouped = mdata.groupby('Stride')
                count = dict.fromkeys([-3, -2, -1, 0, 1])
                for stride, group in grouped:
                    ax[midx].hist(group.values, bins=20, alpha=0.5, label=f'Stride {stride}', color=colors[stride - 1])
                    ax[midx].set_ylim(0, 25)
                    ax[midx].set_xlim(-50, 1100)
                    ax[midx].set_ylabel('Count')
                    ax[midx].set_xlabel('Walking Speed (mm/s)')
                    ax[midx].spines['top'].set_visible(False)
                    ax[midx].spines['right'].set_visible(False)
                    count[stride] = len(group)
                ax[midx].set_title('Mouse %s\nTotal stride counts:\n'
                                   '-3: %s\n-2: %s\n-1: %s\n0: %s\n1: %s' % (
                                   mouse, count[-3], count[-2], count[-1], count[0], count[1]),
                                   pad=-10, y=0.3, x=0.1, loc='left', fontsize=8)
            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
            fig.subplots_adjust(right=0.81)
            fig.subplots_adjust(left=0.09)
            fig.subplots_adjust(top=0.98)
            fig.subplots_adjust(bottom=0.05)

            # check if the folder exists, if not create it
            if not os.path.exists(r"%s\Kinematics_single\WalkingSpeed" % paths['plotting_destfolder']):
                os.makedirs(r"%s\Kinematics_single\WalkingSpeed" % paths['plotting_destfolder'])
            plt.savefig(r"%s\Kinematics_single\WalkingSpeed\WalkingSpeedCounts_%s_%s.png" % (
            paths['plotting_destfolder'], self.conditions, plot_type), format='png')
            plt.close(fig)





