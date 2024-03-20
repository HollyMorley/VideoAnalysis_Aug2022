from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import warnings
import scipy.interpolate as interpolate
import seaborn as sns



class GetData():
    def __init__(self, conditions):
        self.conditions = conditions

    def load_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'multi_kinematics_runXstride')
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
        result.set_index(['Condition', 'MouseID', 'Run', 'Stride', 'FrameIdx', 'Buffer'], inplace=True)
        return result.loc(axis=1)[measure]

    def get_stride_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'StrideInfo')
        for con in self.conditions:
            data[con] = PlottingUtils.remove_prepruns(data, con)
        return data

    def get_interp_data(self, data):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        data = data.swaplevel('Buffer', 'FrameIdx')

        big_list =[]
        # cnt = 0
        for con in self.conditions:
            for mouseID in data[con].index.get_level_values('MouseID').unique():
                for r in data[con].index.get_level_values('Run').unique():
                    for s in data[con].index.get_level_values('Stride').unique():
                        try:
                            start_idx = data.loc(axis=0)[con,mouseID,r,s,'stride'].index[0]
                            end_idx = data.loc(axis=0)[con,mouseID,r,s,'stride'].index[-1]
                            zeroed = data.loc(axis=0)[con,mouseID,r,s,:].index.get_level_values('FrameIdx') - start_idx
                            norm_idx = zeroed / (end_idx - start_idx) * 100
                            new_index = np.linspace(-25, 125, 100)
                            new_vals = np.interp(new_index, norm_idx, data.loc(axis=0)[con,mouseID,r,s].values)
                            interp_ser = pd.Series(data=new_vals, index=pd.MultiIndex.from_product([[con], [mouseID], [r], [s], new_index], names=['Condition', 'MouseID', 'Run', 'Stride', 'RelTime']))
                            big_list.append(interp_ser)
                            # cnt += len(data.loc(axis=0)[con,mouseID,r,s])
                        except:
                            pass
        big_df = pd.concat(big_list)

        return big_df


class PlotKinematics(GetData):
    def __init__(self, conditions):
        super().__init__(conditions)
        self.ls = 14
        self.ts = 12
        #self.stride_info = self.get_stride_data()

    # def getData(self):
    #     data = self.get_measure('traj')
    #     data = self.get_norm_index_values(data)
    #     return data



    def plot_byStride(self, data, measure, params, plottype):
        # param = "bodypart:%s, coord:%s, step_phase:%s, full_stride:%s, speed_correct:%s, aligned:%s, buffer_size:%s" \
        #              %(bodypart, coord, step_phase, full_stride, speed_correct, aligned, buffer_size)

        if plottype == 'SingleMouse_byStride_byPhase':
            cmap = PlottingUtils.set_colormap('ExpPhase')
            colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
            for con in self.conditions:
                fig, ax = plt.subplots(11, 4, figsize=(15, 10))
                Mouse = []
                for sidx, stride in enumerate([-3,-2,-1,0]):
                    for midx, mouseID in enumerate(data.index.get_level_values('MouseID').unique()):
                        ax[midx, sidx] = PlottingUtils.MkFigs().MkFig_TSStride(ax=ax[midx,sidx], xlabel='')
                        for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                            try:
                                phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                                mean = data.loc(axis=0)[con, mouseID, phase_runs, stride].groupby(['RelTime']).mean()
                                sem = data.loc(axis=0)[con, mouseID, phase_runs, stride].groupby(['RelTime']).sem()
                                ax[midx, sidx].plot(mean.index, mean.values, label=phase, color=colors[pidx])
                                ax[midx, sidx].fill_between(mean.index, mean.values - sem.values, mean.values + sem.values, alpha=0.3, color=colors[pidx])
                                #ax[midx, sidx].set_xlabel('% of stride')
                                Mouse.append(mouseID)
                            except:
                                pass
                # format the plot
                for midx, m in enumerate(data.index.get_level_values('MouseID').unique()):
                    ax[midx, 0].set_ylabel(m[-3:], fontsize=10, rotation=45)
                    ax[midx, 0].yaxis.set_label_coords(-0.25, 0.5)
                for sidx, s in enumerate([-3,-2,-1,0]):
                    ax[0, sidx].set_title(s, fontsize=self.ls)
                ax[0, 3].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
                fig.text(0.06, 0.9, 'Mouse', va='center', rotation='horizontal', fontsize=self.ls, fontstyle='italic')
                fig.text(0.05, 0.5, u'%s \xb1sem'%measure, va='center', rotation='vertical', fontsize=self.ls, fontstyle='italic')
                fig.text(0.5, 0.05, '% of stride', ha='center', fontsize=self.ls)
                fig.text(0.5, 0.93, 'Stride', ha='center', fontsize=self.ls)
                fig.subplots_adjust(hspace=0.5)

                # check if the folder exists, if not create it
                if not os.path.exists(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], measure)):
                    os.makedirs(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], measure))
                filepath = PlottingUtils.remove_vowel("%s_%s_%s_%s_byRun" % (measure, con, params, plottype))
                filepath = filepath.replace(':', '=')
                plt.savefig(
                    r"%s\Kinematics_multi\%s\%s.png" % (paths['plotting_destfolder'], measure, filepath), format='png')
                plt.close(fig)

        # if plottype == 'SingleMouse_byStride_byPhase_zscore':

    #
    def plot_angles(self, alldata, param, plottype):
        ###########
        # temp:
        measures = measures_list(0.25)
        param_titles = list(measures['multi_val_measure_list']['signed_angle'].keys())
        param = param_titles[2]
        data = alldata[param]

        if plottype == 'SingleMouse_byStride_byPhase':
            for con in self.conditions:
                fig, ax = plt.subplots(11, 4, figsize=(15, 10))
                Mouse = []
                for sidx, stride in enumerate([-3, -2, -1, 0]):
                    for midx, mouseID in enumerate(data.index.get_level_values('MouseID').unique()):
                        ax[midx, sidx] = PlottingUtils.MkFigs().MkFig_PolarbyTime(ax=ax[midx, sidx])
                        for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                            try:
                                phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                                mean = data.loc(axis=0)[con, mouseID, phase_runs, stride].groupby(['RelTime']).mean()
                            except:
                                pass




#
# def plot_angles(times, angles_deg):
#     """
#         Plots the given angles over time in polar coordinates.
#
#         Parameters:
#             angles_deg (np.array): An array of angles in degrees.
#             times (np.array): An array of the same length as angles_deg representing time.
#         """
#     # Convert angles from degrees to radians
#     # Convert angles from degrees to radians
#     angles_rad = np.radians(angles_deg)
#
#     # Normalize times to have a clear representation
#     times_normalized = (times - times.min()) / (times.max() - times.min())
#
#     # Create a polar plot
#     plt.figure(figsize=(10, 8))
#     ax = plt.subplot(111, polar=True)
#
#     # Plot the angles over time with a color gradient
#     # Use a colormap to represent time; e.g., 'viridis', 'plasma', 'inferno', 'magma'
#     colormap = plt.cm.viridis
#     ax.scatter(angles_rad, times_normalized, c=times_normalized, cmap=colormap, s=10, edgecolor='none')
#
#     # Plot a line connecting the points with changing color
#     points = np.array([angles_rad, times_normalized]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, cmap=colormap, norm=plt.Normalize(0, 1))
#     lc.set_array(times_normalized)
#     lc.set_linewidth(2)
#     ax.add_collection(lc)
#
#     # Customize the plot
#     ax.set_theta_zero_location('N')  # Set 0 degrees to the top
#     ax.set_theta_direction(-1)  # Set the direction to clockwise
#     ax.set_title('Angles Over Time in Polar Coordinates')
#
#     # Add a color bar to indicate time
#     cbar = plt.colorbar(lc, pad=0.1)
#     cbar.set_label('Normalized Time')
#
#     plt.show()
#






def main():
    LowHigh_days_conditions = ['APAChar_LowHigh_Repeats_Wash_Day1', 'APAChar_LowHigh_Repeats_Wash_Day2',
                               'APAChar_LowHigh_Repeats_Wash_Day3']
    plotting = PlotKinematics(LowHigh_days_conditions)
    # for m in measure:
    #     data = plotting.get_measure(m)
    #     plotting.plot_trajectories(data)


if __name__ == '__main__':
    main()
    print("Finished saving plots!! Hope they look good (don't freak out if they don't though!) :)")