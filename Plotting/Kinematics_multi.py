from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.stats import sem
import warnings
import scipy.interpolate as interpolate
import seaborn as sns
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


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

    def interpolate_columns(self, col, new_index, norm_idx):
        return np.interp(new_index, norm_idx, col.values)

    def process_data(self, args):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        con, mouseID, r, s, data, swing_only, SwSt = args
        try:
            if swing_only:
                start_idx = SwSt.loc(axis=0)[con, mouseID, r][np.logical_and(
                    np.any(SwSt.loc(axis=0)[con, mouseID, r].loc(axis=1)[['ForepawToeL', 'ForepawToeR'], 'StepCycle'] == 1, axis=1),
                    SwSt.loc(axis=0)[con, mouseID, r].loc(axis=1)['Stride_no'] == s)].index.get_level_values('FrameIdx')
            else:
                start_idx = data.loc(axis=0)[con, mouseID, r, s, 'stride'].index[0]

            end_idx = data.loc(axis=0)[con, mouseID, r, s, 'stride'].index[-1]
            zeroed = data.loc(axis=0)[con, mouseID, r, s, :].index.get_level_values('FrameIdx') - start_idx
            norm_idx = zeroed / (end_idx - start_idx) * 100
            new_index = np.linspace(-25, 125, 100) if not swing_only else np.linspace(0, 100, 50)
            interpolate_partial = partial(self.interpolate_columns, new_index=new_index, norm_idx=norm_idx)
            interpolated_data = data.loc(axis=0)[con, mouseID, r, s].apply(interpolate_partial, axis=0)
            interpolated_data = interpolated_data.set_index(pd.MultiIndex.from_product([[con], [mouseID], [r], [s], new_index],
                                                                    names=['Condition', 'MouseID', 'Run', 'Stride',
                                                                           'RelTime']))
            return interpolated_data
        except:
            pass

    def get_interp_data(self, data, swing_only=False):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        data = data.swaplevel('Buffer', 'FrameIdx')

        stride_info = self.get_stride_data()
        SwSt = pd.concat(stride_info, names=['Condition'])

        args_list = []
        for con in self.conditions:
            for mouseID in data.loc(axis=0)[con].index.get_level_values('MouseID').unique():
                for r in data.loc(axis=0)[con].index.get_level_values('Run').unique():
                    for s in data.loc(axis=0)[con].index.get_level_values('Stride').unique():
                        try:
                            args_list.append((con, mouseID, r, s, data, swing_only, SwSt))
                        except:
                            pass
        print("Interopolating data...")
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(pool.imap(self.process_data, args_list), total=len(args_list), desc="Processing", dynamic_ncols=True))

        filtered_series = [s for s in results if s is not None]
        big_series = pd.concat(filtered_series, axis=0, ignore_index=False)
        return big_series


class PlotKinematics(GetData):
    def __init__(self, conditions):
        super().__init__(conditions)
        self.ls = 14
        self.ts = 12
        self.stride_info = self.get_stride_data()

    #######################################################################################################################
    # Data Functions
    #######################################################################################################################

    def calculate_and_plot_allmeasures(self, buffer):
        all_measures_list =  measures_list(buffer)
        for measure in all_measures_list["multi_val_measure_list"].keys():
            print('##############################################################################################\n'
                  'Plotting measure: %s\n'
                  '##############################################################################################' % (measure))
            data = self.get_measure(measure)
            if measure != 'signed_angle':
                if measure != 'instantaneous_swing_velocity':
                    interp_data = self.get_interp_data(data, swing_only=False)
                    for param in itertools.product(*all_measures_list['multi_val_measure_list'][measure].values()):
                        param_names = list(all_measures_list['multi_val_measure_list'][measure].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                        if f"buffer_size:{buffer}" in formatted_params:
                            data_param = interp_data[formatted_params]
                            self.plot_byStride(data_param, measure, formatted_params, plottype='SingleMouse_byStride_byPhase')
                        else:
                            pass
                            ## there is no buffer, need a separate plotting function for this
                else:
                    interp_data = self.get_interp_data(data, swing_only=True)
                    for param in itertools.product(*all_measures_list['multi_val_measure_list'][measure].values()):
                        param_names = list(all_measures_list['multi_val_measure_list'][measure].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                        data_param = interp_data[formatted_params]
                        self.plot_InstVel(data_param, formatted_params, plottype='byStride_byPhase')
            else:
                interp_data = self.get_interp_data(data, swing_only=False)

                for param in all_measures_list['multi_val_measure_list'][measure].keys():
                    param_details = all_measures_list['multi_val_measure_list'][measure][param]
                    data_param = interp_data[param]
                    # if buffer is included
                    if param_details[-1] == buffer:
                        self.plot_byStride(data_param, 'signed_angle', param, plottype='SingleMouse_byStride_byPhase')
                    # if buffer is not included
                    elif param_details[-1] == 0:
                        self.plot_angles(data_param, param, plottype='SingleMouse_byStride_byPhase')
                        self.plot_angles(data_param, param, plottype='byStride_byPhase')


    #######################################################################################################################
    # Plotting Functions
    #######################################################################################################################
    def plot_InstVel(self, data, params, plottype):
        if plottype == 'byStride_byPhase':
            cmap = PlottingUtils.set_colormap('ExpPhase')
            colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
            for con in self.conditions:
                fig, ax = plt.subplots(4, 1, figsize=(10, 5))
                for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                    for sidx, stride in enumerate([-3, -2, -1, 0]):
                        try:
                            phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                            mean = data.loc(axis=0)[con, :, phase_runs, stride].groupby(['RelTime']).mean()
                            sem = data.loc(axis=0)[con, :, phase_runs, stride].groupby(['RelTime']).sem()
                            ax[sidx].plot(mean.index, mean.values, label=phase, color=colors[pidx])
                            ax[sidx].fill_between(mean.index, mean.values - sem.values, mean.values + sem.values, alpha=0.3, color=colors[pidx])
                            ax[sidx].set_xticks([])
                            ax[sidx].spines['top'].set_visible(False)
                            ax[sidx].spines['right'].set_visible(False)
                            ax[sidx].set_title(f"Stride {stride}", y=0.98, x=0.9, fontsize=self.ls, fontstyle='italic')
                        except:
                            pass
                ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ls)
                ax[-1].set_xlabel('% of swing', fontsize=self.ls)
                ax[-1].set_xticks([0, 25, 50, 75, 100])
                ax[-1].set_xticklabels(['0', '25', '50', '75', '100'], fontsize=self.ts)
                ax[1].set_ylabel('Instantaneous Swing Velocity (mm/s)', fontsize=self.ls, y=-0.25)

                fig.subplots_adjust(hspace=0.5)
                fig.subplots_adjust(left=0.1)
                fig.subplots_adjust(right=0.8)

                # check if the folder exists, if not create it
                if not os.path.exists(r"%s\Kinematics_multi\instantaneous_velocity" % (paths['plotting_destfolder'])):
                    os.makedirs(r"%s\Kinematics_multi\instantaneous_velocity" % (paths['plotting_destfolder']))
                con_form = str(con).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                con_form = con_form.replace(',', '-')
                filepath = PlottingUtils.remove_vowel("%s_%s_%s" % (params, con_form, plottype))
                filepath = filepath.replace(':', '=').replace(' ', '')

                plt.savefig(
                    r"%s\Kinematics_multi\instantaneous_velocity\%s.png" % (paths['plotting_destfolder'], filepath), format='png')
                plt.close(fig)


    def plot_byStride(self, data, measure, params, plottype):
        ###### data here is the interpolated data

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
                con_form = str(con).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                con_form = con_form.replace(',', '-')
                filepath = PlottingUtils.remove_vowel("%s_%s_%s_%s_byRun" % (measure, con_form, params, plottype))
                filepath = filepath.replace(':', '=').replace(' ', '')
                plt.savefig(
                    r"%s\Kinematics_multi\%s\%s.png" % (paths['plotting_destfolder'], measure, filepath), format='png')
                plt.close(fig)

        # if plottype == 'SingleMouse_byStride_byPhase_zscore':


    def plot_angles(self, data, param, plottype):
        cmaps = ["YlOrRd", "YlGn", "PuBu", "PuRd"]

        if plottype == 'SingleMouse_byStride_byPhase':
            for con in self.conditions:
                fig, ax = plt.subplots(4, 11, figsize=(35, 12), subplot_kw={'projection': 'polar'})
                Mouse = []
                for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                    for sidx, stride in enumerate([-3, -2, -1, 0]):
                        for midx, mouseID in enumerate(data.index.get_level_values('MouseID').unique()):
                            try:
                                phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                                data_rad = np.radians(data)
                                mean = data_rad.loc(axis=0)[con, mouseID, phase_runs, stride].groupby(['RelTime']).mean()
                                times_normalised = mean.index

                                ax[sidx, midx].scatter(mean.values, times_normalised, c=times_normalised, cmap=cmaps[pidx], s=3, edgecolor='none')

                                # plot a line connecting the points with changing colour
                                points = np.array([mean, times_normalised]).T.reshape(-1, 1, 2)
                                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                                lc = LineCollection(segments, cmap=cmaps[pidx], norm=plt.Normalize(0, 100))
                                lc.set_array(times_normalised)
                                lc.set_linewidth(2)
                                ax[sidx, midx].add_collection(lc)

                                ax[sidx, midx].set_theta_zero_location('S')  # Set 0 degrees to the top
                                ax[sidx, midx].set_theta_direction(-1)  # Set the direction to clockwise

                                if sidx == 0:
                                    ax[sidx, midx].set_title(mouseID, fontsize=self.ls, y=1.3)

                                Mouse.append(mouseID)
                            except:
                                print("Error plotting  mouse: %s, stride: %s, phase: %s" %(mouseID, stride, phase))
                                pass

                    # Add a color bar for each phase
                    cbar_ax = fig.add_axes(
                        [0.93 + pidx * 0.01, 0.1, 0.01, 0.73], label=f"cbar_{pidx}")  # Adjust the position of each color bar
                    cbar = plt.colorbar(lc, cax=cbar_ax)
                    cbar.set_ticks([])
                    if pidx == 3:
                        cbar.set_label('Percentage Stride', fontsize=self.ls)
                        cbar.set_ticks([0, 25, 50, 75, 100])
                    cbar_ax.set_title(phase, fontsize=self.ls, rotation=90)

                for sidx, s in enumerate([-3,-2,-1,0]):
                    ax[sidx, 0].set_ylabel(s, fontsize=self.ls, rotation=45)
                    ax[sidx, 0].yaxis.set_label_coords(-0.5, 0.5)
                fig.text(0.008, 0.89, 'Stride', va='center', rotation='horizontal', fontsize=self.ls, fontstyle='italic')

                fig.subplots_adjust(hspace=0.1)
                fig.subplots_adjust(wspace=0.4)
                fig.subplots_adjust(bottom=0.05)
                fig.subplots_adjust(left=0.05)
                fig.subplots_adjust(right=0.9)

                # check if the folder exists, if not create it
                if not os.path.exists(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], param)):
                    os.makedirs(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], param))
                con_form = str(con).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                con_form = con_form.replace(',', '-')
                filepath = PlottingUtils.remove_vowel("SignedAngleSingleMice_%s_%s_%s" % (con_form, param, plottype))
                filepath = filepath.replace(':', '=').replace(' ', '')
                plt.savefig(
                    r"%s\Kinematics_multi\signed_angle\%s.png" % (paths['plotting_destfolder'], filepath), format='png')
                plt.close(fig)

        if plottype == 'byStride_byPhase':
            shade_cmap = PlottingUtils.set_colormap('ExpPhase')
            shade_colors = sns.color_palette(shade_cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
            for con in self.conditions:
                fig, ax = plt.subplots(1, 4, figsize=(25, 10), subplot_kw={'projection': 'polar'})
                for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                    for sidx, stride in enumerate([-3, -2, -1, 0]):
                        phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                        data_rad = np.radians(data)
                        mean = data_rad.loc(axis=0)[con, :, phase_runs, stride].groupby(['RelTime']).mean()
                        sem = data_rad.loc(axis=0)[con, :, phase_runs, stride].groupby(['RelTime']).sem()
                        times_normalised = mean.index

                        ax[sidx].scatter(mean.values, times_normalised, c=times_normalised, cmap=cmaps[pidx], s=5, edgecolor='none')

                        # plot a line connecting the points with changing colour
                        points = np.array([mean, times_normalised]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = LineCollection(segments, cmap=cmaps[pidx], norm=plt.Normalize(0, 100))
                        lc.set_array(times_normalised)
                        lc.set_linewidth(2)
                        ax[sidx].add_collection(lc)

                        # add shaded area for SEM
                        ax[sidx].fill_betweenx(times_normalised, mean.values - sem.values, mean.values + sem.values,
                                               alpha=0.3, color=shade_colors[pidx])

                        ax[sidx].set_theta_zero_location('S')  # Set 0 degrees to the top
                        ax[sidx].set_theta_direction(-1)  # Set the direction to clockwise

                    cbar_ax = fig.add_axes(
                        [0.93 + pidx * 0.01, 0.1, 0.01, 0.73], label=f"cbar_{pidx}")
                    cbar = plt.colorbar(lc, cax=cbar_ax)
                    cbar.set_ticks([])
                    if pidx == 3:
                        cbar.set_label('Percentage Stride', fontsize=self.ls)
                        cbar.set_ticks([0, 25, 50, 75, 100])
                    cbar_ax.set_title(phase, fontsize=self.ls, rotation=90)

                    for sidx, s in enumerate([-3, -2, -1, 0]):
                        ax[sidx].set_title(s, fontsize=self.ls, y=1.2, fontstyle='italic')
                    fig.text(0.5, 0.87, 'Stride', va='center', rotation='horizontal', fontsize=self.ls, fontstyle='italic')

                # check if the folder exists, if not create it
                if not os.path.exists(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], param)):
                    os.makedirs(r"%s\Kinematics_multi\%s" % (paths['plotting_destfolder'], param))
                con_form = str(con).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                con_form = con_form.replace(',', '-')
                filepath = PlottingUtils.remove_vowel("SignedAngleMiceAverage_%s_%s_%s" % (con_form, param, plottype))
                filepath = filepath.replace(':', '=').replace(' ', '')
                plt.savefig(
                    r"%s\Kinematics_multi\signed_angle\%s.png" % (paths['plotting_destfolder'], filepath), format='png')
                plt.close(fig)


def main():
    LowHigh_days_conditions = ['APAChar_LowHigh_Repeats_Wash_Day1', 'APAChar_LowHigh_Repeats_Wash_Day2',
                               'APAChar_LowHigh_Repeats_Wash_Day3']
    plotting = PlotKinematics(LowHigh_days_conditions)
    plotting.calculate_and_plot_allmeasures(0.25)


if __name__ == '__main__':
    main()
    print("Finished saving plots!! Hope they look good (don't freak out if they don't though!) :)")