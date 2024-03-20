from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import seaborn as sns
import itertools
import warnings


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
        self.ls = 14
        self.ts = 12

#######################################################################################################################
# Data Functions
#######################################################################################################################

    def back_tail_height(self):
        for bodypart in ['tail', 'back']:
            data = self.get_measure(f"{bodypart}_height")
            print(f"Plotting {bodypart} height...............................................\n")
            self.plot_BackHeight(data, bodypart, 'SingleCon_byExpPhase_byStride', step_phase=0, full_stride=False)
            self.plot_BackHeight(data, bodypart, 'SingleCon_byExpPhase_byStride', step_phase=1, full_stride=False)
            self.plot_BackHeight(data, bodypart, 'SingleCon_byExpPhase_byStride', step_phase=None, full_stride=True)

    def walking_speed(self):
        data = self.get_measure('walking_speed')
        print(f"Plotting walking speed...............................................\n")
        for plot_type in ['AcrossDays_byStride', 'AllDays_byMouse']:
            for label in ['Tail1', 'Back6']:
                self.plot_WalkingSpeed_counts(data, plot_type, bodypart=label, speed_correct=True)
                self.plot_WalkingSpeed_counts(data, plot_type, bodypart=label, speed_correct=False)
        for param in ['bodypart:Back6, speed_correct:False', 'bodypart:Tail1, speed_correct:False']:
            for plottype in ['AcrossDays_AcrossPhase', 'AcrossPhase_AcrossDays']:
                self.plot_byStride(data, 'walking_speed', param, plottype)
            for plottype in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays']:
                self.plot_byRun(data, 'walking_speed', param, plottype)

    def no_param_measures(self):
        measures = ['stride_duration', 'stance_duration', 'swing_duration', 'cadence', 'duty_factor', 'double_support', 'coo_euclidean', 'swing_velocity',
                    'stride_length']
        for m in measures:
            data = self.get_measure(m)
            print(f"Plotting {m}...............................................\n")
            params = 'speed_correct:False' if m in measures[-2:] else 'no_param'
            for plot_type in ['AcrossDays_AcrossPhase', 'AcrossPhase_AcrossDays']:
                self.plot_byStride(data, m, params, plot_type)
            for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays']:
                self.plot_byRun(data, m, params, plot_type)
            for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays', 'All']:
                self.plot_bySpeed(data, m, params, plot_type)

    def param_measures(self, buffer):
        all_measures_list = measures_list(buffer)
        for measure in ['x', 'y', 'z', 'coo_xyz', 'bos_stancestart', 'ptp_amplitude_stride', 'body_distance',
                        'back_skew', 'limb_rel_to_body']:
            data = self.get_measure(measure)
            print(f"Plotting {measure}...............................................\n")
            for param in itertools.product(*all_measures_list['single_val_measure_list'][measure].values()):
                param_names = list(all_measures_list['single_val_measure_list'][measure].keys())
                formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                if f"buffer_size:{buffer}" not in formatted_params:
                    if "speed_correct:True" not in formatted_params:
                        if np.logical_or('full_stride' in formatted_params, 'step_phase' in formatted_params):
                            if np.logical_or(np.logical_and('full_stride:True' in formatted_params,
                                                            'step_phase:None' in formatted_params),
                                             np.logical_and('full_stride:False' in formatted_params,
                                                            'step_phase:None' not in formatted_params)):
                                for plot_type in ['AcrossDays_AcrossPhase', 'AcrossPhase_AcrossDays']:
                                    self.plot_byStride(data, measure, formatted_params, plot_type)
                                for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays']:
                                    self.plot_byRun(data, measure, formatted_params, plot_type)
                                for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays', 'All']:
                                    self.plot_bySpeed(data, measure, formatted_params, plot_type)
                        else:
                            for plot_type in ['AcrossDays_AcrossPhase', 'AcrossPhase_AcrossDays']:
                                self.plot_byStride(data, measure, formatted_params, plot_type)
                            for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays']:
                                self.plot_byRun(data, measure, formatted_params, plot_type)
                            for plot_type in ['AcrossDays_AcrossStride', 'AcrossStride_AcrossDays', 'All']:
                                self.plot_bySpeed(data, measure, formatted_params, plot_type)


#######################################################################################################################
# Plotting Functions
#######################################################################################################################

    def plot_WalkingSpeed_counts(self, alldata, plot_type, bodypart='Tail1', speed_correct=True, colormap='viridis'):
        param_name = "bodypart:%s, speed_correct:%s" %(bodypart, speed_correct)
        data = alldata.loc(axis=1)[param_name]
        cmap = plt.get_cmap(colormap)

        # remove square brackets from self.conditions to put into directory path as string
        conditions = str(self.conditions).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
        conditions = conditions.replace(',', '-')

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
                                con.split('_')[-1], count[-3], count[-2], count[-1], count[0], count[1]), pad=-10, y=0.8, x=0.1, loc='left',
                                fontsize=10)
            ax[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
            fig.subplots_adjust(right=0.87)
            fig.subplots_adjust(left=0.08)

            #check if the folder exists, if not create it
            if not os.path.exists(r"%s\Kinematics_single\walking_speed" % paths['plotting_destfolder']):
                os.makedirs(r"%s\Kinematics_single\walking_speed" % paths['plotting_destfolder'])
            filepath = PlottingUtils.remove_vowel("WalkingSpeedCounts_%s_speedcorr=%s_bodypart=%s_%s" %(conditions, speed_correct, bodypart, plot_type))
            plt.savefig(r"%s\Kinematics_single\walking_speed\%s.png" %
                        (paths['plotting_destfolder'], filepath), format='png')
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
            if not os.path.exists(r"%s\Kinematics_single\walking_speed" % paths['plotting_destfolder']):
                os.makedirs(r"%s\Kinematics_single\walking_speed" % paths['plotting_destfolder'])
            filepath = PlottingUtils.remove_vowel("WalkingSpeedCounts_%s_speedcorr=%s_bodypart=%s_%s" %(conditions, speed_correct, bodypart, plot_type))
            plt.savefig(r"%s\Kinematics_single\walking_speed\%s.png" % (
            paths['plotting_destfolder'], filepath), format='png')
            plt.close(fig)



    def plot_BackHeight(self, alldata, bodypart, plot_type, step_phase, full_stride, buffersize=0, all_vals=False):
        if bodypart == 'back':
            labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12']
            params = []
            for b in labels:
                param_name = "back_label:%s, step_phase:%s, all_vals:%s, full_stride:%s, buffer_size:%s" % (b, step_phase, all_vals, full_stride, buffersize)
                params.append(param_name)
            front = 'Nose'
            back = 'Tail'
        else:
            labels = ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12']
            params = []
            for b in labels:
                param_name = "tail_label:%s, step_phase:%s, all_vals:%s, full_stride:%s, buffer_size:%s" % (b, step_phase, all_vals, full_stride, buffersize)
                params.append(param_name)
            front = 'Base'
            back = 'Tip'
        data = alldata.loc(axis=1)[params]
        data.columns = labels

        if plot_type == 'SingleCon_byExpPhase_byStride':
            for con in self.conditions:
                colors = sns.color_palette("hls", len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
                grouped = data.loc(axis=0)[con].groupby('Stride')
                num_strides = len(data.loc(axis=0)[con].index.get_level_values('Stride').unique())
                mice = data.loc(axis=0)[con].index.get_level_values(level='MouseID').unique()
                fig, ax = plt.subplots(num_strides, 1, figsize=(10, 15))
                for stride, group in grouped:
                    sidx = stride + 3
                    for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                        phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                        phase_means_bymice = group.loc(axis=0)[mice, phase_runs].groupby('MouseID').mean()
                        phase_mean = phase_means_bymice.mean(axis=0)[::-1]
                        phase_sem = phase_means_bymice.sem(axis=0)[::-1]
                        ax[sidx] = PlottingUtils.MkFigs().MkFig_BackShape(ax[sidx], xlabel='',ylabel='')
                        ax[sidx].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.8)
                        ax[sidx].plot(np.arange(1, 13), phase_mean, label=phase, color=colors[pidx])
                        ax[sidx].fill_between(np.arange(1,13), phase_mean - phase_sem, phase_mean + phase_sem, alpha=0.3, color=colors[pidx])
                        if bodypart == 'back':
                            ax[sidx].set_ylim(18, 38)
                        elif bodypart == 'tail':
                            ax[sidx].set_ylim(10, 32)
                        ax[sidx].set_ylabel(u'Z (mm) \xb1sem ')
                        ax[sidx].set_title(f"Stride {stride}", y=0.98, x=0.9, fontsize=self.ls)
                    ax[-1].set_xlabel('%s label' %bodypart, fontsize=self.ls)
                    ax[-1].annotate(front, xytext=(0.85, -0.12), xy=(1, -0.12), xycoords='axes fraction', ha='center',
                                va='center',
                                fontstyle='italic', fontsize=self.ls, arrowprops=dict(arrowstyle='-|>'))
                    ax[-1].annotate(back, xytext=(0.15, -0.12), xy=(0, -0.12), xycoords='axes fraction', ha='center',
                                va='center',
                                fontstyle='italic', fontsize=self.ls, arrowprops=dict(arrowstyle='-|>'))
                ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
                fig.subplots_adjust(right=0.81)
                fig.subplots_adjust(left=0.09)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(bottom=0.05)

                # check if the folder exists, if not create it
                if not os.path.exists(r"%s\Kinematics_single\%s_height" % (paths['plotting_destfolder'], bodypart)):
                    os.makedirs(r"%s\Kinematics_single\%s_height" % (paths['plotting_destfolder'], bodypart))
                filepath = PlottingUtils.remove_vowel("%s_bodypart=%s_stepphase=%s_fullstride=%s_%s" %(con, bodypart, step_phase, full_stride, plot_type))
                plt.savefig(
                    r"%s\Kinematics_single\%s_height\%s.png" % (paths['plotting_destfolder'], bodypart, filepath),format='png')
                plt.close(fig)

    def plot_byStride(self, measuredata, measure, params, plottype):
        data = measuredata.loc(axis=1)[params]
        data_byStride = data.reset_index().pivot_table(index=['Condition', 'MouseID', 'Run'], columns='Stride',values=params)

        if plottype == 'AcrossDays_AcrossPhase':
            fig, ax = plt.subplots(3, 1, figsize=(10, 7))
            cmap = PlottingUtils.set_colormap('ExpPhase')
            colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
            mice = data_byStride.index.get_level_values('MouseID').unique()
            phase_means = []
            phase_sems = []
            for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                # calculate mean for each mouse and phase (ie mean across runs in that phase for each mouse and condition)
                mice_phase_mean = data_byStride.loc(axis=0)[self.conditions, mice, phase_runs].groupby(['Condition', 'MouseID']).mean()
                # calculate the mean/sem of all mice for each condition
                mice_mean = mice_phase_mean.groupby('Condition').mean()
                micesem = mice_phase_mean.groupby('Condition').sem()
                phase_means.append(mice_mean)
                phase_sems.append(micesem)
            phase_means_swap = pd.concat(phase_means, keys=expstuff['condition_exp_runs']['APACharRuns']['Short'].keys(), axis=0).swaplevel(0, 1)
            phase_sems_swap = pd.concat(phase_sems, keys=expstuff['condition_exp_runs']['APACharRuns']['Short'].keys(), axis=0).swaplevel(0, 1)

            for i, con in enumerate(self.conditions):
                ax[i] = PlottingUtils.MkFigs().MkFig_byStride(ax[i], ylabel='', xlabel='')
                for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                    ax[i].plot(phase_means_swap.loc(axis=0)[con, phase].index, phase_means_swap.loc(axis=0)[con, phase], label=phase, color=colors[pidx])
                    ax[i].fill_between(phase_means_swap.loc(axis=0)[con, phase].index, phase_means_swap.loc(axis=0)[con, phase] - phase_sems_swap.loc(axis=0)[con, phase], phase_means_swap.loc(axis=0)[con, phase] + phase_sems_swap.loc(axis=0)[con, phase], alpha=0.3, color=colors[pidx])
                    ax[i].set_title(con.split('_')[-1], fontsize=self.ls, y=0.95, x=0.9, fontstyle='italic')
            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
            ax[-1].set_xlabel('Stride', fontsize=self.ls)
            ax[1].set_ylabel(u'%s \xb1sem'%measure, y=0.5, fontsize=self.ls)

        if plottype == 'AcrossPhase_AcrossDays':
            mice = data.index.get_level_values(level='MouseID').unique()
            cmap = PlottingUtils.set_colormap('Day')
            colors = sns.color_palette(cmap, len(self.conditions))
            fig, ax = plt.subplots(4, 1, figsize=(10, 7))
            for pidx, phase in enumerate(expstuff['condition_exp_runs']['APACharRuns']['Short'].keys()):
                phase_runs = expstuff['condition_exp_runs']['APACharRuns']['Short'][phase]
                # calculate mean for each mouse and phase (ie mean across runs in that phase for each mouse and condition)
                mice_phase_mean = data_byStride.loc(axis=0)[self.conditions, mice, phase_runs].groupby(['Condition', 'MouseID']).mean()
                # calculate the mean/sem of all mice for each condition
                mice_mean = mice_phase_mean.groupby('Condition').mean()
                mice_sem = mice_phase_mean.groupby('Condition').sem()
                # plot
                ax[pidx] = PlottingUtils.MkFigs().MkFig_byStride(ax[pidx], ylabel='', xlabel='')
                for cidx, con in enumerate(self.conditions):
                    ax[pidx].plot(mice_mean.loc[con].index, mice_mean.loc[con], label=con.split('_')[-1], color=colors[cidx])
                    ax[pidx].fill_between(mice_mean.loc[con].index, mice_mean.loc[con] - mice_sem.loc[con], mice_mean.loc[con] + mice_sem.loc[con], alpha=0.3, color=colors[cidx])
                    ax[pidx].set_title(phase, fontsize=self.ls, y=0.95, x=0.9, fontstyle='italic')
            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
            ax[-1].set_xlabel('Stride', fontsize=self.ls)
            ax[1].set_ylabel(u'%s \xb1sem'%measure, y=-0.5, fontsize=self.ls)

        fig.subplots_adjust(right=0.85)
        fig.subplots_adjust(left=0.1)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(bottom=0.1)
        fig.subplots_adjust(hspace=0.5)

        # check if the folder exists, if not create it
        if not os.path.exists(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure)):
            os.makedirs(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure))

        conditions = str(self.conditions).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
        conditions = conditions.replace(',', '-')

        filepath = PlottingUtils.remove_vowel("%s_%s_%s_%s_byStride" %(measure, conditions, params, plottype))
        filepath = filepath.replace(':', '=').replace(' ', '')
        # delete spaces from the filepath
        filepath = filepath

        plt.savefig(
            r"%s\Kinematics_single\%s\%s.png" % (paths['plotting_destfolder'], measure, filepath), format='png')
        plt.close(fig)

    def plot_byRun(self, measuredata, measure, params, plottype):
        data = measuredata.loc(axis=1)[params]
        conditions = str(self.conditions).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
        conditions = conditions.replace(',', '-')

        if plottype == 'AcrossDays_AcrossStride':
            cmap = PlottingUtils.set_colormap('Stride')
            colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
            fig, ax = plt.subplots(len(self.conditions), 1, figsize=(10, 5))
            for i, con in enumerate(self.conditions):
                ax[i] = PlottingUtils.MkFigs().MkFig_TSRuns_with_ExpPhase_blocks(ax[i],xlabel='',xlabels_visible=False)
                for sidx, stride in enumerate(data.index.get_level_values(level='Stride').unique()):
                    mean = data.loc(axis=0)[con,:,:,stride].groupby('Run').mean()
                    sem = data.loc(axis=0)[con,:,:,stride].groupby('Run').sem()
                    ax[i].plot(mean.index.get_level_values('Run'), mean, label=stride, color=colors[sidx])
                    ax[i].fill_between(mean.index.get_level_values('Run'), mean - sem, mean + sem, alpha=0.3, color=colors[sidx])
                    ax[i].set_title(con.split('_')[-1], fontsize=self.ls, y=0.95, x=0.9, fontstyle='italic')
            ax[-1].set_xlabel('Run', fontsize=self.ls)
            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
            ax[1].set_ylabel(u'%s \xb1sem'%measure, y=0.5, fontsize=self.ls)

        if plottype == 'AcrossStride_AcrossDays':
            cmap = PlottingUtils.set_colormap('Day')
            colors = sns.color_palette(cmap, len(self.conditions))
            fig, ax = plt.subplots(4, 1, figsize=(10, 7))
            for sidx, stride in enumerate(data.index.get_level_values(level='Stride').unique()):
                ax[sidx] = PlottingUtils.MkFigs().MkFig_TSRuns_with_ExpPhase_blocks(ax[sidx],xlabel='',xlabels_visible=False)
                for i, con in enumerate(self.conditions):
                    mean = data.loc(axis=0)[con, :, :, stride].groupby('Run').mean()
                    sem = data.loc(axis=0)[con, :, :, stride].groupby('Run').sem()
                    ax[sidx].plot(mean.index.get_level_values('Run'), mean, label=con.split('_')[-1], color=colors[i])
                    ax[sidx].fill_between(mean.index.get_level_values('Run'), mean - sem, mean + sem, alpha=0.3, color=colors[i])
                    ax[sidx].set_title(f"Stride {stride}", fontsize=self.ls, y=0.95, x=0.9, fontstyle='italic')
            ax[-1].set_xlabel('Run', fontsize=self.ls)
            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
            ax[1].set_ylabel(u'%s \xb1sem' % measure, y=-0.5, fontsize=self.ls)

        fc_box = '0.9'
        lw_box = 0.1
        lw_arrow = 1
        ytext = -.9
        yarrow = -.55
        ax[-1].annotate('Baseline', xytext=(0.135, ytext), xy=(0.135, yarrow), xycoords='axes fraction',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
        ax[-1].annotate('APA', xytext=(0.5, ytext), xy=(0.5, yarrow), xycoords='axes fraction', ha='center',
                        va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=13, lengthB=1', lw=lw_arrow))
        ax[-1].annotate('Baseline', xytext=(0.865, ytext), xy=(0.865, yarrow), xycoords='axes fraction',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                        arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
        fig.subplots_adjust(right=0.85)
        fig.subplots_adjust(left=0.1)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(bottom=0.2)
        fig.subplots_adjust(hspace=0.5)

        # check if the folder exists, if not create it
        if not os.path.exists(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure)):
            os.makedirs(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure))
        filepath = PlottingUtils.remove_vowel("%s_%s_%s_%s_byRun" %(measure, conditions, params, plottype))
        filepath = filepath.replace(':', '=')
        plt.savefig(
            r"%s\Kinematics_single\%s\%s.png" % (paths['plotting_destfolder'], measure, filepath), format='png')
        plt.close(fig)

    def plot_bySpeed(self, measuredata, measure, params, plottype):
        data = measuredata.loc(axis=1)[params]
        speeddata = self.get_measure('walking_speed').loc(axis=1)['bodypart:Tail1, speed_correct:False']

        num_bins = 20
        bins = pd.cut(speeddata, bins=num_bins, labels=False)
        bin_edges = np.linspace(speeddata.min(), speeddata.max(), num_bins + 1)  # Calculate bin edges
        grouped_data = pd.DataFrame({'speed': speeddata.values, 'measure': data.values}, index=data.index)
        grouped_data['speed_bin'] = bins

        conditions = str(self.conditions).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
        #replace comma with dash
        conditions = conditions.replace(',', '-')

        if np.all(speeddata.index == data.index):
            if plottype == 'All':
                mean = grouped_data.groupby('speed_bin')['measure'].mean()
                sem = grouped_data.groupby('speed_bin')['measure'].sem()
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                ax = PlottingUtils.MkFigs().MkFig_bySpeed(ax, ylabel=u'%s \xb1sem' % measure)
                ax.plot(bin_edges[mean.index.astype(int)], mean, label=measure)
                ax.fill_between(bin_edges[mean.index.astype(int)], mean - sem, mean + sem, alpha=0.3)
                # ax.set_xticklabels([str(int(round(x))) for x in bin_edges[:-1]], fontsize=self.ts)

            if plottype == 'AcrossDays_AcrossStride':
                cmap = PlottingUtils.set_colormap('Stride')
                colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
                fig, ax = plt.subplots(len(self.conditions), 1, figsize=(6, 8))
                for i, con in enumerate(self.conditions):
                    ax[i] = PlottingUtils.MkFigs().MkFig_bySpeed(ax[i], xlabel='')
                    for sidx, stride in enumerate(data.index.get_level_values(level='Stride').unique()):
                        mean = grouped_data.loc(axis=0)[con, :, :, stride].groupby('speed_bin')['measure'].mean()
                        sem = grouped_data.loc(axis=0)[con, :, :, stride].groupby('speed_bin')['measure'].sem()
                        ax[i].plot(bin_edges[mean.index.astype(int)], mean, label=stride, color=colors[sidx])
                        ax[i].fill_between(bin_edges[mean.index.astype(int)], mean - sem, mean + sem, alpha=0.3, color=colors[sidx])
                        ax[i].set_title(con.split('_')[-1], fontsize=self.ts, y=0.95, x=0.9, fontstyle='italic')
                        # ax[sidx].set_xticklabels([str(int(round(x))) for x in bin_edges[:-1]], fontsize=self.ts)
                ax[-1].set_xlabel('Speed (mm/s)', fontsize=self.ls)
                ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
                ax[1].set_ylabel(u'%s \xb1sem' % measure, y=0.5, fontsize=self.ls)
                fig.subplots_adjust(right=0.82)
                fig.subplots_adjust(left=0.12)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(bottom=0.1)
                fig.subplots_adjust(hspace=0.3)

            if plottype == 'AcrossStride_AcrossDays':
                cmap = PlottingUtils.set_colormap('Day')
                colors = sns.color_palette(cmap, len(expstuff['condition_exp_runs']['APACharRuns']['Short']))
                fig, ax = plt.subplots(4, 1, figsize=(6, 10))
                for sidx, stride in enumerate(data.index.get_level_values(level='Stride').unique()):
                    ax[sidx] = PlottingUtils.MkFigs().MkFig_bySpeed(ax[sidx], xlabel='')
                    for i, con in enumerate(self.conditions):
                        mean = grouped_data.loc(axis=0)[con, :, :, stride].groupby('speed_bin')['measure'].mean()
                        sem = grouped_data.loc(axis=0)[con, :, :, stride].groupby('speed_bin')['measure'].sem()
                        #ax[sidx].plot(mean.index, mean, label=con.split('_')[-1], color=colors[i])
                        ax[sidx].plot(bin_edges[mean.index.astype(int)], mean, label=con.split('_')[-1], color=colors[i])
                        ax[sidx].fill_between(bin_edges[mean.index.astype(int)], mean - sem, mean + sem, alpha=0.3, color=colors[i])
                        ax[sidx].set_title(f"Stride {stride}", fontsize=self.ts, y=0.95, x=0.9, fontstyle='italic')
                       # ax[sidx].set_xticks(np.arange(0,num_bins,1))
                       # ax[sidx].set_xticklabels([str(int(round(x))) for x in bin_edges[:-1]], fontsize=self.ts)
                ax[-1].set_xlabel('Speed (mm/s)', fontsize=self.ls)
                ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), fontsize=self.ts)
                ax[1].set_ylabel(u'%s \xb1sem' % measure, y=-0.5, fontsize=self.ls)
                fig.subplots_adjust(right=0.8)
                fig.subplots_adjust(left=0.12)
                fig.subplots_adjust(top=0.97)
                fig.subplots_adjust(bottom=0.07)
                fig.subplots_adjust(hspace=0.3)

            # check if the folder exists, if not create it
            if not os.path.exists(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure)):
                os.makedirs(r"%s\Kinematics_single\%s" % (paths['plotting_destfolder'], measure))
            filepath = PlottingUtils.remove_vowel("%s_%s_%s_%s_bySpeed" %(measure, conditions, params, plottype))
            filepath = filepath.replace(':', '=')
            plt.savefig(
                r"%s\Kinematics_single\%s\%s.png" % (paths['plotting_destfolder'], measure,filepath), format='png')
            plt.close(fig)
        else:
            raise ValueError('The index of the speed data does not match the index of the measure data')



def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    LowHigh_days_conditions = ['APAChar_LowHigh_Repeats_Wash_Day1', 'APAChar_LowHigh_Repeats_Wash_Day2',
                               'APAChar_LowHigh_Repeats_Wash_Day3']
    plotting = PlotKinematics(LowHigh_days_conditions)

    # Run plots
    plotting.back_tail_height()
    plotting.walking_speed()
    plotting.no_param_measures()
    plotting.param_measures(buffer=0.25)

if __name__ == '__main__':
    main()
    print("Finished saving plots!! Hope they look good (don't freak out if they don't though!) :)")