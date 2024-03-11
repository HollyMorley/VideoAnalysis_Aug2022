from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder


class GetData():
    def __init__(self, conditions):
        self.conditions = conditions

    def load_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'behaviour_run')
        for con in self.conditions:
            data[con] = PlottingUtils.remove_prepruns(data, con)
        return data

    def get_measure(self, measure):
        data = self.load_data()
        measure_all = []
        for con in self.conditions:
            for mouseID in data[con].index.get_level_values(level=0).unique():
                measure_data = data[con].loc(axis=0)[mouseID].loc(axis=1)[measure]
                columns = pd.MultiIndex.from_product([[con], [mouseID]], names=['Condition', 'MouseID'])
                measure_df = pd.DataFrame(measure_data.values, columns=columns, index=measure_data.index)
                measure_all.append(measure_df)
        all_data_points = pd.concat(measure_all, axis=1).sort_index()
        return all_data_points

class PlotBehaviour(GetData):
    def __init__(self, conditions):
        super().__init__(conditions)
        # self.data = GetData.load_data(conditions)

    def wait_time_bar_line(self, plot_type, zoom=False, var='sem', cmap_name='Blues'):
        """
        Plot the wait time data as a bar plot or line plot
        :param plot_type: 'mice_across_days' or 'all_mice_across_runs_across_days'
        :param var:
        :param cmap_name:
        :return:
        """
        all_data_points = self.get_measure('wait_time')
        cmap = plt.get_cmap(cmap_name)

        if plot_type == 'mice_across_days':
            ### mean of ALL data points across runs
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            vals = all_data_points.mean(axis=1)
            vars = None
            if var == 'sem':
                vars = pd.Series(index=vals.index, data=np.ma.filled(sem(all_data_points.astype(float), axis=1, nan_policy='omit'), np.nan))
            elif var == 'std':
                vars = all_data_points.std(axis=1)
            ax = PlottingUtils.MkFigs().MkFig_TSRuns_with_ExpPhase_blocks(ax=ax)
            ax.bar(vals.index, vals.values, label=vals.name, color=cmap(0.6))
            ax.errorbar(vals.index, vals.values, yerr=vars.values, fmt='none', ecolor='black', elinewidth=1, capsize=2)
            ax.set_ylabel('Wait time (s)', fontsize=14)
            plt.savefig(r"%s\Behaviour\Wait_times\WaitTime_%s_%s.png" % (paths['plotting_destfolder'], self.conditions, plot_type), format='png')
            plt.close(fig)

        elif plot_type == 'all_mice_across_runs_across_days':
            ### single traces across runs for each mouse, with a plot for each day
            cmap = plt.get_cmap('rainbow')
            fig, ax = plt.subplots(3, 1, figsize=(10,10))
            for i, con in enumerate(self.conditions):
                vals = all_data_points[con]
                ax[i] = PlottingUtils.MkFigs().MkFig_TSRuns_with_ExpPhase_blocks(ax=ax[i], xlabels_visible=False)
                for mouse, midx in enumerate(vals.columns.get_level_values(level='MouseID').unique()):
                    ax[i].plot(vals.index, vals[midx], color=cmap(np.linspace(0, 1, len(vals.columns.get_level_values(level='MouseID').unique()))[mouse]), label="m%s" % midx[-3:])
                    ax[i].set_ylim(0, 10) if zoom else ax[i].set_ylim(0, 50)
                    if i == len(self.conditions) - 1:
                        fc_box = '0.9'
                        lw_box = 0.1
                        lw_arrow = 1
                        text_y_pos = -0.45
                        arrow_y_pos = -0.25
                        ax[i].annotate('Baseline', xytext=(0.135, text_y_pos), xy=(0.135, arrow_y_pos), xycoords='axes fraction',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                                   arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
                        ax[i].annotate('APA', xytext=(0.5, text_y_pos), xy=(0.5, arrow_y_pos), xycoords='axes fraction', ha='center',
                                   va='center',
                                   bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                                   arrowprops=dict(arrowstyle='-[, widthB=13, lengthB=1', lw=lw_arrow))
                        ax[i].annotate('Baseline', xytext=(0.865, text_y_pos), xy=(0.865, arrow_y_pos), xycoords='axes fraction',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='square', fc=fc_box, lw=lw_box),
                                   arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=lw_arrow))
                        ax[i].set_xlabel('Run', fontsize=14)
                    ax[i].set_title('%s' % con.split('_')[-1], fontsize=14, y=0.9)
                    ax[i].set_ylabel('Wait time (s)', fontsize=14)

            ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
            fig.subplots_adjust(right=0.89)
            fig.subplots_adjust(left=0.1)

            plt.savefig(r"%s\Behaviour\Wait_times\WaitTime_%s_%s.png" % (paths['plotting_destfolder'], self.conditions, plot_type), format='png')
            plt.close(fig)



    def paw_pref(self, plot_type, pref_type):
        """
        plot_type: 'mice_across_days', 'all_mice_across_runs_singlecon'
        pref_type: 'start', 'transition'
        """
        pref = None
        if pref_type == 'start':
            pref = self.get_measure('start_paw_pref')
        elif pref_type == 'trans':
            pref = self.get_measure('trans_paw_pref')

        if plot_type == 'mice_across_days':
            # find the percentage of right paw starts for each mouse on each day
            start_pref_paw = pref.replace({1: 'ForePawToeL', 2: 'ForePawToeR'})
            paw_counts = start_pref_paw.apply(pd.Series.value_counts, axis=0)
            paw_percentage = paw_counts.div(paw_counts.sum(axis=0), axis=1) * 100

            # plot the percentage of right paw starts for each mouse on each day
            cmap = plt.get_cmap('rainbow')
            r_percent = paw_percentage.T['ForePawToeR'].reset_index().rename(columns={0: 'Value'}).pivot(index='MouseID', columns='Condition', values='ForePawToeR')
            fig,ax = plt.subplots(1,1,figsize=(6,8))
            for midx, mouse in enumerate(r_percent.index):
                # plot line for each mouse across the three days with a different colour for each mouse
                ax.plot(np.linspace(1,3,3), r_percent.loc[mouse].values, color=cmap(np.linspace(0,1,len(r_percent.index))[midx]), label="m%s"%mouse[-3:])
            ax.set_ylim(0,100)
            if pref_type == 'start':
                ax.set_ylabel('% runs initiated with right paw', fontsize=14)
            elif pref_type == 'trans':
                ax.set_ylabel('% belt transitions with right paw', fontsize=14)
            ax.set_xticks(np.linspace(1,3,3))
            ax.set_xticklabels(['Day1', 'Day2', 'Day3'], fontsize=12)
            ax.set_yticks(np.arange(0,101,10))
            ax.set_yticklabels(np.arange(0,101,10), fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', borderaxespad=0., fontsize=12)
            fig.subplots_adjust(left=0.15)
            fig.subplots_adjust(right=0.7)

            plt.savefig(r"%s\Behaviour\Paw_preferences\PawPref_%s_%s_%s.png"%(paths['plotting_destfolder'], pref_type, self.conditions, plot_type), format='png')
            plt.close(fig)

        elif plot_type == 'all_mice_across_runs_singlecon':
            # for each condition, plot the left right paw preference for each mouse across all runs
            for con in self.conditions:
                start_pref_singlecon = pref[con]
                fig, ax = plt.subplots(1, len(start_pref_singlecon.columns), figsize=(15, 10))
                cmap = plt.get_cmap('rainbow')
                for midx, mouse in enumerate(start_pref_singlecon.columns):
                    ax[midx] = PlottingUtils.MkFigs().MkFig_PawPref(ax[midx])
                    ax[midx].plot(start_pref_singlecon[mouse].values, start_pref_singlecon[mouse].index, color=cmap(np.linspace(0, 1, len(start_pref_singlecon.columns))[midx]))
                    ax[midx].scatter(start_pref_singlecon[mouse].values, start_pref_singlecon[mouse].index, color=cmap(np.linspace(0, 1, len(start_pref_singlecon.columns))[midx]))
                    ax[midx].axhline(y=10.5, color='black', linestyle='--', alpha=0.3)
                    ax[midx].axhline(y=30.5, color='black', linestyle='--', alpha=0.3)
                    ax[midx].set_title("%s"%mouse[-3:], fontsize=14)
                    ax[midx].set_xticks([1, 2])
                    ax[midx].set_xticklabels(['L', 'R'], fontsize=14)
                    ax[midx].set_ylim(0,41)
                    ax[midx].set_yticks(np.arange(0,41,5))
                    ax[midx].set_yticklabels(np.arange(0,41,5), fontsize=12)
                ax[0].set_ylabel('Run', fontsize=16)
                if pref_type == 'start':
                    fig.text(0.5, 0.05, 'Paw preference at run initiation', ha='center', fontsize=16)
                elif pref_type == 'trans':
                    fig.text(0.5, 0.05, 'Paw preference at belt transition', ha='center', fontsize=16)
                fig.subplots_adjust(left=0.05)
                fig.subplots_adjust(right=0.95)
                fig.subplots_adjust(wspace=0.2)
                fig.suptitle(con, fontsize=16)

                plt.savefig(r"%s\Behaviour\Paw_preferences\PawPref_%s_%s_%s.png"%(paths['plotting_destfolder'], pref_type, con, plot_type), format='png')
                plt.close(fig)

    def paw_matching(self, plot_type):
        """
        plot_type:
        """
        matching = self.get_measure('start_to_trans_paw_matching')

        if plot_type == 'mice_across_days':
            matching_words = matching.replace({0: 'Mismatch', 1: 'Match'})
            matching_counts = matching_words.apply(pd.Series.value_counts, axis=0)
            matching_percentage = matching_counts.div(matching_counts.sum(axis=0), axis=1) * 100

            cmap = plt.get_cmap('rainbow')
            match_percent = matching_percentage.T['Match'].reset_index().rename(columns={0: 'Value'}).pivot(index='MouseID', columns='Condition', values='Match')
            fig,ax = plt.subplots(1,1,figsize=(6,8))
            for midx, mouse in enumerate(match_percent.index):
                # plot line for each mouse across the three days with a different colour for each mouse
                ax.plot(np.linspace(1,3,3), match_percent.loc[mouse].values, color=cmap(np.linspace(0,1,len(match_percent.index))[midx]), label="m%s"%mouse[-3:])
            ax.set_ylim(0,100)
            ax.set_ylabel('% runs with matching paw', fontsize=16)
            ax.set_xticks(np.linspace(1,3,3))
            ax.set_xticklabels(['Day1', 'Day2', 'Day3'], fontsize=14)
            ax.set_yticks(np.arange(0,101,10))
            ax.set_yticklabels(np.arange(0,101,10), fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', borderaxespad=0., fontsize=12)
            fig.subplots_adjust(left=0.15)
            fig.subplots_adjust(right=0.7)

            plt.savefig(r"%s\Behaviour\Paw_preferences\PawMatching_%s_%s.png"%(paths['plotting_destfolder'], self.conditions, plot_type), format='png')
            plt.close(fig)

        if plot_type == 'all_mice_across_runs_singlecon':
            # for each condition, plot the paw matching for each mouse across all runs
            for con in self.conditions:
                matching_singlecon = matching[con]
                fig, ax = plt.subplots(1, len(matching_singlecon.columns), figsize=(15, 10))
                cmap = plt.get_cmap('rainbow')
                for midx, mouse in enumerate(matching_singlecon.columns):
                    ax[midx] = PlottingUtils.MkFigs().MkFig_PawPref(ax[midx])
                    ax[midx].plot(matching_singlecon[mouse].values, matching_singlecon[mouse].index, color=cmap(np.linspace(0, 1, len(matching_singlecon.columns))[midx]))
                    ax[midx].scatter(matching_singlecon[mouse].values, matching_singlecon[mouse].index, color=cmap(np.linspace(0, 1, len(matching_singlecon.columns))[midx]))
                    ax[midx].axhline(y=10.5, color='black', linestyle='--', alpha=0.3)
                    ax[midx].axhline(y=30.5, color='black', linestyle='--', alpha=0.3)
                    ax[midx].set_title("%s"%mouse[-3:], fontsize=14)
                    ax[midx].set_xticks([0,1])
                    ax[midx].set_xlim(-0.5,1.5)
                    ax[midx].set_xticklabels(['Mismatch', 'Match'], fontsize=14, rotation=45)
                    ax[midx].set_ylim(0,41)
                    ax[midx].set_yticks(np.arange(0,41,5))
                    ax[midx].set_yticklabels(np.arange(0,41,5), fontsize=12)
                    # met x tick labels diagonal
                ax[0].set_ylabel('Run', fontsize=16)
                fig.text(0.5, 0.04, 'Run initiation to transition paw syncing', ha='center', fontsize=16)
                fig.subplots_adjust(left=0.05)
                fig.subplots_adjust(right=0.95)
                fig.subplots_adjust(bottom=0.18)
                fig.subplots_adjust(wspace=0.2)
                fig.suptitle(con, fontsize=16)

                plt.savefig(r"%s\Behaviour\Paw_preferences\PawMatching_%s_%s.png"%(paths['plotting_destfolder'], con, plot_type), format='png')
                plt.close(fig)

    def paw_pref_mutual_info(self):
        """
        MI defined as We define the MI as the relative entropy between the joint distribution of the two variables and the product of their marginal distributions.
        When the MI is 0, then knowing the values of x does not tells us anything about y, and vice versa, that is knowing y, does not tell us anything about x.
        plot_type:
        """
        start_pref = self.get_measure('start_paw_pref')
        transition_pref = self.get_measure('trans_paw_pref')
        for con in self.conditions:
            start_pref_con = start_pref[con]
            transition_pref_con = transition_pref[con]
            mi_values = {}
            for mouse_id in start_pref_con.columns:
                start_pref_con_mouse = start_pref_con[mouse_id].dropna()
                transition_pref_con_mouse = transition_pref_con[mouse_id].dropna()
                mi = mutual_info_score(start_pref_con_mouse, transition_pref_con_mouse)
                mi_values[mouse_id] = mi
            shortened_name_mi = {key[-3:]: value for key, value in mi_values.items()}
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.bar(shortened_name_mi.keys(), shortened_name_mi.values())
            ax.set_ylim(0, 0.1)
            ax.set_xticks(range(len(shortened_name_mi)))
            ax.set_xticklabels(shortened_name_mi.keys() ,fontsize=12, rotation=45)
            ax.set_xlabel('Mouse', fontsize=14)
            ax.set_ylabel('Mutual Information', fontsize=14)
            ax.set_title(con, fontsize=14)
            fig.subplots_adjust(left=0.17)
            fig.subplots_adjust(bottom=0.15)

            plt.savefig(
                r"%s\Behaviour\Paw_preferences\SteppingPawMutualInfo_%s.png" % (paths['plotting_destfolder'], con),
                format='png')
            plt.close(fig)

def main():
    LowHigh_days_conditions = ['APAChar_LowHigh_Repeats_Wash_Day1','APAChar_LowHigh_Repeats_Wash_Day2','APAChar_LowHigh_Repeats_Wash_Day3']
    plotting = PlotBehaviour(LowHigh_days_conditions)
    for time in ['start','trans']:
        for type in ['mice_across_days', 'all_mice_across_runs_singlecon']:
            plotting.paw_pref(type, time)
            plotting.paw_matching(type)
    plotting.paw_pref_mutual_info()
    for type in ['mice_across_days', 'all_mice_across_runs_across_days']:
        plotting.wait_time_bar_line(type)


if __name__ == '__main__':
    main()
    print("Finished saving plots!! Hope they look good (don't freak out if they don't though!) :)")