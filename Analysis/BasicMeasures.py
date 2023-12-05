from Helpers.Config_23 import *
from Helpers import Structural_calculations
from Helpers import utils
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt


class CalculateMeasuresByStride():
    def __init__(self, data, con, mouseID, r, stride_start, stride_end, stepping_limb):
        self.data = data
        self.con = con
        self.mouseID = mouseID
        self.r = r
        self.stride_start = stride_start
        self.stride_end = stride_end
        self.stepping_limb = stepping_limb

        # calculate sumarised dataframes
        df_s = self.data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']]
        df_f = self.data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']]
        self.df_s = df_s.droplevel(['Run', 'RunStage'])
        self.df_f = df_f.droplevel(['Run', 'RunStage'])

    def stride_duration(self): # ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stride_frames = data_chunk.index[-1] - data_chunk.index[0]
        result = (stride_frames/fps)*1000
        return result

    def walking_speed(self): # px/ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        x_displacement = data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[-1] - data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[0]
        stride_duration = self.stride_duration()
        result = x_displacement/stride_duration
        return result

    def cadence(self):
        stride_duration = self.stride_duration()
        result = 1/stride_duration
        return result

    def swing_velocity(self): # px/ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        swing_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_length = swing.loc(axis=1)['x'].iloc[-1] - swing.loc(axis=1)['x'].iloc[0]
        swing_frames = swing.index[-1] - swing.index[0]
        swing_duration = (swing_frames/fps)*1000
        result = swing_length/swing_duration
        return result

    def stride_length(self): # px
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        result = data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[-1] - data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[0]
        return result

    def stance_duration(self): # px/ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stance_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        stance = data_chunk.loc(axis=1)[self.stepping_limb][stance_mask]
        stance_frames = stance.index[-1] - stance.index[0]
        result = (stance_frames / fps) * 1000
        return result

    def duty_factor(self): # %
        stance_duration = self.stance_duration()
        stride_duration = self.stride_duration()
        result = (stance_duration / stride_duration) *100
        return result

    # def trajectory AND instantaneous swing vel

    def coo_x(self): #px ##### not sure about this?????
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        swing_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        tail_x = swing.loc(axis=0)[mid_t].loc(axis=0)['Tail1', 'x']
        limb_x = swing.loc(axis=0)[mid_t].loc(axis=0)[self.stepping_limb, 'x']
        result = limb_x - tail_x
        return result

    def coo_y(self): #px ##### not sure about this?????
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        swing_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        tail_y = swing.loc(axis=0)[mid_t].loc(axis=0)['Tail1', 'y']
        limb_y = swing.loc(axis=0)[mid_t].loc(axis=0)[self.stepping_limb, 'y']
        result = limb_y - tail_y
        return result

    def bos_ref_stance(self): # mm
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        xpos = self.df_s.loc(axis=0)[self.stride_start].loc[[self.stepping_limb, 'ForepawToe%s' % lr],'x']
        ypos = self.df_f.loc(axis=0)[self.stride_start].loc[[self.stepping_limb, 'ForepawToe%s' % lr],'x']
        triang, pixel_sizes = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).map_pixel_sizes_to_belt('Front', 'Front')
        real_position = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).find_interpolated_pixel_size(xpos.values, ypos.values, pixel_sizes, triang)
        front_real_y_pos = ypos*real_position
        result = abs(front_real_y_pos[self.stepping_limb] - front_real_y_pos['ForepawToe%s' % lr]).values[0]
        return result

    # def bos_hom_stance(self):

    # def tail1_displacement(self):

    def double_support(self): # %
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        homo_swing_frame_mask = data_chunk.loc(axis=1)['ForepawToe%s' % lr,'StepCycle'] ==1
        if any(homo_swing_frame_mask):
            homo_swing_frame = data_chunk.index[homo_swing_frame_mask][0]
            ref_stance_frame = data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((homo_swing_frame - ref_stance_frame)/stride_duration)*100
        else:
            result = 0
        return result


    def tail1_ptp_amplitude_stride(self): # px
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        peak = data_chunk.loc(axis=1)['Tail1','y'].max()
        trough = data_chunk.loc(axis=1)['Tail1','y'].min()
        result = peak - trough
        return result

    def tail1_speed(self): # px/ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        x_displacement = data_chunk.loc(axis=1)['Tail1', 'x'].iloc[-1] - \
                         data_chunk.loc(axis=1)['Tail1', 'x'].iloc[0]
        stride_duration = self.stride_duration()
        result = x_displacement / stride_duration
        return result

    def body_length_stance(self): # px
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = data_chunk.loc(axis=1)['Back12', 'likelihood'] > pcutoff
        back_mask = back1_mask & back12_mask
        mask = back_mask & stsw_mask
        back1x = data_chunk.loc(axis=1)['Back1', 'x'][mask]
        back12x = data_chunk.loc(axis=1)['Back12', 'x'][mask]
        results = back1x - back12x
        results_mean = results.mean()
        return results_mean

    def body_length_swing(self): # px
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = data_chunk.loc(axis=1)['Back12', 'likelihood'] > pcutoff
        back_mask = back1_mask & back12_mask
        mask = back_mask & stsw_mask
        back1x = data_chunk.loc(axis=1)['Back1', 'x'][mask]
        back12x = data_chunk.loc(axis=1)['Back12', 'x'][mask]
        results = back1x - back12x
        results_mean = results.mean()
        return results_mean



class Save():
    def __init__(self, conditions):
        self.conditions = conditions
        self.data = utils.Utils().GetDFs(conditions,reindexed_loco=True)
        self.CalculateMeasuresByStride = CalculateMeasuresByStride

    def find_pre_post_transition_strides(self, con, mouseID, r):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        view = 'Side'

        pre_frame = self.data[con][mouseID][view].loc(axis=0)[r, 'RunStart'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[-1]
        post_frame = self.data[con][mouseID][view].loc(axis=0)[r, 'Transition'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[0]
        trans_limb_mask = post_frame - pre_frame == -1
        stepping_limb = np.array(['ForepawToeL', 'ForepawToeR'])[trans_limb_mask]
        if len(stepping_limb) == 1:
            stepping_limb = stepping_limb[0]
        else:
            raise ValueError('wrong number of stepping limbs identified')

        limbs_mask_post = (self.data[con][mouseID][view].loc(axis=0)[r, ['Transition', 'RunEnd']].loc(axis=1)[['ForepawToeR','ForepawToeL'], 'likelihood'] > pcutoff).any(axis=1)

        stance_mask_pre = self.data[con][mouseID][view].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_pre = self.data[con][mouseID][view].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1
        stance_mask_post = self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_post = self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1

        stance_idx_pre = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][stance_mask_pre].tail(3))
        swing_idx_pre = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][swing_mask_pre].tail(3))
        stance_idx_post = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][stance_mask_post & limbs_mask_post].head(2))
        swing_idx_post = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][swing_mask_post & limbs_mask_post].head(2))

        stance_idx_pre['Stride_no'] = np.sort(np.arange(1,len(stance_idx_pre)+1)*-1)
        swing_idx_pre['Stride_no'] = np.sort(np.arange(1,len(swing_idx_pre)+1)*-1)
        stance_idx_post['Stride_no'] = np.arange(0,len(stance_idx_post))
        swing_idx_post['Stride_no'] = np.arange(0,len(swing_idx_post))


        # Combine pre and post DataFrames
        combined_df = pd.concat([stance_idx_pre,swing_idx_pre, stance_idx_post, swing_idx_post]).sort_index(level='FrameIdx')

        return combined_df

    def find_pre_post_transition_strides_ALL_RUNS(self, con, mouseID):
        view = 'Side'
        SwSt = []
        for r in self.data[con][mouseID][view].index.get_level_values(level='Run').unique().astype(int):
            try:
                stsw = self.find_pre_post_transition_strides(con=con,mouseID=mouseID,r=r)
                SwSt.append(stsw)
            except:
                print('Cant get stsw for run %s' %r)
        SwSt_df = pd.concat(SwSt)

        return SwSt_df

    #def plot_discrete_RUN_x_STRIDE_subplots(self, SwSt, con, mouseID, view):
    def get_discrete_measures_byrun_bystride(self, SwSt, con, mouseID):
        st_mask = (SwSt.loc(axis=1)[['ForepawToeR', 'ForepawToeL'],'StepCycle'] == 0).any(axis=1)
        stride_borders = SwSt[st_mask]

        # create df for single mouse to put measures into
        levels = [[mouseID],np.arange(0, 42), [-3, -2, -1, 0, 1]]
        multi_index = pd.MultiIndex.from_product(levels, names=['MouseID','Run', 'Stride'])
        measure_list_flat = [value for sublist in measure_list.values() for value in sublist]
        measures = pd.DataFrame(index=multi_index,columns=measure_list_flat)

        for r in tqdm(stride_borders.index.get_level_values(level='Run').unique()):
            stepping_limb = np.array(['ForepawToeR','ForepawToeL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawToeR','ForepawToeL']].count() > 1).values][0]
            try:
                for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no'][:-1]):
                    stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                    stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1]

                    class_instance = self.CalculateMeasuresByStride(self.data, con, mouseID, r, stride_start, stride_end,stepping_limb)

                    for m in measure_list_flat:
                        try:
                            method_name = m
                            if hasattr(class_instance, method_name) and callable(getattr(class_instance, method_name)):
                                method = getattr(class_instance, method_name)
                                result = method()
                                measures.loc(axis=0)[mouseID,r, s][m] = result
                            else:
                                print('Something went wrong for r: %s, stride: %s, measure: %s' %(r,s,m))
                        except:
                            print('cant plot measure %s' % m)
            except:
                print('cant plot stride %s' %s)

        return measures

    def get_discrete_measures_byrun_bystride_ALLMICE(self, con):
        mice = list(self.data[con].keys())

        mouse_measures_ALL = []
        for midx, mouseID in enumerate(mice):
            try:
                print('Calculating measures for %s (%s/%s)...' %(mouseID,midx,len(mice)-1))
                SwSt = self.find_pre_post_transition_strides_ALL_RUNS(con, mouseID)
                mouse_measures = self.get_discrete_measures_byrun_bystride(SwSt=SwSt, con=con, mouseID=mouseID)
                mouse_measures_ALL.append(mouse_measures)
            except:
                print('cant plot mouse %s' %mouseID)
        mouse_measures_ALL = pd.concat(mouse_measures_ALL)

        return mouse_measures_ALL

class plotting(): # currently not compatible for extended experiments
    def __init__(self, conditions):
        self.conditions = conditions # all conditions must be individually listed

    def load_measures_files(self, measure_organisation ='discreet_runXstride'):
        measures = dict.fromkeys(self.conditions)
        for con in self.conditions:
            print('Loading measures dataframes for condition: %s' %con)
            segments = con.split('_')
            filename = 'allmice_allmeasures_%s.h5' %measure_organisation
            if 'Day' not in con and 'Wash' not in con:
                conname, speed = segments[0:2]
                measure_filepath = r'%s\%s_%s\%s' %(paths['filtereddata_folder'], conname, speed, filename)
            else:
                conname, speed, repeat, wash, day = segments
                measure_filepath = r'%s\%s_%s\%s\%s\%s\%s' %(paths['filtereddata_folder'], conname, speed, repeat, wash, day, filename)
            measures[con] = pd.read_hdf(measure_filepath)
        return measures

    def plot(self, plot_type):
        measures = self.load_measures_files()

        if plot_type == 'discreet_strideXrun' and len(measures) == 1:
            self.plot_discrete_measures_singlecon_strideXrun(measures) #

    def plot_discrete_measures_singlecon_strideXrun(self, measures, chunk_size=10):
        """
        Plots measures from a single condition with stride on x-axis and a single line for each run.
        If measures for multiple conditions given it will save multiple plots.
        :param measures: dict of dfs filled with measure values for each mouse and run
        """
        stride_no = [-3, -2, -1, 0, 1]
        measure_list_flat = [value for sublist in measure_list.values() for value in sublist]

        for con in self.conditions:
            conname = con.split('_')[0]
            run_nos = expstuff['condition_exp_lengths']['%sRuns' %conname]
            run_nos_filled = np.concatenate([np.array([prepruns]), np.array(run_nos)]).cumsum() #+ prepruns
            prepruns = 5 if 'HighLow' in con else 2
            mice = list(measures[con].index.get_level_values(level='MouseID').unique())

            # colors = [['blue'] * run_nos[0], ['red'] * run_nos[1], ['green'] * run_nos[2]]
            # colors = [item for sublist in colors for item in sublist]
            blues = utils.Utils().get_cmap((run_nos[0]//chunk_size) + 2, 'Blues')
            reds = utils.Utils().get_cmap((run_nos[1]//chunk_size) + 2, 'Reds')
            greens = utils.Utils().get_cmap((run_nos[2]//chunk_size) + 2, 'Greens')
            colors = np.vstack((blues, reds, greens))

            for m in measure_list_flat:
                # for mouseID in mice:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                for i in range(0, len(run_nos)):
                    measure_mean = []
                    color_mean = []
                    for ridx, r in enumerate(np.arange(run_nos_filled[i],run_nos_filled[i+1])):
                        gradient = 1/(run_nos_filled[i+1] - run_nos_filled[i])
                        measure = measures[con].xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                        measure_mean.append(measure)
                        color_mean.append(colors[i][0]((ridx//chunk_size)+1))
                        ax.plot(stride_no, measure, color=colors[i][0]((ridx//chunk_size)+1), alpha=0.5, linewidth=1)
                    mean = pd.concat(measure_mean).groupby('Stride').mean()
                    ax.plot(stride_no, mean, color='red', linewidth=2, alpha=1, label='apa')

    def plot_discrete_measures_runXstride(self,df, mean_only=True):
        mice = list(df.index.get_level_values(level='MouseID').unique())
        stride_no = [-3,-2,-1,0]
        x = np.arange(1,41)
        measure_list_flat = [value for sublist in measure_list.values() for value in sublist]

        for m in measure_list_flat:
            fig, axs = plt.subplots(4, 1, figsize=(8, 10))
            for sidx, s in enumerate(stride_no):
                all_mice = []
                for mouseID in mice:
                    measure = df.loc(axis=0)[mouseID].xs(s,axis=0,level='Stride').loc(axis=1)[m]
                    measure_trim = measure.values[2:]
                    all_mice.append(measure)
                    if not mean_only:
                        axs[sidx].plot(x, measure_trim, color='b', alpha=0.2)
                    axs[sidx].set_title(s)
                all_mice_df = pd.concat(all_mice,axis=1)
                mean = all_mice_df.mean(axis=1).values[2:]
                axs[sidx].plot(x, mean, color='b', alpha=1)

            # formatting
            for ax in axs:
                ax.set_xlim(0,41)
                ax.axvline(10, alpha=0.5, color='black', linestyle='--')
                ax.axvline(30, alpha=0.5, color='black', linestyle='--')

            fig.suptitle(m)
            axs[3].set_xlabel('Run')

            plt.savefig(r'%s\Limb_parameters_bystride\Day2\RunXStrideNo_%s.png' % (
                paths['plotting_destfolder'], m),
                        bbox_inches='tight', transparent=False, format='png')

    def plot_discrete_measures_strideXrun(self, df):
        mice = list(df.index.get_level_values(level='MouseID').unique())
        stride_no = [-3, -2, -1, 0, 1]
        measure_list_flat = [value for sublist in measure_list.values() for value in sublist]
        colors = [['blue']*10,['red']*20,['lightskyblue']*10]
        colors = [item for sublist in colors for item in sublist]
        baseline = np.arange(2, 12)
        apa = np.arange(12, 32)
        washout = np.arange(32, 42)
        print(":):):):):)")

        for m in measure_list_flat:
            # for mouseID in mice:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for r in np.arange(2,42):
                measure = df.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                ax.plot(stride_no, measure, color=colors[r-2], alpha=0.2)

            measure_mean = []
            for r in baseline:
                measure = df.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                measure_mean.append(measure)
            mean = pd.concat(measure_mean).groupby('Stride').mean()
            ax.plot(stride_no, mean, color='blue', linewidth=2, alpha=1, label='baseline')

            measure_mean = []
            for r in apa:
                measure = df.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                measure_mean.append(measure)
            mean = pd.concat(measure_mean).groupby('Stride').mean()
            ax.plot(stride_no, mean, color='red', linewidth=2, alpha=1, label='apa')

            measure_mean = []
            for r in washout:
                measure = df.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                measure_mean.append(measure)
            mean = pd.concat(measure_mean).groupby('Stride').mean()
            ax.plot(stride_no, mean, color='lightskyblue', linewidth=2, alpha=1, label='washout')

            fig.suptitle('%s' %(m))
            ax.set_xticks(stride_no)
            ax.set_xlabel('Stride number')
            ax.axvline(0, alpha=0.5, color='black', linestyle='--')
            fig.legend()

            plt.savefig(r'%s\Limb_parameters_bystride\Day2\StrideNoXRun_%s.png' % (
                paths['plotting_destfolder'], m),
                        bbox_inches='tight', transparent=False, format='png')

# df.to_hdf(r"%s\APAChar_LowHigh\Repeats\Wash\Day1\allmice_allmeasures_discreet_runXstride.h5" % (paths['filtereddata_folder']), key='measures%s' %v, mode='w')
    # def plot_continuous_measure_raw_aligned_to_transition(self,results):
    #
    # def

# from Analysis import BasicMeasures
# conditions = ['APAChar_LowHigh_Repeats_Wash_Day1']
# con = conditions[0]
# plotting = BasicMeasures.plotting(conditions)
# df = plotting.get_discrete_measures_byrun_bystride_ALLMICE(con)





