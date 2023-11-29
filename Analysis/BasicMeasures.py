from Helpers.Config_23 import *
from Helpers import Structural_calculations
from Helpers import utils
import numpy as np
import pandas as pd
import warnings
from scipy.stats import skew, shapiro, levene


class calculate_body():
    def __init__(self, conditions):
        self.conditions = conditions

    def BodyLength(self, data, con, mouseID, r,view='Side', mean=False, phase=None, frame_window=None):
        if mean == True:
            runphase = 'RunStart' if phase == 'apa' or phase == 'pre' else 'Transition'

            back1_mask = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back1', 'likelihood'] > pcutoff
            back12_mask = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back12', 'likelihood'] > pcutoff

            back1 = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back1', 'x'][back1_mask]
            back12 = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back12', 'x'][back12_mask]

            body_length_mean = np.mean(back1 - back12)
            body_length_std = np.std(back1 - back12)
            body_length_sem = np.std(back1 - back12)/np.sqrt(len(back1))
            body_length_cv = body_length_std / body_length_mean

            results = {
                'mean': body_length_mean,
                'std': body_length_std,
                'sem': body_length_sem,
                'cv': body_length_cv
            }
        else:
            back1_mask = data[con][mouseID][view].loc(axis=0)[r,['RunStart','Transition']].loc(axis=1)['Back1', 'likelihood'] > pcutoff
            back12_mask = data[con][mouseID][view].loc(axis=0)[r,['RunStart','Transition']].loc(axis=1)['Back12', 'likelihood'] > pcutoff
            back_mask = back1_mask & back12_mask

            back1x = data[con][mouseID][view].loc(axis=0)[r,['RunStart','Transition']].loc(axis=1)['Back1', 'x'][back_mask]
            back12x = data[con][mouseID][view].loc(axis=0)[r,['RunStart','Transition']].loc(axis=1)['Back12', 'x'][back_mask]

            # back1y = data[con][mouseID]['Overhead'].loc(axis=0)[r, ['RunStart', 'Transition']].loc(axis=1)['Back1', 'x'][back_mask]
            # back12y = data[con][mouseID]['Overhead'].loc(axis=0)[r, ['RunStart', 'Transition']].loc(axis=1)['Back12', 'x'][back_mask]
            # real_x_back1 = Structural_calculations.GetRealDistances(data, con, mouseID).find_interpolated_pixel_size(
            #     back1x.values, back1y.values, pixel_sizes, triang)
            # real_x_back12 = Structural_calculations.GetRealDistances(data, con, mouseID).find_interpolated_pixel_size(
            #     back12x.values, back12y.values, pixel_sizes, triang)

            results = back1x - back12x

        return results

    def BackHeight(self, data, con, mouseID, view, r, phase, frame_window, calculation):
        runphase = 'RunStart' if phase == 'apa' or phase == 'pre' else 'Transition'

        back_mask = (data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)[['Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12'], 'likelihood'] > pcutoff).all(axis=1)
        start_plat_R_mask = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff
        transition_R_mask = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff

        start_plat_R_mean = np.mean(data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['StartPlatR', 'y'][start_plat_R_mask])
        transition_R_mean = np.mean(data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)['TransitionR', 'y'][transition_R_mask])

        belt_level = np.mean([start_plat_R_mean,transition_R_mean])
        back_y = data[con][mouseID][view].loc(axis=0)[r, runphase, frame_window].loc(axis=1)[['Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12'], 'y'][back_mask]
        back_y_heights = belt_level - back_y

        if calculation == 'skew':
            result = skew(back_y_heights,axis=1)
        # elif calculation == 'height':
        #     result =

        result_mean = np.mean(result)


class plotting():
    def __init__(self, conditions):
        self.conditions = conditions
        self.data = utils.Utils().GetDFs(conditions,reindexed_loco=True)

    def find_pre_post_transition_strides(self, con, mouseID, view, r):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
        final_swing_end_mask = np.logical_and(limbs_mask_post,self.data[con][mouseID][view].loc(axis=0)[r, ['Transition', 'RunEnd']].loc(axis=1)[stepping_limb, 'StepCycleFill'] == 1)

        stance_idx_pre = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][stance_mask_pre].tail(3))
        swing_idx_pre = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][swing_mask_pre].tail(3))
        stance_idx_post = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][stance_mask_post & limbs_mask_post].head(2))
        swing_idx_post = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][swing_mask_post & limbs_mask_post].head(2))
        final_swing_idx = pd.DataFrame(self.data[con][mouseID][view].loc(axis=0)[r, ['Transition', 'RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][final_swing_end_mask].tail(1))

        stance_idx_pre['Stride_no'] = np.sort(np.arange(1,len(stance_idx_pre)+1)*-1)
        swing_idx_pre['Stride_no'] = np.sort(np.arange(1,len(swing_idx_pre)+1)*-1)
        stance_idx_post['Stride_no'] = np.arange(1,len(stance_idx_post)+1)
        swing_idx_post['Stride_no'] = np.arange(1,len(swing_idx_post)+1)
        final_swing_idx['Stride_no'] = 100


        # Combine pre and post DataFrames
        all_stances = pd.concat([stance_idx_pre, stance_idx_post, final_swing_idx]).sort_index(level='FrameIdx')

        combined_df = pd.concat([stance_idx_pre,swing_idx_pre, stance_idx_post, swing_idx_post, final_swing_idx]).sort_index(level='FrameIdx')

        return combined_df

    def find_pre_post_transition_strides_ALL_RUNS(self, con, mouseID, view):
        SwSt = []
        for r in self.data[con][mouseID][view].index.get_level_values(level='Run').unique().astype(int):
            try:
                stsw = self.find_pre_post_transition_strides(con=con,mouseID=mouseID,view=view,r=r)
                SwSt.append(stsw)
            except:
                print('Cant get stsw for run %s' %r)
        SwSt_df = pd.concat(SwSt)

        return SwSt_df

    #def plot_discrete_RUN_x_STRIDE_subplots(self, SwSt, con, mouseID, view):
    def get_discrete_measures_wholestride(self, SwSt, con, mouseID, view, measure_list):
        st_mask = (SwSt.loc(axis=1)[['ForepawToeR', 'ForepawToeL'],'StepCycle'] == 0).any(axis=1)
        end_mask = SwSt.loc(axis=1)['Stride_no'] == 100

        stride_borders_mask = st_mask | end_mask
        stride_borders = SwSt[stride_borders_mask]

        for r in stride_borders.index.get_level_values(level='Run').unique():
            stepping_limb = np.array(['ForepawToeR','ForepawToeL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawToeR','ForepawToeL']].count() > 1).values][0]
            for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no'][:-1]):
                stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1]

                data_chunk = self.data[con][mouseID][view].loc(axis=0)[r, ['RunStart','Transition','RunEnd'],np.arange(stride_start,stride_end)]




    # def plot_continuous_measure_raw_aligned_to_transition(self,results):
    #
    # def





