from Helpers.Config_23 import *
from Helpers import Structural_calculations
from Helpers import utils
#from scipy.stats import skew, shapiro, levene
import scipy.stats as stats
import numpy as np
import pandas as pd
import warnings
import os
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
        # self.df_s = df_s.droplevel(['Run', 'RunStage'])
        # self.df_f = df_f.droplevel(['Run', 'RunStage'])

    def get_data_summary(self, limb_type):
        limb_data = self.data[self.con][self.mouseID][limb_type].loc(axis=0)[
            self.r, ['RunStart', 'Transition', 'RunEnd']]
        return limb_data.droplevel(['Run', 'RunStage'])

    def get_data_chunk(self, limb_type):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        return data_chunk.loc(axis=1)[self.stepping_limb, limb_type]

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
        # todo reduce size of these functions
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

    def back_skew_stance(self): ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        slope, intercept = self.get_belt_line()
        back_mask = (data_chunk.loc(axis=1)[
                        ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                         'Back11',
                         'Back12'], 'likelihood'] > pcutoff).droplevel(level='coords',axis=1)
        belt_heights = (data_chunk.loc(axis=1)[
                           ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                            'Back11', 'Back12'], 'x'][stsw_mask] * slope + intercept).droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        back_heights = data_chunk.loc(axis=1)[
            ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11',
             'Back12'], 'y'][stsw_mask].droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        mult = np.arange(1, 13)
        true_back_height = belt_heights - back_heights
        com = (true_back_height*mult).sum(axis=1)/true_back_height.sum(axis=1)
        result = np.median(mult) - com
        result_mean = np.mean(result)
        return result_mean

    def back_skew_swing(self): ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        slope, intercept = self.get_belt_line()
        back_mask = (data_chunk.loc(axis=1)[
                        ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                         'Back11',
                         'Back12'], 'likelihood'] > pcutoff).droplevel(level='coords',axis=1)
        belt_heights = (data_chunk.loc(axis=1)[
                           ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                            'Back11', 'Back12'], 'x'][stsw_mask] * slope + intercept).droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        back_heights = data_chunk.loc(axis=1)[
            ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11',
             'Back12'], 'y'][stsw_mask].droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        mult = np.arange(1, 13)
        true_back_height = belt_heights - back_heights
        com = (true_back_height*mult).sum(axis=1)/true_back_height.sum(axis=1)
        result = np.median(mult) - com
        result_mean = np.mean(result)
        return result_mean

    def back_curvature_stance(self): ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        slope, intercept = self.get_belt_line()
        back_mask = (data_chunk.loc(axis=1)[
                        ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                         'Back11','Back12'], 'likelihood'] > pcutoff).droplevel(level='coords',axis=1)
        belt_heights = (data_chunk.loc(axis=1)[
                           ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                            'Back11', 'Back12'], 'x'][stsw_mask] * slope + intercept).droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        back_heights = data_chunk.loc(axis=1)[
            ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11',
             'Back12'], 'y'][stsw_mask].droplevel(level='coords',axis=1)[back_mask].iloc[:, ::-1]
        true_back_height = belt_heights - back_heights
        result = true_back_height.mean(axis=0)
        return result

    def body_tilt_stance(self): # positive means back12 is lower than back1
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = data_chunk.loc(axis=1)['Back12', 'likelihood'] > pcutoff
        back_mask = back1_mask & back12_mask
        mask = back_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        backx = data_chunk.loc(axis=1)[['Back1','Back12'], 'x'][mask].droplevel('coords',axis=1)
        backy = data_chunk.loc(axis=1)[['Back1','Back12'], 'y'][mask]
        belty = backx*slope_belt + intercept_belt
        true_backy = belty - backy.droplevel('coords',axis=1)

        slope = (true_backy.loc(axis=1)['Back1'] - true_backy.loc(axis=1)['Back12'])/(backx.loc(axis=1)['Back1'] - backx.loc(axis=1)['Back12'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def body_tilt_swing(self):  # positive means back12 is lower than back1
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = data_chunk.loc(axis=1)['Back12', 'likelihood'] > pcutoff
        back_mask = back1_mask & back12_mask
        mask = back_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        backx = data_chunk.loc(axis=1)[['Back1', 'Back12'], 'x'][mask].droplevel('coords', axis=1)
        backy = data_chunk.loc(axis=1)[['Back1', 'Back12'], 'y'][mask]
        belty = backx * slope_belt + intercept_belt
        true_backy = belty - backy.droplevel('coords', axis=1)

        slope = (true_backy.loc(axis=1)['Back1'] - true_backy.loc(axis=1)['Back12']) / (
                    backx.loc(axis=1)['Back1'] - backx.loc(axis=1)['Back12'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def head_tilt_stance(self):  # positive means back12 is lower than back1
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        nose_mask = data_chunk.loc(axis=1)['Nose', 'likelihood'] > pcutoff
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        head_mask = back1_mask & nose_mask
        mask = head_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        headx = data_chunk.loc(axis=1)[['Nose', 'Back1'], 'x'][mask].droplevel('coords', axis=1)
        heady = data_chunk.loc(axis=1)[['Nose', 'Back1'], 'y'][mask]
        belty = headx * slope_belt + intercept_belt
        true_heady = belty - heady.droplevel('coords', axis=1)

        slope = (true_heady.loc(axis=1)['Nose'] - true_heady.loc(axis=1)['Back1']) / (
                    headx.loc(axis=1)['Nose'] - headx.loc(axis=1)['Back1'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def head_tilt_swing(self):  # positive means back12 is lower than back1
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        nose_mask = data_chunk.loc(axis=1)['Nose', 'likelihood'] > pcutoff
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        head_mask = back1_mask & nose_mask
        mask = head_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        headx = data_chunk.loc(axis=1)[['Nose', 'Back1'], 'x'][mask].droplevel('coords', axis=1)
        heady = data_chunk.loc(axis=1)[['Nose', 'Back1'], 'y'][mask]
        belty = headx * slope_belt + intercept_belt
        true_heady = belty - heady.droplevel('coords', axis=1)

        slope = (true_heady.loc(axis=1)['Nose'] - true_heady.loc(axis=1)['Back1']) / (
                    headx.loc(axis=1)['Nose'] - headx.loc(axis=1)['Back1'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def tail_tilt_stance(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        tail1_mask = data_chunk.loc(axis=1)['Tail1', 'likelihood'] > pcutoff
        tail12_mask = data_chunk.loc(axis=1)['Tail12', 'likelihood'] > pcutoff
        tail_mask = tail1_mask & tail12_mask
        mask = tail_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        tailx = data_chunk.loc(axis=1)[['Tail1', 'Tail12'], 'x'][mask].droplevel('coords', axis=1)
        taily = data_chunk.loc(axis=1)[['Tail1', 'Tail12'], 'y'][mask]
        belty = tailx * slope_belt + intercept_belt
        true_taily = belty - taily.droplevel('coords', axis=1)

        slope = (true_taily.loc(axis=1)['Tail1'] - true_taily.loc(axis=1)['Tail12']) / (
                    tailx.loc(axis=1)['Tail1'] - tailx.loc(axis=1)['Tail12'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def tail_tilt_swing(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        tail1_mask = data_chunk.loc(axis=1)['Tail1', 'likelihood'] > pcutoff
        tail12_mask = data_chunk.loc(axis=1)['Tail12', 'likelihood'] > pcutoff
        tail_mask = tail1_mask & tail12_mask
        mask = tail_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        tailx = data_chunk.loc(axis=1)[['Tail1', 'Tail12'], 'x'][mask].droplevel('coords', axis=1)
        taily = data_chunk.loc(axis=1)[['Tail1', 'Tail12'], 'y'][mask]
        belty = tailx * slope_belt + intercept_belt
        true_taily = belty - taily.droplevel('coords', axis=1)

        slope = (true_taily.loc(axis=1)['Tail1'] - true_taily.loc(axis=1)['Tail12']) / (
                    tailx.loc(axis=1)['Tail1'] - tailx.loc(axis=1)['Tail12'])
        angle = np.rad2deg(np.arctan(slope))
        result_mean = angle.mean()

        return result_mean

    def neck_height_stance(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        neck_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        mask = neck_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        neckx = data_chunk.loc(axis=1)['Back1', 'x'][mask]
        necky = data_chunk.loc(axis=1)['Back1', 'y'][mask]
        belty = neckx * slope_belt + intercept_belt
        true_necky = belty - necky
        result_mean = true_necky.mean()
        return result_mean

    def neck_height_swing(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        neck_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        mask = neck_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        neckx = data_chunk.loc(axis=1)['Back1', 'x'][mask]
        necky = data_chunk.loc(axis=1)['Back1', 'y'][mask]
        belty = neckx * slope_belt + intercept_belt
        true_necky = belty - necky
        result_mean = true_necky.mean()
        return result_mean

    def tail1_height_stance(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        tail1_mask = data_chunk.loc(axis=1)['Tail1', 'likelihood'] > pcutoff
        mask = tail1_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        tail1x = data_chunk.loc(axis=1)['Tail1', 'x'][mask]
        tail1y = data_chunk.loc(axis=1)['Tail1', 'y'][mask]
        belty = tail1x * slope_belt + intercept_belt
        true_tail1y = belty - tail1y
        result_mean = true_tail1y.mean()
        return result_mean

    def tail1_height_swing(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        tail1_mask = data_chunk.loc(axis=1)['Tail1', 'likelihood'] > pcutoff
        mask = tail1_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        tail1x = data_chunk.loc(axis=1)['Tail1', 'x'][mask]
        tail1y = data_chunk.loc(axis=1)['Tail1', 'y'][mask]
        belty = tail1x * slope_belt + intercept_belt
        true_tail1y = belty - tail1y
        result_mean = true_tail1y.mean()
        return result_mean

    def midback_height_stance(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        back6_mask = data_chunk.loc(axis=1)['Back6', 'likelihood'] > pcutoff
        back7_mask = data_chunk.loc(axis=1)['Back7', 'likelihood'] > pcutoff
        back_mask = back6_mask & back7_mask
        mask = back_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        midbackx = data_chunk.loc(axis=1)[['Back6','Back7'], 'x'][mask].droplevel('coords', axis=1)
        midbacky = data_chunk.loc(axis=1)[['Back6','Back7'], 'y'][mask]
        belty = midbackx * slope_belt + intercept_belt
        true_midbacky = belty - midbacky.droplevel('coords', axis=1)
        mean_midbacky = true_midbacky.mean(axis=1)
        result_mean = mean_midbacky.mean()
        return result_mean

    def midback_height_swing(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        back6_mask = data_chunk.loc(axis=1)['Back6', 'likelihood'] > pcutoff
        back7_mask = data_chunk.loc(axis=1)['Back7', 'likelihood'] > pcutoff
        back_mask = back6_mask & back7_mask
        mask = back_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()
        midbackx = data_chunk.loc(axis=1)[['Back6','Back7'], 'x'][mask].droplevel('coords', axis=1)
        midbacky = data_chunk.loc(axis=1)[['Back6','Back7'], 'y'][mask]
        belty = midbackx * slope_belt + intercept_belt
        true_midbacky = belty - midbacky.droplevel('coords', axis=1)
        mean_midbacky = true_midbacky.mean(axis=1)
        result_mean = mean_midbacky.mean()
        return result_mean

    def get_belt_line(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        start = data_chunk.loc(axis=1)['StartPlatR',['x','y']].mean().droplevel(level='bodyparts')
        trans = data_chunk.loc(axis=1)['TransitionR',['x','y']].mean().droplevel(level='bodyparts')
        slope, intercept, _, _, _ = stats.linregress([start['x'], trans['x']], [start['y'], trans['y']])
        return slope, intercept

    def limb_rel_to_body_stance(self):
        data_chunk = self.df_s.loc(axis=0)[self.stride_start]
        x_vals = data_chunk.loc(axis=0)[['Back1','Back12',self.stepping_limb], 'x']
        x_vals_zeroed = x_vals - x_vals['Back12']
        x_vals_norm_to_neck = x_vals_zeroed/x_vals_zeroed['Back1']
        result = x_vals_norm_to_neck[self.stepping_limb].values[0]
        return result


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

class plotting():
    # todo class currently not compatible for extended experiments
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

    def plot_discrete_measures_singlecon_strideXrun(self, m, con, conname, measures, chunk_size,plot_filepath,all_days):
        stride_no = [-3, -2, -1, 0, 1]
        prepruns = 5 if 'HighLow' in con else 2
        run_nos = expstuff['condition_exp_lengths']['%sRuns' % conname]
        run_nos_filled = np.concatenate([np.array([prepruns]), np.array(run_nos)]).cumsum()  # + prepruns

        blues = utils.Utils().get_cmap((run_nos[0] // chunk_size) + 2, 'Blues')
        reds = utils.Utils().get_cmap((run_nos[1] // chunk_size) + 2, 'Reds')
        greens = utils.Utils().get_cmap((run_nos[2] // chunk_size) + 2, 'Greens')
        colors = np.vstack((blues, reds, greens))

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i in range(0, len(run_nos)):
            measure_mean = []
            color_mean = []
            for ridx, r in enumerate(np.arange(run_nos_filled[i], run_nos_filled[i + 1])):
                if all_days == True:
                    measure_day = measures.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby(['Stride','Day']).mean()
                    measure = measure_day.unstack(level='Day')
                else:
                    measure = measures.xs(r, axis=0, level='Run').loc(axis=1)[m].groupby('Stride').mean()
                measure_mean.append(measure)
                color_mean.append(colors[i][0]((ridx // chunk_size) + 1))
                # ax.plot(stride_no, measure, color=colors[i][0]((ridx // chunk_size) + 1), alpha=0.5, linewidth=1)
                ax.plot(measure, color=colors[i][0]((ridx // chunk_size) + 1), alpha=0.5, linewidth=1)

            for c in range(run_nos[i] // chunk_size):
                mean = pd.concat(measure_mean[chunk_size * c:chunk_size * (c + 1)]).groupby('Stride').mean()
                label = '%s_%s' % (expstuff['exp_chunks']['ExpPhases'][i], c)
                ax.plot(stride_no, mean, color=colors[i][0](c + 1), linewidth=4, alpha=1, label=label)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(measures.loc(axis=1)[m].groupby(['Run', 'Stride']).mean().quantile(0.01),
                    measures.loc(axis=1)[m].groupby(['Run', 'Stride']).mean().quantile(0.99))
        ax.set_xticks(stride_no)
        ax.set_xlabel('Stride number')
        ax.axvline(0, alpha=0.5, color='black', linestyle='--')
        fig.legend()
        fig.suptitle('')  # Clear the default supertitle
        fig.text(0.5, 0.95, m, ha='center', va='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.90, con, ha='center', va='center', fontsize=12)

        if not os.path.exists(plot_filepath):
            os.makedirs(plot_filepath)
        fig.savefig(r'%s\StrideNoXRun_chunksize=%s_%s.png' % (plot_filepath, chunk_size, m),
                    bbox_inches='tight', transparent=False, format='png')


    def saveplots_discrete_measures_singlecon_strideXrun(self, measures, chunk_size=10, all_days_mean=False, all_days=False):
        """
        Plots measures from a single condition with stride on x-axis and a single line for each run.
        If measures for multiple conditions given it will save multiple plots.
        :param measures: dict of dfs filled with measure values for each mouse and run
        """
        stride_no = [-3, -2, -1, 0, 1]
        measure_list_flat = [value for sublist in measure_list.values() for value in sublist]

        if not all_days_mean:
            for con in self.conditions:
                segments = con.split('_')
                if 'Day' not in con and 'Wash' not in con:
                    conname, speed = segments[0:2]
                    plot_filepath = r'%s\%s_%s' % (paths['plotting_destfolder'], conname, speed)
                else:
                    conname, speed, repeat, wash, day = segments
                    plot_filepath = r'%s\%s_%s\%s\%s\%s' % (paths['plotting_destfolder'], conname, speed, repeat, wash, day)

                for m in measure_list_flat:
                    self.plot_discrete_measures_singlecon_strideXrun(m=m,con=con,conname=conname,
                                                                     measures=measures[con],
                                                                     chunk_size=chunk_size, plot_filepath=plot_filepath,all_days=all_days)
        else:
            measures_all = pd.concat(measures.values(), keys=measures.keys())
            measures_all.index.set_names('Day', level=0, inplace=True)
            segments = self.conditions[0].split('_')
            con = '_'.join(segments[:-1])
            conname, speed, repeat, wash, _ = segments
            plot_filepath = r'%s\%s_%s\%s\%s' % (paths['plotting_destfolder'], conname, speed, repeat, wash)

            for m in measure_list_flat:
                self.plot_discrete_measures_singlecon_strideXrun(m=m, con=con, conname=conname,
                                                                       measures=measures_all,
                                                                       chunk_size=chunk_size, plot_filepath=plot_filepath,all_days=all_days)




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





