from Helpers.Config_23 import *
from Helpers import Structural_calculations
from Helpers import utils
#from scipy.stats import skew, shapiro, levene
import scipy.stats as stats
import numpy as np
import pandas as pd
import warnings
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


class CalculateMeasuresByStride():
    def __init__(self, data, con, mouseID, r, stride_start, stride_end, stepping_limb, maps):
        self.data, self.con, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb, self.maps = data, con, mouseID, r, stride_start, stride_end, stepping_limb, maps

        # calculate sumarised dataframes
        self.df_s = self.data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])
        self.df_f = self.data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])
        self.df_o = self.data[con][mouseID]['Overhead'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])

    def stride_duration(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stride_frames = data_chunk.index[-1] - data_chunk.index[0]
        return (stride_frames/fps)*1000

    def walking_speed(self): # px/ms
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        x_displacement = data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[-1] - data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[0]
        return x_displacement/self.stride_duration()

    def cadence(self):
        stride_duration = self.stride_duration()
        return 1/stride_duration

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
        real_position = self.maps['map'].find_interpolated_pixel_size(xpos.values, ypos.values, self.maps['pixel_sizes']['front_f'], self.maps['triang']['front_f'])
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

    def calculate_body_length(self, step_phase): # px
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
        back1_mask = data_chunk.loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = data_chunk.loc(axis=1)['Back12', 'likelihood'] > pcutoff
        back_mask = back1_mask & back12_mask
        mask = back_mask & stsw_mask
        back1x = data_chunk.loc(axis=1)['Back1', 'x'][mask]
        back12x = data_chunk.loc(axis=1)['Back12', 'x'][mask]
        results = back1x - back12x
        return results.mean()

    def body_length_stance(self): # px
        return self.calculate_body_length(0)

    def body_length_swing(self): # px
        return self.calculate_body_length(1)

    def calculate_back(self, step_phase):  ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
        slope, intercept = self.get_belt_line()
        back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                       'Back11', 'Back12']
        back_mask = (data_chunk.loc(axis=1)[back_labels, 'likelihood'] > pcutoff).droplevel(level='coords', axis=1)
        belt_heights = \
        (data_chunk.loc(axis=1)[back_labels, 'x'][stsw_mask] * slope + intercept).droplevel(level='coords', axis=1)[
            back_mask].iloc[:, ::-1]
        back_heights = data_chunk.loc(axis=1)[back_labels, 'y'][stsw_mask].droplevel(level='coords', axis=1)[
                           back_mask].iloc[:, ::-1]
        return belt_heights - back_heights
        #return true_back_height.mean(axis=0)

    def calculate_back_skew(self, step_phase): ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        mult = np.arange(1, 13)
        true_back_height = self.calculate_back(step_phase)
        com = (true_back_height * mult).sum(axis=1) / true_back_height.sum(axis=1)
        return np.mean(np.median(mult) - com)

    def back_skew_stance(self):
        return self.calculate_back_skew(0)

    def back_skew_swing(self):
        return self.calculate_back_skew(1)

   # def calculate_back_curvature(self):

    # def back_curvature_stance(self):
    #     return self.calculate_back_curvature(0)
    #
    # def back_curvature_swing(self):
    #     return self.calculate_back_curvature(1)

    def calculate_body_tilt(self, body_parts, step_phase):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
        part_mask = np.all(data_chunk.loc(axis=1)[body_parts, 'likelihood'] > pcutoff, axis=1)
        mask = part_mask & stsw_mask
        slope_belt, intercept_belt = self.get_belt_line()

        part_x = data_chunk.loc(axis=1)[body_parts, 'x'][mask].droplevel('coords', axis=1)
        part_y = data_chunk.loc(axis=1)[body_parts, 'y'][mask]
        belty = part_x * slope_belt + intercept_belt
        true_part_y = belty - part_y.droplevel('coords', axis=1)

        slope = (true_part_y.loc(axis=1)[body_parts[0]] - true_part_y.loc(axis=1)[body_parts[1]]) / (
                part_x.loc(axis=1)[body_parts[0]] - part_x.loc(axis=1)[body_parts[1]])
        angle = np.rad2deg(np.arctan(slope))
        return angle.mean()

    def body_tilt_stance(self): # positive means back12 is lower than back1
        return self.calculate_body_tilt(['Back1','Back12'], 0)

    def body_tilt_swing(self): # positive means back12 is lower than back1
        return self.calculate_body_tilt(['Back1','Back12'], 1)

    def head_tilt_stance(self):
        return self.calculate_body_tilt(['Nose','Back1'], 0)

    def head_tilt_swing(self):
        return self.calculate_body_tilt(['Nose','Back1'], 1)

    def tail_tilt_stance(self):
        return self.calculate_body_tilt(['Tail1','Tail12'], 0)

    def tail_tilt_swing(self):
        return self.calculate_body_tilt(['Tail1','Tail12'], 1)

    # def calculate_body_z(self, body_part, step_phase, yref):
    #     '''
    #     Returns true (subtracted from belt line) height (in z-plane) of one or more body part/s
    #     :param body_part:
    #     :param step_phase:
    #     :return:
    #     '''
    #     # data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     # stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
    #     # part_mask = np.all(data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
    #     #     if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff
    #     # mask = part_mask & stsw_mask
    #     data_chunk, data_chunk_yref, mask = self.get_body_part_coordinates(body_part, step_phase, yref)
    #     slope_belt, intercept_belt = self.get_belt_line()
    #     xpos = data_chunk.loc(axis=1)[body_part, 'x'][mask].droplevel('coords', axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'x'][mask]
    #     zpos = data_chunk.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'y'][mask]
    #     if yref == 'front':
    #         ypos = data_chunk_yref.loc(axis=1)[body_part, 'x'][mask].droplevel('coords', axis=1).mean(axis=1) \
    #             if isinstance(body_part, list) else data_chunk_yref.loc(axis=1)[body_part, 'x'][mask]
    #         real_px_size = self.maps['map'].find_interpolated_pixel_size(xpos.values, ypos.values,
    #                                                                      self.maps['pixel_sizes']['side_f'],
    #                                                                      self.maps['triang']['side_f'])
    #     elif yref == 'overhead':
    #         ypos = data_chunk_yref.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1).mean(axis=1) \
    #             if isinstance(body_part, list) else data_chunk_yref.loc(axis=1)[body_part, 'y'][mask]
    #         real_px_size = self.maps['map'].find_interpolated_pixel_size(xpos.values, ypos.values,
    #                                                                      self.maps['pixel_sizes']['side_o'],
    #                                                                      self.maps['triang']['side_o'])
    #     belty = part_x * slope_belt + intercept_belt
    #     true_part_y = belty - part_y
    #     return true_part_y.mean(axis=0)

    def neck_z_stance(self):
        return self.calculate_body_z('Back1', 0)

    def neck_z_swing(self):
        return self.calculate_body_z('Back1', 1)

    def tail_z_stance(self):
        return self.calculate_body_z('Tail1', 0)

    def tail_z_swing(self):
        return self.calculate_body_z('Tail1', 1)

    def midback_z_stance(self):
        height = self.calculate_body_z(['Back6','Back7'], 0)
        return height.mean()

    def midback_z_swing(self):
        height = self.calculate_body_z(['Back6','Back7'], 1)
        return height.mean()

    def stepping_limb_z_stance(self):
        return self.calculate_body_z(self.stepping_limb, 0)

    def stepping_limb_z_swing(self):
        return self.calculate_body_z(self.stepping_limb, 1)

    def contra_limb_z_stance(self):
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        body_part = 'ForepawToe%s' % lr
        return self.calculate_body_z(body_part, 0)

    def contra_limb_z_swing(self):
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        body_part = 'ForepawToe%s' % lr
        return self.calculate_body_z(body_part, 1)

    # def get_body_part_coordinates(self, body_part, step_phase, yref, zref='side', sub_y_bodypart=None):
    #     data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     if yref == 'front':
    #         data_chunk_yref = self.df_f.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     elif yref == 'overhead':
    #         data_chunk_yref = self.df_o.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     else:
    #         raise ValueError("Invalid value for yref. It should be either 'front' or 'overhead'.")
    #
    #     stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
    #     part_mask = np.all(data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff
    #     if sub_y_bodypart is not None:
    #         part_yref = sub_y_bodypart
    #     else:
    #         part_yref = body_part
    #     part_yref_mask = np.all(data_chunk_yref.loc(axis=1)[part_yref, 'likelihood'] > pcutoff, axis=1) \
    #             if isinstance(part_yref, list) else data_chunk_yref.loc(axis=1)[part_yref, 'likelihood'] > pcutoff
    #     mask = np.logical_and.reduce((part_mask, part_yref_mask, stsw_mask))
    #
    #     xpos = data_chunk.loc(axis=1)[body_part, 'x'][mask].droplevel('coords', axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'x'][mask]
    #     if zref == 'side':
    #         zpos = data_chunk.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1) \
    #             if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'y'][mask]
    #     elif zref == 'front':
    #         zpos = data_chunk_yref.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1) \
    #             if isinstance(body_part, list) else data_chunk_yref.loc(axis=1)[body_part, 'y'][mask]
    #     if yref == 'front':
    #         ypos = data_chunk_yref.loc(axis=1)[part_yref, 'x'][mask].droplevel('coords', axis=1).mean(axis=1) \
    #             if isinstance(part_yref, list) else data_chunk_yref.loc(axis=1)[part_yref, 'x'][mask]
    #     elif yref == 'overhead':
    #         ypos = data_chunk_yref.loc(axis=1)[part_yref, 'y'][mask].droplevel('coords', axis=1).mean(axis=1) \
    #             if isinstance(part_yref, list) else data_chunk_yref.loc(axis=1)[part_yref, 'y'][mask]
    #
    #     return {'x': xpos, 'y': ypos, 'z': zpos}
    #
    # def temp_get_body_part_coordinates_SINGLEVIEW(self, body_part, view, step_phase):
    #     if view == 'side':
    #         data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #         data_chunk_other = self.df_f.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     elif view == 'front':
    #         data_chunk = self.df_f.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #         data_chunk_other = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #     elif view == 'overhead':
    #         data_chunk = self.df_o.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #         data_chunk_other = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
    #
    #     stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
    #     other_mask = np.all(data_chunk_other.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
    #         if isinstance(body_part, list) else data_chunk_other.loc(axis=1)[body_part, 'likelihood'] > pcutoff
    #     part_mask = np.all(data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff
    #     mask = np.logical_and.reduce((part_mask, other_mask, stsw_mask))
    #
    #     xpos = data_chunk.loc(axis=1)[body_part, 'x'][mask].droplevel('coords', axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'x'][mask]
    #     ypos = data_chunk.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1) \
    #         if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'y'][mask]
    #
    #     return xpos, ypos

    def convert_coords_to_mm(self, x, y, z, coord, view, yref):
        real_px_size = []
        real_coord = []
        if view == 'Side':
            if yref == 'front':
                real_px_size = self.maps['map'].find_interpolated_pixel_size(x.values, y.values,
                                                                             self.maps['pixel_sizes']['side_f'],
                                                                             self.maps['triang']['side_f'])
            elif yref == 'overhead':
                real_px_size = self.maps['map'].find_interpolated_pixel_size(x.values, y.values,
                                                                             self.maps['pixel_sizes']['side_o'],
                                                                             self.maps['triang']['side_o'])

            if coord == 'x' or 'z' and coord != 'y':
                real_coord = eval(coord) * real_px_size
            else:
                raise ValueError('Incompatible view and target coordinate combination')
        if view == 'Front':
            real_px_size = self.maps['map'].find_interpolated_pixel_size(x.values, y.values,
                                                                         self.maps['pixel_sizes']['front_f'],
                                                                         self.maps['triang']['front_f'])
            if coord == 'y' or 'z' and coord != 'x':
                real_coord = eval(coord) * real_px_size
            else:
                raise ValueError('Incompatible view and target coordinate combination')

        return real_coord

    def calculate_body_z(self, body_part, step_phase, yref): # todo confirm this is only from side view (relevant for belt calculations)
        '''
        Returns true (subtracted from belt line) height (in z-plane) of one or more body part/s
        :param body_part:
        :param step_phase:
        :return:
        '''
        slope_belt, intercept_belt = self.get_belt_line()
        coords = self.get_body_part_coordinates(body_part, step_phase, yref)
        z_real = self.convert_coords_to_mm(x=coords['x'], y=coords['y'], z=coords['z'], coord='z', view='Side',
                                               yref=yref)
        beltz = coords['x'] * slope_belt + intercept_belt
        beltz_real = self.maps['pixel_sizes']['side_f'].min() * beltz
        return beltz_real - z_real

    def calculate_body_x(self, body_part, step_phase, yref): # mm
        coords = self.get_body_part_coordinates(body_part, step_phase, yref)
        x_real = self.convert_coords_to_mm(x=coords['x'],y=coords['y'],z=coords['z'],coord='x',view='Side',yref=yref)

        ### figure out start line bit
        mask = np.logical_and.reduce((self.df_s.loc(axis=1)['StartPlatL','likelihood'] > pcutoff,
                                      self.df_s.loc(axis=1)['StartPlatR','likelihood'] > pcutoff,
                                      self.df_f.loc(axis=1)['StartPlatL','likelihood'] > pcutoff,
                                      self.df_f.loc(axis=1)['StartPlatR','likelihood'] > pcutoff))
        lx = self.df_s.loc(axis=1)['StartPlatL','x'][mask]
        rx = self.df_s.loc(axis=1)['StartPlatR','x'][mask]
        ly = self.df_f.loc(axis=1)['StartPlatL','x'][mask]
        ry = self.df_f.loc(axis=1)['StartPlatR','x'][mask]
        real_px_size_l = self.maps['map'].find_interpolated_pixel_size(lx.values, ly.values,
                                                                     self.maps['pixel_sizes']['side_f'],
                                                                     self.maps['triang']['side_f'])
        real_px_size_r = self.maps['map'].find_interpolated_pixel_size(rx.values, ry.values,
                                                                       self.maps['pixel_sizes']['side_f'],
                                                                       self.maps['triang']['side_f'])
        real_startL = np.mean(real_px_size_l * lx)
        real_startR = np.mean(real_px_size_r * rx)
        slope, intercept, _, _, _ = stats.linregress([r['x'], l['x']], [r['y'], l['y']])


        return x_real

    def neck_x_displacement_stance(self):
        x = self.calculate_body_x('Back1', 0, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def neck_x_displacement_swing(self):
        x = self.calculate_body_x('Back1', 1, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def tail_x_displacement_stance(self):
        x = self.calculate_body_x('Tail1', 0, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def tail_x_displacement_swing(self):
        x = self.calculate_body_x('Tail1', 1, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def midback_x_displacement_stance(self):
        x = self.calculate_body_x(['Back6','Back7'], 0, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def midback_x_displacement_swing(self):
        x = self.calculate_body_x(['Back6','Back7'], 1, 'overhead')
        return x.iloc[-1] - x.iloc[0]

    def stepping_limb_x_displacement_stance(self):
        x = self.calculate_body_x(self.stepping_limb, 0, 'front')
        return x.iloc[-1] - x.iloc[0]

    def stepping_limb_x_displacement_swing(self):
        x = self.calculate_body_x(self.stepping_limb, 1, 'front')
        return x.iloc[-1] - x.iloc[0]

    def contra_limb_x_displacement_stance(self):
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        body_part = 'ForepawToe%s' % lr
        x = self.calculate_body_x(body_part, 0, 'front')
        return x.iloc[-1] - x.iloc[0]

    def contra_limb_x_displacement_swing(self):
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        body_part = 'ForepawToe%s' % lr
        x = self.calculate_body_x(body_part, 1, 'front')
        return x.iloc[-1] - x.iloc[0]


    # def calculate_body_y(toe_x, toe_y, ankle_x, ankle_y):
    #     # Calculate differences in x and y coordinates
    #     dx = ankle_x - toe_x
    #     dy = ankle_y - toe_y
    #
    #     # Calculate the angle in radians
    #     angle_radians = math.atan2(dy, dx)
    #
    #     # Convert angle from radians to degrees
    #     angle_degrees = math.degrees(angle_radians)
    #
    #     # Ensure the angle is between -180 and 180 degrees
    #     angle_degrees = (angle_degrees + 180) % 360 - 180
    #
    #     return angle_degrees


    # def neck_x_traj_stance(self):
    #     x = self.calculate_body_x('Back1', 0, 'overhead')
    #     return x.values
    #
    # def neck_x_traj_swing(self):
    #     x = self.calculate_body_x('Back1', 1, 'overhead')
    #     return x.values
    #
    # def tail_x_traj_stance(self):
    #     x = self.calculate_body_x('Tail1', 0, 'overhead')
    #     return x.values
    #
    # def tail_x_traj_swing(self):
    #     x = self.calculate_body_x('Tail1', 1, 'overhead')
    #     return x.values
    #
    # def midback_x_traj_stance(self):
    #     x = self.calculate_body_x(['Back6','Back7'], 0, 'overhead')
    #     return x.values
    #
    # def midback_x_traj_swing(self):
    #     x = self.calculate_body_x(['Back6','Back7'], 1, 'overhead')
    #     return x.values
    #
    # def stepping_limb_x_traj_stance(self):
    #     x = self.calculate_body_x(self.stepping_limb, 0, 'front')
    #     return x.values
    #
    # def stepping_limb_x_traj_swing(self):
    #     x = self.calculate_body_x(self.stepping_limb, 1, 'front')
    #     return x.values
    #
    # def contra_limb_x_traj_stance(self):
    #     lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
    #     body_part = 'ForepawToe%s' % lr
    #     x = self.calculate_body_x(body_part, 0, 'front')
    #     return x.values
    #
    # def contra_limb_x_traj_swing(self):
    #     lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
    #     body_part = 'ForepawToe%s' % lr
    #     x = self.calculate_body_x(body_part, 1, 'front')
    #     return x.values


    # to do add function looking at recah angle of stepping paw


    def get_belt_line(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        start = data_chunk.loc(axis=1)['StartPlatR',['x','y']].mean().droplevel(level='bodyparts')
        trans = data_chunk.loc(axis=1)['TransitionR',['x','y']].mean().droplevel(level='bodyparts')
        slope, intercept, _, _, _ = stats.linregress([start['x'], trans['x']], [start['y'], trans['y']])
        return slope, intercept

    def get_start_line(self):
        data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        l = data_chunk.loc(axis=1)['StartPlatL',['x','y']].mean().droplevel(level='bodyparts')
        r = data_chunk.loc(axis=1)['StartPlatR',['x','y']].mean().droplevel(level='bodyparts')
        slope, intercept, _, _, _ = stats.linregress([r['x'], l['x']], [r['y'], l['y']])
        return slope, intercept

    def limb_rel_to_body_stance(self): # back1 is 1, back 12 is 0, further forward than back 1 is 1+
        data_chunk = self.df_s.loc(axis=0)[self.stride_start]
        x_vals = data_chunk.loc(axis=0)[['Back1','Back12',self.stepping_limb], 'x']
        x_vals_zeroed = x_vals - x_vals['Back12']
        x_vals_norm_to_neck = x_vals_zeroed/x_vals_zeroed['Back1']
        result = x_vals_norm_to_neck[self.stepping_limb].values[0]
        return result

    def temp_transformation_of_coordinates(self):
        real_coords = self.maps['map'].get_real_space_coordinates()
        cam_coords_sf = self.maps['map'].get_comb_camera_coords('front')
        h = self.maps['map'].get_homography_matrix(cam_coords_sf, real_coords)

        utils.Utils().plot_polygon_with_numberered_pts(real_coords)
        labels = ['Nose', 'ForepawToeR', 'ForepawToeL', 'TransitionL', 'TransitionR', 'StartPlatR']

        for l in labels:
            target_coords = self.get_body_part_coordinates(l, 1, 'front','side')
            target_coords_3d = [[[target_coords['x'][i]], [target_coords['y'][i]], [1]] for i in target_coords['x'].keys()]

            x, y = self.maps['map'].get_transformed_coordinates(h, target_coords_3d)
            colour_vals = np.arange(0,len(x))

            ax = plt.gca()
            ax.scatter(x,y, c=colour_vals, cmap='cool')
            ax.annotate(l, xy=(x[-1],y[-1]), xytext=(10,10),textcoords='offset points', arrowprops=dict(facecolor='black', arrowstyle='->'))

    def temp_find_perspective_convergence(self, view):
        s, f, o = self.maps['map'].assign_coordinates()
        # Define the endpoints of the two vertical lines
        if view == 'side':
            side_cam_coords = np.array([(s[key]['x'], s[key]['y']) for key in s.keys()])
            side_cam_coords = np.delete(side_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = side_cam_coords[[0, 3]]
            vertical_line2_endpoints = side_cam_coords[[1, 2]]
        elif view == 'front':
            front_cam_coords = np.array([(f[key]['x'], f[key]['y']) for key in f.keys()])
            front_cam_coords = np.delete(front_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = front_cam_coords[[0, 1]]
            vertical_line2_endpoints = front_cam_coords[[3, 2]]

        # Calculate the slope and intercept of each vertical line (mx + b form)
        slope1 = (vertical_line1_endpoints[1, 1] - vertical_line1_endpoints[0, 1]) / (
                    vertical_line1_endpoints[1, 0] - vertical_line1_endpoints[0, 0])
        intercept1 = vertical_line1_endpoints[0, 1] - slope1 * vertical_line1_endpoints[0, 0]
        slope2 = (vertical_line2_endpoints[1, 1] - vertical_line2_endpoints[0, 1]) / (
                    vertical_line2_endpoints[1, 0] - vertical_line2_endpoints[0, 0])
        intercept2 = vertical_line2_endpoints[0, 1] - slope2 * vertical_line2_endpoints[0, 0]
        # Calculate the x-coordinate of the intersection point
        intersection_x = (intercept2 - intercept1) / (slope1 - slope2)
        # Calculate the y-coordinate of the intersection point
        intersection_y = slope1 * intersection_x + intercept1

        return intersection_x, intersection_y


    def temp_find_z(self, label1, label2, h, yref, zref, offset):
        Z = []
        for l in [label1, label2]:
            target_coords = self.get_body_part_coordinates(l, 1, yref, zref)
            if zref == 'side':
                target_coords_3d = [[[target_coords['z'][i] + offset], [target_coords['y'][i]], [1]] for i in
                                target_coords['x'].keys()]
                x, _ = self.maps['map'].get_transformed_coordinates(h, target_coords_3d)
                Z.append(x)
            elif zref == 'front':
                target_coords_3d = [[[target_coords['x'][i]], [target_coords['z'][i] + offset], [1]] for i in
                                target_coords['x'].keys()]
                _, y = self.maps['map'].get_transformed_coordinates(h, target_coords_3d)
                Z.append(y)
        print(Z[0] - Z[1])

    def temp_plot_tracked_xy_points_on_real_space(self, labels, yref):
        real_coords, cam_coords, h = self.get_coords_and_h(yref)

        utils.Utils().plot_polygon_with_numberered_pts(real_coords)
        for l in labels:
            target_coords = self.get_body_part_coordinates(l, 1, yref=yref, zref='side')
            target_coords_3d = [[[target_coords['x'][i]], [target_coords['y'][i]], [1]] for i in
                                target_coords['x'].keys()]

            x, y = self.maps['map'].get_transformed_coordinates(h, target_coords_3d)

            colour_vals = np.arange(0, len(x))

            ax = plt.gca()
            ax.scatter(x, y, c=colour_vals, cmap='cool')
            ax.annotate(l, xy=(x[-1], y[-1]), xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.show()

    def temp_plot_x_diff_in_real_space(self, label1, label2, yref):
        real_coords, cam_coords, h = self.get_coords_and_h(yref)

        X = []
        Y = []
        for lidx, l in enumerate([label1, label2]):
            target_coords = self.get_body_part_coordinates(l, 1, yref=yref, zref='side')
            target_coords_3d = [[[target_coords['x'][i]], [target_coords['y'][i]], [1]] for i in
                                target_coords['x'].keys()]
            x, y = self.maps['map'].get_transformed_coordinates(h, target_coords_3d)
            X.append(x)
            Y.append(y)

        plt.figure()
        plt.plot(X[0], X[0] - X[1])
        plt.title('%s - %s' %(label1, label2))
        #plt.ylim(-1, 5)
        ax = plt.gca()
        ax.set_ylabel('distance (mm)')
        ax.set_xlabel('X position (mm)')
        plt.show()

    # # def get_coords_and_h(self, yref):
    # #     real_coords = self.maps['map'].get_real_space_coordinates()
    # #     cam_coords = self.maps['map'].get_comb_camera_coords(yref)
    # #     h = self.maps['map'].get_homography_matrix(cam_coords, real_coords)
    # #     return real_coords, cam_coords, h
    #
    # def get_coords_and_h(self, yref):
    #     real_coords = self.maps['map'].get_real_space_coordinates()
    #     xref_cam_coords = self.maps['map'].get_cam_coords('side')
    #     yref_cam_coords = self.maps['map'].get_cam_coords(yref)
    #     hx = self.maps['map'].get_homography_matrix(real_coords, xref_cam_coords)
    #     hy = self.maps['map'].get_homography_matrix(real_coords, yref_cam_coords)
    #     return real_coords, xref_cam_coords, yref_cam_coords, hx, hy


    def temp_transform_from_real_to_cam_to_real(self, yref, length=20, depth=5):
        real_coords, xref_cam_coords, yref_cam_coords, hx, hy = self.get_coords_and_h(yref)

        grid = np.array([[[x], [y], [z]] for x in range(0, structural_stuff['belt_length_sideviewrange'] -
                                                        structural_stuff['belt_length_sideviewend'], length) for y in
                         range(0, structural_stuff['belt_width'], depth) for z in [1]])

        x_side, y_side = self.maps['map'].get_transformed_coordinates(hx, grid)
        x_front, y_front = self.maps['map'].get_transformed_coordinates(hy, grid)

        plt.figure()
        plt.scatter(x_side, x_front)
        plt.title('2 - transformed sideXfront grid')
        grid_side_3d = np.array([[[x_side[i][0]], [y_side[i][0]], [1]] for i in range(len(x_side))])
        grid_front_3d = np.array([[[x_front[i][0]], [y_front[i][0]], [1]] for i in range(len(x_front))])

        x_side_real, y_side_real = self.maps['map'].get_transformed_coordinates(np.linalg.inv(hx), grid_side_3d)
        x_front_real, y_front_real = self.maps['map'].get_transformed_coordinates(np.linalg.inv(hy), grid_front_3d)
        plt.figure()
        plt.scatter(x_side_real, y_front_real)
        plt.title('3 - transformed real grid')
        grid_comb_real_3d = np.array([[[x_side_real[i][0]], [y_front_real[i][0]], [1]] for i in range(len(x_side_real))])


    def temp_plot_tracked_xy_points_on_real_space_TWO(self, labels, yref):
        real_coords, xref_cam_coords, yref_cam_coords, hx, hy = self.get_coords_and_h(yref)

        num_limbs = len(labels)
        cmap = plt.cm.get_cmap('cool', num_limbs)
        colors = [cmap(i) for i in range(num_limbs)]

        utils.Utils().plot_polygon_with_numberered_pts(real_coords)
        for idx, l in enumerate(labels):
            xref_body_x, xref_body_y  = self.temp_get_body_part_coordinates_SINGLEVIEW(l, 'side', 1)
            yref_body_x, yref_body_y = self.temp_get_body_part_coordinates_SINGLEVIEW(l, yref, 1)

            xref_3d = np.array([[[xref_body_x.iloc[i]], [xref_body_y.iloc[i]], [1]] for i in range(len(xref_body_x))])
            yref_3d = np.array([[[yref_body_x.iloc[i]], [yref_body_y.iloc[i]], [1]] for i in range(len(yref_body_x))])

            xref_real_x, xref_real_y = self.maps['map'].get_transformed_coordinates(np.linalg.inv(hx), xref_3d)
            yref_real_x, yref_real_y = self.maps['map'].get_transformed_coordinates(np.linalg.inv(hy), yref_3d)

            colour_vals = np.arange(0, len(xref_real_x))

            ax = plt.gca()
            # ax.scatter(xref_real_x, yref_real_y, c=colour_vals, cmap='cool')
            ax.scatter(xref_real_x, yref_real_y, color=colors[idx])
            ax.annotate(l, xy=(xref_real_x[-1], yref_real_y[-1]), xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.show()


class Map():
    def __init__(self, data, con, mouseID, r, stride_start, stride_end, stepping_limb, maps):
        self.data, self.con, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb, self.maps = data, con, mouseID, r, stride_start, stride_end, stepping_limb, maps

        # calculate sumarised dataframes
        self.df_s = self.data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])
        self.df_f = self.data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])
        self.df_o = self.data[con][mouseID]['Overhead'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage'])


    def get_body_part_coordinates(self, body_part, view, step_phase):
        if view == 'side':
            data_chunk = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
            data_chunk_other = self.df_f.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        elif view == 'front':
            data_chunk = self.df_f.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
            data_chunk_other = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        elif view == 'overhead':
            data_chunk = self.df_o.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
            data_chunk_other = self.df_s.loc(axis=0)[np.arange(self.stride_start, self.stride_end)]
        else:
            raise ValueError('Incorrect view given')

        stsw_mask = data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
        other_mask = np.all(data_chunk_other.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
            if isinstance(body_part, list) else data_chunk_other.loc(axis=1)[body_part, 'likelihood'] > pcutoff
        part_mask = np.all(data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff, axis=1) \
            if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'likelihood'] > pcutoff
        mask = np.logical_and.reduce((part_mask, other_mask, stsw_mask))

        xpos = data_chunk.loc(axis=1)[body_part, 'x'][mask].droplevel('coords', axis=1) \
            if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'x'][mask]
        ypos = data_chunk.loc(axis=1)[body_part, 'y'][mask].droplevel('coords', axis=1) \
            if isinstance(body_part, list) else data_chunk.loc(axis=1)[body_part, 'y'][mask]

        return xpos, ypos

    def get_h(self, yref):
        real_coords = self.maps['map'].get_real_space_coordinates()
        cam_coords = self.maps['map'].get_comb_camera_coords(yref)
        h = self.maps['map'].get_homography_matrix(cam_coords, real_coords)
        return h

    def get_xy_coordinates(self, labels, dfx, dfy, yref):
        ydim = 'y' if yref == 'overhead' else 'x'
        mask = np.logical_and(dfx.loc(axis=1)[labels, 'likelihood'] > pcutoff,
                              dfy.loc(axis=1)[labels, 'likelihood'] > pcutoff).droplevel(axis=1, level='coords')
        x = dfx.loc(axis=1)[labels, 'x'][mask]
        y = dfy.loc(axis=1)[labels, ydim][mask]
        y.rename(columns={'x': 'y'}, level='coords', inplace=True)
        XYc = pd.concat([x, y], axis=1)
        return XYc

    def calculate_real_xy(self, XYc, h):
        columns = pd.MultiIndex.from_product([XYc.columns.get_level_values(level='bodyparts').unique(), ['x', 'y', 'z']], names=['bodyparts', 'coords'])
        XYZw = pd.DataFrame(index=XYc.index, columns=columns)
        for l in XYc.columns.get_level_values(level='bodyparts').unique():
            xy_3d = np.array([[[XYc.loc(axis=1)[l,'x'].iloc[i]], [XYc.loc(axis=1)[l,'y'].iloc[i]], [1]] for i in range(len(XYc))])
            xw, yw = self.maps['map'].get_transformed_coordinates(h, xy_3d)
            XYZw[l, 'x'] = xw
            XYZw[l, 'y'] = yw #condense the above
        return XYZw

    def calculate_real_z(self, h, XYc, XYZw, offset):
        for l in XYZw.columns.get_level_values(level='bodyparts').unique():
            sidecam_y = self.df_s.loc(axis=1)[l,'y']
            zy_3d = np.array(
                [[[sidecam_y.iloc[i] + offset], [XYc.loc(axis=1)[l, 'y'].iloc[i]], [1]] for i in range(len(XYc))])
            zw, _ = self.maps['map'].get_transformed_coordinates(h, zy_3d)
            XYZw[l, 'z'] = zw
        return XYZw


    def find_perspective_convergence(self, view):
        s, f, o = self.maps['map'].assign_coordinates()
        # Define the endpoints of the two vertical lines
        if view == 'side':
            side_cam_coords = np.array([(s[key]['x'], s[key]['y']) for key in s.keys()])
            side_cam_coords = np.delete(side_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = side_cam_coords[[0, 3]]
            vertical_line2_endpoints = side_cam_coords[[1, 2]]
        elif view == 'front':
            front_cam_coords = np.array([(f[key]['x'], f[key]['y']) for key in f.keys()])
            front_cam_coords = np.delete(front_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = front_cam_coords[[0, 1]]
            vertical_line2_endpoints = front_cam_coords[[3, 2]]

        # Calculate the slope and intercept of each vertical line (mx + b form)
        slope1 = (vertical_line1_endpoints[1, 1] - vertical_line1_endpoints[0, 1]) / (
                    vertical_line1_endpoints[1, 0] - vertical_line1_endpoints[0, 0])
        intercept1 = vertical_line1_endpoints[0, 1] - slope1 * vertical_line1_endpoints[0, 0]
        slope2 = (vertical_line2_endpoints[1, 1] - vertical_line2_endpoints[0, 1]) / (
                    vertical_line2_endpoints[1, 0] - vertical_line2_endpoints[0, 0])
        intercept2 = vertical_line2_endpoints[0, 1] - slope2 * vertical_line2_endpoints[0, 0]
        # Calculate the x-coordinate of the intersection point
        intersection_x = (intercept2 - intercept1) / (slope1 - slope2)
        # Calculate the y-coordinate of the intersection point
        intersection_y = slope1 * intersection_x + intercept1

        return intersection_x, intersection_y

    def get_real_xyz(self):
        hf = self.get_h('front')
        ho = self.get_h('overhead')

        sideXfront_XYc = self.get_xy_coordinates(labels=label_list['sideXfront'], dfx=self.df_s, dfy=self.df_f, yref='front')
        sideXoverhead_XYc = self.get_xy_coordinates(labels=label_list['sideXoverhead'], dfx=self.df_s, dfy=self.df_o, yref='overhead')

        sideXfront_XYw = self.calculate_real_xy(XYc=sideXfront_XYc, h=hf)
        sideXoverhead_XYw = self.calculate_real_xy(XYc=sideXoverhead_XYc, h=ho)

        offset, _ = self.find_perspective_convergence('side')

        sideXfront_XYZw = self.calculate_real_z(h=hf, XYc=sideXfront_XYc, XYZw=sideXfront_XYw, offset=offset)
        sideXoverhead_XYZw = self.calculate_real_z(h=ho, XYc=sideXoverhead_XYc, XYZw=sideXoverhead_XYw, offset=offset)

        XYZw = pd.concat([sideXoverhead_XYZw, sideXfront_XYZw], axis=1)
        return XYZw


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

        # calculate space maps
        map = Structural_calculations.GetRealDistances(self.data, con, mouseID)
        triang_side_f, pixel_sizes_side_f = map.map_pixel_sizes_to_belt('Side', 'Front') # pixel sizes in y-plane - using front view as reference for depth
        triang_side_o, pixel_sizes_side_o = map.map_pixel_sizes_to_belt('Side', 'Overhead') # pixel sizes in y-plane - using overhead view as reference for depth (pixel sizes are same as above)
        triang_front_f, pixel_sizes_front_f = map.map_pixel_sizes_to_belt('Front', 'Front') # pixel sizes in x-plane - using side view as reference for depth (same map with side as reference for y plane depth)

        maps = {
            'map': map,
            'triang':{'side_f': triang_side_f, 'front_f': triang_front_f, 'side_o': triang_side_o},
            'pixel_sizes': {'side_f': pixel_sizes_side_f, 'front_f': pixel_sizes_front_f, 'side_o': pixel_sizes_side_o}
        }

        for r in tqdm(stride_borders.index.get_level_values(level='Run').unique()):
            stepping_limb = np.array(['ForepawToeR','ForepawToeL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawToeR','ForepawToeL']].count() > 1).values][0]
            try:
                for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no'][:-1]):
                    stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                    stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1] - 1 # todo check i am right to consider the previous frame the end frame

                    class_instance = self.CalculateMeasuresByStride(self.data, con, mouseID, r, stride_start, stride_end, stepping_limb, maps)

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





