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
    def __init__(self, XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb, speed_correct):
        self.XYZw, self.con, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb, self.speed_correct = XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb, speed_correct

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage']).loc(axis=0)[np.arange(self.stride_start,self.stride_end)]

    ####################################################################################################################
    ### General utility functions
    ####################################################################################################################

    def correct_z(self, z_coord): # TEMP!!!!!!!!
        """
        Subtract real z coordinate from belt z to find change in z from belt level. This is only necessary beacuse the 3D mapping is not finished yet and z is an estimate
        :param z_coord: pd series with z coordinates of body part
        :return: difference in z
        """
        idx = z_coord.index
        try:
            beltline = self.XYZw[self.con][self.mouseID].loc(axis=0)[idx].loc(axis=1)[['StartPlatR', 'StartPlatL','TransitionR',
                                                                            'TransitionL'],'z'].mean(axis=1)
            z = z_coord.sub(beltline, axis=0) * -1
        except:
            beltline = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,['RunStart','Transition','RunEnd'],idx].loc(axis=1)[
                ['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], 'z'].mean(axis=1)
            z = z_coord.sub(beltline, axis=0).droplevel([0,1],axis=0) * -1
        return z

    def calculate_belt_speed(self):
        transition_idx = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,'Transition'].index[0]
        con_speed = re.findall('[A-Z][^A-Z]*', self.con.split('_')[1])
        belt_speed = expstuff['speeds'][con_speed[0]]*10 if self.stride_start < transition_idx else expstuff['speeds'][con_speed[1]]*10
        return belt_speed

    def calculate_belt_x_displacement(self):
        belt_speed = self.calculate_belt_speed()
        time_ms = (self.stride_end - self.stride_start)/fps
        distance_mm = belt_speed * time_ms
        return distance_mm

    def get_buffer_chunk(self, buffer_size):
        """
        Get stride data with x% of stride length in buffer frames either side
        :param buffer_size: percentage as decimal of stride length that want in frames either side of start and end of stride
        :return: the new data chunk
        """
        stride_length = self.stride_end - self.stride_start
        start = self.stride_start - round(stride_length * buffer_size)
        end = self.stride_end + round(stride_length * buffer_size)
        buffer_chunk = self.XYZw[self.con][self.mouseID].loc(axis=0)[
            self.r, ['RunStart', 'Transition', 'RunEnd'], np.arange(start, end)]
        return buffer_chunk

    ####################################################################################################################
    ### Single value only calculations
    ####################################################################################################################

    ########### DURATIONS ###########:

    def stride_duration(self):
        stride_frames = self.data_chunk.index[-1] - self.data_chunk.index[0]
        return (stride_frames / fps) * 1000

    def stance_duration(self):
        stance_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        stance = self.data_chunk.loc(axis=1)[self.stepping_limb][stance_mask]
        stance_frames = stance.index[-1] - stance.index[0]
        return (stance_frames / fps) * 1000

    def cadence(self):
        stride_duration = self.stride_duration()
        return 1/stride_duration

    def duty_factor(self): # %
        stance_duration = self.stance_duration()
        stride_duration = self.stride_duration()
        result = (stance_duration / stride_duration) *100
        return result

    def double_support(self): # %
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stride_length = self.stride_end - self.stride_start
        contr_swing_posframe_mask = self.data_chunk.loc(axis=1)['ForepawToe%s' % lr,'StepCycle'] ==1
        contr_swing_negframe_mask = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,['RunStart', 'Transition', 'RunEnd'],np.arange(self.stride_start-round(stride_length/4),self.stride_start)].loc(axis=1)['ForepawToe%s' % lr, 'StepCycle'] == 1
        if any(contr_swing_posframe_mask):
            contr_swing_frame = self.data_chunk.index[contr_swing_posframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame)/stride_duration)*100
        elif any(contr_swing_negframe_mask):
            contr_swing_frame = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,['RunStart', 'Transition', 'RunEnd'],np.arange(self.stride_start-round(stride_length/4),self.stride_start)].index[contr_swing_negframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame) / stride_duration) * 100
        else:
            result = 0
        return result

    #todo 3 & 4 support patterns

    ########### SPEEDS ###########:

    def walking_speed(self, speed_correct):
        x_displacement = self.data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[-1] - self.data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[0]
        walking_speed = x_displacement/self.stride_duration()
        if not speed_correct:
            return walking_speed
        else:
            return walking_speed - self.calculate_belt_speed()

    def swing_velocity(self, speed_correct):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = self.data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_length = swing.loc(axis=1)['x'].iloc[-1] - swing.loc(axis=1)['x'].iloc[0]
        swing_frames = swing.index[-1] - swing.index[0]
        swing_duration = (swing_frames/fps)*1000
        swing_vel = swing_length/swing_duration
        if not speed_correct:
            return swing_vel
        else:
            return swing_vel - self.calculate_belt_speed()

    def tail1_speed(self, speed_correct):
        x_displacement = self.data_chunk.loc(axis=1)['Tail1', 'x'].iloc[-1] - \
                         self.data_chunk.loc(axis=1)['Tail1', 'x'].iloc[0]
        stride_duration = self.stride_duration()
        speed = x_displacement / stride_duration
        if not speed_correct:
            return speed
        else:
            return speed - self.calculate_belt_speed()

    ########### DISTANCES ###########:

    def stride_length(self, speed_correct):
        length = self.data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[-1] - self.data_chunk.loc(axis=1)[self.stepping_limb,'x'].iloc[0]
        if not speed_correct:
            return length
        else:
            return length - self.calculate_belt_x_displacement()

    def x(self, bodypart, speed_correct, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x = buffer_chunk.loc(axis=1)[bodypart, 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            x = self.data_chunk.loc(axis=1)[bodypart, 'x'][stsw_mask]
        x = x - self.calculate_belt_speed() if speed_correct else x
        if all_vals:
            return x
        else:
            return x.mean()

    def y(self, bodypart, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            y = buffer_chunk.loc(axis=1)[bodypart, 'y']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            y = self.data_chunk.loc(axis=1)[bodypart, 'y'][stsw_mask]
        if all_vals:
            return y
        else:
            return y.mean()

    def z(self, bodypart, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            z = self.correct_z(buffer_chunk.loc(axis=1)[bodypart, 'z'])
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            z = self.correct_z(self.data_chunk.loc(axis=1)[bodypart, 'z'][stsw_mask])
        if all_vals:
            return z
        else:
            return z.mean()

    ########### BODY-RELATVE DISTANCES ###########:

    def coo(self,xyz):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        mid_back = swing.loc(axis=0)[mid_t].loc(axis=0)['Back6', xyz]
        limb = swing.loc(axis=0)[mid_t].loc(axis=0)[self.stepping_limb, xyz]
        if xyz == 'z':
            mid_back = self.correct_z(mid_back)
            limb = self.correct_z(limb)
        return limb - mid_back

    def bos_ref_stance(self,all_vals):
        """
        Y distance between front paws during stepping limb stance
        :param all_vals: If true, returns all values from stride
        :return:
        """
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        st_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 0
        bos = abs(self.data_chunk.loc(axis=1)[self.stepping_limb,'y'][st_mask] - self.data_chunk.loc(axis=1)[ 'ForepawToe%s' % lr,'y'][st_mask])
        if all_vals:
            return bos
        else:
            return bos.values[0]

    def tail1_ptp_amplitude_stride(self):
        peak = self.correct_z(self.data_chunk.loc(axis=1)['Tail1','z'].max())
        trough = self.correct_z(self.data_chunk.loc(axis=1)['Tail1','z'].min())
        return peak - trough

    def body_length(self, step_phase, all_vals, full_stride, buffer_size=0.25): ### make this adjustable for full stride too
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            back1x = buffer_chunk.loc(axis=1)['Back1', 'x']
            back12x = buffer_chunk.loc(axis=1)['Back12', 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            back1x = self.data_chunk.loc(axis=1)['Back1', 'x'][stsw_mask]
            back12x = self.data_chunk.loc(axis=1)['Back12', 'x'][stsw_mask]
        length = back1x - back12x
        if all_vals:
            return length
        else:
            return length.mean()

    def back_height(self, step_phase, all_vals, full_stride, buffer_size=0.25):
        back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                       'Back11', 'Back12']
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            back_heights = self.correct_z(buffer_chunk.loc(axis=1)[back_labels, 'z'].droplevel(level='coords', axis=1).iloc[:, ::-1])
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            back_heights = self.correct_z(self.data_chunk.loc(axis=1)[back_labels, 'z'][stsw_mask].droplevel(level='coords', axis=1).iloc[:, ::-1])
        if all_vals:
            return back_heights
        else:
            return back_heights.mean(axis=0)

    ########### BODY-RELATIVE POSITIONING ###########:

    def back_skew(self, step_phase, all_vals, full_stride, buffer_size=0.25): ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        mult = np.arange(1, 13)
        true_back_height = self.back_height(step_phase, all_vals=all_vals, full_stride=full_stride, buffer_size=buffer_size)
        COM = (true_back_height * mult).sum(axis=1) / true_back_height.sum(axis=1) # calculate centre of mass
        skew = np.median(mult) - COM
        if all_vals:
            return skew
        else:
            return skew.mean()

    def limb_rel_to_body(self, step_phase, all_vals, full_stride, buffer_size=0.25): # back1 is 1, back 12 is 0, further forward than back 1 is 1+
        ### WARNING: while mapping is not fixed, this will be nonsense as back and legs mapped separately
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x = buffer_chunk.loc(axis=1)[['Back1','Back12',self.stepping_limb], 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            x = self.data_chunk.loc(axis=1)[['Back1','Back12',self.stepping_limb], 'x'][stsw_mask]
        x_zeroed = x - x['Back12']
        x_norm_to_neck = x_zeroed/x_zeroed['Back1']
        position = x_norm_to_neck[self.stepping_limb].values[0]
        if all_vals:
            return position
        else:
            return position.mean()

    def angle_3d(self, bodypart1, bodypart2, reference_axis, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Calculate the angle between two body parts relative to a reference axis.
        :param bodypart1 (str): First body part
        :param bodypart2 (str): Second body part
        :param reference_axis (array): Reference axis in the form of a 3D vector.
        :param step_phase: 0 or 1 for stance or swing, respectively
        :param all_vals: True or False for returning all values in stride or averaging, respectively
        :param full_stride: True or False for analysing all frames from the stride and not splitting into st or sw
        :param buffer_size: Proportion of stride in franes to add before and end as a buffer, 0 to 1
        :return (angle): Angle between the two body parts and the reference axis (in degrees).
        """
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            coord_1 = buffer_chunk.loc(axis=1)[bodypart1, ['x','y','z']].droplevel('bodyparts', axis=1)
            coord_2 = buffer_chunk.loc(axis=1)[bodypart2, ['x','y','z']].droplevel('bodyparts', axis=1)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            coord_1 = self.data_chunk.loc(axis=1)[bodypart1, ['x','y','z']][stsw_mask].droplevel('bodyparts', axis=1)
            coord_2 = self.data_chunk.loc(axis=1)[bodypart2, ['x','y','z']][stsw_mask].droplevel('bodyparts', axis=1)
        # Correct z
        coord_1['z'] = self.correct_z(coord_1['z'])
        coord_2['z'] = self.correct_z(coord_2['z'])
        # Calculate the vectors from body1 to body2
        vectors_body1_to_body2 = coord_2 - coord_1
        # Calculate the dot product between the vectors and the reference axis
        dot_products = np.dot(vectors_body1_to_body2, reference_axis)
        # Calculate the magnitudes of the vectors
        magnitudes_body1_to_body2 = np.linalg.norm(vectors_body1_to_body2, axis=1)
        magnitude_reference_axis = np.linalg.norm(reference_axis)
        # Calculate the cosine of the reach angle
        cosine_reach_angle = dot_products / (magnitudes_body1_to_body2 * magnitude_reference_axis)
        # Calculate the reach angle in radians
        angle_radians = np.arccos(cosine_reach_angle)
        # Convert the reach angle to degrees
        angle_degrees = np.degrees(angle_radians)
        if all_vals:
            return angle_degrees
        else:
            return angle_degrees.mean()

    # def body_tilt(self, body_parts, step_phase, all_vals, full_stride, buffer_size=0.25):
    #     if full_stride:
    #         buffer_chunk = self.get_buffer_chunk(buffer_size)
    #         z = buffer_chunk.loc(axis=1)[body_parts, 'z'].droplevel('coords', axis=1)
    #         x = buffer_chunk.loc(axis=1)[body_parts, 'x'].droplevel('coords', axis=1)
    #     else:
    #         stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
    #         z = self.data_chunk.loc(axis=1)[body_parts, 'z'][stsw_mask].droplevel('coords', axis=1)
    #         x = self.data_chunk.loc(axis=1)[body_parts, 'z'][stsw_mask].droplevel('coords', axis=1)
    #
    #     slope = (z.loc(axis=1)[body_parts[0]] - z.loc(axis=1)[body_parts[1]]) / (
    #             x.loc(axis=1)[body_parts[0]] - x.loc(axis=1)[body_parts[1]])
    #     angle = np.rad2deg(np.arctan(slope))
    #     if all_vals:
    #         return angle
    #     else:
    #         return angle.mean()




    # todo add function looking at recah angle of stepping paw

    #todo def trajectory AND instantaneous swing vel

    #todo def bos_hom_stance(self):

    #todo def tail1_displacement(self):



    ####################################################################################################################
    ### Multi value calculations
    ####################################################################################################################


    def body_length_stance(self): # px
        return self.calculate_body_length(0)

    def body_length_swing(self): # px
        return self.calculate_body_length(1)





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




class GetMeasuresByStride_AllVal(CalculateMeasuresByStride):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GetMeasuresByStride_SingleVal(CalculateMeasuresByStride):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class Save():
    def __init__(self, conditions):
        self.conditions = conditions
        #self.data = utils.Utils().GetDFs(conditions,reindexed_loco=True)
        self.XYZw = utils.Utils().Get_XYZw_DFs(conditions)
        self.CalculateMeasuresByStride = CalculateMeasuresByStride

    def find_pre_post_transition_strides(self, con, mouseID, r, numstrides=4):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        pre_frame = self.XYZw[con][mouseID].loc(axis=0)[r, 'RunStart'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[-1]
        post_frame = self.XYZw[con][mouseID].loc(axis=0)[r, 'Transition'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[0]
        trans_limb_mask = post_frame - pre_frame == -1
        stepping_limb = np.array(['ForepawToeL', 'ForepawToeR'])[trans_limb_mask]
        if len(stepping_limb) == 1:
            stepping_limb = stepping_limb[0]
        else:
            raise ValueError('wrong number of stepping limbs identified')

        # limbs_mask_post = (self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition', 'RunEnd']].loc(axis=1)[['ForepawToeR','ForepawToeL'], 'likelihood'] > pcutoff).any(axis=1)
        #
        stance_mask_pre = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_pre = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1
        stance_mask_post = self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_post = self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1

        stance_idx_pre = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][stance_mask_pre].tail(numstrides))
        swing_idx_pre = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][swing_mask_pre].tail(numstrides))
        stance_idx_post = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][stance_mask_post].head(2))
        swing_idx_post = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][swing_mask_post].head(2))

        stance_idx_pre['Stride_no'] = np.sort(np.arange(1,len(stance_idx_pre)+1)*-1)
        swing_idx_pre['Stride_no'] = np.sort(np.arange(1,len(swing_idx_pre)+1)*-1)
        stance_idx_post['Stride_no'] = np.arange(0,len(stance_idx_post))
        swing_idx_post['Stride_no'] = np.arange(0,len(swing_idx_post))


        # Combine pre and post DataFrames
        combined_df = pd.concat([stance_idx_pre,swing_idx_pre, stance_idx_post, swing_idx_post]).sort_index(level='FrameIdx')

        return combined_df

    def find_pre_post_transition_strides_ALL_RUNS(self, con, mouseID):
        #view = 'Side'
        SwSt = []
        for r in self.XYZw[con][mouseID].index.get_level_values(level='Run').unique().astype(int):
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
                    stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1] - 1 # todo check i am right to consider the previous frame the end frame

                    class_instance = self.CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)

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
        mice = list(self.XYZw[con].keys())

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





