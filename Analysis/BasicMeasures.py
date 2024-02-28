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
from scipy.signal import savgol_filter
import itertools




class CalculateMeasuresByStride():
    def __init__(self, XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb):
        self.XYZw, self.con, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb = \
            XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage']).loc(axis=0)[np.arange(self.stride_start,self.stride_end+1)]

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

    def calculate_belt_speed(self): # mm/s
        transition_idx = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,'Transition'].index[0]
        con_speed = re.findall('[A-Z][^A-Z]*', self.con.split('_')[1])
        belt_speed = expstuff['speeds'][con_speed[0]]*10 if self.stride_start < transition_idx else expstuff['speeds'][con_speed[1]]*10
        return belt_speed

    def calculate_belt_x_displacement(self):
        belt_speed = self.calculate_belt_speed()
        time_s = (self.stride_end - self.stride_start)/fps
        distance_mm = belt_speed * time_s # mm/s / s
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
        """
        :return: Duration in seconds
        """
        stride_frames = self.data_chunk.index[-1] - self.data_chunk.index[0]
        return (stride_frames / fps) # * 1000

    def stance_duration(self):
        """
        :return: Duration in seconds
        """
        stance_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 0
        stance = self.data_chunk.loc(axis=1)[self.stepping_limb][stance_mask]
        stance_frames = stance.index[-1] - stance.index[0]
        return (stance_frames / fps) # * 1000

    def swing_duration(self):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        swing = self.data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_frames = swing.index[-1] - swing.index[0]
        return (swing_frames / fps)  # * 1000

    def cadence(self):
        stride_duration = self.stride_duration()
        return 1/stride_duration

    def duty_factor(self): # %
        stance_duration = self.stance_duration()
        stride_duration = self.stride_duration()
        result = (stance_duration / stride_duration) * 100
        return result

    ########### SPEEDS ###########:

    def walking_speed(self, bodypart, speed_correct):
        """
        :param bodypart: Either Tail1 or Back6
        :return: Speed in mm/s
        """
        x_displacement = self.data_chunk.loc(axis=1)[bodypart,'x'].iloc[-1] - self.data_chunk.loc(axis=1)[bodypart,'x'].iloc[0] # mm
        walking_speed_mm_s = x_displacement/self.stride_duration()
        if not speed_correct:
            return walking_speed_mm_s
        else:
            return walking_speed_mm_s - self.calculate_belt_speed()

    def swing_velocity(self, speed_correct):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = self.data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_length = swing.loc(axis=1)['x'].iloc[-1] - swing.loc(axis=1)['x'].iloc[0]
        swing_frames = swing.index[-1] - swing.index[0]
        swing_duration = (swing_frames/fps)# * 1000
        swing_vel = swing_length/swing_duration
        if not speed_correct:
            return swing_vel
        else:
            return swing_vel - self.calculate_belt_speed()

    def instantaneous_swing_velocity(self, speed_correct, xyz, smooth=False):
        """
        Derivative of swing trajectory
        :return: dataframe of velocities for x, y and z
        """
        swing_trajectory = self.traj(self.stepping_limb, step_phase=1,full_stride=False,speed_correct=speed_correct, buffer_size=0)
        time_interval = self.swing_duration()
        d_xyz = swing_trajectory.diff(axis=0)
        v_xyz = d_xyz/time_interval
        if smooth:
            # Optionally, smooth the velocities using Savitzky-Golay filter
            v_xyz = savgol_filter(v_xyz, window_length=3, polyorder=1)
        if not speed_correct:
            return v_xyz[xyz]
        else:
            if xyz == 'x':
                return v_xyz[xyz] - self.calculate_belt_speed()
            else:
                pass


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
        x = x - self.calculate_belt_x_displacement() if speed_correct else x
        if all_vals:
            return x
        else:
            return x.iloc[-1] - x.iloc[0]

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
            return y.iloc[-1] - y.iloc[0]

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
            return z.iloc[-1] - z.iloc[0]

    def traj(self, bodypart, step_phase, full_stride, speed_correct, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            xyz = buffer_chunk.loc(axis=1)[bodypart, ['x','y','z']].droplevel('bodyparts',axis=1)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            xyz = self.data_chunk.loc(axis=1)[bodypart, ['x','y','z']][stsw_mask].droplevel('bodyparts',axis=1)
        xyz['z'] = self.correct_z(xyz['z'])
        if speed_correct:
            xyz['x'] = xyz['x'] - self.calculate_belt_x_displacement()
        xyz_aligned = xyz - xyz.iloc[0]
        return xyz_aligned


    ########### BODY-RELATVE DISTANCES ###########:

    def coo_xyz(self,xyz):
        """
        Centre of oscillation in x, y, OR z of paw with respect to Back6
        :param xyz:
        :return:
        """
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb,'StepCycleFill'] == 1
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        mid_back = swing.loc(axis=0)[mid_t].loc(axis=0)['Back6', xyz]
        limb = swing.loc(axis=0)[mid_t].loc(axis=0)[self.stepping_limb, xyz]
        if xyz == 'z':
            mid_back = self.correct_z(mid_back)
            limb = self.correct_z(limb)
        return limb - mid_back

    def coo_euclidean(self):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == 1
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        bodypart1 = swing.loc(axis=1)['Back6', ['x','y','z']].droplevel('bodyparts', axis=1)
        bodypart2 = swing.loc(axis=1)[self.stepping_limb, ['x','y','z']].droplevel('bodyparts', axis=1)
        bodypart1['z'] = self.correct_z(bodypart1['z'])
        bodypart2['z'] = self.correct_z(bodypart2['z'])
        bodypart1 = bodypart1.loc(axis=0)[mid_t]
        bodypart2 = bodypart2.loc(axis=0)[mid_t]
        distance = np.sqrt((bodypart2['x'] - bodypart1['x'])**2 + (bodypart2['y'] - bodypart1['y'])**2 + (bodypart2['z'] - bodypart1['z'])**2)
        return distance

    def bos_stancestart(self, ref_or_contr, y_or_euc):
        """
        Base of support - Y distance between front paws at start of *stepping limb* or *contralateral limb* stance
        :param all_vals: If true, returns all values from stride
        @:param ref_or_contr: which limb stance start use as timepoint for analysis
        :return (float): base of support
        """
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stance_limb = self.stepping_limb if ref_or_contr == 'ref' else 'ForepawToe%s' %lr
        st_mask = self.data_chunk.loc(axis=1)[stance_limb, 'StepCycleFill'] == 0
        steppinglimb = self.data_chunk.loc(axis=1)[self.stepping_limb][st_mask]
        contrlimb = self.data_chunk.loc(axis=1)[ 'ForepawToe%s' % lr][st_mask]
        if y_or_euc == 'y':
            bos = abs(steppinglimb['y'] - contrlimb['y'])
        else:
            bos = np.sqrt((contrlimb['x'] - steppinglimb['x']) ** 2 + (contrlimb['y'] - steppinglimb['y']) ** 2 + (contrlimb['z'] - steppinglimb['z']) ** 2)
        return bos.values[0]

    def ptp_amplitude_stride(self, bodypart):
        peak = self.correct_z(self.data_chunk.loc(axis=1)[bodypart,'z'].max())
        trough = self.correct_z(self.data_chunk.loc(axis=1)[bodypart,'z'].min())
        return peak - trough

    def body_distance(self, bodyparts, step_phase, all_vals, full_stride, buffer_size=0.25): ### eg body length, midback to forepaw
        bodypart1, bodypart2 = bodyparts
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x1 = buffer_chunk.loc(axis=1)[bodypart1, 'x']
            x2 = buffer_chunk.loc(axis=1)[bodypart2, 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'StepCycleFill'] == step_phase
            x1 = self.data_chunk.loc(axis=1)[bodypart1, 'x'][stsw_mask]
            x2 = self.data_chunk.loc(axis=1)[bodypart2, 'x'][stsw_mask]
        length = abs(x1 - x2)
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

    ########### BODY-RELATIVE TIMINGS/PHASES ###########:

    def double_support(self): # %
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stride_length = self.stride_end - self.stride_start
        contr_swing_posframe_mask = self.data_chunk.loc(axis=1)['ForepawToe%s' % lr,'StepCycle'] ==1
        contr_swing_negframe_mask = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,['RunStart', 'Transition', 'RunEnd'],np.arange(self.stride_start-round(stride_length/4),self.stride_start)].loc(axis=1)['ForepawToe%s' % lr, 'StepCycle'] == 1
        if any(contr_swing_posframe_mask):
            contr_swing_frame = self.data_chunk.index[contr_swing_posframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame)/stride_duration) * 100
        elif any(contr_swing_negframe_mask):
            contr_swing_frame = self.XYZw[self.con][self.mouseID].loc(axis=0)[self.r,['RunStart', 'Transition', 'RunEnd'],np.arange(self.stride_start-round(stride_length/4),self.stride_start)].index[contr_swing_negframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame) / stride_duration) * 100
        else:
            result = 0
        return result

    #todo 3 & 4 support patterns

    #todo def stance_phase(self):

    #todo tail and nose phases

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

    def signed_angle(self, reference_vector, plane_normal, bodyparts, buffer_size=0.25):
        """
        Calculates the signed angle between the vector AB and a reference vector when viewed from a given plane.
        Positive angles are clockwise, and negative angles are anticlockwise from the reference vector.

        I'm assuming your use cases will mostly care about angles when viewed form a specific plane (i.e. side view, top
        view, front view, or any other plane you defined through plane_normal). Use the right hand rule for figuring out the
        appropriate normal vector for any plane.

        E.g. for looking at from side plane (x and z coords) the plane_normal would be [0, 1, 0], and if you want angles
        relative to vertical, reference_vector = [0, 0, 1]; relative to horizontal belt line would be [1,0,0].

        Parameters:
            A (np.array): An (n x 3) array of points.
            B (np.array): An (n x 3) array of points.
            reference_vector (np.array): The reference vector within the plane, defining the "vertical" direction.
            plane_normal (np.array): The normal vector of the plane.

        Returns:
            np.array: An array of signed angles in degrees.

        Example use (you'll have to add the z corrections etc):
            path_ankle = '~/Downloads/Ankle.csv'
            path_toe   = '~/Downloads/Toe.csv'

            A = load_coords(path_ankle)
            B = load_coords(path_toe)
            A = A[['x','y','z']].to_numpy()
            B = B[['x', 'y', 'z']].to_numpy()

            # Define the reference vector (z axis)
            reference_vector = np.array([0, 0, -1])

            # Define the plane normal (Y-axis, for the XZ plane - i.e. side view)
            plane_normal = np.array([0, 1, 0])

            angles_deg = calculate_signed_angle(A, B, reference_vector, plane_normal)
            plot_angles(np.arange(30), angles_deg)
        """
        bodypart1, bodypart2 = bodyparts
        buffer_chunk = self.get_buffer_chunk(buffer_size)
        coord_1 = buffer_chunk.loc(axis=1)[bodypart1, ['x','y','z']].droplevel('bodyparts', axis=1)
        coord_2 = buffer_chunk.loc(axis=1)[bodypart2, ['x','y','z']].droplevel('bodyparts', axis=1)
        # Correct z
        coord_1['z'] = self.correct_z(coord_1['z'])
        coord_2['z'] = self.correct_z(coord_2['z'])

        # Calculate vectors from points A to B
        A = coord_1.to_numpy()
        B = coord_2.to_numpy()
        vectors_AB = B - A

        # Project vectors_AB and reference_vector onto the plane
        # Subtracting the outer product basically removes the component in vectors_AB that is aligned to plane_normal,
        # leaving only the projection onto the reference plane
        vectors_AB_projected = vectors_AB - np.outer(
            np.dot(vectors_AB, plane_normal) / np.linalg.norm(plane_normal) ** 2,
            plane_normal)
        # For clarity, the above line is equivalent to calculating projections onto x, y, and z dims, then just looking at
        # the two relevant dimensions defining your reference plane.
        # vectors_AB_projectedX = np.dot(vectors_AB, [1,0,0])
        # vectors_AB_projectedY = np.dot(vectors_AB, [0,1,0])
        # vectors_AB_projectedZ = np.dot(vectors_AB, [0,0,1])
        # then if your reference plane is in x and z (side view), vectors_AB_projected is just:
        # [vectors_AB_projectedX, zeros column, vectors_AB_projectedZ]

        reference_vector_projected = reference_vector - np.dot(reference_vector, plane_normal) / np.linalg.norm(
            plane_normal) ** 2 * plane_normal

        # Normalize the projected vectors
        vectors_AB_projected_normalized = vectors_AB_projected / np.linalg.norm(vectors_AB_projected, axis=1)[:,
                                                                 np.newaxis]
        reference_vector_projected_normalized = reference_vector_projected / np.linalg.norm(reference_vector_projected)

        # Calculate the angle magnitudes
        dot_products = np.dot(vectors_AB_projected_normalized, reference_vector_projected_normalized)
        angles_rad = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip for numerical stability

        # Determine the sign of the angles using the cross product and the plane normal
        cross_products = np.cross(vectors_AB_projected_normalized, reference_vector_projected_normalized)
        angle_signs = np.sign(np.dot(cross_products,
                                     plane_normal))  # this is a simple way to calculate if angles should be + or -; see right hand rule

        # Apply signs to the angles
        signed_angles_rad = angles_rad * angle_signs

        # Convert to degrees
        signed_angles_deg = np.degrees(signed_angles_rad)

        return pd.DataFrame(signed_angles_deg,index=coord_1)

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


        # #Calculate the vectors from body1 to body2
        #
        # v1 = coord_1 - reference_axis
        # v2 = coord_2 - reference_axis
        #
        # v1_on_plane = v1 - np.dot(v1, reference_axis) * reference_axis
        # v2_on_plane = v2 - np.dot(v2, reference_axis) * reference_axis
        #
        # plane_normal = np.cross(v1_on_plane, v2_on_plane)
        # angle_sign = np.sign(np.dot(plane_normal, reference_axis)) #####!!!
        #
        # angle_radians = np.arctan2(angle_sign * np.linalg.norm(np.cross(v1_on_plane, v2_on_plane)), np.dot(v1_on_plane, v2_on_plane))
        # # Convert angle to degrees and ensure it is in the 0-360 range
        # angle_degrees = np.degrees(angle_radians) % 360
        # # return angle_degrees

        vectors_body1_to_body2 = coord_2 - coord_1

        #start morio
        # sign_angle = np.sign(np.cross(vectors_body1_to_body2, reference_axis))
        # sign_angle = sign_angle[:,1] * sign_angle[:,2]
        #end morio

        # Calculate the dot product between the vectors and the reference axis
        dot_products = np.dot(vectors_body1_to_body2, reference_axis)
        # Calculate the magnitudes of the vectors
        magnitudes_body1_to_body2 = np.linalg.norm(vectors_body1_to_body2, axis=1)
        magnitude_reference_axis = np.linalg.norm(reference_axis)
        # Calculate the cosine of the reach angle
        cosine_reach_angle = dot_products / (magnitudes_body1_to_body2 * magnitude_reference_axis)
        # Calculate the reach angle in radians
        angle_radians = np.arcsin(cosine_reach_angle)# * sign_angle
        # Convert the reach angle to degrees
        angle_degrees = np.degrees(angle_radians) + 90

        if all_vals:
            return angle_degrees
        else:
            return angle_degrees.mean()


# class RunMeasures(CalculateMeasuresByStride):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)


class RunMeasures(CalculateMeasuresByStride):
    """
    calc_obj = CalculateMeasuresByStride(XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
    run_measures = RunMeasures(measures_dict, calc_obj)
    results = run_measures.run()
    """
    def __init__(self, measures_dict, calc_obj, stepping_limb):
        super().__init__(*calc_obj)  # Initialize parent class with the provided arguments
        self.measures_dict = measures_dict
        self.stepping_limb = stepping_limb

    def run(self, datatype, measures):
        result = None
        name = None
        vals = pd.DataFrame()
        for function in measures[datatype].keys():
            if any(measures[datatype][function]):
                if function != 'signed_angle':
                    for param in itertools.product(*measures[datatype][function].values()):
                        result = getattr(self, function)(*param)

                        param_names = list(measures[datatype][function].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                        name = f"{function}({formatted_params})"

                else:
                    for param in measures[datatype][function].keys():
                        result = getattr(self, function)(*measures['single_val_measure_list'][function][param])
                        name = f"{function}({param})"
            else:
                # when no parameters required
                result = getattr(self, function)()
                name = function
            vals[name] = result
        return vals

    def get_all_results(self):
        lr_contr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        lr_ipsi = utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')

        measures = measures_list(stepping_limb=self.stepping_limb, lr_contr=lr_contr, lr_ipsi=lr_ipsi)

        single_val = self.run('single_val_measure_list', measures)
        multi_val = self.run('multi_val_measure_list', measures)

        return single_val, multi_val


class Save():
    def __init__(self, conditions):
        self.conditions = conditions
        #self.data = utils.Utils().GetDFs(conditions,reindexed_loco=True)
        self.XYZw = utils.Utils().Get_XYZw_DFs(conditions)
        self.CalculateMeasuresByStride = CalculateMeasuresByStride

    def find_pre_post_transition_strides(self, con, mouseID, r, numstrides=3):
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
        measures_single = pd.DataFrame(index=multi_index) #,columns=measure_list_flat)
        measures_multi = pd.DataFrame(index=multi_index) #,columns=measure_list_flat)


        for r in tqdm(stride_borders.index.get_level_values(level='Run').unique()):
            stepping_limb = np.array(['ForepawToeR','ForepawToeL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawToeR','ForepawToeL']].count() > 1).values][0]
            try:
                for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no'][:-1]):
                    stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                    stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1] - 1 # todo check i am right to consider the previous frame the end frame

                    #class_instance = self.CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
                    calc_obj = CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
                    measures_dict = measures_list(stepping_limb=stepping_limb,lr_contr=utils.Utils().picking_left_or_right(stepping_limb, 'contr'),lr_ipsi=utils.Utils().picking_left_or_right(stepping_limb, 'ipsi'))

                    run_measures = RunMeasures(measures_dict, calc_obj)
                    singlevals, multivals = run_measures.get_all_results()
                    measures_single.loc(axis=0)[mouseID, r, s] = singlevals
                    measures_multi.loc(axis=0)[mouseID, r, s] = multivals

                    # for m in measure_list_flat:
                    #     try:
                    #         method_name = m
                    #         if hasattr(class_instance, method_name) and callable(getattr(class_instance, method_name)):
                    #             method = getattr(class_instance, method_name)
                    #             result = method()
                    #             measures.loc(axis=0)[mouseID,r, s][m] = result
                    #         else:
                    #             print('Something went wrong for r: %s, stride: %s, measure: %s' %(r,s,m))
                    #     except:
                    #         print('cant plot measure %s' % m)
            except:
                print('cant plot stride %s' %s)

        return measures_single, measures_multi

    def get_discrete_measures_byrun_bystride_ALLMICE(self, con):
        mice = list(self.XYZw[con].keys())

        mouse_measures_single_ALL = []
        mouse_measures_multi_ALL = []
        for midx, mouseID in enumerate(mice):
            try:
                print('Calculating measures for %s (%s/%s)...' %(mouseID,midx,len(mice)-1))
                SwSt = self.find_pre_post_transition_strides_ALL_RUNS(con, mouseID)
                mouse_measures_single, mouse_measures_multi = self.get_discrete_measures_byrun_bystride(SwSt=SwSt, con=con, mouseID=mouseID)
                mouse_measures_single_ALL.append(mouse_measures_single)
                mouse_measures_multi_ALL.append(mouse_measures_multi)
            except:
                print('cant plot mouse %s' %mouseID)
        mouse_measures_single_ALL = pd.concat(mouse_measures_single_ALL)
        mouse_measures_multi_ALL = pd.concat(mouse_measures_multi_ALL)

        return mouse_measures_single_ALL, mouse_measures_multi_ALL


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


conditions = ['APAChar_LowHigh_Repeats_Wash_Day1']
save = Save(conditions)
single, multi = save.get_discrete_measures_byrun_bystride_ALLMICE(conditions[0])


