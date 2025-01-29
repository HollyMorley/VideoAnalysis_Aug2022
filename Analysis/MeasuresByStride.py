import warnings
import pandas as pd
import itertools
from scipy.signal import savgol_filter
import re

from Helpers.Config_23 import *
from Helpers import utils



class CalculateMeasuresByStride():
    def __init__(self, XYZw, mouseID, r, stride_start, stride_end, stepping_limb, conditions):
        self.XYZw, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb, self.conditions = \
            XYZw, mouseID, r, stride_start, stride_end, stepping_limb, conditions

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[mouseID].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage']).loc(axis=0)[np.arange(self.stride_start ,self.stride_end +1)]

    ####################################################################################################################
    ### General utility functions
    ####################################################################################################################

    def calculate_belt_speed(self): # mm/s # todo this doesnt seem to factor in run number/experimental stage??
        speed_condition = self.conditions['speed']
        run_phases = expstuff['condition_exp_runs'][self.conditions['exp']][self.conditions['repeat_extend']]
        for phase_name, run_array in run_phases.items():
            if self.r in run_array:
                run_phase = phase_name
        if 'Baseline' in run_phase or 'Washout' in run_phase:
            belt1_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
            belt2_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
        elif 'APA' in run_phase:
            belt1_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
            belt2_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[1]] * 10 # convert to mm
        else:
            raise ValueError(f'Run phase {run_phase} not recognised')

        transition_idx = self.XYZw[self.mouseID].loc(axis=0)[self.r ,'Transition'].index[0]
        belt_speed = belt1_speed if self.stride_start < transition_idx else belt2_speed

        return belt_speed

    def calculate_belt_x_displacement(self):
        belt_speed = self.calculate_belt_speed()
        time_s = (self.stride_end - self.stride_start ) /fps
        distance_mm = belt_speed * time_s # mm/s / s
        return distance_mm

    def get_buffer_chunk(self, buffer_size):
        """
        Get stride data with x% of stride length in buffer frames either side
        :param buffer_size: percentage as decimal of stride length that want in frames either side of start and end of stride
        :return: the new data chunk
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        stride_length = self.stride_end - self.stride_start
        start = self.stride_start - round(stride_length * buffer_size)
        end = self.stride_end + round(stride_length * buffer_size)
        buffer_chunk = self.XYZw[self.mouseID].loc(axis=0)[
            self.r, ['RunStart', 'Transition', 'RunEnd'], np.arange(start, end)]
        return buffer_chunk

    def convert_notoe_to_toe(self, limb_name):
        limb_name_clean = re.sub(r'_.*$', '', limb_name)
        pattern = r'^(.*?)([RL])$'
        replacement = r'\1' + 'Toe' + r'\2'
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)
        return limb_name_modified

    def convert_toe_to_notoe(self, limb_name):
        # Step 1: Remove any trailing suffix starting with an underscore
        limb_name_clean = re.sub(r'_.*$', '', limb_name)

        # Step 2: Remove 'Toe' before the directional suffix ('R' or 'L')
        pattern = r'^(.*?)(Toe)([RL])$'
        replacement = r'\1\3'  # Replace with the first and third captured groups
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)

        return limb_name_modified

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
        stance_mask = self.data_chunk.loc(axis=1)[self.stepping_limb ,'SwSt'] == locostuff['swst_vals_2025']['st']
        stance = self.data_chunk.loc(axis=1)[self.stepping_limb][stance_mask]
        stance_frames = stance.index[-1] - stance.index[0]
        return (stance_frames / fps) # * 1000

    def swing_duration(self):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_frames = swing.index[-1] - swing.index[0]
        return (swing_frames / fps)  # * 1000

    def cadence(self):
        stride_duration = self.stride_duration()
        return 1/ stride_duration

    def duty_factor(self):  # %
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
        x_displacement = self.data_chunk.loc(axis=1)[bodypart, 'x'].iloc[-1] - \
                         self.data_chunk.loc(axis=1)[bodypart, 'x'].iloc[0]  # mm
        walking_speed_mm_s = x_displacement / self.stride_duration()
        if not speed_correct:
            return walking_speed_mm_s
        else:
            return walking_speed_mm_s - self.calculate_belt_speed()

    def swing_velocity(self, speed_correct):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing = self.data_chunk.loc(axis=1)[limb_name][swing_mask]
        swing_length = swing.loc(axis=1)['x'].iloc[-1] - swing.loc(axis=1)['x'].iloc[0]
        swing_frames = swing.index[-1] - swing.index[0]
        swing_duration = (swing_frames / fps)  # * 1000
        swing_vel = swing_length / swing_duration
        if not speed_correct:
            return swing_vel
        else:
            return swing_vel - self.calculate_belt_speed()

    def instantaneous_swing_velocity(self, speed_correct, xyz, smooth=False):
        """
        Derivative of swing trajectory
        :return: dataframe of velocities for x, y and z
        """
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing_trajectory = self.traj(limb_name, coord=xyz, step_phase='0', full_stride=False,
                                     speed_correct=False, aligned=False,
                                     buffer_size=0)  ####todo check speed correct should be false!
        time_interval = self.swing_duration()
        d_xyz = swing_trajectory.diff()
        v_xyz = d_xyz / time_interval
        if smooth:
            # Optionally, smooth the velocities using Savitzky-Golay filter
            v_xyz = savgol_filter(v_xyz, window_length=3, polyorder=1)
        if not speed_correct:
            return v_xyz
        else:
            if xyz == 'x':
                return v_xyz - self.calculate_belt_speed()
            else:
                return v_xyz

    ########### DISTANCES ###########:

    def stride_length(self, speed_correct):
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        length = self.data_chunk.loc(axis=1)[limb_name, 'x'].iloc[-1] - \
                 self.data_chunk.loc(axis=1)[limb_name, 'x'].iloc[0]
        if not speed_correct:
            return length
        else:
            return length - self.calculate_belt_x_displacement()

    def x(self, bodypart, speed_correct, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x = buffer_chunk.loc(axis=1)[bodypart, 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            x = self.data_chunk.loc(axis=1)[bodypart, 'x'][stsw_mask]
        x = x - self.calculate_belt_x_displacement() if speed_correct else x
        if all_vals:
            return x.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return x.iloc[-1] - x.iloc[0]

    def y(self, bodypart, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            y = buffer_chunk.loc(axis=1)[bodypart, 'y']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            y = self.data_chunk.loc(axis=1)[bodypart, 'y'][stsw_mask]
        if all_vals:
            return y.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return y.iloc[-1] - y.iloc[0]

    def z(self, bodypart, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            z = buffer_chunk.loc(axis=1)[bodypart, 'z']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            z = self.data_chunk.loc(axis=1)[bodypart, 'z'][stsw_mask]
        if all_vals:
            return z.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return z.iloc[-1] - z.iloc[0]

    def traj(self, bodypart, coord, step_phase, full_stride, speed_correct, aligned, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            xyz = buffer_chunk.loc(axis=1)[bodypart, ['x', 'y', 'z']].droplevel('bodyparts', axis=1).droplevel(
                ['Run', 'RunStage'], axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            xyz = self.data_chunk.loc(axis=1)[bodypart, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)
        if speed_correct:
            xyz['x'] = xyz['x'] - self.calculate_belt_x_displacement()
        if aligned:
            xyz = xyz - xyz.loc(axis=0)[
                self.stride_start]  ### todo cant align traj to first position when that is a nan
        return xyz[coord]

    ########### BODY-RELATVE DISTANCES ###########:

    def coo_xyz(self, xyz):
        """
        Centre of oscillation in x, y, OR z of paw with respect to Back6
        :param xyz:
        :return:
        """
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        mid_back = swing.loc(axis=1)['Back6', xyz]  # .loc(axis=0)[mid_t]
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        limb = swing.loc(axis=1)[limb_name, xyz]

        return limb[mid_t] - mid_back[mid_t]

    def coo_euclidean(self):
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        bodypart1 = swing.loc(axis=1)['Back6', ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        bodypart2 = swing.loc(axis=1)[limb_name, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        bodypart1 = bodypart1.loc(axis=0)[mid_t]
        bodypart2 = bodypart2.loc(axis=0)[mid_t]
        distance = np.sqrt((bodypart2['x'] - bodypart1['x']) ** 2 + (bodypart2['y'] - bodypart1['y']) ** 2 + (
                    bodypart2['z'] - bodypart1['z']) ** 2)
        return distance

    def bos_stancestart(self, ref_or_contr, y_or_euc):
        """
        Base of support - Y distance between front paws at start of *stepping limb* or *contralateral limb* stance
        :param all_vals: If true, returns all values from stride
        @:param ref_or_contr: which limb stance start use as timepoint for analysis
        :return (float): base of support
        """
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stance_limb = self.stepping_limb if ref_or_contr == 'ref' else 'Forepaw%s' % lr
        st_mask = self.data_chunk.loc(axis=1)[stance_limb, 'SwSt'] == locostuff['swst_vals_2025']['st']
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        steppinglimb = self.data_chunk.loc(axis=1)[limb_name][st_mask]
        contrlimb = self.data_chunk.loc(axis=1)['ForepawToe%s' % lr][st_mask]
        if y_or_euc == 'y':
            bos = abs(steppinglimb['y'] - contrlimb['y'])
        else:
            bos = np.sqrt((contrlimb['x'] - steppinglimb['x']) ** 2 + (contrlimb['y'] - steppinglimb['y']) ** 2 + (
                        contrlimb['z'] - steppinglimb['z']) ** 2)
        return bos.values[0]

    def ptp_amplitude_stride(self, bodypart):
        coords = self.data_chunk.loc(axis=1)[bodypart, 'z']
        peak = coords.max()
        trough = coords.min()
        return peak - trough

    def body_distance(self, bodyparts, step_phase, all_vals, full_stride,
                      buffer_size=0.25):  ### eg body length, midback to forepaw
        bodypart1, bodypart2 = bodyparts
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x1 = buffer_chunk.loc(axis=1)[bodypart1, 'x']
            x2 = buffer_chunk.loc(axis=1)[bodypart2, 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            x1 = self.data_chunk.loc(axis=1)[bodypart1, 'x'][stsw_mask]
            x2 = self.data_chunk.loc(axis=1)[bodypart2, 'x'][stsw_mask]
        length = abs(x1 - x2)
        if all_vals:
            return length.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return length.mean()

    def back_height(self, back_label, step_phase, all_vals, full_stride, buffer_size=0.25):
        # back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
        #                'Back11', 'Back12']
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            back_heights = buffer_chunk.loc(axis=1)[back_label, 'z'].droplevel(['Run', 'RunStage'],
                                                                                axis=0)  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            back_heights = self.data_chunk.loc(axis=1)[back_label, 'z'][
                stsw_mask]  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        if all_vals:
            # return back_heights.droplevel(['Run','RunStage'],axis=0)
            return back_heights
        else:
            return back_heights.mean(axis=0)

    def tail_height(self, tail_label, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            tail_heights = buffer_chunk.loc(axis=1)[tail_label, 'z'].droplevel(['Run', 'RunStage'],
                                                                                axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            tail_heights = self.data_chunk.loc(axis=1)[tail_label, 'z'][
                stsw_mask]  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        if all_vals:
            # return back_heights.droplevel(['Run','RunStage'],axis=0)
            return tail_heights
        else:
            return tail_heights.mean(axis=0)

    ########### BODY-RELATIVE TIMINGS/PHASES ###########:

    def double_support(self):  # %
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stride_length = self.stride_end - self.stride_start
        contr_swing_posframe_mask = self.data_chunk.loc(axis=1)['Forepaw%s' % lr, 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']
        contr_swing_negframe_mask = self.XYZw[self.mouseID].loc(axis=0)[
                                        self.r, ['RunStart', 'Transition', 'RunEnd'], np.arange(
                                            self.stride_start - round(stride_length / 4), self.stride_start)].loc(
            axis=1)['Forepaw%s' % lr, 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']
        if any(contr_swing_posframe_mask):
            contr_swing_frame = self.data_chunk.index.get_level_values(level='FrameIdx')[contr_swing_posframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame) / stride_duration) * 100
        elif any(contr_swing_negframe_mask):
            contr_swing_frame = self.XYZw[self.mouseID].loc(axis=0)[
                self.r, ['RunStart', 'Transition', 'RunEnd'], np.arange(self.stride_start - round(stride_length / 4),
                                                                        self.stride_start)].index.get_level_values(
                level='FrameIdx')[contr_swing_negframe_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            stride_duration = self.stride_duration()
            result = ((contr_swing_frame - ref_stance_frame) / stride_duration) * 100
        else:
            result = 0
        return result

    # todo 3 & 4 support patterns

    # todo def stance_phase(self):

    # todo tail and nose phases

    ########### BODY-RELATIVE POSITIONING ###########:

    def back_skew(self, step_phase, all_vals, full_stride,
                  buffer_size=0.25):  ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                       'Back11', 'Back12']
        mult = np.arange(1, 13)
        true_back_height = []
        for b in reversed(back_labels):
            true_back_height_label = self.back_height(b, step_phase, all_vals=True, full_stride=full_stride,
                                                      buffer_size=buffer_size)
            true_back_height.append(true_back_height_label)
        true_back_height = pd.concat(true_back_height, axis=1)
        COM = (true_back_height * mult).sum(axis=1) / true_back_height.sum(axis=1)  # calculate centre of mass
        skew = np.median(mult) - COM
        if all_vals:
            return skew
        else:
            return skew.mean()

    def limb_rel_to_body(self, time, step_phase, all_vals, full_stride,
                         buffer_size=0.25):  # back1 is 1, back 12 is 0, further forward than back 1 is 1+
        ### WARNING: while mapping is not fixed, this will be nonsense as back and legs mapped separately
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x = buffer_chunk.loc(axis=1)[['Back1', 'Back12', limb_name], 'x'].droplevel(['Run', 'RunStage'],
                                                                                                 axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            x = self.data_chunk.loc(axis=1)[['Back1', 'Back12', limb_name], 'x'][stsw_mask]
        x_zeroed = x - x['Back12']
        x_norm_to_neck = x_zeroed / x_zeroed['Back1']
        position = x_norm_to_neck[limb_name]
        position = pd.Series(position.values.flatten(), index=position.index)
        if all_vals:
            return position
        else:
            if time == 'start':
                return position.iloc[0]
            elif time == 'end':
                return position.iloc[-1]

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
        coord_1 = buffer_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        coord_2 = buffer_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)

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
        signed_angles_deg = signed_angles_deg.flatten()

        return pd.Series(signed_angles_deg, index=coord_1.index.get_level_values(level='FrameIdx'))

    def angle_3d(self, bodypart1, bodypart2, reference_axis, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Calculate the angle between two body parts relative to a reference axis.
        :param bodypart1 (str): First body part
        :param bodypart2 (str): Second body part
        :param reference_axis (array): Reference axis in the form of a 3D vector.
        :param step_phase: 0 or 1 for swing or stance , respectively #todo this was swapped from the original code 20250128
        :param all_vals: True or False for returning all values in stride or averaging, respectively
        :param full_stride: True or False for analysing all frames from the stride and not splitting into st or sw
        :param buffer_size: Proportion of stride in franes to add before and end as a buffer, 0 to 1
        :return (angle): Angle between the two body parts and the reference axis (in degrees).
        """
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            coord_1 = buffer_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
            coord_2 = buffer_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            coord_1 = self.data_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)
            coord_2 = self.data_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)

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

        # start morio
        # sign_angle = np.sign(np.cross(vectors_body1_to_body2, reference_axis))
        # sign_angle = sign_angle[:,1] * sign_angle[:,2]
        # end morio

        # Calculate the dot product between the vectors and the reference axis
        dot_products = np.dot(vectors_body1_to_body2, reference_axis)
        # Calculate the magnitudes of the vectors
        magnitudes_body1_to_body2 = np.linalg.norm(vectors_body1_to_body2, axis=1)
        magnitude_reference_axis = np.linalg.norm(reference_axis)
        # Calculate the cosine of the reach angle
        cosine_reach_angle = dot_products / (magnitudes_body1_to_body2 * magnitude_reference_axis)
        # Calculate the reach angle in radians
        angle_radians = np.arcsin(cosine_reach_angle)  # * sign_angle
        # Convert the reach angle to degrees
        angle_degrees = np.degrees(angle_radians) + 90

        if all_vals:
            return angle_degrees.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return angle_degrees.mean()


########################################################################
########################################################################
class RunMeasures(CalculateMeasuresByStride):
    """
    calc_obj = CalculateMeasuresByStride(XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
    run_measures = RunMeasures(measures_dict, calc_obj)
    results = run_measures.run()
    """
    def __init__(self, measures, calc_obj, buffer_size, stride):
        super().__init__(calc_obj.XYZw, calc_obj.mouseID, calc_obj.r,
                         calc_obj.stride_start, calc_obj.stride_end, calc_obj.stepping_limb, calc_obj.conditions)  # Initialize parent class with the provided arguments
        self.measures = measures
        self.buffer_size = buffer_size
        self.stride = stride

    def setup_df(self, datatype, measures):
        col_names = []
        for function in measures[datatype].keys():
            if any(measures[datatype][function]):
                if function != 'signed_angle':
                    for param in itertools.product(*measures[datatype][function].values()):
                        param_names = list(measures[datatype][function].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                        col_names.append((function, formatted_params))
                else:
                    for combo in measures[datatype]['signed_angle'].keys():
                        col_names.append((function, combo))
            else:
                col_names.append((function, 'no_param'))

        col_names_trimmed = []
        for c in col_names:
            if np.logical_and('full_stride:True' in c[1], 'step_phase:None' not in c[1]):
                pass
            elif np.logical_and('full_stride:False' in c[1], 'step_phase:None' in c[1]):
                pass
            else:
                col_names_trimmed.append(c)

        buffered_idx = self.get_buffer_chunk(self.buffer_size).index.get_level_values(level='FrameIdx')
        ## add in new index with either 'buffer' or 'stride' based on whether that frame is in the buffer or between stride start and end
        buffer_mask = np.logical_and(buffered_idx >= self.stride_start, buffered_idx <= self.stride_end)
        idx_type = np.where(buffer_mask, 'stride', 'buffer')


        if datatype == 'single_val_measure_list':
            vals = pd.DataFrame(index=[0], columns=pd.MultiIndex.from_tuples(col_names_trimmed, names=['Measure', 'Params']))
        elif datatype == 'multi_val_measure_list':
            mult_index = pd.MultiIndex.from_arrays([buffered_idx, idx_type], names=['FrameIdx', 'Buffer'])
            vals = pd.DataFrame(index=mult_index, columns=pd.MultiIndex.from_tuples(col_names_trimmed, names=['Measure', 'Params']))
        return vals

    def run_calculations(self, datatype, measures):
        vals = self.setup_df(datatype,measures)

        for function in measures[datatype].keys():
            if any(measures[datatype][function]):
                if function != 'signed_angle':
                    for param in itertools.product(*measures[datatype][function].values()):
                        param_names = list(measures[datatype][function].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))

                        if np.logical_and('full_stride:True' in formatted_params, 'step_phase:None' not in formatted_params):
                            pass
                        elif np.logical_and('full_stride:False' in formatted_params, 'step_phase:None' in formatted_params):
                            pass
                        else:
                            result = getattr(self, function)(*param)
                            if datatype == 'single_val_measure_list':
                                vals.loc(axis=1)[(function, formatted_params)] = result
                            elif datatype == 'multi_val_measure_list':
                                # idx_mask = vals.index.get_level_values(level='Buffered_idx').isin(result.index)
                                # full_idx = vals.index[idx_mask]
                                vals.loc[result.index, (function, formatted_params)] = result.values

                else:
                    for combo in measures[datatype]['signed_angle'].keys():
                        result = getattr(self, function)(*measures[datatype][function][combo])
                        # if datatype == 'single_val_measure_list':
                        #     vals.loc(axis=1)[(function, combo)] = result
                        # elif datatype == 'multi_val_measure_list':
                        vals.loc[result.index, (function, combo)] = result.values

            else:
                # when no parameters required
                result = getattr(self, function)()
                if datatype == 'single_val_measure_list':
                    vals.loc(axis=1)[(function, 'no_param')] = result
                elif datatype == 'multi_val_measure_list':
                    vals.loc[result.index, (function, 'no_param')] = result.values

        return vals

    def add_single_idx(self, data):
        single_idx = pd.MultiIndex.from_tuples([(self.mouseID, int(self.r), self.stride)],
                                               names=['MouseID', 'Run', 'Stride'])
        data.set_index(single_idx, inplace=True)
        return data

    def add_multi_idx(selfself, data, single_data):
        single_idx = single_data.index
        data_idx = [data.index.get_level_values(level='FrameIdx'), data.index.get_level_values(level='Buffer')]
        multi_index_tuples = [(a[0], a[1], a[2], data_idx[0][b], data_idx[1][b]) for a in single_idx for b in np.arange(len(data_idx[0]))]
        multi_index = pd.MultiIndex.from_tuples(multi_index_tuples,
                                                names=['MouseID', 'Run', 'Stride', 'FrameIdx', 'Buffer'])
        data.set_index(multi_index, inplace=True)
        return data

    def get_all_results(self):
        single_val = self.run_calculations('single_val_measure_list', self.measures)
        multi_val = self.run_calculations('multi_val_measure_list', self.measures)

        # add in multi indexes
        single_val_indexed = self.add_single_idx(single_val)
        mult_val_indexed = self.add_multi_idx(multi_val,single_val_indexed)

        return single_val_indexed, mult_val_indexed
