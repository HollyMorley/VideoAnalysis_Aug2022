import numpy as np
import os
import pandas as pd
import Helpers.utils as utils
import Helpers.GetRuns as GetRuns
import Velocity_v2 as Velocity
from Helpers.Config_23 import *
from Helpers import Structural_calculations
import scipy
from scipy.signal import savgol_filter
from scipy import stats
from tqdm import tqdm
import math
import warnings

class LocomotionDefinition():
    def __init__(self, data, con, mouseID): # MouseData is input ie list of h5 files
        self.data = data
        self.con = con
        self.mouseID = mouseID

    def getLocoPeriods(self, view='Side', fillvalues=True, n=30):
        """
        Add in new columns to dataframe for step cycles for all limbs + fra,es where mouse is sitting
        :param markerstuff:
        :param view:
        :param fillvalues:
        :param n:
        :return:
        """
        warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
        warnings.filterwarnings("ignore", message="The behavior of indexing on a MultiIndex with a nested sequence of labels is deprecated and will change in a future version.")
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']
        windowsize = math.ceil((fps / n) / 2.) * 2
        triang_o, pixel_sizes_o = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).map_pixel_sizes_to_belt('Side', 'Overhead')
        triang_f, pixel_sizes_f = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).map_pixel_sizes_to_belt('Side', 'Front')
        velstuff = Velocity.Velocity(self.data, self.con, self.mouseID).\
            getVelocityInfo(vel_zeroed=False,
                            xaxis='x',
                            f=range(0,int(self.data[self.con][self.mouseID][view].index.get_level_values(level='Run').unique().max()+1)),
                            windowsize=windowsize,
                            triang=triang_o,
                            pixel_sizes=pixel_sizes_o)

        # take record of which runs include a period of sitting
        sitting_log = self.getPotentialSittingRuns(view, velstuff)
        sitting_r_mask = np.isin(self.data[self.con][self.mouseID][view].index.get_level_values(level='Run'), sitting_log)
        sitting_r = sitting_r_mask*1
        for v in ['Side', 'Front', 'Overhead']:
            self.data[self.con][self.mouseID][v].loc(axis=1)['Sitting'] = sitting_r

        for l in limblist:
            print('\n{}Finding locomotor periods for %s...{}\n------------------------------------------>>>'.format('\033[1m', '\033[0m') %l)
            StepCycleAll = []
            StepCycleFilledAll = []
            for r in self.data[self.con][self.mouseID][view].index.get_level_values(level='Run').unique().astype(int):
                try:
                    if 'Fore' in l:
                        #swst_dict = self.get_limb_swst_bothcam(self.data,self.con,self.mouseID,r,l,velstuff,markerstuff,sitting_log)
                        swst_dict = self.get_limb_swst_frontcam(r=r, l=l, velstuff=velstuff, windowsize=windowsize, triang=triang_f, pixel_sizes=pixel_sizes_f)
                    elif 'Hind' in l:
                        swst_dict = self.get_limb_swst_sidecam(r=r,l=l,velstuff=velstuff,triang=triang_f,pixel_sizes=pixel_sizes_f,sitting_log=sitting_log)

                    # put first swing and stance frames into df
                    StepCycle = np.full([len(self.data[self.con][self.mouseID][view].loc(axis=0)[r])], np.nan)
                    swingmask = np.isin(self.data[self.con][self.mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['swingstart'])
                    stancemask = np.isin(self.data[self.con][self.mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, swst_dict['stancestart'])

                    StepCycle[stancemask] = 0
                    StepCycle[swingmask] = 1

                    swingmask_bkwd = np.isin(self.data[self.con][self.mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['swingstart_bkwd'])
                    #stancemask_bkwd = np.isin(self.data[self.con][self.mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['stancestart_bkwd'])

                    #StepCycle[stancemask_bkwd] = 2
                    StepCycle[swingmask_bkwd] = 3

                    if not np.any(swingmask & stancemask):
                        if fillvalues == True:
                            lastval = np.nan
                            newcycle = np.full([len(StepCycle)], np.nan)
                            for i, val in enumerate(StepCycle):
                                if val == 0 or val == 1:
                                    lastval = val
                                    newcycle[i] = val
                                elif np.isnan(val):
                                    newcycle[i] = lastval
                            StepCycleFilledAll.append(newcycle)
                    else:
                        raise ValueError(
                            'There is overlap between detection of first stance and first swing frame! This can''t be possible')

                    StepCycleAll.append(StepCycle)


                except:
                    print('Cant get sw/st data for run: %s, limb: %s' % (r, l))
                    StepCycle = np.full([len(self.data[self.con][self.mouseID][view].loc(axis=0)[r])], np.nan)
                    StepCycleAll.append(StepCycle)
                    StepCycleFilledAll.append(StepCycle)

            StepCycleAll_flt = np.concatenate(StepCycleAll).ravel()
            for v in ['Side', 'Front', 'Overhead']:
                self.data[self.con][self.mouseID][v].loc(axis=1)[l, 'StepCycle'] = StepCycleAll_flt
            if fillvalues == True:
                StepCycleFilledAll_flt = np.concatenate(StepCycleFilledAll).ravel()
                for v in ['Side', 'Front', 'Overhead']:
                    self.data[self.con][self.mouseID][v].loc(axis=1)[l, 'StepCycleFill'] = StepCycleFilledAll_flt

        return self.data[self.con][self.mouseID]

    def getPotentialSittingRuns(self, view, velstuff):
        # take record of which runs include a period of sitting WITHOUT limb data
        sitting_log_mask = []
        for r in self.data[self.con][self.mouseID][view].index.get_level_values(level='Run').unique().astype(int):
            try:
                slow_mask = velstuff['runs_lowess'][r][1] < 8
                if sum(slow_mask) > 10:
                    mouse_stat_blocks = utils.Utils().find_blocks(velstuff['runs_lowess'][r][1][slow_mask].index.get_level_values(level='FrameIdx'), 10, 20)
                    if np.any(mouse_stat_blocks):
                        sitting_log_mask.append(True)
                    else:
                        sitting_log_mask.append(False)
                else:
                    sitting_log_mask.append(False)
            except:
                sitting_log_mask.append(True)  # mark as sitting if cant get velocity data
        sitting_log = self.data[self.con][self.mouseID][view].index.get_level_values(level='Run').unique().astype(int)[
            sitting_log_mask].to_numpy()

        return sitting_log

    def get_limb_swst_frontcam(self, r, l, velstuff, windowsize, triang, pixel_sizes, view='Front'):
        ############ use triang with 'Front' Front' to find real y movement
        commonidx = velstuff['runs_lowess'][int(r), 1].index.get_level_values(level='FrameIdx')
        front_mask = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'likelihood'].values > 0.99
        front_y = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'y'][front_mask]

        y = - front_y.values
        t = front_y.index.get_level_values(level='FrameIdx')
        model = np.poly1d(np.polyfit(t, y, 2))
        front_y_rot = y - model(t)
        front_y_rot_ser = pd.Series(data=front_y_rot, index=t)
        front_y_rot_ser = front_y_rot_ser + abs(front_y_rot_ser.min())

        #stance_start_idx, stance_end_idx = self.find_stat_blocks(vel=front_y_rot_ser, gap_thresh=5, frame_diff_thresh=0.5, speed_thresh=0, allowance=0.5)
        stance_start_idx, stance_end_idx = stance_start_idx, stance_end_idx = self.find_stat_blocks(vel=front_y_rot_ser, gap_thresh=5, frame_diff_thresh=0.5, speed_thresh=0, allowance=0.5)
        # check for backwards steps
        v_x = Velocity.Velocity(self.data, self.con, self.mouseID).getVelocity_specific_limb(l, 'Side', r, windowsize, triang, pixel_sizes)
        n_peaks, neg_info = scipy.signal.find_peaks(-1 * v_x.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], commonidx], distance=10, height=100,prominence=200)
        neg_peaks_idx = v_x.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], commonidx].index.get_level_values(level='FrameIdx')[n_peaks]

        swingstart_bkwd = []
        if any(neg_peaks_idx):
            for i in neg_peaks_idx:
                neg_swing = np.logical_and(i > stance_end_idx, i < np.roll(stance_start_idx, -1))
                if np.any(neg_swing):
                    position = np.where(neg_swing)[0][0]
                    swingstart_bkwd.append(stance_end_idx[position])
                    stance_end_idx = np.delete(stance_end_idx, position)

        swst_dict = {
            'stancestart': stance_start_idx,
            'swingstart': stance_end_idx,
            #'stancestart_bkwd': [],
            'swingstart_bkwd': swingstart_bkwd
        }

        return swst_dict

    def get_limb_swst_sidecam(self, r, l, velstuff, triang, pixel_sizes, sitting_log, view='Side', n=30):
        windowsize = math.ceil((fps / n) / 2.) * 2
        commonidx = velstuff['runs_lowess'][int(r), 1].index.get_level_values(level='FrameIdx')
        mask = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
                   l, 'likelihood'].values > pcutoff  # change so that NO LONGER excluding when mouse stationary (at this stage)

        # get vel data for whole mouse, remove sitting periods from available data, and identify runs with sitting
        mouse_stat_mask = abs(velstuff['runs_lowess'][r][1].diff()) < 0.1
        if sum(mouse_stat_mask) > 100:
            mouse_stat_blocks = utils.Utils().find_blocks(
                velstuff['runs_lowess'][r][1][mouse_stat_mask].index.get_level_values(level='FrameIdx'), 10, 20)
            if np.any(mouse_stat_blocks):
                mouse_stat_blocks_total = []
                for b in range(len(mouse_stat_blocks)):
                    idxs = np.arange(mouse_stat_blocks[b][0], mouse_stat_blocks[b][1])
                    mouse_stat_blocks_total.append(idxs)
                mouse_stat_blocks_total = np.concatenate(mouse_stat_blocks_total)
                mouse_stat_blocks_mask = np.isin(self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition',
                                                                                          'RunEnd'], commonidx].index.get_level_values(
                    level='FrameIdx'), mouse_stat_blocks_total)
                mask = mask & ~mouse_stat_blocks_mask

        ####### NEW ##########
        # fill in the gaps of missing data to avoid erronous velocity data being calculated
        #if l == 'ForepawToeL':
        interpolated_x = self.fill_in_limb_x_gaps(r,l,mask,commonidx)
        index_mask = np.isin(self.data[self.con][self.mouseID][view].index.get_level_values(level='FrameIdx'), interpolated_x.index)
        temp_data = self.data[self.con][self.mouseID][view].loc(axis=1)[l, 'x'].values
        temp_data[index_mask] = interpolated_x.values
        self.data[self.con][self.mouseID][view].loc(axis=1)[l, 'x'] = temp_data

        # get vel data for limb
        vel_limb = Velocity.Velocity(self.data, self.con, self.mouseID).getVelocity_specific_limb(l, view, r, windowsize, triang, pixel_sizes)  # get the velocity data for this body part (limb)
        vel_limb = vel_limb.loc[['RunStart', 'Transition', 'RunEnd'], interpolated_x.index.get_level_values(level='FrameIdx')]

        # trim velocity data by where nose still in frame at end of run (some runs seem to go beyond this)
        nose_mask = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)['Nose', 'likelihood'] > pcutoff
        last_frame = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[nose_mask][-1]
        vel_end_mask = vel_limb.index.get_level_values(level='FrameIdx') <= last_frame
        vel_limb = vel_limb[vel_end_mask]

        # find swing stance pattern
        swst_sequence = self.get_swst_by_speed(l, r, vel_limb, sitting_log)

        # find if there are any backwards steps and relabel
        if np.any(swst_sequence == 'neg_peak'):
            backwards_pos = np.where(swst_sequence['val_type'] == 'neg_peak')[0]
            for i in range(len(backwards_pos)):
                backwards_swing_pos = backwards_pos[i] - 1
                backwards_stance_pos = backwards_pos[i] + 1
                swst_sequence.iloc[backwards_swing_pos] = 'swing_bkwd'
                swst_sequence.iloc[backwards_stance_pos] = 'stance_bkwd'

        # remove the peaks data from the final dataframe
        swst = swst_sequence[swst_sequence['val_type'].isin(['stance', 'swing', 'stance_bkwd', 'swing_bkwd'])]

        stancestart = swst.index[(swst.loc(axis=1)['val_type'].values == 'stance').flatten()]
        swingstart = swst.index[(swst.loc(axis=1)['val_type'].values == 'swing').flatten()]
        stancestart_bkwd = swst.index[(swst.loc(axis=1)['val_type'].values == 'stance_bkwd').flatten()]
        swingstart_bkwd = swst.index[(swst.loc(axis=1)['val_type'].values == 'swing_bkwd').flatten()]

        swst_dict = {
            'stancestart': stancestart,
            'swingstart': swingstart,
            'stancestart_bkwd': stancestart_bkwd,
            'swingstart_bkwd': swingstart_bkwd
        }

        return swst_dict

    def fill_in_limb_x_gaps(self, r, l, mask, commonidx, view='Side'):
        outlier_mask = utils.Utils().find_outliers(xdf=self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'], mask=True)
        x = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].values[mask & ~outlier_mask]
        t = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask & ~outlier_mask]
        present = pd.DataFrame(data=x, index=t)[0]
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t, x)
        missing_t = self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[~mask | outlier_mask]
        missing_x = cs(self.data[self.con][self.mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[~mask | outlier_mask])
        missing = pd.DataFrame(data=missing_x, index=missing_t)[0]
        end_mask = np.logical_and(missing.index <= t[-1], missing.index >= t[1])
        missing = missing[end_mask]
        interpolated_x = pd.concat([present, missing], axis=0).sort_index()
        return interpolated_x

    def get_swst_by_speed(self,l,r,vel_limb,sitting_log):
        vel_limb_cropped_nan = vel_limb[~vel_limb.isna()]
        vel_limb_smoothed = vel_limb_cropped_nan.rolling(5, min_periods=0, center=True).sum()

        ############################### find peaks of velocity data to find swings #####################################
        peaks, info = scipy.signal.find_peaks(vel_limb_smoothed, distance=15, height=2500, prominence=100) # heigh changed from 30 to 25 on 14/8/23
        n_peaks, neg_info = scipy.signal.find_peaks(-1*vel_limb_smoothed, distance=10, height=2500, prominence=100) # height changed from 10 to 25 on 15/8/23
        peaks_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[peaks]
        neg_peaks_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[n_peaks]

        # find rising and falling periods
        try:
            falling = vel_limb_smoothed[peaks].values - vel_limb_smoothed[peaks+10].values
        except:
            falling = vel_limb_smoothed[peaks[:-1]].values - vel_limb_smoothed[peaks[:-1]+10].values
        try:
            rising = vel_limb_smoothed[peaks].values - vel_limb_smoothed[peaks-10].values
        except:
            rising = vel_limb_smoothed[peaks[1:]].values - vel_limb_smoothed[peaks[1:] - 10].values
        if np.all(falling > 0) and np.all(rising > 0):
            pass
        elif np.all(falling[:-1] > 0) and np.all(rising[1:] > 0):
            print('First or last peak is close to edge for run %s, leaving it as true.' %r)
        else:
            raise ValueError('This is not a real peak for swing')

        # ################## find stationary blocks in velocity data to find beginning and end of stance #################
        # #### new ###
        # slope, intercept, r_value, p_value, std_err = stats.linregress(vel_limb_smoothed.index.get_level_values(level='FrameIdx'), vel_limb_smoothed.values)
        # rot_vel_limb_smoothed = vel_limb_smoothed.values - ((vel_limb_smoothed.index.get_level_values(level='FrameIdx') - vel_limb_smoothed.index.get_level_values(level='FrameIdx')[0]) * slope)
        # rot_vel_limb_smoothed = pd.Series(data=rot_vel_limb_smoothed, index=vel_limb_smoothed.index)
        # rot_vel_limb_smoothed = rot_vel_limb_smoothed + abs(rot_vel_limb_smoothed.min())
        ############

        #stance_start_idx, stance_end_idx = self.find_speed_changes(vel_limb_smoothed)
        stance_start_idx, stance_end_idx = self.find_stat_periods(vel_limb_smoothed)
        # if l == 'HindpawToeL':
        #     stance_start_idx, stance_end_idx = self.find_stat_blocks(vel_limb_smoothed, gap_thresh=5, frame_diff_thresh=200)
        #     stance_start_idx_bkup, stance_end_idx_bkup = self.find_stat_blocks(rot_vel_limb_smoothed, gap_thresh=5, frame_diff_thresh=200)
        # else:
        #     stance_start_idx, stance_end_idx = self.find_stat_blocks(vel_limb_smoothed, gap_thresh=5)
        #     stance_start_idx_bkup, stance_end_idx_bkup = self.find_stat_blocks(rot_vel_limb_smoothed, gap_thresh=5)

        # check if any of these values are the result of gaps in the data and remove accordingly
        frame_blocks = utils.Utils().find_blocks(vel_limb_smoothed.index.get_level_values(level='FrameIdx'),10,0)
        matching_values_start = np.intersect1d(stance_start_idx[1:], frame_blocks[:, 0])
        if len(matching_values_start) > 0:
            for i in range(len(matching_values_start)):
                current_start = vel_limb_smoothed[np.where(vel_limb_smoothed.index.get_level_values(level='FrameIdx') == matching_values_start[i])[0][0]]
                prev_start = vel_limb_smoothed[np.where(vel_limb_smoothed.index.get_level_values(level='FrameIdx') == matching_values_start[i])[0][0] - 1]
                if abs(current_start - prev_start) < 10:
                    stance_start_idx = np.setdiff1d(stance_start_idx, matching_values_start[i])
        matching_values_end = np.intersect1d(stance_end_idx, frame_blocks[:, 1])
        if len(matching_values_end) > 0:
            stance_end_idx = np.setdiff1d(stance_end_idx, matching_values_end)

        ########################## form dfs of each data point to then combine together ################################
        pos_peaks = pd.DataFrame(data=['pos_peak'] * len(peaks_idx), index=peaks_idx, columns=['val_type'])
        neg_peaks = pd.DataFrame(data=['neg_peak'] * len(neg_peaks_idx), index=neg_peaks_idx, columns=['val_type'])
        stance = pd.DataFrame(data=['stance'] * len(stance_start_idx), index=stance_start_idx, columns=['val_type'])
        swing = pd.DataFrame(data=['swing'] * len(stance_end_idx), index=stance_end_idx, columns=['val_type'])
        # stance_bkup = pd.DataFrame(data=['stance'] * len(stance_start_idx_bkup), index=stance_start_idx_bkup, columns=['val_type'])
        # swing_bkup = pd.DataFrame(data=['swing'] * len(stance_end_idx_bkup), index=stance_end_idx_bkup, columns=['val_type'])
        swst_sequence = pd.concat([pos_peaks, neg_peaks, stance, swing])
        swst_sequence = swst_sequence.sort_index()
        # swst_sequence_bkup = pd.concat([pos_peaks, neg_peaks, stance_bkup, swing_bkup])
        # swst_sequence_bkup = swst_sequence_bkup.sort_index()

        # #### pre-check the pattern to remove erronous neg peaks for FL which occur as a result of poor data quality and interpolation
        # if l == 'ForepawToeL':
        #     to_del_neg = []
        #     for i in range(0,len(neg_peaks_idx)):
        #         # if the previous value is stance and the next is swing then this must be wrong so delete (if previous is swing and next stance this would indicate a missing peak, aka backwards swing)
        #         if swst_sequence.shift(-1).loc[neg_peaks_idx[i]].values == 'swing' and swst_sequence.shift(1).loc[neg_peaks_idx].values == 'stance':
        #             to_del_neg.append(neg_peaks_idx[i])
        #     if np.any(to_del_neg):
        #         swst_sequence = swst_sequence.drop(to_del_neg)
        #         # swst_sequence_bkup = swst_sequence_bkup.drop(to_del_neg)


        ########################################### check sequence #####################################################
        #try:
        self.check_swst_pattern_limb_speed(swst_sequence,r,sitting_log)
            #print("The swing-stance pattern is followed in run %s" % r)
        # except:
        #     self.check_swst_pattern_limb_speed(swst_sequence_bkup,r,sitting_log) # use rotated values in case missing st or sw. Data will be less accurate but detections there at least
        #     #print("The swing-stance pattern is followed in run %s ** ONLY WITH ROTATION - LESS ACCURATE **" %r)

        return swst_sequence

    def find_speed_changes(self, vel):
        frame_shift =  vel - vel.shift(-1)
        frame_shift_abs = abs(frame_shift)
        peaks, info = scipy.signal.find_peaks(frame_shift_abs, distance=1, height=400, prominence=100)  # heigh changed from 30 to 25 on 14/8/23
        peaks_idx = frame_shift_abs.index.get_level_values(level='FrameIdx')[peaks]
        pos_mask = frame_shift.loc(axis=0)[['RunStart','Transition','RunEnd'], peaks_idx] > 0
        stance_start_idx = frame_shift.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], peaks_idx][pos_mask].index.get_level_values('FrameIdx')
        swing_start_idx = frame_shift.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], peaks_idx][~pos_mask].index.get_level_values('FrameIdx')

        return stance_start_idx, swing_start_idx

    def find_stat_periods(self, vel, gap_thresh=5, frame_diff_thresh=700): #, speed_thresh=0, allowance=0.5):
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(vel.index.get_level_values(level='FrameIdx'), vel.values)
        index = np.arange(vel.index.get_level_values(level='FrameIdx').min(),
                          vel.index.get_level_values(level='FrameIdx').max())
        vel_interp = pd.Series(cs(index), index=index)
        frame_shift = vel_interp - vel_interp.shift(-1)
        stationary_mask = np.logical_and(np.abs(frame_shift) < frame_diff_thresh, vel_interp < vel_interp.mean() + vel_interp.sem()*16)
        # stationary_mean = np.mean(vel[stationary_mask].values)
        # stationary_sem = np.std(vel[stationary_mask].values) / np.sqrt(len(vel[stationary_mask]))
        # stationary_window_mask = np.logical_and(vel.values > speed_thresh, vel.values < stationary_mean + stationary_sem * allowance)
        # stance_raw = vel[stationary_window_mask]
        stance_raw = vel_interp[stationary_mask]
        stance_start_idx = np.array(utils.Utils().find_blocks(stance_raw.index, gap_thresh, 10))[:, 0] # was 5
        stance_end_idx = np.array(utils.Utils().find_blocks(stance_raw.index, gap_thresh, 10))[:, 1] # was 5

        return stance_start_idx, stance_end_idx

    def find_stat_blocks(self, vel, gap_thresh, frame_diff_thresh=5, speed_thresh=2, allowance=0.5):
        frame_shift = vel.shift(1) - vel
        stationary_mask = np.abs(frame_shift) < frame_diff_thresh
        stationary_mean = np.mean(vel[stationary_mask].values)
        stationary_sem = np.std(vel[stationary_mask].values) / np.sqrt(len(vel[stationary_mask]))
        stationary_window_mask = np.logical_and(vel.values > speed_thresh,
                                                vel.values < stationary_mean + stationary_sem * allowance)
        stance_raw = vel[stationary_window_mask]
        stance_start_idx = np.array(
            utils.Utils().find_blocks(stance_raw.index.get_level_values(level='FrameIdx'), gap_thresh, 2))[:,
                           0]  # was 5
        stance_end_idx = np.array(
            utils.Utils().find_blocks(stance_raw.index.get_level_values(level='FrameIdx'), gap_thresh, 2))[:,
                         1]  # was 5

        return stance_start_idx, stance_end_idx

    def check_swst_pattern_limb_speed(self,swst_sequence,r,sitting_log):
        ## should alternate between stance and swing but there should also be a peak (pos or neg) detected after swing to prove the validity of this swing
        val_types = swst_sequence['val_type'].tolist()
        # Define the pattern
        pattern = ["stance", "swing", ["pos_peak", "neg_peak"]]
        # Check the pattern for each value in the list
        for i in range(len(val_types) - 1):
            current_val = val_types[i]
            next_val = val_types[i + 1]
            if current_val == "stance" and next_val != "swing":
                if r not in sitting_log:
                    raise ValueError("Pattern mismatch: 'stance' should be followed by 'swing'.")
            elif current_val == "swing" and next_val not in pattern[2]:
                if r not in sitting_log:
                    raise ValueError("Pattern mismatch: 'swing' should be followed by either 'pos_peak' or 'neg_peak'.")
            elif current_val in pattern[2] and next_val != "stance":
                if next_val == "pos_peak":
                    print('Two peaks detected for run %s but probably just slowed down, assuming ok but check!' %r)
                else:
                    if r not in sitting_log:
                        raise ValueError("Pattern mismatch: 'pos_peak' or 'neg_peak' should be followed by 'stance'.")
        #print("The swing-stance pattern is followed in run %s" %r)

    def find_first_transitioning_paw(self, r, markerstuff):
        # improve this - instead or as well as looking at when nose is over the transition, look at when either paw is over this point
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']

        nose_mask = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart','Transition','RunEnd']].loc(axis=1)['Nose','likelihood'] > pcutoff
        nose_x = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['Nose', 'x'][nose_mask]

        trans_x_far = markerstuff['DualBeltMarkers']['x_edges'][2]
        trans_x_near = markerstuff['DualBeltMarkers']['x_edges'][3]

        # find when nose passes the transition point
        nose_trans_mask_far = nose_x > trans_x_far
        nose_trans_idx_far = nose_x[nose_trans_mask_far].index.get_level_values(level='FrameIdx')[0]
        nose_trans_mask_near = nose_x > trans_x_near
        nose_trans_idx_near = nose_x[nose_trans_mask_near].index.get_level_values(level='FrameIdx')[0]
        nose_transition_thresh_idx = int((nose_trans_idx_near + nose_trans_idx_far)/2)

        # find when **a** paw passes the transition point
        l_paw_mask = np.logical_and(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx') > nose_transition_thresh_idx)
        r_paw_mask = np.logical_and(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx') > nose_transition_thresh_idx)
        l_paw_x = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)['ForepawToeL', 'x'][l_paw_mask]
        r_paw_x = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)['ForepawToeR', 'x'][r_paw_mask]
        trans_x_mid = np.mean([trans_x_far,trans_x_near])
        l_paw_first = l_paw_x[l_paw_x > trans_x_mid].index.get_level_values(level='FrameIdx')[0]
        r_paw_first = r_paw_x[r_paw_x > trans_x_mid].index.get_level_values(level='FrameIdx')[0]

        transition_thresh_idx = l_paw_first if l_paw_first < r_paw_first else r_paw_first

        st_idx_i, st_limb_i = np.where(np.logical_or(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limblist[:2],'StepCycle'] == 0, self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limblist[:2],'StepCycle'] == 2))
        limb = np.array(limblist)[st_limb_i]
        idx = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx')[st_idx_i]

        st_mask = idx >= transition_thresh_idx

        transitioning_limb = limb[st_mask][0]
        transitioning_idx = idx[st_mask][0]

        return transitioning_idx, transitioning_limb

    def update_transition_idx(self,markerstuff):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        RunStages_new = []
        FrameIdx_new = []
        for r in self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='Run').unique():
            try:
                transitioning_idx, transitioning_limb = self.find_first_transitioning_paw(r,markerstuff)
                sorted_data = self.data[self.con][self.mouseID]['Side'].sort_index(level=['Run', 'RunStage'], inplace=False)
                current_trans_idx = sorted_data.loc(axis=0)[r, 'Transition'].index[0]

                idx_copy = pd.Series(data=self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='RunStage'),index=self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx'))

                if transitioning_idx < current_trans_idx:
                    new_trans_mask = np.logical_and(
                        self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx') >= transitioning_idx,
                        self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx') < current_trans_idx)
                    idx_copy[new_trans_mask] = 'Transition'
                elif transitioning_idx > current_trans_idx:
                    new_run_mask = np.logical_and(
                        self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx') < transitioning_idx,
                        self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx') >= current_trans_idx)
                    idx_copy[new_run_mask] = 'RunStart'
                elif transitioning_idx == current_trans_idx:
                    pass
                new_idx = pd.MultiIndex.from_arrays([idx_copy.values, np.array(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx'))])
                #test = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.reindex(new_idx)
                runstages, frameidx = zip(*new_idx)
                RunStages_new.append(np.array(runstages))
                FrameIdx_new.append(np.array(frameidx))
            except:
                RunStages_new.append(np.array(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='RunStage')))
                FrameIdx_new.append(np.array(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx')))

        for v in ['Side','Overhead','Front']:
            self.data[self.con][self.mouseID][v].loc(axis=1)['RunStage'] = np.concatenate(RunStages_new).ravel() ############################## DO THIS IN LOOP FOR ALL VIEWS
            self.data[self.con][self.mouseID][v].loc(axis=1)['FrameIdx'] = np.concatenate(FrameIdx_new).ravel() ############################## DO THIS IN LOOP FOR ALL VIEWS
            self.data[self.con][self.mouseID][v].loc(axis=1)['Run'] = self.data[self.con][self.mouseID][v].index.get_level_values(level='Run') ############################## DO THIS IN LOOP FOR ALL VIEWS

            self.data[self.con][self.mouseID][v].set_index(['Run', 'RunStage', 'FrameIdx'], append=False, inplace=True)

        return self.data[self.con][self.mouseID]


class LocomotionSaveDfs():
    def __init__(self):
        super().__init__()

    def create_new_file_with_loco_and_updated_transitions(self,pathname,data,con,mouseID,fillvalues=True):
        ######### need to add in a check for if files already exist (have a overwriting parameter) ##########
        warnings.filterwarnings("ignore", message="PerformanceWarning: your performance may suffer as PyTables will pickle object types that it cannot map directly to c-types")
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID]['Side'])
        LocDef = LocomotionDefinition(data, con, mouseID)
        data[con][mouseID] = LocDef.getLocoPeriods(fillvalues=fillvalues)
        data[con][mouseID] = LocDef.update_transition_idx(markerstuff=markerstuff)

        # save new file format
        for v in ['Side','Overhead','Front']:
            if pathname.count('side') == 1:
                if v != 'Side':
                    pathname_cam = pathname.replace('side', v.lower()).split('.')[0]
                else:
                    pathname_cam = pathname.split('.')[0]
            data[con][mouseID][v].to_hdf("%s__IdxCorr.h5" % (pathname_cam), key='Runs%s' %v, mode='w')


    def create_BATCH_OF_NEW_FILES_with_loco_and_updated_transitions(self,all=True,condition='APAChar_LowHigh',exptype='',wash='',day=''):
        ######### need to add in a check for if files already exist (have a overwriting parameter) ##########
        if all == True:
            dir_selection = paths['filtereddata_folder']
            # get all subdirectories
            data_subdirs = []
            for root, subdirs, files in os.walk(dir_selection):
                if not subdirs:
                    if not root.endswith(('temp_bin', 'consistent_speed_up')):
                        data_subdirs.append(root)
        else:
            dir_selection = os.path.join(paths['filtereddata_folder'], condition, exptype, wash, day)
            # get all subdirectories
            data_subdirs = []
            for root, subdirs, files in os.walk(dir_selection):
                if not any(sbdr in ['Day1', 'Day2', 'Day3', 'Wash', 'NoWash','Repeats','Extended','APAChar_LowHigh'] for sbdr in subdirs): # exclude parent dirs
                    if not root.endswith(('temp_bin', 'consistent_speed_up')):
                        data_subdirs.append(root)
        data_subdirs.sort()

        for s in tqdm(data_subdirs):
            files = utils.Utils().GetlistofH5files(directory=s)
            if np.any(files):
                if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
                    con = '_'.join(list(filter(lambda x: len(x) > 0, s.split('\\')))[7:])
                    data = utils.Utils().GetDFs([con])
                    print(
                        '------------------------------------------------------------------------------------------------\n{}Starting analysing condition:{} %s    ------------------->\n------------------------------------------------------------------------------------------------'.format(
                            '\033[1m', '\033[0m', '\033[1m', '\033[0m') % con)
                    ### for each triplet of side, front and overhead
                    for f in files['Side']:
                        if not f.endswith('_IdxCorr.h5'):
                            mouseID = f.split('\\')[-1].split('_')[3]
                            print('##################################################################################################\n{}Updating index for Condition:{} %s{}, MouseID:{} %s\n##################################################################################################'.format('\033[1m', '\033[0m', '\033[1m', '\033[0m') %(con,mouseID))
                            self.create_new_file_with_loco_and_updated_transitions(f,data,con,mouseID)
            files = None # reset so can be subsequently checked in next loop
