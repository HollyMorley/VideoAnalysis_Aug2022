import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
import Helpers.utils as utils
import Helpers.GetRuns as GetRuns
import Velocity
from Helpers.Config import *
import Plot
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm
import math
import warnings
# from statsmodels.nonparametric.smoothers_lowess import lowess


class Locomotion():

    def __init__(self):
        super().__init__()


    def findsigmoid(self, type, st, r, l, data, xy):
        ctr_window = 10
        fwd_window = 30

        if type == 'ctr':
            mask = data.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], np.arange(st - ctr_window, st + ctr_window)].loc(axis=1)[
                       l, 'likelihood'].values > pcutoff
            data = data.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], np.arange(st - ctr_window, st + ctr_window)].loc(axis=1)[l, xy].values[mask]
            frame = np.arange(st - ctr_window, st + ctr_window)[mask]
            p0 = [max(data), np.median(frame), 1, min(data)]
        elif type == 'fwd':
            mask = data.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], np.arange(st, st + fwd_window)].loc(axis=1)[
                       l, 'likelihood'].values > pcutoff
            data = data.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], np.arange(st, st + fwd_window)].loc(axis=1)[l, xy].values[mask]
            frame = np.arange(st, st + fwd_window)[mask]
            p0 = [max(data), np.median(frame), 1, min(data)]
        else:
            raise ValueError()

        return data, frame, p0


    def findplateau(self, frame, popt):
        halfwindow = int(len(frame)/2)
        plateaumask = (np.gradient(utils.Utils().sigmoid(frame, *popt), frame)/max(np.gradient(utils.Utils().sigmoid(frame, *popt), frame)))[halfwindow:] < 0.5 # find where normalised gradient is lower than 0.5 in second half (downward curve in change)
        newfirststance = frame[halfwindow:][plateaumask][0]
        return newfirststance

    def findNewVal(self, r, l, i, ydata, p0, frame):
        try:
            popt, pcov = curve_fit(utils.Utils().sigmoid, frame, ydata, p0, method='dogbox')
            # check for 's' sigmoid shape...
            if np.logical_and.reduce((utils.Utils().sigmoid(frame, *popt)[-1] == max(utils.Utils().sigmoid(frame, *popt)),
                                      utils.Utils().sigmoid(frame, *popt)[0] == min(utils.Utils().sigmoid(frame, *popt)),
                                      max(utils.Utils().sigmoid(frame, *popt)) != min(utils.Utils().sigmoid(frame, *popt)),
                                      np.all(np.diff(utils.Utils().sigmoid(frame, *popt)[-1:-4:-1]) < 0.0001),
                                      np.all(np.diff(utils.Utils().sigmoid(frame, *popt)[0:3]) < 0.0001))):
                try:
                    newfirststance = self.findplateau(frame=frame, popt=popt)
                    satisfied = True

                except:
                    print('cant divide for run: %s, limb: %s, stance: %s --> Deleted' % (r, l, i))
                    newfirststance = []
                    satisfied = False

            else:
                newfirststance = []
                satisfied = False
        except:
            print('Cant find any curve to refine for run: %s, limb: %s, stance: %s --> Deleted (1)' % (r, l, i))
            newfirststance = []
            satisfied = False
        return [newfirststance, satisfied]


    def findPeakByRot(self, x, t, w, p):
        peaks, info = scipy.signal.find_peaks(x, rel_height=1, width=w, distance=10, prominence=p)
        stancestart = np.array(t[peaks])  # find first frame
        return stancestart


    def findalternating(self, df, expected_phase):
        missing = []
        for i in range(1, len(df)):
            if df.iloc[i][0] == 'stance_bkwd':
                # this breaks the alternating pattern so need to reset it
                expected_phase = 'swing'
            elif df.iloc[i][0] == 'swing_bkwd':
                expected_phase = 'stance_bkwd'
            elif df.iloc[i][0] != expected_phase:
                #print(f"Missing {expected_phase} phase at frame {df.index[i - 1]}")
                missing.append([expected_phase, df.index[i - 1]])
                # keep expected phase the same as missing this value
                expected_phase = 'stance' if expected_phase == 'stance' else 'stance'
            else:
                # alternate the expected phase
                expected_phase = 'swing' if expected_phase == 'stance' else 'stance'
        return missing


    def find_y_swst(self, y, t, slope, velstuff, vel_limb): ### find mean by when limb moving vs not moving
        stationary_limb_mask = vel_limb < 20
        moving_limb_idx = vel_limb[~stationary_limb_mask].index.get_level_values(level='FrameIdx')

        new_y = y - ((t - t[0]) * slope) # y corrected horizontally
        min_dist = 1.5
        block_thresh = 10
        new_y_reset = new_y.reset_index(name='ydata')
        frame_range = pd.DataFrame({'FrameIdx': range(new_y_reset['FrameIdx'].min(), new_y_reset['FrameIdx'].max() + 2)})
        merged_df = frame_range.merge(new_y_reset, on='FrameIdx', how='left')
        interpolated_df = merged_df.interpolate(method='linear')
        interpolated_df = interpolated_df.set_index(['Run', 'RunStage', 'FrameIdx'])

        # # remove stationary limb frames temporarily to calculate a more accurate mean of y for distinguishing stance/swing
        interpolated_moving_mask = np.isin(interpolated_df.index.get_level_values(level='FrameIdx').to_numpy(), moving_limb_idx.to_numpy())
        interpolated_df_movingonly = interpolated_df[interpolated_moving_mask]
        interpolated_df_notmoving = interpolated_df[~interpolated_moving_mask]
        # moving_mean = interpolated_df_movingonly.mean()
        moving_mean = interpolated_df_movingonly.quantile([0.75])

        mask = np.logical_and.reduce(
            (abs(interpolated_df.shift(1) - interpolated_df) < min_dist,  # NB gives as a list of tuples
             abs(interpolated_df - interpolated_df.shift(-1)) < min_dist,
             interpolated_df > moving_mean.values)).flatten()
        new_t = interpolated_df.index.get_level_values(level='FrameIdx')
        y_swst_blocks = utils.Utils().find_blocks(new_t[mask], block_thresh)
        y_swst_blocks = np.array(y_swst_blocks)
        y_swst_blocks = y_swst_blocks[y_swst_blocks[:, 1] - y_swst_blocks[:, 0] > 3]
        swst_names = len(y_swst_blocks) * ['stance', 'swing']
        y_swstdf = pd.DataFrame(data=swst_names, index=np.array(y_swst_blocks).flatten())

        # chuck out sw/st when limb stationary
        stance_mask = (y_swstdf == 'stance').values.flatten()
        stance_idx = y_swstdf.index[stance_mask].to_numpy()


        return y_swstdf

    def find_initial_x_swst_withrot(self, x, t, slope):
        ### first tilt the x plot
        array_length = len(t)
        # create new arrays to store the regression results for each quadrant
        slope_array = []
        # iterate through each quadrant
        for i in range(2):
            for j in range(2):
                # calculate the start and end indices for the current quadrant
                start_idx = i * (array_length // 2) + j * (array_length // 4)
                end_idx = start_idx + (array_length // 4)
                # call stats.linregress() on the current quadrant
                tnew = t[start_idx:end_idx]
                xnew = x[start_idx:end_idx]
                slope, intercept, r_value, p_value, std_err = stats.linregress(tnew, xnew)
                # append the regression results to their respective arrays
                slope_array.append(slope)

        ### Then find peaks
        if np.all(abs(np.diff(slope_array)) < 1):
            xcorr_rot0 = (x - slope * t) - intercept
            xcorr_rot90 = (x - slope * t * 0.5) - intercept

            # find start of stance
            stancestart = self.findPeakByRot(xcorr_rot90, t, w=5, p=20)
            stancestart_rot0 = self.findPeakByRot(xcorr_rot0, t, w=1, p=10)

            # find start of swing
            swingstart = self.findPeakByRot(-xcorr_rot0, t, w=5, p=20)
        else:
            stancestart_qs = []
            stancestart_rot0_qs = []
            swingstart_qs = []
            for i in range(len(slope_array)):
                rot0 = (x - slope_array[i] * t) - intercept
                rot90 = (x - slope_array[i] * t * 0.5) - intercept

                # find start of stance using quadrant regression value
                st = self.findPeakByRot(rot90, t, w=5, p=20)
                st_rot0 = self.findPeakByRot(rot0, t, w=1, p=10)
                sw = self.findPeakByRot(-rot0, t, w=5, p=20)
                stancestart_qs.append(st)
                stancestart_rot0_qs.append(st_rot0)
                swingstart_qs.append(sw)

            # concatenate the arrays into one
            st_rot_qs_concat = np.concatenate(stancestart_rot0_qs)
            st_qs_concat = np.concatenate(stancestart_qs)
            sw_qs_concat = np.concatenate(swingstart_qs)
            arr_concat = [st_rot_qs_concat, st_qs_concat, sw_qs_concat]

            # find the unique values and their counts
            unique_vals_st_rot, counts_st_rot = np.unique(st_rot_qs_concat, return_counts=True)
            unique_vals_st, counts_st = np.unique(st_qs_concat, return_counts=True)
            unique_vals_sw, counts_sw = np.unique(sw_qs_concat, return_counts=True)
            unique_vals = [unique_vals_st_rot, unique_vals_st, unique_vals_sw]

            # initialize a dictionary to store the summary values
            summary_dict = [{}, {}, {}]
            # loop over the unique values
            for i in [0, 1, 2]:
                for val in unique_vals[i]:
                    try:
                        # find the indices of the similar values
                        idx = np.where(np.abs(arr_concat[i] - val) <= 10)[0]
                        # calculate the mean of the similar values as an integer
                        mean_val = int(np.round(np.mean(arr_concat[i][idx])))
                        # add the summary value to the dictionary
                        summary_dict[i][val] = mean_val
                    except:
                        print('kashsklahl')
            # get a list of unique values
            stancestart_rot0 = list(set(list(summary_dict[0].values())))
            stancestart = list(set(list(summary_dict[1].values())))
            swingstart = list(set(list(summary_dict[2].values())))

        stancestart_rot0.sort()
        stancestart.sort()
        swingstart.sort()

        return stancestart_rot0, stancestart, swingstart

    def check_swst_pattern(self, swingstart, stancestart, stancestart_rot0, normy, xS, r, y, l):
        # Check whether pattern of swing vs stance is right
        # this assumes there are never any missing swing values!!
        sw = pd.DataFrame(data=['swing'] * len(swingstart), index=swingstart)
        st = pd.DataFrame(data=['stance'] * len(stancestart), index=stancestart)
        swstdf = sw.append(st)
        swstdf = swstdf.sort_index()
        buffer_phase = 'swing' if swstdf.iloc[0][0] == 'stance' else 'stance'
        buffer_row_start = pd.DataFrame({0: buffer_phase}, index=[swstdf.index[0] - 1])
        df = pd.concat([swstdf, buffer_row_start])
        df = df.sort_index()

        # Iterate through data and check for missing values
        expected_phase = 'swing' if buffer_phase == 'stance' else 'stance'
        missing = self.findalternating(df, expected_phase)
        len_missing = len(missing)

        # check for values where mouse stepped backwards
        backwards_frames = np.asarray(((xS - xS.shift(8))[(xS - xS.shift(8)) < 0]).index)
        if np.any(backwards_frames):
            backwards_swst = []
            for f in swstdf.index:
                if np.logical_and(np.any(backwards_frames - f > 8), np.any(backwards_frames - f < 25)):
                    backwards_swst.append(f)

        # iterate through missing list and keep or delete swing/stance value and update missing list accordingly
        i = 0
        # for i in range(len(missing)):
        while i < len_missing:
            if missing[i][0] == 'swing_bkwd':
                # i dont think this can actually happen??
                print('This shouldnt happen!!!!!!!')
            elif missing[i][0] == 'stance':
                if np.any(backwards_frames) and missing[i][1] in backwards_swst:
                    for b in backwards_swst:
                        if missing[i][1] == b:
                            swstdf.loc[b] = 'swing_bkwd'
                            df.loc(axis=0)[b] = 'swing_bkwd'
                            swstdf.iloc[swstdf.index.get_loc(b) + 1] = 'stance_bkwd'
                            df.iloc[swstdf.index.get_loc(b) + 1] = 'stance_bkwd'
                            newmissing = self.findalternating(df, expected_phase)
                            for n in range(len(newmissing)):
                                missing.append(newmissing[n])
                            len_missing = len(missing)
                elif np.any(stancestart_rot0 > missing[i][1]):  # check if there are any rotated stance values following the last correct value
                    currentswing = np.where(swingstart == missing[i][1])[0][0]
                    nextswing = currentswing + 1
                    if currentswing < (len(swingstart) - 1):  # check if last correct swing value is NOT the final swing
                        try:
                            poss_stance = np.array(stancestart_rot0)[np.logical_and(missing[i][1] < stancestart_rot0, np.array(stancestart_rot0) < swingstart[nextswing])][0] # find a stancestart_rot0 value that is between the swing values bordering the missing stance

                            ### Check if this poss stance value is followed by an increase in y (denoting moving downwards)
                            tchange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance, poss_stance + 10)].index.get_level_values(level='FrameIdx')
                            ychange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance, poss_stance + 10)].values
                            slope, intercept, r_value, p_value, std_err = stats.linregress(tchange, ychange)
                            if slope > 0: # check if
                                swstdf.loc(axis=0)[poss_stance] = 'stance'
                                df.loc(axis=0)[poss_stance] = 'stance'
                                swstdf = swstdf.sort_index()
                                df = df.sort_index()
                                #newmissing = self.findalternating(df, expected_phase)
                                # for n in range(len(newmissing)):
                                #     missing.append(newmissing[n])
                                # len_missing = len(missing)
                                missing = self.findalternating(df, expected_phase)
                                len_missing = len(missing)
                                #print('Missing stance value replaced for run: %s, limb: %s' % (r, l))
                            else:
                                swstdf.drop(index=swingstart[nextswing], inplace=True)
                                df.drop(index=swingstart[nextswing], inplace=True)
                                # newmissing = self.findalternating(df, expected_phase)
                                # for n in range(len(newmissing)):
                                #     missing.append(newmissing[n])
                                # len_missing = len(missing)
                                missing = self.findalternating(df, expected_phase)
                                len_missing = len(missing)
                                # print(
                                #     'Can\'t find a rotated stance value - TEMP: deleting this swing value (%s) as assuming incorrect' %
                                #     swingstart[nextswing])
                        except:
                            swstdf.drop(index=swingstart[currentswing], inplace=True)
                            df.drop(index=swingstart[currentswing], inplace=True)
                            # newmissing = self.findalternating(df, expected_phase)
                            # for n in range(len(newmissing)):
                            #     missing.append(newmissing[n])
                            # len_missing = len(missing)
                            missing = self.findalternating(df, expected_phase)
                            len_missing = len(missing)
                            # print(
                            #     'Can\'t find a rotated stance value - TEMP: deleting this swing value (%s) as assuming incorrect' %
                            #     swingstart[nextswing])
                    else:
                        try:
                            poss_stance = stancestart_rot0[missing[i][1] < stancestart_rot0][0]
                            swstdf.loc(axis=0)[poss_stance] = 'stance'
                            df.loc(axis=0)[poss_stance] = 'stance'
                            swstdf = swstdf.sort_index()
                            df = df.sort_index()
                            # newmissing = self.findalternating(df, expected_phase)
                            # for n in range(len(newmissing)):
                            #     missing.append(newmissing[n])
                            # len_missing = len(missing)
                            missing = self.findalternating(df, expected_phase)
                            len_missing = len(missing)
                            #print('Missing stance value replaced for run: %s, limb: %s' % (r, l))
                        except:
                            swstdf.drop(index=swingstart[currentswing], inplace=True)
                            df.drop(index=swingstart[currentswing], inplace=True)
                            # newmissing = self.findalternating(df, expected_phase)
                            # for n in range(len(newmissing)):
                            #     missing.append(newmissing[n])
                            # len_missing = len(missing)
                            missing = self.findalternating(df, expected_phase)
                            len_missing = len(missing)
                            # print(
                            #     'Can\'t find a rotated stance value (end) - TEMP: deleting this swing value as assuming incorrect')
                else:
                    print('Can\'t find any upcoming rotated stance values')
            elif missing[i][0] == 'swing':
                if np.any(backwards_frames) and missing[i][1] in backwards_swst:
                    for b in backwards_swst:
                        if missing[i][1] == b:
                            swstdf.loc[b] = 'swing_bkwd'
                            df.loc(axis=0)[b] = 'swing_bkwd'
                            swstdf.iloc[swstdf.index.get_loc(b) + 1] = 'stance_bkwd'
                            df.iloc[swstdf.index.get_loc(b) + 1] = 'stance_bkwd'
                            missing = self.findalternating(df, expected_phase)
                            len_missing = len(missing)
                else:
                    print(
                        'Not sure how to deal with missing swing values! For now going to assume they are right and this represents extra stance values')
                    currentstance = np.where(stancestart == missing[i][1])[0][0]
                    poss_stance_todel = stancestart[currentstance]
                    tchange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance_todel, poss_stance_todel + 10)].index.get_level_values(level='FrameIdx')
                    ychange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance_todel, poss_stance_todel + 10)].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(tchange, ychange)
                    if slope < 0:  # get rid of that stance value as the limb is still in swing
                        swstdf.drop(index=poss_stance_todel, inplace=True)
                        df.drop(index=poss_stance_todel, inplace=True)
                        # newmissing = self.findalternating(df, expected_phase)
                        # for n in range(len(newmissing)):
                        #     missing.append(newmissing[n])
                        # len_missing = len(missing)
                        missing = self.findalternating(df, expected_phase)
                        len_missing = len(missing)
                    else:
                        print('This stance value seems fine (frame: %s)' % poss_stance_todel)
                        stno = swstdf.index.get_loc(poss_stance_todel)
                        c = 1
                        try:
                            while swstdf.iloc[stno + c].values == 'stance':
                                poss_stance_todel = swstdf.iloc[stno + c].name
                                tchange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance_todel, poss_stance_todel + 15)].index.get_level_values(level='FrameIdx')
                                ychange = y.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(poss_stance_todel, poss_stance_todel + 15)].values
                                slope, intercept, r_value, p_value, std_err = stats.linregress(tchange, ychange)
                                if slope < 0:  # get rid of that stance value as the limb is still in swing
                                    swstdf.drop(index=poss_stance_todel, inplace=True)
                                    df.drop(index=poss_stance_todel, inplace=True)
                                    # newmissing = self.findalternating(df, expected_phase)
                                    # for n in range(len(newmissing)):
                                    #     missing.append(newmissing[n])
                                    # len_missing = len(missing)
                                    missing = self.findalternating(df, expected_phase)
                                    len_missing = len(missing)
                                else:
                                    print('This stance value seems fine (frame: %s)' % poss_stance_todel)
                                c += 1
                        except:
                            pass
            else:
                print('Not sure why I got here :/')
            i += 1

        # if last value is a swing, check that can't find any stance values after
        if swstdf.values[-1][0] == 'swing':
            lastswing = swstdf.index[-1]
            finalstmask = stancestart_rot0 > lastswing
            posslaststances = np.array(stancestart_rot0)[finalstmask]
            onground = normy.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], posslaststances + 2] > 0
            if sum(onground) == 1:
                laststance = posslaststances[onground][0]
                finalrow = pd.DataFrame({0: 'stance'}, index=[laststance])
                swstdf = pd.concat([swstdf, finalrow])
            if sum(onground) > 1:
                laststance = posslaststances[onground][0]
                finalrow = pd.DataFrame({0: 'stance'}, index=[laststance])
                swstdf = pd.concat([swstdf, finalrow])
                print('FOUND MULTIPLE LAST STANCES FOR R: %s - PROBABLY MISSING SWING HERE!!' % r)
            if np.logical_and(sum(onground) == 0, len(posslaststances) > 0):
                tchange = normy.loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], range(posslaststances[0], posslaststances[0] + 10)].index.get_level_values(level='FrameIdx')
                ychange = normy.loc(axis=0)[
                    r, ['RunStart', 'Transition', 'RunEnd'], range(posslaststances[0], posslaststances[0] + 10)].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(tchange, ychange)
                if slope > 0:  # ie paw is going downwards
                    laststance = posslaststances[0]
                    finalrow = pd.DataFrame({0: 'stance'}, index=[laststance])
                    swstdf = pd.concat([swstdf, finalrow])

        return swstdf

    def findslope(self, chunk, return_value):
        slope, intercept, r_value, p_value, std_err = stats.linregress(chunk.index.get_level_values(level='FrameIdx'),
                                                                       chunk.values)
        if return_value == 'slope':
            return slope
        if return_value == 'intercept':
            return intercept
        elif return_value == 'r_value':
            return r_value
        elif return_value == 'p_value':
            return p_value
        elif return_value == 'std_err':
            return std_err

    def calculate_slope_height(self, x, vel_slope, vel_intercept):
        lastframe = x.index.get_level_values(level='FrameIdx')[-1]
        firstframe = x.index.get_level_values(level='FrameIdx')[0]
        slope_height = (lastframe * vel_slope.xs(lastframe, level='FrameIdx') + vel_intercept.xs(lastframe, level='FrameIdx').values) - (firstframe * vel_slope.xs(lastframe, level='FrameIdx') + vel_intercept.xs(lastframe,level='FrameIdx'))
        return slope_height


    def remove_extra_swst(self, df, swst_name, Poss_Stance_x):
        remove_idx = []
        i = 0
        while i < len(df.loc(axis=1)[0].values) - 1:
        #for i in range(len(df.loc(axis=1)[0].values) - 1):
            if df.loc(axis=1)[0].values[i] == df.loc(axis=1)[0].values[i + 1] == swst_name:
                if i > 0 and df.loc(axis=1)[0].values[i - 1] == swst_name:
                    if swst_name == 'swing':
                        #### check if any stances in Poss missing stance help this
                        this_idx = df.loc(axis=1)[0].index[i]
                        next_idx = df.loc(axis=1)[0].index[i + 1]
                        found_stance_mask = np.logical_and(Poss_Stance_x.index < next_idx, Poss_Stance_x.index > this_idx)
                        found_stance = Poss_Stance_x[found_stance_mask]
                        if np.any(found_stance):
                            df = df.append(found_stance)
                            df = df.sort_index()
                            i -= 1
                        else:
                            raise ValueError('More than 2 %s found together and cant find an extra stance' % swst_name)  # Skip if there are more than 2 swings in a row
                    else:
                        raise ValueError('More than 2 %s found together' % swst_name)  # Skip if there are more than 2 swings in a row

                else:
                    if df.loc(axis=1)['source'].values[i] == 'y':
                        remove_idx.append([i, df.index[i]])
                    elif df.loc(axis=1)['source'].values[i + 1] == 'y':
                        remove_idx.append([i + 1, df.index[i + 1]])
                    elif swst_name == 'swing':
                        #### check if any stances in Poss missing stance help this
                        this_idx = df.loc(axis=1)[0].index[i]
                        next_idx = df.loc(axis=1)[0].index[i + 1]
                        found_stance_mask = np.logical_and(Poss_Stance_x.index < next_idx,
                                                           Poss_Stance_x.index > this_idx)
                        found_stance = Poss_Stance_x[found_stance_mask]
                        if np.any(found_stance):
                            df = df.append(found_stance)
                            df = df.sort_index()
                            i -= 1
                        else:
                            raise ValueError('Multiple %s but cant find any y based %s' % (swst_name, swst_name))
                    else:
                        raise ValueError('Multiple %s but cant find any y based %s' % (swst_name, swst_name))

            elif df.loc(axis=1)[0].values[i] == '%s_bkwd' %swst_name: #and df.loc(axis=1)[0].values[i + 1] == swst_name: # if this value is the 'bkwd' version of the next value
                ################### deal with bkwd values and any surrounding detected swings/stances
                if df.loc(axis=1)[0].values[i-1] == swst_name:
                    remove_idx.append([i-1, df.index[i-1]])
                elif df.loc(axis=1)[0].values[i+1] == swst_name:
                    remove_idx.append([i+1, df.index[i+1]])

            elif df.loc(axis=1)[0].values[i] == swst_name and df.loc(axis=1)[0].values[i+1] == '%s_bkwd' %swst_name:
                pass # because already dealing with this when get to it

            i += 1


        remove_idx_arr = np.array(remove_idx)
        try:
            to_keep = np.delete(np.arange(len(df)), remove_idx_arr[:, 0])
            df = df.iloc[to_keep]
        except:
            df = df


        return df

    def get_limb_speed(self,l,r,data,con,mouseID,view,t,markerstuff,vel_limb):
        ### get velocity data for this limb
        # vel_limb = Velocity.Velocity().getVelocity_specific_limb(l, r, data, con, mouseID, view,
        #                                                          # get the velocity data for this body part (limb)
        #                                                          math.ceil((fps / 30) / 2.) * 2, markerstuff)
        vel_limb_mask = np.isin(vel_limb.index.get_level_values(level='FrameIdx'), t)
        vel_limb_cropped = vel_limb[vel_limb_mask]
        vel_limb_cropped_nan = vel_limb_cropped[~vel_limb_cropped.isna()]
        from statsmodels.nonparametric.smoothers_lowess import lowess
        vel_limb_smoothed = pd.DataFrame(index=vel_limb_cropped_nan.index, data=np.array(lowess(vel_limb_cropped_nan,vel_limb_cropped_nan.index.get_level_values(level='FrameIdx'), frac=.04))[:,1])
        vel_limb_smoothed = vel_limb_smoothed[0]
        if l == 'ForepawToeR' or l == 'ForepawToeL':
            val = vel_limb_cropped_nan
        elif l == 'HindpawToeR' or l == 'HindpawToeL':
            val = vel_limb_smoothed
        vel_slope = val.rolling(window=10).apply(lambda h: self.findslope(h, 'slope'))
        vel_intercept = val.rolling(window=10).apply(lambda h: self.findslope(h, 'intercept'))
        vel_slope_height = val.rolling(window=10).apply(lambda x: self.calculate_slope_height(x, vel_slope, vel_intercept))
        upwards_sloping_vals = vel_slope_height[vel_slope_height > 30].index.get_level_values(level='FrameIdx')
        upwards_sloping_blocks = utils.Utils().find_blocks(upwards_sloping_vals, 10)
        Swing_borders = []
        for i in range(len(upwards_sloping_blocks)):
            swing_borders = [upwards_sloping_blocks[i][0] - 10, upwards_sloping_blocks[i][1]]
            Swing_borders.append(swing_borders)

        # check what direction the changes in limb speed are going in that are captured by Swing_borders
        peaks, info = scipy.signal.find_peaks(vel_limb_smoothed, distance=10, height=50)
        neg_peaks, neg_info = scipy.signal.find_peaks(vel_limb_smoothed * -1, distance=10, height=10)
        neg_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[neg_peaks]
        step_direction = np.array(['fwd'] * len(Swing_borders))
        for i in range(len(neg_idx)):
            bkwds_mask = np.logical_and(np.array(neg_idx)[i] < np.array(Swing_borders)[:,1], np.array(neg_idx)[i] > np.array(Swing_borders)[:,0])
            step_direction[bkwds_mask] = 'bwd'

        return Swing_borders, step_direction

    def getLocoPeriods(self, data, con, mouseID, xy, fillvalues=True, view='Side', n=30):
        warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']
        windowsize = math.ceil((fps / n) / 2.) * 2
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        velstuff = Velocity.Velocity().getVelocityInfo(data, con, mouseID, zeroed=False, view=view, xaxis='x', windowsize=windowsize, markerstuff=markerstuff, f=range(0,int(data[con][mouseID][view].index.get_level_values(level='Run').unique().max()+1)))

        for l in tqdm(limblist):
            StepCycleAll = []
            StepCycleFilledAll = []
            for r in data[con][mouseID][view].index.get_level_values(level='Run').unique():
                try:
                    commonidx = velstuff['runs_lowess'][int(r), 1].index.get_level_values(level='FrameIdx')
                    mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'likelihood'].values > pcutoff  # change so that NO LONGER excluding when mouse stationary (at this stage)
                    x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].values[mask]
                    y = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'y'][mask]
                    t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask]
                    xS = pd.Series(data=x, index=t)
                    vel_limb = Velocity.Velocity().getVelocity_specific_limb(l, r, data, con, mouseID, view, math.ceil((fps / 30) / 2.) * 2, markerstuff) # get the velocity data for this body part (limb)
                    vel_limb = vel_limb.loc[['RunStart','Transition','RunEnd']]

                    ###### Velocity ######
                    # get velocity data for this limb and use this to find the general area sw/st should fall in
                    Swing_borders, step_direction = self.get_limb_speed(l,r,data,con,mouseID,view,t,markerstuff,vel_limb)

                    ###### Y data ######
                    # get y values where y has been normalised to 180 deg line (so can find stationary periods better)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                    normy = y - ((slope * t).values + intercept)
                    # find y stsw
                    y_swstdf = self.find_y_swst(y, t, slope, velstuff['runs_lowess'][int(r), 1], vel_limb)

                    ###### X data ######
                    # find x swst
                    stancestart_rot0, stancestart, swingstart = self.find_initial_x_swst_withrot(x, t, slope)
                    # check if there is an alternating pattern (in x here) and correct if not
                    swstdf = self.check_swst_pattern(swingstart, stancestart, stancestart_rot0, normy, xS, r, y, l)

                    ######## compare all values, priority: speed swing borders, y, x #########
                    Real_Swing = []
                    Poss_Missing_Swing_y = []
                    Poss_Stance_x = []
                    Bkwds = []
                    #Poss_Missing_Swing_x = []
                    for i in range(len(Swing_borders)):
                        if step_direction[i] == 'fwd':
                            ## find if there are any y detections within this swing border
                            y_detected = np.where(np.logical_and(y_swstdf.index > Swing_borders[i][0], y_swstdf.index < Swing_borders[i][1]))
                            if np.any(y_detected):
                                real_swing = y_swstdf.iloc[y_detected[0]]
                                if len(y_detected[0]) == 1:
                                    Real_Swing.append(real_swing)
                                else:
                                    detected_swing = y_swstdf.iloc[y_detected].values.flatten() == 'swing'
                                    detected_swing_index = y_swstdf.iloc[y_detected[0]][detected_swing]
                                    if len(detected_swing_index) == 1:
                                        Real_Swing.append(detected_swing_index)
                                    else:
                                        raise ValueError('There are multiple swings detected, not sure what to do!')
                            else:
                                # note any possible x-found swings in this period
                                x_detected = np.where(np.logical_and(swstdf.index > Swing_borders[i][0], swstdf.index < Swing_borders[i][1]))[0]
                                if np.any(x_detected):
                                    if len(swstdf.iloc[x_detected][swstdf.iloc[x_detected]=='swing']) == 1: # if only found one swing value (might catch a stance value so cant just say len == 1)
                                        Real_Swing.append(swstdf.iloc[x_detected][swstdf.iloc[x_detected]=='swing'])
                                        # note possible missed stances for this swing
                                        this_stance_idx = x_detected[(swstdf.iloc[x_detected]=='swing').values.flatten()][0]
                                        try:
                                            border_stances = swstdf.iloc[[this_stance_idx - 1,this_stance_idx + 1]]
                                        except:
                                            try:
                                                border_stances = swstdf.iloc[[this_stance_idx - 1, this_stance_idx - 1]]
                                            except:
                                                border_stances = swstdf.iloc[[this_stance_idx + 1, this_stance_idx + 1]]
                                        Poss_Stance_x.append(border_stances)

                                        # poss_swing_x = swstdf.iloc[x_detected[0]]
                                        # Poss_Swing_x.append(poss_swing_x)
                                else:
                                    # note middle frame between the borders to later check if this should be a detection
                                    poss_missing_swing_y = pd.DataFrame(data=['swing'], index=[np.mean(Swing_borders[i]).astype(int)])
                                    Poss_Missing_Swing_y.append(poss_missing_swing_y)

                        elif step_direction[i] == 'bwd':
                            # check if bkwds swing/stance detected in x
                            if np.any(swstdf == 'stance_bkwd') and np.any(swstdf == 'swing_bkwd'):  # are any detected bkwds values somewhere in the swing borders
                                if sum(swstdf.values == 'stance_bkwd')[0] == 1 and sum(swstdf.values == 'swing_bkwd')[0] == 1: # is it just 1 of each??
                                    bkwds = swstdf[swstdf[0].str.contains('bkwd', case=False)]
                                    Bkwds.append(bkwds)
                                else:
                                    raise ValueError('There is more than 1 bkwd value for run %s, limb %s - not sure what to do!' %(r,l))
                            else:
                                # raise ValueError('Cant find bkwds values in x data where velocity data shows a bkwds movement...? Run: %s, limb: %s' % (r, l))
                                print('Cant find bkwds values in x data where velocity data shows a bkwds movement...? Run: %s, limb: %s' % (r, l))

                    Real_Swing = pd.concat(Real_Swing)
                    if np.any(Poss_Missing_Swing_y):
                        Poss_Missing_Swing_y = pd.concat(Poss_Missing_Swing_y)
                    if np.any(Bkwds):
                        Bkwds = pd.concat(Bkwds)
                    if np.any(Poss_Stance_x):
                        Poss_Stance_x = pd.concat(Poss_Stance_x)
                    #Poss_Swing_x = pd.concat(Poss_Swing_x)

                    # if a possible y swing value is at the beginning of the run just add it as vel data probably just missed it
                    if len(Poss_Missing_Swing_y) == 1 and np.all(Poss_Missing_Swing_y.index[0] < Real_Swing.index):
                        Real_Swing = pd.concat([Real_Swing, pd.DataFrame(data=['swing'], index=Poss_Missing_Swing_y.index)])
                        Real_Swing.sort_index(inplace=True)

                    Real_Swing['source'] = 'vel'
                    y_swstdf['source'] = 'y'

                    # add in bkwds values and y data
                    if np.any(Bkwds) and len(Bkwds) == 2:
                        Bkwds['source'] = 'interpolated_y'
                        Real_Swing = pd.concat([Real_Swing, Bkwds])
                        Real_Swing.sort_index(inplace=True)
                    comb_vel_y_df = pd.concat([Real_Swing, y_swstdf])
                    comb_vel_y_df = comb_vel_y_df.sort_index()

                    # remove extra swings
                    comb_vel_y_df = self.remove_extra_swst(comb_vel_y_df,'swing',Poss_Stance_x)
                    # remove extra stances
                    comb_vel_y_df = self.remove_extra_swst(comb_vel_y_df,'stance',Poss_Stance_x)

                    '''
                    # compare stsw from y and x data
                    matched_pairs = []
    
                    # Iterate over the frames of swstdf
                    for frame_x in swstdf.index:
                        value_x = swstdf.loc[frame_x, 0]  # Assuming the single column is labeled as 0
    
                        # Find the closest frame in y_swstdf within the desired range
                        closest_frame_y = None
                        min_frame_diff = float('inf')
    
                        for frame_y in y_swstdf.index:
                            value_y = y_swstdf.loc[frame_y, 0]  # Assuming the single column is labeled as 0
    
                            if abs(frame_x - frame_y) <= 100:
                                frame_diff = abs(frame_x - frame_y)
    
                                # Check if the current frame is closer than the previous closest frame
                                if frame_diff < min_frame_diff:
                                    min_frame_diff = frame_diff
                                    closest_frame_y = frame_y
    
                        # Add the matched pair to the list
                        if closest_frame_y is not None:
                            matched_pairs.append((frame_x, closest_frame_y))
    
                    unique_values, counts = np.unique(np.array(matched_pairs)[:, 1], return_counts=True)
                    duplicate_positions = np.where(counts > 1)[0]
                    duplicate_values = unique_values[duplicate_positions]
                    if any(duplicate_values):
                        print('There are duplicate values which indicate a MISSING VALUE in run %s (mouse: %s) at frame(/s): %s' %(r,mouseID,duplicate_values))
                    #################################################################################
                    '''

                    # # temp - code to compare number/pattern of sw/st from x and y - depending on which one has more values, the comparison will either be against x or y
                    # if len(y_swstdf) > len(swstdf):  # shorter = maindf
                    #     subdf = y_swstdf.copy(deep=True)
                    #     maindf = swstdf.copy(deep=True)
                    # else:
                    #     subdf = swstdf.copy(deep=True)
                    #     maindf = y_swstdf.copy(deep=True)
                    # dist = np.array(maindf.index)[:, np.newaxis] - np.array(
                    #     subdf.index)  # shorter down the rows, longer across the columns
                    # dist = dist.astype(float)
                    # dist[abs(dist) > 30] = np.nan
                    # potential_closest = np.nanargmin(abs(dist), axis=1)
                    # mask = subdf.iloc[potential_closest].values == maindf.values  # compare ????
                    # if np.all(mask):
                    #     pass
                    # else:
                    #     mismatch = np.where(mask == False)[0]
                    #     if np.any(mismatch):
                    #         mismatched_swstdf = subdf.iloc[mismatch]
                    #         for swst_idx, swst_name in enumerate(mismatched_swstdf.values):
                    #             if swst_name == 'swing_bkwd' or swst_name == 'stance_bkwd':
                    #                 print('Backward stance/swing detected for limb: %s, run: %s' % (l, r))
                    #             else:
                    #                 Mismatch.append(r)
                    #                 print(
                    #                     'Mismatch between x- and y- detected stance and swing time points for limb: %s, run: %s' % (
                    #                     l, r))
                    #     else:
                    #         print(
                    #             'Mismatch between x- and y- detected stance and swing time points for limb: %s, run: %s' % (
                    #             l, r))
                    #         Mismatch.append(r)



                    #TEMP
                    plt.figure()
                    plt.plot(t, x)
                    plt.vlines(x=comb_vel_y_df[comb_vel_y_df[0]=='swing'].index.values.astype(int), ymin=100, ymax=1900, colors='red')
                    plt.vlines(x=comb_vel_y_df[comb_vel_y_df[0]=='stance'].index.values.astype(int), ymin=100, ymax=1900, colors='green')
                    plt.vlines(x=comb_vel_y_df[comb_vel_y_df[0] == 'swing_bkwd'].index.values.astype(int), ymin=100, ymax=1900, colors='darkred', linestyle='--')
                    plt.vlines(x=comb_vel_y_df[comb_vel_y_df[0] == 'stance_bkwd'].index.values.astype(int), ymin=100, ymax=1900, colors='darkgreen', linestyle='--')
                    plt.title('Run: %s\nLimb: %s' %(r,l))
                    plt.savefig(r'%s\Locomotion,%s_%s_%s.png' % (plotting_destfolder,mouseID,r,l),bbox_inches='tight', transparent=False, format='png')
                except:
                    print('cant plot run: %s, limb: %s' %(r,l))

                # # OLD
                # diff = stancestart[:, np.newaxis] - swingstart
                # diffswing = swingstart[:, np.newaxis] - stancestart
                # posALL = []
                # for i in range(0, diff.shape[0]):
                #     pos = sum(diff[i] > 0)
                #     posALL.append(pos)
                #
                # # find and try and replace any missing values
                # miss = np.array([])
                # if np.any(np.diff(posALL) > 1):
                #     miss = np.array(np.where(np.diff(posALL) > 1)[0] + 1)  # find the expected position of any missing stances
                #
                # if np.any(miss): # if there are any missing values found
                #     # check stancestart_rot0
                #     for i in miss:
                #         poss_st = np.logical_and(stancestart_rot0 > stancestart[i-1], stancestart_rot0 < stancestart[i]) # find possible values
                #         if np.any(poss_st):
                #             if sum(poss_st) == 1:
                #                 newvalmask = np.logical_and(stancestart_rot0 > stancestart[i-1], stancestart_rot0 < stancestart[i])
                #                 stancestart = np.concatenate((stancestart, stancestart_rot0[newvalmask]))
                #                 stancestart.sort()
                #             else:
                #                 print('Found %s extra possible values for stance_bkup to fill in missing gap in run %s, limb %s' %(sum(poss_st), r, l))
                #         else:
                #             print('Unable to find stance values for run: %s and limb: %s' %(r,l))
                #              # maybe need to estitmate stance value here
                #
                # extr_indices = []
                # if np.any(np.diff(posALL) < 1):
                #     #extr = np.array(np.where(np.diff(posALL) < 1)[0])  # find the first instance of a swing phase with multiple stances
                #     unique_vals, counts = np.unique(posALL, return_counts=True)
                #     non_unique_vals = unique_vals[counts > 1]  # get the non-unique values
                #     extr_indices = np.where(np.isin(posALL, non_unique_vals))[0]
                #

                # Remove erronous stancestarts AND refine frames
                # markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
                # todelALL = []
                # for i in range(0, len(stancestart)):
                #     xs = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], stancestart[i]].loc(axis=1)[l, 'x'].values[0] # find x value in single frame where l is in stance
                #
                #     # Check that paw is not in transition area...
                #     if xs < markerstuff['DualBeltMarkers']['x_edges'][2] or xs > markerstuff['DualBeltMarkers']['x_edges'][3] + 5:
                #
                #         # 1st try: find curve and refine using main identified stance values
                #         ydata, frame, p0 = self.findsigmoid(type='ctr', st=stancestart[i], r=r, l=l, data=data[con][mouseID][view], xy='y')
                #         newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)
                #
                #         # 2nd try: find curve and refine using back up stance values (from tilting)
                #         if satisfied is False:
                #             # Find if there are any stance_bkup values nearby
                #             diffs = abs(stancestart[i] - stancestart_rot0)  # find difference between current stance val and all the stance_bkup vals
                #             if np.min(diffs[np.nonzero(diffs)]) < 30: # if there is a bkup value that is within 30 frames and isnt the same value....
                #                 nearmask = np.where(np.min(diffs[np.nonzero(diffs)]) == diffs)[0][0] # create mask to show where the closest stance_bkup value that isnt the same value as stance is
                #                 nearest = stancestart_rot0[nearmask]
                #                 ydata, frame, p0 = self.findsigmoid(type='ctr', st=nearest, r=r, l=l, data=data[con][mouseID][view], xy='y')
                #                 newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)
                #
                #         # 3rd try: find curve and refine using stance values from 30 frames ahead
                #         if satisfied is False:
                #             lastidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[mask][-1]
                #             if stancestart[i] + 30 < lastidx:
                #                 ydata, frame, p0 = self.findsigmoid(type='fwd', st=stancestart[i], r=r, l=l, data=data[con][mouseID][view], xy='y') # check for 30 frames ahead of stance point just in case it cant find the curve as it is further ahead
                #                 newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)
                #             else:
                #                 satisfied = False
                #
                #         # 4th try: if can't find a sigmoid with either ctr or fwd aproach, find if there is a horizontal line between stance and it's next swing. Could just be a missing data issue
                #         if satisfied is False:
                #             if np.any(stancestart[i] - swingstart < 0):
                #                 try:
                #                     swingid = np.where(stancestart[i] - swingstart < 0)[0][0]
                #                     slope, intercept, r_value, p_value, std_err = stats.linregress(
                #                         xS.loc[stancestart[i]:swingstart[swingid]].index,
                #                         xS.loc[stancestart[i]:swingstart[swingid]])
                #                     if slope < 2:
                #                         newfirststance = stancestart[i]
                #                         satisfied = True
                #                     else:
                #                         satisfied = False
                #                 except:
                #                     print('Couldnt find horizontal line for run: %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                #                     satisfied = False
                #             else:
                #                 try:
                #                     finalidx = xS.index[-1]
                #                     slope, intercept, r_value, p_value, std_err = stats.linregress(
                #                         xS.loc[stancestart[i]:finalidx].index,
                #                         xS.loc[stancestart[i]:finalidx])
                #                     if slope < 2:
                #                         newfirststance = stancestart[i]
                #                         satisfied = True
                #                     else:
                #                         satisfied = False
                #                 except:
                #                     print('Couldnt find horizontal line for run (using final value of run): %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                #                     satisfied = False
                #
                #
                #         if satisfied is True:
                #             stancestart[i] = newfirststance
                #         else:
                #             # remove this stancestart value from the stancestart array
                #             todelALL.append(i)
                #             print('Cant find any way to validify this value for run: %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                #             if i in extr_indices:
                #                 print('This stance was possibly an erronous extra stance')
                #
                #     else:
                #         pass
                #
                # stancestart = np.delete(stancestart, todelALL)

                ## update swingstart and stancestart
                # gait_data = swstdf if xy == 'x' else y_swstdf
                gait_data = y_swstdf if xy == 'y' else comb_vel_y_df
                stancestart = gait_data.index[(gait_data.loc(axis=1)[0].values == 'stance').flatten()]
                swingstart = gait_data.index[(gait_data.loc(axis=1)[0].values == 'swing').flatten()]
                stancestart_bkwd = gait_data.index[(gait_data.loc(axis=1)[0].values == 'stance_bkwd').flatten()]
                swingstart_bkwd =  gait_data.index[(gait_data.loc(axis=1)[0].values == 'swing_bkwd').flatten()]


                # put first swing and stance frames into df
                StepCycle = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                swingmask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, swingstart)
                stancemask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, stancestart)
                swingmask_bkwd = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, swingstart_bkwd)
                stancemask_bkwd = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, stancestart_bkwd)

                StepCycle[stancemask] = 0
                StepCycle[swingmask] = 1
                StepCycle[stancemask_bkwd] = 2
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
                    raise ValueError('There is overlap between detection of first stance and first swing frame! This can''t be possible')

                StepCycleAll.append(StepCycle)

            StepCycleAll_flt = np.concatenate(StepCycleAll).ravel()
            for v in ['Side', 'Front', 'Overhead']:
                data[con][mouseID][v].loc(axis=1)[l, 'StepCycle'] = StepCycleAll_flt
            if fillvalues == True:
                StepCycleFilledAll_flt = np.concatenate(StepCycleFilledAll).ravel()
                for v in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][v].loc(axis=1)[l, 'StepCycleFill'] = StepCycleFilledAll_flt

        return data[con][mouseID]

    def getStanceSwingFrames(self, data, con, mouseID, view, l, r=None):
        if r is not None:
            stancemask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'] == 0
            swingmask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'] == 1
            stancemask_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'] == 2
            swingmask_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'] == 3

            stanceidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask]
            swingidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask]
            stanceidx_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask_bkwd]
            swingidx_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask_bkwd]

            stancex = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[stancemask]
            swingx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[swingmask]
            stancex_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[stancemask_bkwd]
            swingx_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[swingmask_bkwd]

            stancey_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[stancemask_bkwd]
            swingy_bkwd = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[swingmask_bkwd]
        else:
            stancemask = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 0
            swingmask = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 1
            stancemask_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 2
            swingmask_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 3

            stanceidx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask]
            swingidx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask]
            stanceidx_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask_bkwd]
            swingidx_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask_bkwd]

            stancex = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[stancemask]
            swingx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[swingmask]
            stancex_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[stancemask_bkwd]
            swingx_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[swingmask_bkwd]

            stancey = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[stancemask]
            swingy = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[swingmask]
            stancey_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[stancemask_bkwd]
            swingy_bkwd = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[swingmask_bkwd]

        stsw = {
            'Stance': {
                'idx': stanceidx,
                'x': stancex,
                'y': stancey
            },
            'Swing': {
                'idx': swingidx,
                'x': swingx,
                'y': swingy
            },
            'Stance_bkwd': {
                'idx': stanceidx_bkwd,
                'x': stancex_bkwd,
                'y': stancey_bkwd
            },
            'Swing_bkwd': {
                'idx': swingidx_bkwd,
                'x': swingx_bkwd,
                'y': swingy_bkwd
            }
        }

        return stsw


    def plotStepCycleRun(self, data, con, mouseID, r, view='Side'):
        '''

        :param data:
        :param con:
        :param mouseID:
        :param r: enter as list
        :param view:
        :return:
        '''
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']
        colors = {'ForepawToeR': 'darkblue', 'ForepawToeL': 'lightblue', 'HindpawToeR': 'darkgreen',
                  'HindpawToeL': 'lightgreen'}

        plt.figure()
        for l in limblist:
            mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'likelihood'] > pcutoff
            x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[mask]
            t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask]

            stsw = self.getStanceSwingFrames(data, con, mouseID, view, l, r)

            # # find derivative of change in angle between two points in x
            # dx = np.diff(t)
            # dy = np.diff(x)
            # slope = dx / dy
            # angle = np.arctan(slope)
            # angle_degrees = np.rad2deg(angle)
            # degreediff = np.diff(angle_degrees)

            #plt.figure()
            plt.plot(t, x, color=colors[l])
            plt.scatter(stsw['Stance']['idx'], stsw['Stance']['x'], marker='s', label="%s-Stance" %l, color=colors[l])
            plt.scatter(stsw['Swing']['idx'], stsw['Swing']['x'], marker='o', label="%s-Swing" %l, color=colors[l])
            #plt.vlines(t[2:][degreediff < -25], ymin=-100, ymax=2000, colors='red')

           # plt.scatter(t, x)

        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        transitionx = (markerstuff['DualBeltMarkers']['x_edges'][2] + markerstuff['DualBeltMarkers']['x_edges'][3])/2
        plt.hlines(y=transitionx, xmin=t[0], xmax=t[-1])
        plt.vlines(x=data[con][mouseID][view].loc(axis=0)[r, 'Transition'].index[0], ymin=x[0], ymax=x[-1])

        plt.legend()
        plt.title('Run: %s' %r)
        plt.xlabel('Frame number')
        plt.ylabel('x position (px)')
        plt.show()

    def get_xmotion_singleRun(self, data, con, mouseID, view, r, l, velstuff):
        commonidx = velstuff['runs_lowess'][int(r), 1].index.get_level_values(level='FrameIdx')
        mask = np.logical_and(
            data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
                l, 'likelihood'].values > pcutoff, velstuff['runs_lowess'][int(r), 1].values > 10)
        x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            l, 'x'].values[mask]
        y = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            l, 'y'].values[mask]
        t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            l, 'x'].index.get_level_values(level='FrameIdx')[mask]

        x_trans = data[con][mouseID][view].loc(axis=0)[r, 'Transition'].loc(axis=1)[l, 'x'].values[0]
        # y_trans = data[con][mouseID][view].loc(axis=0)[r, 'Transition'].loc(axis=1)[l, 'y'].values[0]
        t_trans = data[con][mouseID][view].loc(axis=0)[r, 'Transition'].loc(axis=1)[l, 'x'].index[0]

        xplot = ((t - t_trans) / fps) * 1000  # converts frames to ms
        yplot = x - x_trans

        return xplot, yplot

    def plot_overlay_runs_singleMouse(self, data, con, mouseID, expphase, view='Side', n=30):
        warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance.")

        position = np.where(np.array(ExpPhases) == expphase)[0][0]

        limblist = ['ForepawToeL', 'ForepawToeR', 'HindpawToeL', 'HindpawToeR']
        windowsize = math.ceil((fps / n) / 2.) * 2
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        velstuff = Velocity.Velocity().getVelocityInfo(data, con, mouseID, zeroed=False, view=view, xaxis='x', windowsize=windowsize, markerstuff=markerstuff, f=range(0, int(data[con][mouseID][view].index.get_level_values(level='Run').unique().max() + 1)))

        if 'APAChar_' in con:
            colors = utils.Utils().get_cmap(APACharRuns[position], 'cool')
            run_start = utils.Utils().find_phase_starts(APACharRuns)[position]
            run_num = APACharRuns[position]
        else:
            print('configure for other types of experiments')

        fig, axs = plt.subplots(2, 2, figsize=(15,10))

        for lidx, l in enumerate(limblist):
            row = lidx // 2
            col = lidx % 2

            firstval = run_start + 2
            lastval = firstval + run_num
            for r in range(firstval, lastval):
                try:
                    xplot, yplot = self.get_xmotion_singleRun(data,con,mouseID,view,r,l,velstuff)

                    axs[row,col].plot(xplot, yplot, color=colors(r - firstval))
                    axs[row, col].set_title(l, fontsize=18, y=0.95) #.set_weight('bold')
                except:
                    print('couldnt plot run %s' % r)

            # plot dotted lines to emphasise transition point
            axs[row, col].axvline(0, linestyle='--', color='black', alpha=0.3)
            axs[row, col].axhline(0, linestyle='--', color='black', alpha=0.3)

            # plot formatting
            axs[row,col].set_xlabel('Time from transition (ms)', fontsize=14)
            axs[row,col].set_ylabel('X position from transition (pixels)', fontsize=14)
            axs[row,col].set_xlim(-1000,250)
            axs[row,col].set_ylim(-1500,550)
            axs[row,col].tick_params(axis='both', labelsize=12)
            axs[row,col].spines['top'].set_visible(False)
            axs[row,col].spines['right'].set_visible(False)

        fig.suptitle('%s\n%s' %(expphase,mouseID), fontsize=20, y=1)

        # Create a ScalarMappable object for colorbar
        sm = ScalarMappable(cmap=colors)
        sm.set_array([])  # Set an empty array for the colorbar

        # Add a colorbar
        cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])  # Create an axes at the rightmost side of the figure
        cb = fig.colorbar(sm, cax=cax)  # Use the ScalarMappable object for the colorbar

        # Set the tick locations and labels of the colorbar
        cbar_len = lastval - firstval
        cb.set_ticks(np.linspace((1/cbar_len)/2, 1-(1/cbar_len)/2, cbar_len))
        cb.set_ticklabels(range(firstval-1, lastval-1))
        # Add a title to the colorbar
        cb.set_label('Run Number', fontsize=14)

        #fig.tight_layout()

        fig.suptitle('%s  -  %s' % (expphase, mouseID), fontsize=20, y=1) # do after tight layout as a cheat way of getting more space underneath

        fig.savefig(r'%s\Locomotion_OVERLAY, %s, %s, %s.png' % (plotting_destfolder, con, mouseID, expphase), bbox_inches='tight', transparent=True, format='png')


    def plot_overlay_runs_allPhases_allMice(self, conditions=['APAChar_LowHigh_Repeats_Wash_Day1','APAChar_LowHigh_Repeats_Wash_Day2','APAChar_LowHigh_Repeats_Wash_Day3'], expphase=['Baseline','APA','Washout']):
        data = Plot.Plot().GetDFs(conditions)

        for con in conditions:
            for midx, mouseID in enumerate(data[con].keys()):
                for eidx, e in enumerate(expphase):
                    try:
                        self.plot_overlay_runs_singleMouse(data=data,con=con,mouseID=mouseID,expphase=e)
                    except:
                        print('Cant plot for mouse: %s and expphase: %s' %(mouseID))



    def plot_mean_loco_allLimbs_perMouse(self, data, con, mouseID, expphase, variance='std', x='t', y='x', view='Side', n=30):
        warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance.")

        limblist = ['ForepawToeR', 'HindpawToeR'] #['ForepawToeL', 'ForepawToeR', 'HindpawToeL', 'HindpawToeR']
        colors = {'ForepawToeR': 'darkblue', 'ForepawToeL': 'lightblue', 'HindpawToeR': 'darkgreen',
                  'HindpawToeL': 'lightgreen'}
        windowsize = math.ceil((fps / n) / 2.) * 2
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        velstuff = Velocity.Velocity().getVelocityInfo(data, con, mouseID, zeroed=False, view=view, xaxis='x', windowsize=windowsize, markerstuff=markerstuff, f=range(0, int(data[con][mouseID][view].index.get_level_values(level='Run').unique().max() + 1)))

        fig, axs = plt.subplots(1, len(expphase), figsize=(35,10))
        #plt.figure()

        for eidx, e in enumerate(expphase):
            try:
                position = np.where(np.array(ExpPhases) == e)[0][0]

                if 'APAChar_' in con:
                    # colors = utils.Utils().get_cmap(APACharRuns[position], 'cool')
                    run_start = utils.Utils().find_phase_starts(APACharRuns)[position]
                    run_num = APACharRuns[position]
                else:
                    print('configure for other types of experiments')

                for lidx, l in enumerate(limblist):
                    try:
                        row = lidx // 2
                        col = lidx % 2

                        firstval = run_start + 2
                        lastval = firstval + run_num
                        All_x = []
                        All_y = []
                        for r in range(firstval, lastval):
                            try:
                                xplot, yplot = self.get_xmotion_singleRun(data, con, mouseID, view, r, l, velstuff)

                                All_x.append(xplot)
                                All_y.append(yplot)

                            except:
                                print('couldnt find run %s' % r)

                        upper_bound = min([run[-1] for run in All_x])
                        lower_bound = max([run[0] for run in All_x])
                        increment = (1 / fps) * 1000

                        # run_lengths = [len(all[np.logical_and(all >= lower_bound, all <= upper_bound)]) for all in All_x]
                        # longest_run = np.where(np.array(run_lengths) == max(run_lengths))[0][0]

                        #common_x = All_x[longest_run][np.logical_and(All_x[longest_run] >= lower_bound, All_x[longest_run] <= upper_bound)] # what if frame missing from first run???

                        full_list = list(set().union(*All_x))
                        full_list.sort()

                        test = pd.DataFrame(index=full_list, columns=range(len(All_x)))

                        copy_All_x = All_x.copy()
                        copy_All_y = All_y.copy()
                        for i in range(len(All_y)):
                            All_x[i] = copy_All_x[i][np.logical_and(copy_All_x[i] >= lower_bound, copy_All_x[i] <= upper_bound)]
                            All_y[i] = copy_All_y[i][np.logical_and(copy_All_x[i] >= lower_bound, copy_All_x[i] <= upper_bound)]
                            test.loc[All_x[i], i] = All_y[i]

                        #plt.figure()
                        axs[eidx].plot(test.index, test.mean(axis=1), color=colors[l], label=l)
                        if variance == 'sem':
                            axs[eidx].fill_between(test.index, test.mean(axis=1) - test.std(axis=1)/np.sqrt(test.count(axis=1)), test.mean(axis=1) + test.std(axis=1)/np.sqrt(test.count(axis=1)),
                                          interpolate=False, alpha=0.15, color=colors[l])
                        elif variance == 'std':
                            axs[eidx].fill_between(test.index, test.mean(axis=1) - test.std(axis=1), test.mean(axis=1) + test.std(axis=1), interpolate=False, alpha=0.15, color=colors[l])
                    except:
                        print('couldnt plot limb %s' %l)
            except:
                print('couldnt plot phase %s' %e)
            # plot dotted lines to emphasise transition point
            axs[eidx].axvline(0, linestyle='--', color='black', alpha=0.3)
            axs[eidx].axhline(0, linestyle='--', color='black', alpha=0.3)

            # plot formatting
            axs[eidx].set_title(e, fontsize=18, loc='center', pad=-80)
            axs[eidx].set_ylabel('X position from transition (pixels)', fontsize=16)
            axs[eidx].set_xlabel('Time from transition (ms)', fontsize=16)
            axs[eidx].tick_params(axis='both', labelsize=14)
            axs[eidx].spines['top'].set_visible(False)
            axs[eidx].spines['right'].set_visible(False)


            # # plot formatting
            # axs[eidx].title('%s (+/-%s)\n%s\n%s'%(expphase,variance,mouseID,con.split('_')[-1]), fontsize=16)
            # axs[eidx].xlabel('Time from transition (ms)', fontsize=14)
            # axs[eidx].ylabel('X position from transition (pixels)', fontsize=14)
            # axs[eidx].legend(title='Limb', bbox_to_anchor=(0.03, 0.97), loc='upper left', fontsize=12)
            # axs[eidx].tick_params(axis='both', labelsize=12)

        axs[0].legend(bbox_to_anchor=(0.03, 0.97), loc='upper left', fontsize=14)
        fig.suptitle('%s  -  %s  -  +/-%s' % (mouseID, con.split('_')[-1], variance), fontsize=20, y=1)

        max_x = max(ax.get_xlim()[1] for ax in axs)
        min_x = min(ax.get_xlim()[0] for ax in axs)
        max_y = max(ax.get_ylim()[1] for ax in axs)
        min_y = min(ax.get_ylim()[0] for ax in axs)
        # Set the x and y limits of each subplot to the maximum limits
        for ax in axs:
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        fig.tight_layout()

        fig.savefig(r'%s\Locomotion, %s, %s, %s.png' % (plotting_destfolder, con, mouseID, variance), bbox_inches='tight', transparent=True, format='png')


#common_x = All_x[0][np.logical_and(All_x[0] >= lower_bound, All_x[0] <= upper_bound)]


    def plot_mean_loco_allLimbs_ALLMICE(self, conditions, expphase=['Baseline','APA','Washout']):
        data = Plot.Plot().GetDFs(conditions)

        for con in conditions:
            for midx, mouseID in enumerate(data[con].keys()):
                    try:
                        self.plot_mean_loco_allLimbs_perMouse(data=data,con=con,mouseID=mouseID,expphase=expphase)
                    except:
                        print('Cant plot for mouse: %s and expphase: %s' %(mouseID))
'''
                ########################################################################################################
                ############################## FINDING BELT SPEED IN EACH CONDITION ####################################
                ########################################################################################################
                if 'APAChar' in con:
                    runs = APACharRuns
                elif 'Perception' in con:
                    runs = APAPerRuns
                elif 'VMT' in con:
                    runs = APAVmtRuns

                beltspeeds = utils.Utils().getSpeedConditions(con=con)

                ....

                belt1speed = None
                belt2speed = None
                # Calculate the velocity
                velocity = np.gradient(x, t)
                .....
                
                ##### find first stance frames and first swing frames
                markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

                if 'APAChar' in con or 'Perception' in con: ############################ !!!!!!!!!!!!!! COULD ADD IN HERE AN AND STATEMENT TO SAY AND NO E.G. REPEAT OR EXTENDED TO SIGNIFY ITS THE ORIGINAL EXP FORMAT !!!!!!!!!!!!! #########################
                    # define perception test as always low-high but this isnt in con name
                    if 'Perception' in con:
                        beltspeeds = ('Low', 'High')

                    # based on the phase of the experiment, define the speed of each belt
                    belt1speed = speeds[beltspeeds[0]]
                    if r < runs[0] or runs[0] + runs[1] <= r < runs[0] + runs[1] + runs[2]:
                        belt2speed = speeds[beltspeeds[0]]
                    elif runs[0] <= r < runs[0] + runs[1]:
                        belt2speed = speeds[beltspeeds[1]]
                    else:
                        raise ValueError('There are too many runs here')
                elif 'VMT' in con:
                    if 'pd' in con:
                        belt1speed = speeds[beltspeeds[0]]
                        belt2speed = speeds[beltspeeds[1]]
                    elif 'ac' in con:
                        print('!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!\nWARNING MESSAGE: Are you still using mid as the transformed belt2 speed for both ac and pd??? If not, change the code!')
                        # based on the phase of the experiment, define the speed of each belt
                        belt1speed = speeds[beltspeeds[0]]
                        if r < runs[0] or runs[0] + runs[1] <= r < runs[0] + runs[1] + runs[2]:
                            belt2speed = speeds[beltspeeds[1]]
                        elif runs[0] <= r < runs[0] + runs[1]:
                            belt2speed = speeds['Mid']
                        else:
                            raise ValueError('There are too many runs here')
                else:
                    raise ValueError('Something has gone wrong with run phase identification....')

                # now need to specify the velocity threshold based on if each limb is past the transition line
                beltmask = x_smooth > markerstuff['DualBeltMarkers']['x_edges'][3] # not sure if i should use x_smooth or raw x here
                belt1speed_px = ((markerstuff['cmtopx'] * belt1speed) / fps) + 1
                belt2speed_px = ((markerstuff['cmtopx'] * belt2speed) / fps) + 1
                prethreshmask = velocity[~beltmask] > belt1speed_px
                postthreshmask = velocity[beltmask] > belt2speed_px
                swingthreshmask = np.concatenate((prethreshmask,postthreshmask))
                
                ########################################################################################################
                ########################################################################################################
'''



