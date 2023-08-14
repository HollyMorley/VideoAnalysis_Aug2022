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
from statsmodels.nonparametric.smoothers_lowess import lowess


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
        y_swst_blocks = utils.Utils().find_blocks(new_t[mask], block_thresh, 1)
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
        warnings.filterwarnings("ignore", message="The behavior of indexing on a MultiIndex with a nested sequence of labels is deprecated and will change in a future version.")
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
                    pass
                    #print('Can\'t find any upcoming rotated stance values')
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
                    # print(
                    #     'Not sure how to deal with missing swing values! For now going to assume they are right and this represents extra stance values')
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
                        #print('This stance value seems fine (frame: %s)' % poss_stance_todel)
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

    def check_swst_pattern_limb_speed(self,swst_sequence,r,sitting_log):
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
                    print('Two peaks detected but probably just slowed down, assuming ok but check!')
                else:
                    if r not in sitting_log:
                        raise ValueError("Pattern mismatch: 'pos_peak' or 'neg_peak' should be followed by 'stance'.")
        print("The swing-stance pattern is followed in run %s" %r)



    def get_limb_speed(self,l,r,t,vel_limb,sitting_log):
        vel_limb_mask = np.isin(vel_limb.index.get_level_values(level='FrameIdx'), t)
        vel_limb_cropped = vel_limb[vel_limb_mask]
        vel_limb_cropped_nan = vel_limb_cropped[~vel_limb_cropped.isna()]
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if l == 'ForepawToeR':
            vel_limb_smoothed = pd.DataFrame(index=vel_limb_cropped_nan.index, data=np.array(lowess(vel_limb_cropped_nan,vel_limb_cropped_nan.index.get_level_values(level='FrameIdx'), frac=.04))[:,1])
        else:
            vel_limb_smoothed = pd.DataFrame(index=vel_limb_cropped_nan.index, data=np.array(lowess(vel_limb_cropped_nan,vel_limb_cropped_nan.index.get_level_values(level='FrameIdx'), frac=.08))[:,1])
        vel_limb_smoothed = vel_limb_smoothed[0]
        # if l == 'ForepawToeR' or l == 'ForepawToeL':
        #     val = vel_limb_cropped_nan
        # elif l == 'HindpawToeR' or l == 'HindpawToeL':
        #     val = vel_limb_smoothed

        # find peaks of swings
        # pos_mask = vel_limb_smoothed > base_speed
        peaks, info = scipy.signal.find_peaks(vel_limb_smoothed, distance=10, height=30, prominence=20)
        n_peaks, neg_info = scipy.signal.find_peaks(-1*vel_limb_smoothed, distance=10, height=10, prominence=10)
        peaks_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[peaks]
        neg_peaks_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[n_peaks]
        left_base_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[info['left_bases']]
        right_base_idx = vel_limb_smoothed.index.get_level_values(level='FrameIdx')[info['right_bases']]

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
        else:
            raise ValueError('This is not a real peak for swing')

        # find stationary blocks
        frame_shift = vel_limb_smoothed.shift(1) - vel_limb_smoothed
        stationary_mask = np.abs(frame_shift) < 5
        stationary_mean = np.mean(vel_limb_smoothed[stationary_mask].values)
        stationary_sem = np.std(vel_limb_smoothed[stationary_mask].values)/np.sqrt(len(vel_limb_smoothed[stationary_mask]))
        stationary_window_mask = np.logical_and(vel_limb_smoothed.values > 2,
                                                vel_limb_smoothed.values < stationary_mean + stationary_sem * 0.5)
        stance_raw = vel_limb_smoothed[stationary_window_mask]
        stance_start_idx = np.array(utils.Utils().find_blocks(stance_raw.index.get_level_values(level='FrameIdx'),10,2))[:,0]
        stance_end_idx = np.array(utils.Utils().find_blocks(stance_raw.index.get_level_values(level='FrameIdx'),10,2))[:,1]

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

        # form dfs of each data point to then combine together
        pos_peaks = pd.DataFrame(data=['pos_peak'] * len(peaks_idx), index=peaks_idx, columns=['val_type'])
        neg_peaks = pd.DataFrame(data=['neg_peak'] * len(neg_peaks_idx), index=neg_peaks_idx, columns=['val_type'])
        stance = pd.DataFrame(data=['stance'] * len(stance_start_idx), index=stance_start_idx, columns=['val_type'])
        swing = pd.DataFrame(data=['swing'] * len(stance_end_idx), index=stance_end_idx, columns=['val_type'])
        swst_sequence = pd.concat([pos_peaks, neg_peaks, stance, swing])
        swst_sequence = swst_sequence.sort_index()

        # check sequence
        self.check_swst_pattern_limb_speed(swst_sequence,r,sitting_log)

        return swst_sequence


    def get_limb_speed_borders(self,l,r,data,con,mouseID,view,t,markerstuff,vel_limb):
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
        upwards_sloping_blocks = utils.Utils().find_blocks(upwards_sloping_vals, 10, 1)
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

    def getLocoPeriods(self, data, con, mouseID, view='Side', fillvalues=True, n=30):
        warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
        warnings.filterwarnings("ignore", message="The behavior of indexing on a MultiIndex with a nested sequence of labels is deprecated and will change in a future version.")
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']
        windowsize = math.ceil((fps / n) / 2.) * 2
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        velstuff = Velocity.Velocity().getVelocityInfo(data, con, mouseID, zeroed=False, view=view, xaxis='x', windowsize=windowsize, markerstuff=markerstuff, f=range(0,int(data[con][mouseID][view].index.get_level_values(level='Run').unique().max()+1)))

        # take record of which runs include a period of sitting
        sitting_log_mask = []
        for r in data[con][mouseID][view].index.get_level_values(level='Run').unique().astype(int):
            mouse_stat_mask = abs(velstuff['runs_lowess'][r][1].diff()) < 0.01
            if sum(mouse_stat_mask) > 100:
                mouse_stat_blocks = utils.Utils().find_blocks(velstuff['runs_lowess'][r][1][mouse_stat_mask].index.get_level_values(level='FrameIdx'), 10, 20)
                if np.any(mouse_stat_blocks):
                    sitting_log_mask.append(True)
                else:
                    sitting_log_mask.append(False)
            else:
                sitting_log_mask.append(False)
        sitting_log = data[con][mouseID][view].index.get_level_values(level='Run').unique().astype(int)[sitting_log_mask].to_numpy()

        for l in tqdm(limblist):
            StepCycleAll = []
            StepCycleFilledAll = []
            for r in data[con][mouseID][view].index.get_level_values(level='Run').unique().astype(int):
                try:
                    swst_dict = self.get_FR_HR_HL_sidecam(data,con,mouseID,r,l,velstuff,markerstuff,sitting_log)

                    # put first swing and stance frames into df
                    StepCycle = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                    swingmask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['swingstart'])
                    stancemask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, swst_dict['stancestart'])

                    StepCycle[stancemask] = 0
                    StepCycle[swingmask] = 1

                    if l != 'ForepawToeL': # cant see this (currently) as only using front cam to detect FL
                        swingmask_bkwd = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['swingstart_bkwd'])
                        stancemask_bkwd = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values,swst_dict['stancestart_bkwd'])

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
                        raise ValueError(
                            'There is overlap between detection of first stance and first swing frame! This can''t be possible')

                    StepCycleAll.append(StepCycle)


                except:
                    print('Cant get sw/st data for run: %s, limb: %s' % (r, l))
                    StepCycle = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                    StepCycleAll.append(StepCycle)

            StepCycleAll_flt = np.concatenate(StepCycleAll).ravel()
            for v in ['Side', 'Front', 'Overhead']:
                data[con][mouseID][v].loc(axis=1)[l, 'StepCycle'] = StepCycleAll_flt
            if fillvalues == True:
                StepCycleFilledAll_flt = np.concatenate(StepCycleFilledAll).ravel()
                for v in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][v].loc(axis=1)[l, 'StepCycleFill'] = StepCycleFilledAll_flt

        return data[con][mouseID]

    def get_FR_HR_HL_sidecam(self, data, con, mouseID, r, l, velstuff, markerstuff, sitting_log, view='Side'):
        commonidx = velstuff['runs_lowess'][int(r), 1].index.get_level_values(level='FrameIdx')
        mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
                   l, 'likelihood'].values > pcutoff  # change so that NO LONGER excluding when mouse stationary (at this stage)

        # get vel data for whole mouse, remove sitting periods from available data, and identify runs with sitting
        mouse_stat_mask = abs(velstuff['runs_lowess'][r][1].diff()) < 0.01
        if sum(mouse_stat_mask) > 100:
            mouse_stat_blocks = utils.Utils().find_blocks(
                velstuff['runs_lowess'][r][1][mouse_stat_mask].index.get_level_values(level='FrameIdx'), 10, 20)
            if np.any(mouse_stat_blocks):
                mouse_stat_blocks_total = []
                for b in range(len(mouse_stat_blocks)):
                    idxs = np.arange(mouse_stat_blocks[b][0], mouse_stat_blocks[b][1])
                    mouse_stat_blocks_total.append(idxs)
                mouse_stat_blocks_total = np.concatenate(mouse_stat_blocks_total)
                mouse_stat_blocks_mask = np.isin(data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition',
                                                                                          'RunEnd'], commonidx].index.get_level_values(
                    level='FrameIdx'), mouse_stat_blocks_total)
                mask = mask & ~mouse_stat_blocks_mask

        ####### NEW ##########
        # fill in the gaps of missing data to avoid erronous velocity data being calculated CURRENTLY ONLY FOR FL
        if l == 'ForepawToeL':
            interpolated_x = self.fill_in_FL_gaps(data,con,mouseID,r,mask,commonidx)
            index_mask = np.isin(data[con][mouseID][view].index.get_level_values(level='FrameIdx'), interpolated_x.index)
            temp_data = data[con][mouseID][view].loc(axis=1)[l, 'x'].values
            temp_data[index_mask] = interpolated_x.values
            data[con][mouseID][view].loc(axis=1)[l, 'x'] = temp_data

        # get vel data for limb
        vel_limb = Velocity.Velocity().getVelocity_specific_limb(l, r, data, con, mouseID, view,
                                                                 math.ceil((fps / 30) / 2.) * 2, markerstuff,
                                                                 xy='x')  # get the velocity data for this body part (limb)
        vel_limb = vel_limb.loc[['RunStart', 'Transition', 'RunEnd']]

        # get basic info
        x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            l, 'x'].values[mask]
        y = \
        data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[l, 'y'][
            mask]
        t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            l, 'x'].index.get_level_values(level='FrameIdx')[mask]

        swst_sequence = self.get_limb_speed(l, r, t, vel_limb, sitting_log)

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

        # plt.figure()
        # plt.plot(t,x)
        # plt.vlines(x=swst[swst.values=='swing'].index.values.astype(int), ymin=100, ymax=1900, colors='red')
        # plt.vlines(x=swst[swst.values=='stance'].index.values.astype(int), ymin=100, ymax=1900, colors='green')
        # plt.vlines(x=swst[swst.values=='swing_bkwd'].index.values.astype(int), ymin=100, ymax=1900, colors='darkred', linestyle='--')
        # plt.vlines(x=swst[swst.values=='stance_bkwd'].index.values.astype(int), ymin=100, ymax=1900, colors='darkgreen', linestyle='--')
        # plt.title(r)
        # path = r'%s\Locomotion\individual_runs\%s\%s\x' % (plotting_destfolder, mouseID, l)
        # isExist = os.path.exists(path)
        # if not isExist:
        #     os.makedirs(path)
        # plt.savefig(r'%s\%s_%s_%s.png' % (path, mouseID, r, l), bbox_inches='tight', transparent=False,
        #             format='png')
        # plt.clf()

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

    def fill_in_FL_gaps(self, data, con, mouseID, r, mask, commonidx, view='Side'):
        # mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[
        #            'ForepawToeL', 'likelihood'] > pcutoff
        x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            'ForepawToeL', 'x'].values[
            mask]
        t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            'ForepawToeL', 'x'].index.get_level_values(level='FrameIdx')[mask]
        present = pd.DataFrame(data=x, index=t)[0]
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t, x)
        missing_t = \
        data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
            'ForepawToeL', 'x'].index.get_level_values(level='FrameIdx')[~mask]
        missing_x = cs(
            data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], commonidx].loc(axis=1)[
                'ForepawToeL', 'x'].index.get_level_values(level='FrameIdx')[~mask])
        missing = pd.DataFrame(data=missing_x, index=missing_t)[0]
        end_mask = np.logical_and(missing.index <= t[-1], missing.index >= t[1])
        missing = missing[end_mask]
        interpolated_x = pd.concat([present, missing], axis=0).sort_index()
        return interpolated_x

    def get_FL_swst_frontcam(self, data, con, mouseID, r, velstuff, markerstuff, beltspeed=6, n=30, view='Front'):
        from statsmodels.nonparametric.smoothers_lowess import lowess
        windowsize = math.ceil((fps / n) / 2.) * 2

        ######################################### TEMP ######################################################
        mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[
                   l, 'likelihood'] > pcutoff
        x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[
            mask]
        t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[
            l, 'x'].index.get_level_values(level='FrameIdx')[mask]
        present = pd.DataFrame(data=x,index=t)[0]
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t, x)
        missing_t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[
            l, 'x'].index.get_level_values(level='FrameIdx')[~mask]
        missing_x = cs(data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[
                           l, 'x'].index.get_level_values(level='FrameIdx')[~mask])
        missing = pd.DataFrame(data=missing_x,index=missing_t)[0]
        end_mask = np.logical_and(missing.index <= t[-1], missing.index >= t[1])
        missing = missing[end_mask]
        interpolated_x = pd.concat([present, missing], axis=0).sort_index()

        plt.figure()
        plt.scatter(interpolated_x.index, interpolated_x.values)
        plt.title(r)

        #####################################################################################################


        # get y data
        vel_y = Velocity.Velocity().getVelocity_specific_limb('ForepawToeL', r, data, con, mouseID, view, math.ceil((fps / 30) / 2.) * 2, markerstuff, xy='y')

        # trim vel_y to remove stationary periods
        vel_y_trimmed = vel_y.reindex(velstuff['runs_raw'][r][1].index)
        vel_y_moving = vel_y_trimmed[velstuff['runs_raw'][r][1] > beltspeed]
        vel_y_moving = vel_y_moving[~vel_y_moving.isna()] # remove na values

        # smooth y
        vel_smooth = pd.DataFrame(index=vel_y_moving.index, data=np.array(lowess(vel_y_moving, vel_y_moving.index.get_level_values(level='FrameIdx'),frac=.08))[:, 1])[0]

        limb_mask = data[con][mouseID][view].loc(axis=1)['ForepawToeL', 'likelihood'].loc(axis=0)[r].loc(axis=0)[
                    vel_smooth.index] > pcutoff #0.99999 # higher cutoff
        vel_smooth = vel_smooth[limb_mask]
        peaks, info = scipy.signal.find_peaks(vel_smooth, distance=10, width=5, prominence=3, wlen=50)

        stancestart = vel_smooth.index.get_level_values(level='FrameIdx')[peaks]
        #swingstart = vel_y.index.get_level_values(level='FrameIdx')[info['right_bases']]

        # refine swing values
        y = data[con][mouseID][view].loc(axis=0)[r].loc(axis=0)[vel_smooth.index].loc(axis=1)['ForepawToeL', 'y'][limb_mask]

        # get slope based on a shorter snippet of y, defined by the area where stances have been found (to try remove messy parts of the data, e.g. stationary sections)
        if len(stancestart > 0):
            if len(stancestart > 1):
                sequence = range(stancestart[0] - 20, stancestart[-1] + 20)
                index_mask = y.index.isin(sequence,level=1)
                short_y = y[index_mask]
                slope, intercept, r_value, p_value, std_err = stats.linregress(short_y.index.get_level_values(level='FrameIdx'), short_y)
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(y.index.get_level_values(level='FrameIdx'), y)

            y_corr = y - ((y.index.get_level_values(level='FrameIdx') - y.index.get_level_values(level='FrameIdx')[0]) * slope)  # y corrected horizontally
            from statsmodels.nonparametric.smoothers_lowess import lowess
            y_smooth = pd.DataFrame(index=y_corr.index, data=np.array(lowess(y_corr, y_corr.index.get_level_values(level='FrameIdx'),frac=.08))[:, 1])
            y_smooth = y_smooth[0]

            peaks, info = scipy.signal.find_peaks(y_smooth, distance=10, wlen=50)
            peaks_idx = y_smooth.index.get_level_values(level='FrameIdx')[peaks]

            dist = peaks_idx.values - stancestart.values[:, np.newaxis]
            dist = dist.astype(float)
            dist[np.logical_or(dist < 0, dist > 20)] = np.nan
            if np.all(np.isnan(dist[-1])):
                dist = dist[:-1]
            potentialClosestpos = np.nanargmin(dist, axis=1)
            closestFound, closestCounts = np.unique(potentialClosestpos, return_counts=True)
            if np.any(closestCounts > 1):
                raise ValueError('found the same peak for two stancestarts, inducating a missing peak')
            stance_peak = peaks_idx[potentialClosestpos]

            swingstart = []
            for i, peak in enumerate(stance_peak):
                target_y = y_smooth.xs(key=stancestart[i], level='FrameIdx').values
                ymask = np.logical_and.reduce((y_smooth.index.get_level_values(level='FrameIdx') > stance_peak[i],
                                               y_smooth.values < target_y))
                if sum(ymask) > 0:
                    new_swing = y_smooth[ymask].index.get_level_values(level='FrameIdx')[0] - 1
                else:
                    target_y = y_smooth.xs(key=stancestart[i]+1, level='FrameIdx').values
                    ymask = np.logical_and.reduce((y_smooth.index.get_level_values(level='FrameIdx') > stance_peak[i],
                                                   y_smooth.values < target_y))
                    if sum(ymask) > 0:
                        new_swing = y_smooth[ymask].index.get_level_values(level='FrameIdx')[0] - 1
                    else:
                        target_y = y_smooth.xs(key=stancestart[i] + 2, level='FrameIdx').values
                        ymask = np.logical_and.reduce((y_smooth.index.get_level_values(level='FrameIdx') > stance_peak[i],
                                                       y_smooth.values < target_y))
                        if sum(ymask) > 0:
                            new_swing = y_smooth[ymask].index.get_level_values(level='FrameIdx')[0] - 1
                        elif i == len(stance_peak) - 1: # if the final stance, there may be no visible following swing
                            pass
                        else:
                            raise ValueError('Cant detect swing for stance %s' %i)
                swingstart.append(new_swing)

            swst_dict = {
                'stancestart': stancestart,
                'swingstart': swingstart
            }
        else:
            raise ValueError('No detections')

        return swst_dict


    def find_first_transitioning_paw(self, data, con, mouseID):
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']
        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID]['Side'])

        for r in data[con][mouseID]['Side'].index.get_level_values(level='Run').unique().astype(int):
            nose_mask = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart','Transition','RunEnd']].loc(axis=1)['Nose','likelihood'] > pcutoff
            nose_x = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['Nose', 'x'][nose_mask]
            nose_y = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['Nose', 'y'][nose_mask]
            #t = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].index.get_level_values(level='FrameIdx')[nose_mask]

            trans_x_far = markerstuff['DualBeltMarkers']['x_edges'][2]
            trans_x_near = markerstuff['DualBeltMarkers']['x_edges'][3]

            # find when nose passes the transition point
            nose_trans_mask_far = nose_x > trans_x_far
            nose_trans_idx_far = nose_x[nose_trans_mask_far].index.get_level_values(level='FrameIdx')[0]
            nose_trans_mask_near = nose_x > trans_x_near
            nose_trans_idx_near = nose_x[nose_trans_mask_near].index.get_level_values(level='FrameIdx')[0]

            #####
            vel_y = Velocity.Velocity().getVelocity_specific_limb('ForepawToeL', r, data, con, mouseID, view, math.ceil((fps / 30) / 2.) * 2, markerstuff, xy='y')
            vel_y_smooth = pd.DataFrame(index=vel_y[5:-3].index, data=np.array(
                lowess(vel_y[5:-3], vel_y[5:-3].index.get_level_values(level='FrameIdx'),
                       frac=.08))[:, 1])
            vel_y_smooth = vel_y_smooth[0]
            peaks, info = scipy.signal.find_peaks(vel_y, distance=10, height=10, prominence=5, wlen=20)
            peaks_idx = vel_y.index.get_level_values(level='FrameIdx')[peaks]
            swingstart = vel_y.index.get_level_values(level='FrameIdx')[info['left_bases']]
            stancestart = vel_y.index.get_level_values(level='FrameIdx')[info['right_bases']]




            fp_toe_L_mask = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart','Transition','RunEnd']].loc(axis=1)['ForepawToeL','likelihood'] > pcutoff
            fp_toe_L_x = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['ForepawToeL', 'x'][fp_toe_L_mask]
            fp_toe_L_y = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['ForepawToeL', 'y'][fp_toe_L_mask]
            fp_toe_L_t = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].index.get_level_values(level='FrameIdx')[fp_toe_L_mask]

            slope, intercept, r_value, p_value, std_err = stats.linregress(fp_toe_L_t, fp_toe_L_y)
            new_FL_y = fp_toe_L_y - ((fp_toe_L_t - fp_toe_L_t[0]) * slope)  # y corrected horizontally

            # fp_toe_R_mask = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart','Transition','RunEnd']].loc(axis=1)['ForepawToeR','likelihood'] > pcutoff
            # fp_toe_R_x = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['ForepawToeR', 'x'][fp_toe_R_mask]
            # fp_toe_R_y = data[con][mouseID]['Front'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].loc(axis=1)['ForepawToeR', 'y'][fp_toe_R_mask]
            # fp_toe_R_t = data[con][mouseID]['Side'].loc(axis=0)[r, ['RunStart', 'Transition','RunEnd']].index.get_level_values(level='FrameIdx')[fp_toe_R_mask]
            #
            # slope, intercept, r_value, p_value, std_err = stats.linregress(fp_toe_R_t, fp_toe_R_y)
            # new_FR_y = fp_toe_R_y - ((fp_toe_R_t - fp_toe_R_t[0]) * slope)  # y corrected horizontally

            from statsmodels.nonparametric.smoothers_lowess import lowess
            FL_smooth = pd.DataFrame(index=new_FL_y.index, data=np.array(
                lowess(new_FL_y, new_FL_y.index.get_level_values(level='FrameIdx'),
                       frac=.04))[:, 1])
            FL_smooth = FL_smooth[0]

            # find stat period for fl
            stat_mask = abs(FL_smooth.diff()) < 0.5
            stat = FL_smooth[stat_mask]
            stat_blocks_poss = utils.Utils().find_blocks(stat.index.get_level_values(level='FrameIdx'),2,8)

            ## temp
            # plt.figure()
            # plt.plot(fp_toe_L_t, FL_smooth, label='L')
            # plt.scatter(fp_toe_L_t, FL_smooth)
            # plt.vlines(nose_trans_idx_far, ymin=145, ymax=190, linestyle='--')
            # plt.vlines(nose_trans_idx_near, ymin=145, ymax=190, linestyle='--')
            # plt.legend()
            # plt.scatter(stat.index.get_level_values(level='FrameIdx'), stat.values, color='red')
            # plt.title(r)

            # get peaks to signify stance periods
            peaks, info = scipy.signal.find_peaks(FL_smooth, distance=10, height=10)
            peaks_idx = FL_smooth.index.get_level_values(level='FrameIdx')[peaks]

            ### find velocity????



            front_poss_trans_mask = data[con][mouseID]['Front'].index.get_level_values(level='FrameIdx') > nose_trans_idx
            front_poss_trans = data[con][mouseID]['Front'].loc(axis=1)['ForepawToeR','y'][front_poss_trans_mask]

            from statsmodels.nonparametric.smoothers_lowess import lowess
            front_poss_trans_smoothed = pd.DataFrame(index=front_poss_trans.index, data=np.array(
                lowess(front_poss_trans, front_poss_trans.index.get_level_values(level='FrameIdx'),
                       frac=.08))[:, 1])
            front_poss_trans_smoothed = front_poss_trans_smoothed[0]

        # look in FRONT now and find the first frame when mouse touches either paw down


    def get_number_plots(self, root_path):
        limbs = ['ForepawToeL', 'ForepawToeR', 'HindpawToeL', 'HindpawToeR']
        # Initialize an empty dictionary to store the file counts
        file_counts = {}

        # Traverse the directory structure
        for mouse_dir in os.listdir(root_path):
            mouse_path = os.path.join(root_path, mouse_dir)
            if os.path.isdir(mouse_path):
                file_counts[mouse_dir] = {}
                for limb in limbs:
                    limb_path = os.path.join(mouse_path, limb, 'x')
                    if os.path.isdir(limb_path):
                        file_counts[mouse_dir][limb] = len(os.listdir(limb_path))
                    else:
                        file_counts[mouse_dir][limb] = 0

        # Create a dataframe from the file counts dictionary
        df = pd.DataFrame(file_counts)

        # Transpose the dataframe and set the mouseID as the index
        df = df.T
        df.index.name = 'mouseID'

        return df

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

            stancey = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[stancemask]
            swingy = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[swingmask]
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



