import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import Helpers.utils as utils
import Helpers.GetRuns as GetRuns
from Helpers.Config import *
import Plot
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm


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


    def getLocoPeriods(self, data, con, mouseID, fillvalues=True, view='Side'):
        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL']

        for l in tqdm(limblist):
            StepCycleAll = []
            StepCycleFilledAll = []
            for r in data[con][mouseID][view].index.get_level_values(level='Run').unique():
                mask = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'likelihood'] > pcutoff
                x = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[mask]
                t = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask]
                xS = pd.Series(data=x, index=t)

                # new swing/ stance id
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, x)
                xcorr_rot0 = (x - slope * t) - intercept
                xcorr_rot90 = (x - slope * t * 0.5) - intercept

                # find start of stance
                stancestart = self.findPeakByRot(xcorr_rot90, t, w=5, p=20)
                stancestart_rot0 = self.findPeakByRot(xcorr_rot0, t, w=1, p=10)

                # find start of swing
                swingstart = self.findPeakByRot(-xcorr_rot0, t, w=5, p=20)

                # Check whether pattern of swing vs stance is right
                # this assumes there are never any missing swing values!!
                diff = stancestart[:, np.newaxis] - swingstart
                miss = np.array([])
                posALL = []
                for i in range(0, diff.shape[0]):
                    pos = sum(diff[i] > 0)
                    posALL.append(pos)

                # find and try and replace any missing values
                if np.any(np.diff(posALL) > 1):
                    miss = np.array(np.where(np.diff(posALL) > 1)[0] + 1)  # find the expected position of any missing stances

                if np.any(miss): # if there are any missing values found
                    # check stancestart_rot0
                    for i in miss:
                        poss_st = np.logical_and(stancestart_rot0 > stancestart[i-1], stancestart_rot0 < stancestart[i]) # find possible values
                        if np.any(poss_st):
                            if sum(poss_st) == 1:
                                newvalmask = np.logical_and(stancestart_rot0 > stancestart[i-1], stancestart_rot0 < stancestart[i])
                                stancestart = np.concatenate((stancestart, stancestart_rot0[newvalmask]))
                                stancestart.sort()
                            else:
                                print('Found %s extra possible values for stance_bkup to fill in missing gap in run %s, limb %s' %(sum(poss_st), r, l))
                        else:
                            print('Unable to find stance values for run: %s and limb: %s' %(r,l))
                             # maybe need to estitmate stance value here

                extr_indices = []
                if np.any(np.diff(posALL) < 1):
                    #extr = np.array(np.where(np.diff(posALL) < 1)[0])  # find the first instance of a swing phase with multiple stances
                    unique_vals, counts = np.unique(posALL, return_counts=True)
                    non_unique_vals = unique_vals[counts > 1]  # get the non-unique values
                    extr_indices = np.where(np.isin(posALL, non_unique_vals))[0]


                # Remove erronous stancestarts AND refine frames
                markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
                todelALL = []
                for i in range(0, len(stancestart)):
                    xs = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd'], stancestart[i]].loc(axis=1)[l, 'x'].values[0] # find x value in single frame where l is in stance

                    # Check that paw is not in transition area...
                    if xs < markerstuff['DualBeltMarkers']['x_edges'][2] or xs > markerstuff['DualBeltMarkers']['x_edges'][3] + 5:

                        # 1st try: find curve and refine using main identified stance values
                        ydata, frame, p0 = self.findsigmoid(type='ctr', st=stancestart[i], r=r, l=l, data=data[con][mouseID][view], xy='y')
                        newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)

                        # 2nd try: find curve and refine using back up stance values (from tilting)
                        if satisfied is False:
                            # Find if there are any stance_bkup values nearby
                            diffs = abs(stancestart[i] - stancestart_rot0)  # find difference between current stance val and all the stance_bkup vals
                            if np.min(diffs[np.nonzero(diffs)]) < 30: # if there is a bkup value that is within 30 frames and isnt the same value....
                                nearmask = np.where(np.min(diffs[np.nonzero(diffs)]) == diffs)[0][0] # create mask to show where the closest stance_bkup value that isnt the same value as stance is
                                nearest = stancestart_rot0[nearmask]
                                ydata, frame, p0 = self.findsigmoid(type='ctr', st=nearest, r=r, l=l, data=data[con][mouseID][view], xy='y')
                                newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)

                        # 3rd try: find curve and refine using stance values from 30 frames ahead
                        if satisfied is False:
                            lastidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[mask][-1]
                            if stancestart[i] + 30 < lastidx:
                                ydata, frame, p0 = self.findsigmoid(type='fwd', st=stancestart[i], r=r, l=l, data=data[con][mouseID][view], xy='y') # check for 30 frames ahead of stance point just in case it cant find the curve as it is further ahead
                                newfirststance, satisfied = self.findNewVal(r, l, i, ydata, p0, frame)
                            else:
                                satisfied = False

                        # 4th try: if can't find a sigmoid with either ctr or fwd aproach, find if there is a horizontal line between stance and it's next swing. Could just be a missing data issue
                        if satisfied is False:
                            if np.any(stancestart[i] - swingstart < 0):
                                try:
                                    swingid = np.where(stancestart[i] - swingstart < 0)[0][0]
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        xS.loc[stancestart[i]:swingstart[swingid]].index,
                                        xS.loc[stancestart[i]:swingstart[swingid]])
                                    if slope < 2:
                                        newfirststance = stancestart[i]
                                        satisfied = True
                                    else:
                                        satisfied = False
                                except:
                                    print('Couldnt find horizontal line for run: %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                                    satisfied = False
                            else:
                                try:
                                    finalidx = xS.index[-1]
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        xS.loc[stancestart[i]:finalidx].index,
                                        xS.loc[stancestart[i]:finalidx])
                                    if slope < 2:
                                        newfirststance = stancestart[i]
                                        satisfied = True
                                    else:
                                        satisfied = False
                                except:
                                    print('Couldnt find horizontal line for run (using final value of run): %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                                    satisfied = False


                        if satisfied is True:
                            stancestart[i] = newfirststance
                        else:
                            # remove this stancestart value from the stancestart array
                            todelALL.append(i)
                            print('Cant find any way to validify this value for run: %s, limb: %s, stance: %s --> Deleted ' % (r, l, i))
                            if i in extr_indices:
                                print('This stance was possibly an erronous extra stance')

                    else:
                        pass

                stancestart = np.delete(stancestart, todelALL)

                # put first swing and stance frames into df
                StepCycle = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                swingmask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, swingstart)
                stancemask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, stancestart)

                StepCycle[stancemask] = 0
                StepCycle[swingmask] = 1

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

            stanceidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask]
            swingidx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask]

            stancex = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[stancemask]
            swingx = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'x'].values[swingmask]

            stancey = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[stancemask]
            swingy = data[con][mouseID][view].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].loc(axis=1)[l, 'y'].values[swingmask]
        else:
            stancemask = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 0
            swingmask = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'] == 1

            stanceidx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[stancemask]
            swingidx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'StepCycle'].index.get_level_values(level='FrameIdx').values[swingmask]

            stancex = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[stancemask]
            swingx = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'x'].values[swingmask]

            stancey = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[stancemask]
            swingy = data[con][mouseID][view].loc[pd.IndexSlice[:, ['RunStart', 'Transition', 'RunEnd']], :].loc(axis=1)[l, 'y'].values[swingmask]

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
        plt.show()

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










        # mask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'likelihood'] > pcutoff
        # x = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].values[mask]
        # y = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'y'].values[mask]
        # t = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].index.get_level_values(
        #     level='FrameIdx')[mask]
        # # Smooth the x-coordinate data
        # x_smooth = savgol_filter(x, window_length=20, polyorder=10) #x_smooth = savgol_filter(x, window_length=5, polyorder=3)
        # y_smooth = savgol_filter(y, window_length=20, polyorder=3)
        #
        # # Calculate the velocity
        # velocity = np.gradient(x_smooth, t)
        #
        # # # Find the high and low peaks
        # # peaks, _ = scipy.signal.find_peaks(velocity, width=3, distance=10, height=2*3.44, rel_height=1) # distance was 25, height was 2 ############# CURRENTLY HEIGHT IS 2X 30CM/S ###############
        # # # find level of the belt
        # # belty = utils.Utils().findConfidentMeans(data=data[con][mouseID][view].loc(axis=0)[r], label='StartPlatL',
        # #                                          coor='y')
        #
        # # # to compare with the y movements of front view...
        # # frontpawview = -y_smooth
        # # slope, intercept, r_value, p_value, std_err = stats.linregress(t, frontpawview)
        # # y_corrected = (frontpawview - slope*t) - intercept
        # # peaks, _ = scipy.signal.find_peaks(y_corrected, width=3, distance=10, rel_height=1)
        #
        #
        # ##### find first stance frames and first swing frames
        # markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
        # maxspeed_px = (markerstuff['cmtopx'] * highspeed)/fps
        # swingthreshmask = velocity > maxspeed_px
        #
        # #Swing
        # window_size = 6
        # rolling_swing = np.zeros(np.shape(swingthreshmask))
        # rolling_swing[:-window_size + 1] = np.sum(np.vstack([swingthreshmask[:window_size][::-1], np.vstack([swingthreshmask[i:i - window_size:-1] for i in range(window_size, len(swingthreshmask))])]), axis=1)
        # # Create a mask for the positions where the pattern occurs
        # swingmask = np.logical_and(rolling_swing == 5, np.logical_not(swingthreshmask))
        # # Find the indices where the mask is True and then pick the next one as this is the first True value where the previous value is False and the next 4 are True
        # swingpositions = np.where(swingmask)[0] + 1
        #
        # # Stance
        # rolling_stance = np.zeros(np.shape(swingthreshmask))
        # rolling_stance[window_size - 1:] = np.sum(np.vstack([swingthreshmask[:window_size][::-1], np.vstack([swingthreshmask[i:i - window_size:-1] for i in range(window_size, len(swingthreshmask))])]), axis=1)
        # # Create a mask for the positions where the pattern occurs
        # stancemask = np.logical_and(rolling_stance == 5, np.logical_not(swingthreshmask))
        # # Find the indices where the mask is True and then pick the next one as this is the first True value where the previous value is False and the next 4 are True
        # stancepositions = np.where(stancemask)[0]
        #
        #
        # Stance = np.full([len(data[con][mouseID][view])], np.nan)
        # Swing = np.full([len(data[con][mouseID][view])], np.nan)
        #
        # Stanceidx = \
        # data[con][mouseID][view].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask][
        #     stancepositions]
        # Swingidx = \
        # data[con][mouseID][view].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask][
        #     swingpositions]
        #
        # ########### do for stance also AND by limb eg FR, FL, HR, HL ###################
        # swingmask = np.isin(data[con][mouseID][view].index.get_level_values(level='FrameIdx').values, Swingidx.values)
        # Swing[swingmask] = 1
        # data[con][mouseID][view].loc(axis=1)['Swing'] = Swing
        #
        # stancemask = np.isin(data[con][mouseID][view].index.get_level_values(level='FrameIdx').values, Stanceidx.values)
        # Stance[stancemask] = 1
        # data[con][mouseID][view].loc(axis=1)['Stance'] = Stance
        #
        #
        # # startplat = utils.Utils().findConfidentMeans(data=data[con][mouseID][view].loc(axis=0)[r], label='StartPlatR', coor='x')
        # # trans = utils.Utils().findConfidentMeans(data=data[con][mouseID][view].loc(axis=0)[r], label='TransitionR', coor='x')
        #
        # # # Plot the velocity as a function of time
        # # plt.plot(t, velocity)
        # # # Plot stars at the high and low peaks
        # # plt.scatter(t[peaks], velocity[peaks], marker='*', color='red')
        # # plt.plot(t, y_corrected, color='green')
        # # plt.scatter(t[peaks], y_corrected[peaks], marker='*', color='red')
        # # plt.vlines(x=data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].index.get_level_values(
        # #     level='FrameIdx')[mask][swingpositions], ymin=-10, ymax=70, colors='purple')
        # #
        # # plt.vlines(x=data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].index.get_level_values(
        # #     level='FrameIdx')[mask][stancepositions], ymin=-10, ymax=70, colors='pink')
        # #
        # # plt.vlines(x=[data[con][mouseID][view].loc(axis=0)[r, 'RunStart'].index[0],
        # #               data[con][mouseID][view].loc(axis=0)[r, 'Transition'].index[0]], ymin=-10, ymax=70,
        # #            colors='black', linestyle='--')
        # # plt.hlines([3.44, 0.689],
        # #            xmin=data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx')[0],
        # #            xmax=data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx')[-1])
        # #
        # # plt.xlabel('Time (s)')
        # # plt.ylabel('Velocity (px/frame)')
        # # # Show the plot
        # # plt.show()



