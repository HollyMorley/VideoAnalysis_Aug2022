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
from scipy import stats


class Locomotion():

    def __init__(self):
        super().__init__()

    def getLocoPeriods(self, data, con, mouseID, view):
        # limbMask = {
        #     'Toe': {
        #         'ForepawToeR': data[con][mouseID][view].loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,
        #         'ForepawToeL': data[con][mouseID][view].loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,
        #         'HindpawToeR': data[con][mouseID][view].loc(axis=1)['HindpawToeR', 'likelihood'] > pcutoff,
        #         'HindpawToeL': data[con][mouseID][view].loc(axis=1)['HindpawToeL', 'likelihood'] > pcutoff
        #     },
        #     'Ankle': {
        #         'ForepawAnkleR': data[con][mouseID][view].loc(axis=1)['ForepawAnkleR', 'likelihood'] > pcutoff,
        #         'ForepawAnkleL': data[con][mouseID][view].loc(axis=1)['ForepawAnkleL', 'likelihood'] > pcutoff,
        #         'HindpawAnkleR': data[con][mouseID][view].loc(axis=1)['HindpawAnkleR', 'likelihood'] > pcutoff,
        #         'HindpawAnkleL': data[con][mouseID][view].loc(axis=1)['HindpawAnkleL', 'likelihood'] > pcutoff
        #     }
        # }

        if 'APAChar' in con:
            runs = APACharRuns
        elif 'Perception' in con:
            runs = APAPerRuns
        elif 'VMT' in con:
            runs = APAVmtRuns

        beltspeeds = utils.Utils().getSpeedConditions(con=con)

        limblist = ['ForepawToeR', 'ForepawToeL', 'HindpawToeR', 'HindpawToeL', 'ForepawAnkleR', 'ForepawAnkleL', 'HindpawAnkleR', 'HindpawAnkleL']

        for r in data[con][mouseID][view].index.get_level_values(level='Run').unique():
            belt1speed = None
            belt2speed = None
            for l in limblist:
                mask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'likelihood'] > pcutoff
                x = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].values[mask]
                t = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[l, 'x'].index.get_level_values(level='FrameIdx')[mask]
                # Smooth the x-coordinate data
                x_smooth = savgol_filter(x, window_length=20, polyorder=10)  # x_smooth = savgol_filter(x, window_length=5, polyorder=3)

                # Calculate the velocity
                velocity = np.gradient(x_smooth, t)

                ##### find first stance frames and first swing frames
                markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view].loc(axis=0)[r])

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
                belt1speed_px = (markerstuff['cmtopx'] * belt1speed) / fps
                belt2speed_px = (markerstuff['cmtopx'] * belt2speed) / fps
                prethreshmask = velocity[~beltmask] > belt1speed_px
                postthreshmask = velocity[beltmask] > belt2speed_px
                swingthreshmask = np.concatenate((prethreshmask,postthreshmask))

                # find Swing
                window_size = 6
                rolling_swing = np.zeros(np.shape(swingthreshmask))
                rolling_swing[:-window_size + 1] = np.sum(np.vstack([swingthreshmask[:window_size][::-1], np.vstack([swingthreshmask[i:i - window_size:-1] for i in range(window_size, len(swingthreshmask))])]), axis=1)
                # Create a mask for the positions where the pattern occurs
                swingmask = np.logical_and(rolling_swing == 5, np.logical_not(swingthreshmask))
                # Find the indices where the mask is True and then pick the next one as this is the first True value where the previous value is False and the next 4 are True
                swingpositions = np.where(swingmask)[0] + 1

                # find Stance
                rolling_stance = np.zeros(np.shape(swingthreshmask))
                rolling_stance[window_size - 1:] = np.sum(np.vstack([swingthreshmask[:window_size][::-1], np.vstack(
                    [swingthreshmask[i:i - window_size:-1] for i in range(window_size, len(swingthreshmask))])]), axis=1)
                # Create a mask for the positions where the pattern occurs
                stancemask = np.logical_and(rolling_stance == 5, np.logical_not(swingthreshmask))
                # Find the indices where the mask is True and then pick the next one as this is the first True value where the previous value is False and the next 4 are True
                stancepositions = np.where(stancemask)[0]

                # put first swing and stance frames into df
                Stance = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                Swing = np.full([len(data[con][mouseID][view].loc(axis=0)[r])], np.nan)
                Stanceidx = t[stancepositions]
                Swingidx = t[swingpositions]
                swingmask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, Swingidx.values)
                Swing[swingmask] = 1
                stancemask = np.isin(data[con][mouseID][view].loc(axis=0)[r].index.get_level_values(level='FrameIdx').values, Stanceidx.values)
                Stance[stancemask] = 1

                for view in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][view].loc[r,(l,'Swing')] = Swing
                    data[con][mouseID][view].loc[r, (l, 'Stance')] = Stance









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



