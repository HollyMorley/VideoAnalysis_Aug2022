import numpy as np
import pandas as pd

from Helpers.Config_23 import *
from Helpers import Structural_calculations


class Velocity:
    def __init__(self, data, con, mouseID): # MouseData is input ie list of h5 files
        self.data = data
        self.con = con
        self.mouseID = mouseID

    def getVelocity_specific_limb(self, limb, view, r, windowsize, triang, pixel_sizes):
        """
        Find velocity of limb of mouse. In side view, speed is in x plane, in front view, speed is in z plane
        :param limb:
        :param view:
        :param r:
        :param windowsize:
        :param triang: mapping of side and front so can see limbs
        :param pixel_sizes:
        :return:
        """
        # Define relevant DLC tracking data
        Tail_mask = np.logical_and(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[
                                       'Tail1', 'likelihood'].values > pcutoff,
                                   self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[
                                       'Tail2', 'likelihood'].values > pcutoff)
        limb_mask_side = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limb, 'likelihood'].values > pcutoff
        limb_mask_front = self.data[self.con][self.mouseID]['Front'].loc(axis=0)[r].loc(axis=1)[limb, 'likelihood'].values > pcutoff
        mask = np.logical_and(Tail_mask, limb_mask_side, limb_mask_front)

        xpos = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limb, 'x'][mask]
        ypos = self.data[self.con][self.mouseID]['Front'].loc(axis=0)[r].loc(axis=1)[limb, 'x'][mask]

        # Get relevant x (x plane) or y (z plane) position of limb depending on view using
        if view == 'Side':
            limb_pos = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limb, 'x'][mask]
        elif view == 'Front':
            limb_pos = self.data[self.con][self.mouseID]['Front'].loc(axis=0)[r].loc(axis=1)[limb, 'y'][mask]

        # convert pixel coordinates to mm
        #triang, pixel_sizes = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).map_pixel_sizes_to_belt(view) #### do this outside of this function!! slow
        real_position = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).find_interpolated_pixel_size(xpos.values, ypos.values, pixel_sizes, triang)
        limb_pos_mm = pd.Series(real_position*limb_pos.values, index=limb_pos.index)

        # Find difference in x or y between frames for limb across a window (of size windowsize) on a rolling basis
        dx = limb_pos_mm.rolling(window=windowsize,center=True,min_periods=0).apply(lambda x: x[-1] - x[0])
        window = limb_pos_mm.index.get_level_values(level='FrameIdx').to_series().rolling(window=windowsize,center=True,min_periods=0).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # calculate speed in mm/s
        dt = (1 / fps) * window
        v = dx / dt

        return v

    def getVelocityInfo(self, vel_zeroed, xaxis, f, windowsize, triang, pixel_sizes):
        """
        Function to return either raw or smoothed velocity profiles for a single/list of runs for a single mouse. Can either be returned with relative time or position values). Can also be zeroed to transition speed and/or time/position point.
        :param vel_zeroed: whether to zero the velocity data to the transitioning velocity
        :param xaxis: The value relative to velocity data to return. 'x' for position or 'time' for time point (in frames)
        :param f: range of run values
        :param triang: mapping of side and overhead
        :param pixel_sizes:
        :return: x and y coordinates (first and second array,respectively) for raw and smoothed velocity data and also transition x and y values for raw and smoothed data
        """

        runs_lowess = list()
        runs_raw = list()
        trans_lowess = list()
        trans_raw = list()
        for r in f:
            try:
                # create column with windowsize values, dependent on the available frames
                v_df = self.calculate_v_mouse(r=r, windowsize=windowsize, triang=triang, pixel_sizes=pixel_sizes)
                v, x, y = v_df['v'], v_df['x'], v_df['y']

                # Check if missed any runbacks
                self.check_missed_rb(v, r)


                # find the frame where transition starts and normalise the time/frame values to the transition (transition is 1)
                transition_idx = v.loc(axis=0)['Transition'].index[0]
                lowest_val = v.index.get_level_values(level='FrameIdx')[0]
                centered_transition_f = (v.index.get_level_values(level='FrameIdx') - lowest_val) / (transition_idx - lowest_val)

                # find the velocity at transition and shift/normalise to transition speed = 0
                transition_v = v.xs(transition_idx, level='FrameIdx').values
                centered_transition_v = v.values - transition_v

                if xaxis == 'x':
                    xplot = x
                elif xaxis == 'time':
                    xplot = pd.Series(data=centered_transition_f,index=x.index)

                if vel_zeroed == True:
                    yplot = pd.Series(data=centered_transition_v,index=v.index)
                else:
                    yplot = v

                from statsmodels.nonparametric.smoothers_lowess import lowess
                lowesslist = lowess(yplot, xplot, frac=.3)
                lowessdf = pd.DataFrame(lowesslist, index=yplot.index[yplot.notnull().values], columns=['x', 'lowess'])

                try:
                    runs_lowess.append([lowessdf['x'], lowessdf['lowess']])
                    runs_raw.append([xplot,yplot])
                    trans_lowess.append([xplot.xs(transition_idx, level=1).values[0],lowessdf['lowess'].xs(transition_idx, level=1).values[0]])
                    trans_raw.append([xplot.xs(transition_idx, level=1).values[0], yplot.xs(transition_idx, level=1).values[0]])
                    del lowessdf, xplot, yplot
                except: ################################ this doesnt work as would just fill from previous run....
                    runs_lowess.append([np.nan,np.nan])
                    runs_raw.append([np.nan,np.nan])
                    trans_lowess.append([np.nan,np.nan])
                    trans_raw.append([np.nan, np.nan])

            except:
                runs_lowess.append([np.nan, np.nan])
                runs_raw.append([np.nan, np.nan])
                trans_lowess.append([np.nan, np.nan])
                trans_raw.append([np.nan, np.nan])
                print('Cant plot run %s, mouse %s' % (r, self.mouseID))

        if len(f) > 1:
            vel = {
                'runs_lowess': np.array(runs_lowess,dtype=object),
                'runs_raw': np.array(runs_raw,dtype=object),
                'trans_lowess': np.array(trans_lowess,dtype=float),
                'trans_raw': np.array(trans_raw, dtype=float)
            }
        else:
            vel = {
                'runs_lowess': runs_lowess,  # np.array(runs_lowess,dtype=object),
                'runs_raw': runs_raw,  # np.array(runs_raw,dtype=object),
                'trans_lowess': trans_lowess,
                'trans_raw': trans_raw
            }
        return vel

    def check_missed_rb(self, v, r):
        if sum(v.loc(axis=0)['RunStart'] < -30) > 30:
            # get indexes of where RunStart should be RunBack
            negmask = v.loc(axis=0)['RunStart'] < -30
            lastnegidx = v.loc(axis=0)['RunStart'][negmask].index[-1]
            firstidx = v.loc(axis=0)['RunStart'].index[0]

            # Change and reassign index
            RunStage = np.array(self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='RunStage'))
            FrameIdx = np.array(self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='FrameIdx'))
            Run = np.array(self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='Run'))

            data_subset = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, 'RunStart', range(firstidx, lastnegidx)]
            Mask = []
            for i in self.data[self.con][self.mouseID]['Side'].index:
                if i in data_subset.index:
                    Mask.append(True)
                else:
                    Mask.append(False)
            RunStage[Mask] = 'RunBack'

            for vw in ['Side', 'Front', 'Overhead']:
                self.data[self.con][self.mouseID][vw].loc(axis=1)['RunStage'] = RunStage
                self.data[self.con][self.mouseID][vw].loc(axis=1)['FrameIdx'] = FrameIdx
                self.data[self.con][self.mouseID][vw].loc(axis=1)['Run'] = Run

                self.data[self.con][self.mouseID][vw].set_index(['Run', 'RunStage', 'FrameIdx'], append=False, inplace=True)

            # update v array to not include the runback
            startidx = v.index.get_level_values(level='FrameIdx')[0]
            v.drop(index=range(startidx, lastnegidx), level='FrameIdx', inplace=True)

            print('Missed runback for %s. Real run starts after frame %s (check this!!)' % (self.mouseID, lastnegidx))

    def calculate_v_mouse(self, r, windowsize, triang, pixel_sizes, limb='Tail1'):
        tailmask_side = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limb, 'likelihood'].values > pcutoff
        tailmask_overhead = self.data[self.con][self.mouseID]['Overhead'].loc(axis=0)[r].loc(axis=1)[limb, 'likelihood'].values > pcutoff
        tailmask = tailmask_side & tailmask_overhead

        xpos = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)[limb, 'x'][tailmask]
        ypos = self.data[self.con][self.mouseID]['Overhead'].loc(axis=0)[r].loc(axis=1)[limb, 'y'][tailmask]

        # convert pixel coordinates to mm
        real_position = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID).find_interpolated_pixel_size(xpos.values, ypos.values, pixel_sizes, triang)
        limb_pos_x_mm = pd.Series(real_position*xpos.values, index=xpos.index)
        limb_pos_y_mm = pd.Series(real_position*ypos.values, index=ypos.index)

        # Find difference in x or y between frames for limb across a window (of size windowsize) on a rolling basis
        dx = limb_pos_x_mm.rolling(window=windowsize,center=True,min_periods=0).apply(lambda x: x[-1] - x[0])
        window = limb_pos_x_mm.index.get_level_values(level='FrameIdx').to_series().rolling(window=windowsize,center=True,min_periods=0).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # calculate speed in mm/s
        dt = (1 / fps) * window
        v = dx / dt

        desired_stages = ['RunStart', 'Transition', 'RunEnd']
        expstage_pattern = [t for t in desired_stages if
                            t in v.index.get_level_values(level='RunStage').unique()]

        df = pd.DataFrame({'v':v,'x':limb_pos_x_mm,'y':limb_pos_y_mm}).loc(axis=0)[expstage_pattern]

        return df

    ### see old Velocity() for plotting functions