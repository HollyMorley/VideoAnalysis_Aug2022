import Helpers.utils as utils
from Helpers.Config_23 import *
from Helpers import GetRuns
import Velocity_v2 as Velocity
from Helpers import Structural_calculations
import numpy as np
import pandas as pd
import scipy
import math

import warnings

class PrepByRunphase:
    def __init__(self, phase, con, mouseID, view, r):
        self.phase = phase  # chunk of run to be analysed, either pre, apa or post
        self.con = con
        self.mouseID = mouseID
        self.view = view
        self.r = r

    ### moved to utils
    #@staticmethod
    # def picking_left_or_right(limb, comparison):
    #     """
    #     Function to return limb side given the side of the limb of interest and whether interested in contra- or ipsi-
    #     lateral limb
    #     :param limb: name of limb, eg 'L' or 'ForepawToeL'. Important thing is the format ends with the limb side L or R
    #     :param comparison: contr or ipsi
    #     :return: 'l' or 'r'
    #     """
    #     options = {
    #         'L': {
    #             'contr': 'R',
    #             'ipsi': 'L'
    #         },
    #         'R': {
    #             'contr': 'L',
    #             'ipsi': 'R'
    #         }
    #     }
    #     limb_lr = limb[-1]
    #
    #     return options[limb_lr][comparison]

    @staticmethod
    def get_distance_travelled_during_frames(df_slice, window):
        pxs_travelled = df_slice.loc(axis=0)[window].loc(axis=1)['Tail1', 'x'].iloc[-1] - \
                        df_slice.loc(axis=0)[window].loc(axis=1)['Tail1', 'x'].iloc[0]
        pxtocm = GetRuns.GetRuns().findMarkers(df_slice,full_df=False)['pxtocm']
        cm_travelled = pxtocm * pxs_travelled

        return cm_travelled

    def run_phase_return(self, df_slice, window, return_distance, return_position):
        cm_travelled = self.get_distance_travelled_during_frames(df_slice, window) if return_distance else None

        if return_position:
            p0 = df_slice.loc(axis=0)[window].loc(axis=1)['Tail1', 'x'].iloc[0]
            p1 = df_slice.loc(axis=0)[window].loc(axis=1)['Tail1', 'x'].iloc[-1]
        else:
            p0 = None
            p1 = None

        return cm_travelled, p0, p1

    def find_phase_step_based(self, df, type_swst, type_ci, stepping_limb, return_distance=False,
                              return_position=False):
        """
        Function to find either the last step phase before transition, the first step phase after transition or the
        preceeding step phases for st/sw, contralateral or ipsilateral limbs and depending on the stepping limb
        :param df: dataframe holding all runs for one mouse
        :param type_swst: swing or stance
        :param type_ci: contra- or ipsi-lateral
        :param stepping_limb: limb which first transitions across belt
        :param return_distance: True/False whether return values for the distance covered in cm during this period
        :param return_position: True/False whether return values for the start and ending position of this period
        :return: array of frame numbers for relevant step cycle(/s) (and distance + position)
        """
        # find which limb focusing on, based on combo of the stepping limb and if we want the ipsi or contra limb
        lr = self.picking_left_or_right(stepping_limb,type_ci)
        df_slice = None
        block = None
        if self.phase == 'apa' or 'pre':
            df_slice = df.loc(axis=0)[self.r, 'RunStart']
            df_vals = df_slice.index[df_slice.loc(axis=1)['ForepawToe%s' %lr, 'StepCycleFill'] \
                                     == locostuff['swst_vals'][type_swst]]
            block = utils.Utils().find_blocks(df_vals, 2, 2)[-1] if self.phase == 'apa' \
                else utils.Utils().find_blocks(df_vals, 2, 2)[:-1]
        elif self.phase == 'post':
            df_slice = df.loc(axis=0)[self.r, 'Transition']
            df_vals = df_slice.index.get_level_values(level='FrameIdx')[df_slice.loc(axis=1) \
                                                                        ['ForepawToe%s' % lr, 'StepCycleFill'] == locostuff['swst_vals'][type_swst]]
            block = utils.Utils().find_blocks(df_vals, 2, 2)[0]
        window = np.arange(block[0], block[1] + 1)

        cm_travelled, p0, p1 = self.run_phase_return(df_slice, window, return_distance, return_position)

        results = {
            'window': window,
            'distance': cm_travelled,
            'position': [p0, p1]
        }
        return results

    def find_phase_time_based(self, df, period=settings['analysis_chunks']['literature_apa_length'],
                              return_distance=False, return_position=False):
        """
        function to find the frames for the literature defined period before stepping
        :param df: dataframe holding all runs for one mouse
        :param period: time period in *ms*
        :param return_distance: True/False whether return values for the distance covered in cm during this period
        :param return_position: True/False whether return values for the start and ending position of this period
        :return: array of frame numbers for specified time period (and distance + position)
        """
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        numframes = round(period / ((1 / fps) * 1000))
        df_slice = None
        window = None

        if self.phase == 'apa' or 'pre':
            df_slice = df.loc(axis=0)[self.r, 'RunStart']
            df_idxs = df_slice.index.get_level_values(level='FrameIdx')
            window = df_idxs[-numframes:] if self.phase == 'apa' else df_idxs[:-25]
        elif self.phase == 'post':
            df_slice = df.loc(axis=0)[self.r,'Transition']
            df_idxs = df_slice.index.get_level_values(level='FrameIdx')
            window = df_idxs[:numframes]

        cm_travelled, p0, p1 = self.run_phase_return(df_slice, window, return_distance, return_position)

        results = {
            'window': window,
            'distance': cm_travelled,
            'position': [p0, p1]
        }
        return results

    def find_phase_speed_based(self, data, triang, pixel_sizes, n=30, excl_speedup=True, return_distance=False, return_position=False):
        """
        function to find the frames for each run phase based on the period where the mouse slows down
        :param data:
        :param n:
        :param excl_speedup: if True, exclude any period where mouse speeds up prior to transitioning
        :param return_distance:
        :param return_position:
        :return:
        """
        windowsize = math.ceil((fps / n) / 2.) * 2
        #triang_o, pixel_sizes_o = Structural_calculations.GetRealDistances(self.data, self.con,self.mouseID).map_pixel_sizes_to_belt('Side','Overhead')
        vel = Velocity.Velocity(self.data, self.con, self.mouseID).getVelocityInfo(
                            vel_zeroed=True,
                            xaxis='time',
                            f=[self.r],
                            windowsize=windowsize,
                            triang=triang,
                            pixel_sizes=pixel_sizes)

        window = None
        cm_travelled = None
        p0 = None
        p1 = None

        if self.phase == 'apa' or 'pre':
            peaks, info = scipy.signal.find_peaks(vel['runs_lowess'][0][1]['RunStart'], prominence=1)
            negpeaks, neginfo = scipy.signal.find_peaks(-vel['runs_lowess'][0][1]['RunStart'], prominence=1)

            if any(peaks) and info['right_bases'][-1] - peaks[-1] > 20: # check if peak present and if right base of peak is sufficiently far away
                speed_drop_idx = vel['runs_lowess'][0][1]['RunStart'].index[peaks[-1]]
                if np.logical_and(any(negpeaks), negpeaks[-1] > peaks[-1]) and excl_speedup:
                    speed_end_idx = vel['runs_lowess'][0][1]['RunStart'].index[negpeaks[-1]]
                else:
                    speed_end_idx = vel['runs_lowess'][0][1]['RunStart'].index[-1] # final frame before transition
                window = np.arange(speed_drop_idx,speed_end_idx) if self.phase == 'apa' else np.arange\
                    (vel['runs_lowess'][0][1]['RunStart'].index[0],speed_drop_idx)
            else:
                print('No slow down period detected for run: %s, mouse: %s, condition: %s' %(self.r,self.mouseID,self.con))
                window = None

        elif self.phase == 'post':
            print('Calculating the post period with the find_speed_block() function does NOT use speed data. '
                  'This is simply the post transition frames')
            start_idx = vel['runs_lowess'][0][1]['Transition'].index[0]
            end_idx = vel['runs_lowess'][0][1]['Transition'].index[-1]
            window = np.arange(start_idx,end_idx)

        if window:
            cm_travelled, p0, p1 = self.run_phase_return(data[self.con][self.mouseID][self.view], window, return_distance, return_position)

        results = {
            'window': window,
            'distance': cm_travelled,
            'position': [p0,p1]
        }
        return results

    def DefineAnalysisChunks(self, type_block, type_swst, type_ci, data, prepruns): #### THIS IS **PER** RUN ####
        """

        :param type_block:
        :param type_swst:
        :param type_ci:
        :param data:
        :param prepruns:
        :return:
        """
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        stepping_limb = np.array(['ForepawToeL', 'ForepawToeR'])[(data[self.con][self.mouseID][self.view].loc(axis=0)[self.r, 'Transition'].loc(axis=1)[['ForepawToeL', 'ForepawToeR'],'StepCycleFill'].iloc[0] == 0).values]
        if len(stepping_limb) > 1:
            raise ValueError('Two stepping limbs detected, this must be a sliding run')
        else:
            stepping_limb = stepping_limb[0]

        phase_info = None # initiate
        if self.r >= prepruns:
            if type_block == 'step':
                phase_info = self.find_phase_step_based(df=data[self.con][self.mouseID][self.view], type_swst=type_swst, type_ci=type_ci, stepping_limb=stepping_limb)
            elif type_block == 'time':
                phase_info = self.find_phase_time_based(df=data[self.con][self.mouseID][self.view])
            elif type_block == 'speed':
                phase_info = self.find_phase_speed_based(data=data)
        return phase_info # dict


class PrepByStride:
    def __init__(self, data, con, mouseID):
        self.data = data
        self.con = con
        self.mouseID = mouseID





