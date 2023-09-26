import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import Helpers.utils as utils
from Helpers.Config_23 import *
from scipy.stats import skew, shapiro, levene
import warnings
import pingouin as pg

class GetAndPrep:
    def __init__(self, conditions):
        self.conditions = conditions

    def GetDFs(self, conditions, reindexed_loco=False):
        '''
        :param conditions: list of experimental conditions want to plot/analyse eg 'APAChar_HighLow', 'APAChar_LowHigh_Day1', 'APAVMT_LowHighac'. NB make sure to include the day if for a condition which has repeats
        :return: dictionary holding all dataframes held under the requested conditions
        '''
        if reindexed_loco:
            file_suffix = 'Runs__IdxCorr'
        else:
            file_suffix = 'Runs'
        print('Conditions to be loaded:\n%s' % conditions)
        data = dict.fromkeys(conditions)
        for conidx, con in enumerate(conditions):
            if 'Day' not in con:
                files = utils.Utils().GetlistofH5files(directory=r"%s\%s" % (paths['filtereddata_folder'], con), filtered=True,
                                                       suffix=file_suffix)
            else:
                splitcon = con.split('_')
                conname = "_".join(splitcon[0:2])
                dayname = splitcon[-1]
                w = splitcon[-2]
                if 'Repeats' in con:
                    files = utils.Utils().GetlistofH5files(
                        directory=r"%s\%s\Repeats\%s\%s" % (paths['filtereddata_folder'], conname, w, dayname), filtered=True,
                        suffix=file_suffix)
                elif 'Extended' in con:
                    files = utils.Utils().GetlistofH5files(
                        directory=r"%s\%s\Extended\%s" % (paths['filtereddata_folder'], conname, dayname), filtered=True,
                        suffix=file_suffix)
                else:
                    files = utils.Utils().GetlistofH5files(
                        directory=r"%s\%s\%s" % (paths['filtereddata_folder'], conname, dayname), filtered=True,
                        suffix=file_suffix)
            mouseIDALL = list()
            dateALL = list()

            # sort lists of filenames so that in same order for reading
            files['Side'].sort()
            files['Front'].sort()
            files['Overhead'].sort()

            for f in range(0, len(files['Side'])):
                mouseID = os.path.basename(files['Side'][f]).split('_')[3]
                mouseIDALL.append(mouseID)
                date = os.path.basename(files['Side'][f]).split('_')[1]
                dateALL.append(date)

            # data['%s' % con] = dict.fromkeys(dateALL)
            data['%s' % con] = dict.fromkeys(mouseIDALL)
            for n, name in enumerate(mouseIDALL):
                # data['%s' % con]['%s' %name] = dict.fromkeys(['Side', 'Front', 'Overhead'])
                data['%s' % con]['%s' % name] = {
                    'Date': dateALL[n],
                    'Side': pd.read_hdf(files['Side'][n]),
                    'Front': pd.read_hdf(files['Front'][n]),
                    'Overhead': pd.read_hdf(files['Overhead'][n])
                }
        return data


    def picking_left_or_right(self,limb,comparison):
        '''
        Function to return limb side given the side of the limb of interest and whether interested in contra- or ipsi-lateral limb
        :param limb: name of limb, eg 'L' or 'ForepawToeL'. Important thing is the format ends with the limb side L or R
        :param comparison: contr or ipsi
        :return: 'l' or 'r'
        '''
        options = {
            'L': {
                'contr': 'R',
                'ipsi': 'L'
            },
            'R': {
                'contr': 'L',
                'ipsi': 'R'
            }
        }
        limb_lr = limb[-1]

        return options[limb_lr][comparison]


    def find_phase_step_based(self,df,r,phase,type_swst,type_ci,stepping_limb,prepruns):
        '''
        Function to find either the last step phase before transition, the first step phase after transition or the preceeding step phases for st/sw, contralateral or ipsilateral limbs and depending on the stepping limb
        :param df: dataframe holding all runs for one mouse
        :param r: run to be analysed
        :param phase: chunk of run to be analysed, either pre, apa or post
        :param type_swst: swing or stance
        :param type_ci: contra- or ipsi-lateral
        :param stepping_limb: limb which first transitions across belt
        :return: array of frame numbers for relevant step cycle(/s)
        '''
        if r >= prepruns:
            lr = self.picking_left_or_right(stepping_limb,type_ci) # find which limb focusing on, based on combo of the stepping limb and if we want the ipsi or contra limb
            if phase == 'apa' or 'pre':
                df_vals = df.loc(axis=0)[r,'RunStart'].index[df.loc(axis=0)[r,'RunStart'].loc(axis=1)['ForepawToe%s' %lr,'StepCycleFill'] == locostuff['swst_vals'][type_swst]]
                block = utils.Utils().find_blocks(df_vals,2,2)[-1] if phase == 'apa' else utils.Utils().find_blocks(df_vals,2,2)[:-1]
            elif phase == 'post':
                df_vals = df.loc(axis=0)[r,'Transition'].index.get_level_values(level='FrameIdx')[df.loc(axis=0)[r, 'Transition'].loc(axis=1)['ForepawToe%s' % lr, 'StepCycleFill'] == locostuff['swst_vals'][type_swst]]
                block = utils.Utils().find_blocks(df_vals, 2,2)[0]
            window = np.arange(block[0], block[1] + 1)

            return window


     def find_time_block(self, df, r, phase, period=settings['analysis_chunks']['literature_apa_length'],return_position=False):
        numframes = round(period / ((1 / fps) * 1000))

        if phase == 'apa' or 'pre':
            df_idxs = df.loc(axis=0)[r,'RunStart'].index.get_level_values(level='FrameIdx')
            window = df_idxs[-numframes:] if phase == 'apa' else df_idxs[:-25]
        elif phase == 'post':
            df_idxs = df.loc(axis=0)[r,'Transition'].index.get_level_values(level='FrameIdx')
            window = df_idxs[:numframes]

        # block = df.loc(axis=0)[r].loc(axis=1)['Quadrant'][df.loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3'][
        #         -settings['analysis_chunks']['APA_lengthruns']:].index.get_level_values(level='FrameIdx')

        #### return which qs or how far mouse travelled in these frames to check if frames comparable
        if not return_position:
            return window

    def DefineAnalysisChunks(self,phase,type_block,type_swst,type_ci,r,data,con,mouseID,view,prepruns): #### THIS IS **PER** RUN ####
        '''

        :param type: ['time', 'last_st_contr', 'last_st_ipsi', 'last_sw_contr', 'last_sw_ipsi', speed]
        :return:
        '''
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        stepping_limb = np.array(['ForepawToeL', 'ForepawToeR'])[(data[con][mouseID][view].loc(axis=0)[r,'Transition'].loc(axis=1)[['ForepawToeL', 'ForepawToeR'],'StepCycleFill'].iloc[0] == 0).values]
        if len(stepping_limb) > 1:
            raise ValueError('Two stepping limbs detected, this must be a sliding run')
        else:
            stepping_limb = stepping_limb[0]

        if type_block == 'step':
            window = self.find_phase_step_based(df=data[con][mouseID][view],r=r,phase=phase,type_swst=type_swst,type_ci=type_ci,stepping_limb=stepping_limb,prepruns=prepruns)
        elif type_block == 'time':
            window = self.find_time_block()

        return window