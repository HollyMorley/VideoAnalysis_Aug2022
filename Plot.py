from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import Helpers.utils as utils
from Helpers.Config import *
from scipy.stats import skew, sem, shapiro, levene
import warnings
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from pathlib import Path
import Helpers.GetRuns as GetRuns
from scipy import stats
from glob import glob
from matplotlib.patches import Rectangle

class Plot():
    def __init__(self):
        super().__init__()

    def GetDFs(self, conditions=[]):
        '''
        :param conditions: list of experimental conditions want to plot/analyse eg 'APAChar_HighLow', 'APAChar_LowHigh_Day1', 'APAVMT_LowHighac'. NB make sure to include the day if for a condition which has repeats
        :return: dictionary holding all dataframes held under the requested conditions
        '''

        print('Conditions to be loaded:\n%s' %conditions)
        data = dict.fromkeys(conditions)
        for conidx, con in enumerate(conditions):
            if 'Day' not in con:
                #files = utils.Utils().GetlistofH5files(directory=r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\%s" %(con), filtered=True)
                files = utils.Utils().GetlistofH5files(directory=r"%s\%s" % (filtereddata_folder, con), filtered=True)
            else:
                splitcon = con.split('_')
                conname = "_".join(splitcon[0:2])
                dayname = splitcon[-1]
                w = splitcon[-2]
                # files = utils.Utils().GetlistofH5files(directory=r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\%s\%s" %(conname, dayname), filtered=True)
                if 'Repeats' in con:
                    files = utils.Utils().GetlistofH5files(directory=r"%s\%s\Repeats\%s\%s" %(filtereddata_folder, conname, w, dayname), filtered=True)
                elif 'Extended' in con:
                    files = utils.Utils().GetlistofH5files(directory=r"%s\%s\Extended\%s" %(filtereddata_folder, conname, dayname), filtered=True)
            mouseIDALL = list()
            dateALL = list()

            # sort lists of filenames so that in same order for reading
            files['Side'].sort()
            files['Front'].sort()
            files['Overhead'].sort()

            for f in range(0,len(files['Side'])):
                mouseID = os.path.basename(files['Side'][f]).split('_')[3]
                mouseIDALL.append(mouseID)
                date = os.path.basename(files['Side'][f]).split('_')[1]
                dateALL.append(date)

            # data['%s' % con] = dict.fromkeys(dateALL)
            data['%s' % con] = dict.fromkeys(mouseIDALL)
            for n, name in enumerate(mouseIDALL):
                #data['%s' % con]['%s' %name] = dict.fromkeys(['Side', 'Front', 'Overhead'])
                data['%s' %con]['%s' %name] = {
                    'Date': dateALL[n],
                    'Side': pd.read_hdf(files['Side'][n]),
                    'Front': pd.read_hdf(files['Front'][n]),
                    'Overhead': pd.read_hdf(files['Overhead'][n])
                }
        return data

    def getMeasures(self, conditions, measure_type, view, timelocked, n_apa1):
        '''
        :param conditions: list of experimental conditions want to plot/analyse eg 'APAChar_HighLow', 'APAChar_LowHigh_Day1', 'APAVMT_LowHighac'. NB make sure to include the day if for a condition which has repeats
        :param measure: which measure you plan to calculate e.g. body length
        :return: dictionary holding specified measure at each run stage [pre, apa, post], both for every run and averaged across each run stage eg baseline.
        '''
        data = self.GetDFs(conditions)
        data_measure = dict.fromkeys(conditions)

        for cidx, con in enumerate(data.keys()):
            # set where to read real runs from (ie remove prep runs)
            if np.logical_and('Char' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                prepruns = preruns_CharLow
            elif 'Char' in con:  # for all other apa characterise conditions
                prepruns = preruns_CharMidHigh
            elif np.logical_and('VMT' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                if np.logical_and('LowHighpd' in con, np.logical_or.reduce(
                        ('1034980' in mouseID, '1034982' in mouseID, '1034983' in mouseID))):
                    prepruns = preruns_CharLow
            elif 'PerceptionTest' in con:
                prepruns = preruns_CharLow ##### ????????????????????????????????
                # finish this + for other conditions
            else:
                raise ValueError('STOP: I havent categorised this file/condition yet in terms of its extra runs')

            data_measure[con] = dict.fromkeys(data[con].keys())
            for midx, mouseID in enumerate(data[con].keys()):
                ########################## Here use locomotion code to idenitfy more accurate timestamps for beginning and transition of run #############################
                # first get the frame numbers for the x frames before and after transition point for every frame and experimental phase
                measureAllRuns_pre = list() # early in run - q1 and 2
                measureAllRuns_apa = list() # just before transition
                measureAllRuns_post = list() # just after transition
                for r in data[con][mouseID]['Side'].index.get_level_values(level='Run').unique():
                    if r >= prepruns:
                        if timelocked == True:
                            preidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q1', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')].index.get_level_values(level='FrameIdx')
                            apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3'][-APA_lengthruns:].index.get_level_values(level='FrameIdx')
                            #apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')][-APA_lengthruns:].index.get_level_values(level='FrameIdx')
                            #apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')].index.get_level_values(level='FrameIdx')
                            postidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q4'][:after_lengthruns].index.get_level_values(level='FrameIdx')
                        else:
                            preidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q1', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')].index.get_level_values(level='FrameIdx')
                            apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3'].index.get_level_values(level='FrameIdx')
                            postidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q4'].index.get_level_values(level='FrameIdx')

                        premask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(preidx)
                        apamask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(apaidx)
                        postmask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(postidx)
                        runmask = np.logical_or.reduce((data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(preidx), #### WARNING: THIS IS MISSING A SECTION BETWEEN PRE AND APA
                                                        data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(apaidx),
                                                        data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(postidx)))

                        # do calculations
                        if measure_type == 'Body Length':
                            measure = self.CalculateBodyLength(data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)
                        if measure_type == 'Back Skew':
                            measure = self.CalculateBack(calculation='skew',  data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)
                        if measure_type == 'Back Height':
                            measure = self.CalculateBack(calculation='height', data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)
                        if measure_type == 'Head Height':
                            measure = self.CalculateHeadHeight(data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)

                        measureAllRuns_pre.append((measure['mean']['pre'], measure['std']['pre'], r))
                        measureAllRuns_apa.append((measure['mean']['apa'], measure['std']['apa'], r))
                        measureAllRuns_post.append((measure['mean']['post'], measure['std']['post'], r))

                measureAllRuns_pre = np.vstack(measureAllRuns_pre)
                measureAllRuns_apa = np.vstack(measureAllRuns_apa)
                measureAllRuns_post = np.vstack(measureAllRuns_post)

                # convert to df so can have accurate run numbers
                measureAllRuns_pre = pd.DataFrame(data=measureAllRuns_pre[:, [0,1]], index=pd.Index(measureAllRuns_pre[:, 2] - 2), columns=['mean','std'])
                measureAllRuns_apa = pd.DataFrame(data=measureAllRuns_apa[:,[0,1]], index=pd.Index(measureAllRuns_apa[:,2]-2), columns=['mean','std'])
                measureAllRuns_post = pd.DataFrame(data=measureAllRuns_post[:,[0,1]], index=pd.Index(measureAllRuns_post[:,2]-2), columns=['mean','std'])
                # check for missing runs and fill in the gaps
                newidx = range(40)
                measureAllRuns_pre = measureAllRuns_pre.reindex(newidx, fill_value=None)
                measureAllRuns_apa = measureAllRuns_apa.reindex(newidx, fill_value=None)
                measureAllRuns_post = measureAllRuns_post.reindex(newidx, fill_value=None)


                #### calculate averages and std for measure in baseline, APA1, APA2, washout ###
                # Baseline runs
                measureBaseline_pre = [np.nanmean(measureAllRuns_pre.loc[:9,'mean']), np.nanstd(measureAllRuns_pre.loc[:9,'mean'])]
                measureBaseline_apa = [np.nanmean(measureAllRuns_apa.loc[:9,'mean']), np.nanstd(measureAllRuns_apa.loc[:9,'mean'])]
                measureBaseline_post = [np.nanmean(measureAllRuns_post.loc[:9,'mean']), np.nanstd(measureAllRuns_post.loc[:9,'mean'])]

                # Washout runs
                measureWashout_pre = [np.nanmean(measureAllRuns_pre.loc[-9:,'mean']), np.nanstd(measureAllRuns_pre.loc[-9:,'mean'])]
                measureWashout_apa = [np.nanmean(measureAllRuns_apa.loc[-9:,'mean']), np.nanstd(measureAllRuns_apa.loc[-9:,'mean'])]
                measureWashout_post = [np.nanmean(measureAllRuns_post.loc[-9:,'mean']), np.nanstd(measureAllRuns_post.loc[-9:,'mean'])]

                if type(n_apa1) == tuple:
                    n_apa1 = n_apa1[0]
                elif type(n_apa1) == int:
                    n_apa1 = n_apa1

                # APA runs (+ APA runs split into two halves)
                if 'Char' in con or 'Perception' in con:
                    numruns = APACharRuns
                elif 'VMT' in con:
                    numruns = APAVmtRuns
                measureAPA_pre = [np.nanmean(measureAllRuns_pre.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_pre.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean'])]
                measureAPA_apa = [np.nanmean(measureAllRuns_apa.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_apa.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean'])]
                measureAPA_post = [np.nanmean(measureAllRuns_post.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_post.loc[numruns[0]:numruns[0]+numruns[1]-1,'mean'])]

                measureAPA_1_pre = [np.nanmean(measureAllRuns_pre.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean']), np.nanstd(measureAllRuns_pre.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean'])]
                measureAPA_1_apa = [np.nanmean(measureAllRuns_apa.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean']), np.nanstd(measureAllRuns_apa.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean'])]
                measureAPA_1_post = [np.nanmean(measureAllRuns_post.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean']), np.nanstd(measureAllRuns_post.loc[numruns[0]:numruns[0]+int(n_apa1)-1,'mean'])]

                measureAPA_2_pre = [np.nanmean(measureAllRuns_pre.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_pre.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean'])]
                measureAPA_2_apa = [np.nanmean(measureAllRuns_apa.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_apa.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean'])]
                measureAPA_2_post = [np.nanmean(measureAllRuns_post.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean']), np.nanstd(measureAllRuns_post.loc[numruns[0]+int(n_apa1):numruns[0]+numruns[1]-1,'mean'])]


                data_measure[con][mouseID] = {
                    'pre': measureAllRuns_pre,
                    'apa': measureAllRuns_apa,
                    'post': measureAllRuns_post,
                    'pre_mean_std': {
                        'baseline': measureBaseline_pre,
                        'apa': measureAPA_pre,
                        'apa_1': measureAPA_1_pre,
                        'apa_2': measureAPA_2_pre,
                        'washout': measureWashout_pre
                    },
                    'apa_mean_std': {
                        'baseline': measureBaseline_apa,
                        'apa': measureAPA_apa,
                        'apa_1': measureAPA_1_apa,
                        'apa_2': measureAPA_2_apa,
                        'washout': measureWashout_apa
                    },
                    'post_mean_std': {
                        'baseline': measureBaseline_post,
                        'apa': measureAPA_post,
                        'apa_1': measureAPA_1_post,
                        'apa_2': measureAPA_2_post,
                        'washout': measureWashout_post
                    }
                }

        return data_measure

    def getMeasures_wholerun(self, conditions, measure_type):
        data = self.GetDFs(conditions)
        data_measure = dict.fromkeys(conditions)

        for cidx, con in enumerate(data.keys()):
            data_measure[con] = dict.fromkeys(data[con].keys())
            for midx, mouseID in enumerate(data[con].keys()):
                # first get the frame numbers for the x frames before and after transition point for every frame and experimental phase
                measureAllRuns = list()
                for r in data[con][mouseID]['Side'].index.get_level_values(level='Run').unique():

                    ###################################### PICK MEASURE ################################################
                    if measure_type == 'Wait Time':
                        measure = self.CalculateWaitTime(data=data, con=con, mouseID=mouseID, r=r)
                    if measure_type == 'Average Run Speed':
                        measure = self.CalculateRunningSpeed_runaverage(data=data, con=con, mouseID=mouseID, r=r)
                    ####################################################################################################

                    measureAllRuns.append(measure)
                measureAllRuns = np.vstack(measureAllRuns)

                if np.logical_and('Char' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                    measureAllRuns = measureAllRuns[preruns_CharLow:]
                    print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' %(preruns_CharLow, mouseID, len(measureAllRuns)))
                elif 'Char' in con: # for all other apa characterise conditions
                    measureAllRuns = measureAllRuns[preruns_CharMidHigh:]
                    print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' %(preruns_CharMidHigh, mouseID, len(measureAllRuns)))
                elif np.logical_and('VMT' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                    if np.logical_and('LowHighpd' in con, np.logical_or.reduce(('1034980' in mouseID, '1034982' in mouseID, '1034983' in mouseID))):
                        measureAllRuns = measureAllRuns[preruns_CharLow:]
                        print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' % (preruns_CharLow, mouseID, len(measureAllRuns)))
                    else:
                        print('No prep runs. Total run number for mouse %s is: %s' %(mouseID, len(measureAllRuns)))
                elif 'PerceptionTest' in con:
                    measureAllRuns = measureAllRuns
                else:
                    raise ValueError('STOP: I havent categorised this file/condition yet in terms of its extra runs')

                #### calculate averages and std for measure in baseline, APA1, APA2, washout ###
                # Baseline runs
                measureBaseline = [np.nanmean(measureAllRuns[:10][:, 0]),
                                       np.nanstd(measureAllRuns[:10][:, 0])]
                # Washout runs
                measureWashout = [np.nanmean(measureAllRuns[-10:][:, 0]),
                                      np.nanstd(measureAllRuns[-10:][:, 0])]

                # APA runs (+ APA runs split into two halves)
                if 'Char' in con or 'Perception' in con:
                    numruns = APACharRuns
                elif 'VMT' in con:
                    numruns = APAVmtRuns
                measureAPA = [np.nanmean(measureAllRuns[numruns[0]:numruns[0] + numruns[1]][:, 0]),
                                  np.nanstd(measureAllRuns[numruns[0]:numruns[0] + numruns[1]][:, 0])]
                measureAPA_1 = [
                    np.nanmean(measureAllRuns[numruns[0]:numruns[0] + int(numruns[1] / 2)][:, 0]),
                    np.nanstd(measureAllRuns[numruns[0]:numruns[0] + int(numruns[1] / 2)][:, 0])]
                measureAPA_2 = [
                    np.nanmean(measureAllRuns[numruns[0] + int(numruns[1] / 2):numruns[0] + numruns[1]][:, 0]),
                    np.nanstd(measureAllRuns[numruns[0] + int(numruns[1] / 2):numruns[0] + numruns[1]][:, 0])]

                data_measure[con][mouseID] = {
                    'all': measureAllRuns,
                    'all_mean_std': {
                        'baseline': measureBaseline,
                        'apa': measureAPA,
                        'apa_1': measureAPA_1,
                        'apa_2': measureAPA_2,
                        'washout': measureWashout
                    }
                }
        return data_measure


    def CalculateBack(self, calculation, data, con, mouseID, r, premask, apamask, postmask, view):
        backmaskall_pre = np.logical_and(
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back7', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back8', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back9', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back10', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back11', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12', 'likelihood'][premask] > pcutoff
        )),
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back2', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back3', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back4', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back5', 'likelihood'][premask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back6', 'likelihood'][premask] > pcutoff
        ))
        )
        backmaskall_apa = np.logical_and(
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back7', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back8', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back9', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back10', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back11', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12', 'likelihood'][apamask] > pcutoff
            )),
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back2', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back3', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back4', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back5', 'likelihood'][apamask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back6', 'likelihood'][apamask] > pcutoff
            ))
        )
        backmaskall_post = np.logical_and(
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back7', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back8', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back9', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back10', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back11', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12', 'likelihood'][postmask] > pcutoff
            )),
            np.logical_and.reduce((
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back2', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back3', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back4', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back5', 'likelihood'][postmask] > pcutoff,
                data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back6', 'likelihood'][postmask] > pcutoff
            ))
        )
        startplatRmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff
        transitionRmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff

        startplatRmean = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['StartPlatR', 'y'][startplatRmask])
        TransitionRmean = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'y'][transitionRmask])

        beltlevel = np.mean([startplatRmean, TransitionRmean])
        backy = dict.fromkeys(['pre','apa','post'])
        masks = {
            'pre': premask,
            'apa': apamask,
            'post': postmask
        }
        backmasks = {
            'pre': backmaskall_pre,
            'apa': backmaskall_apa,
            'post': backmaskall_post
        }
        for idx, i in enumerate(backy.keys()):
            backy[i] = pd.DataFrame([
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back2', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back3', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back4', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back5', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back6', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back7', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back8', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back9', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back10', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back11', 'y'][masks[i]][backmasks[i]],
                beltlevel - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12', 'y'][masks[i]][backmasks[i]]]
                )

        if calculation == 'skew':
            skew_pre_mean = np.mean(skew(backy['pre'],axis=0))
            skew_apa_mean = np.mean(skew(backy['apa'],axis=0))
            skew_post_mean = np.mean(skew(backy['post'],axis=0))

            skew_pre_std = np.std(skew(backy['pre'], axis=0))
            skew_apa_std = np.std(skew(backy['apa'], axis=0))
            skew_post_std = np.std(skew(backy['post'], axis=0))

            data = {
                'mean': {
                    'pre': skew_pre_mean,
                    'apa': skew_apa_mean,
                    'post': skew_post_mean
                },
                'std': {
                    'pre': skew_pre_std,
                    'apa': skew_apa_std,
                    'post': skew_post_std
                }
            }


        if calculation == 'height':
            max_pre_mean = np.mean(backy['pre'].max(axis=0))
            max_apa_mean = np.mean(backy['apa'].max(axis=0))
            max_post_mean = np.mean(backy['post'].max(axis=0))

            max_pre_std = np.std(backy['pre'].max(axis=0))
            max_apa_std = np.std(backy['apa'].max(axis=0))
            max_post_std = np.std(backy['post'].max(axis=0))

            data = {
                'mean': {
                    'pre': max_pre_mean,
                    'apa': max_apa_mean,
                    'post': max_post_mean
                },
                'std': {
                    'pre': max_pre_std,
                    'apa': max_apa_std,
                    'post': max_post_std
                }
            }
        return data



    def CalculateBodyLength(self, data, con, mouseID, r, premask, apamask, postmask, view):
        '''

        :param premask:
        :param apamask:
        :param postmask:
        :param view:
        :return:
        '''
        back1mask_pre = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][premask] > pcutoff
        back12mask_pre = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12', 'likelihood'][premask] > pcutoff
        back1mask_apa = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][apamask] > pcutoff
        back12mask_apa = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','likelihood'][apamask] > pcutoff
        back1mask_post = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1', 'likelihood'][postmask] > pcutoff
        back12mask_post = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','likelihood'][postmask] > pcutoff

        bodylengthmean_pre = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][premask][back1mask_pre] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][premask][back12mask_pre])
        bodylengthmean_apa = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][apamask][back1mask_apa] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][apamask][back12mask_apa])
        bodylengthmean_post = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][postmask][back1mask_post] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][postmask][back12mask_post])

        bodylengthstd_pre = np.std(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][premask][back1mask_pre] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][premask][back12mask_pre])
        bodylengthstd_apa = np.std(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][apamask][back1mask_apa] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][apamask][back12mask_apa])
        bodylengthstd_post = np.std(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back1','x'][postmask][back1mask_post] - data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Back12','x'][postmask][back12mask_post])

        bodylength = {
            'mean': {
                'pre': bodylengthmean_pre,
                'apa': bodylengthmean_apa,
                'post': bodylengthmean_post
            },
            'std': {
                'pre': bodylengthstd_pre,
                'apa': bodylengthstd_apa,
                'post': bodylengthstd_post
            }
        }
        return bodylength


    def CalculateHeadHeight(self, data, con, mouseID, r, premask, apamask, postmask, view):
        # nosemask_pre = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][premask] > pcutoff
        # nosemask_apa = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][apamask] > pcutoff
        # nosemask_post = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][postmask] > pcutoff

        nose_trans_mask_pre = np.logical_and(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][premask] > pcutoff,
                                             np.logical_or(
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'][premask] > pcutoff,
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionL', 'likelihood'][premask] > pcutoff
                                             ))
        nose_trans_mask_apa = np.logical_and(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][apamask] > pcutoff,
                                             np.logical_or(
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'][apamask] > pcutoff,
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionL', 'likelihood'][apamask] > pcutoff
                                             ))
        nose_trans_mask_post = np.logical_and(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'likelihood'][postmask] > pcutoff,
                                             np.logical_or(
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'][postmask] > pcutoff,
                                                 data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionL', 'likelihood'][postmask] > pcutoff
                                             ))

        nose_pre = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'y'][premask][nose_trans_mask_pre]
        nose_apa = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'y'][apamask][nose_trans_mask_apa]
        nose_post = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Nose', 'y'][postmask][nose_trans_mask_post]



        # transitionRmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
        # transitionLmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionL', 'likelihood'] > pcutoff

        transitionR_height_mean = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'y'][transitionRmask])
        transitionL_height_mean = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionL', 'y'][transitionLmask])

        platform_height_av = np.mean([transitionR_height_mean, transitionL_height_mean])

        headheight_mean_pre = nose_pre - platform_height_av
        headheight_mean_apa = nose_apa - platform_height_av
        headheight_mean_post = nose_post - platform_height_av

        headheight_std_pre = nose_pre - platform_height_av
        headheight_std_apa = nose_apa - platform_height_av
        headheight_std_post = nose_post - platform_height_av

        headheight = {
            'mean': {
                'pre': bodylengthmean_pre,
                'apa': bodylengthmean_apa,
                'post': bodylengthmean_post
            },
            'std': {
                'pre': bodylengthstd_pre,
                'apa': bodylengthstd_apa,
                'post': bodylengthstd_post
            }
        }


    def CalculateWaitTime(self, data, con, mouseID, r, view='Side'):
        if 'TrialStart' in data[con][mouseID]['Side'].loc(axis=0)[r].index:
            trialstart_frame = data[con][mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].index[0]
            runstart_frame = data[con][mouseID]['Side'].loc(axis=0)[r, 'RunStart'].index[0]
            waittime_s = (runstart_frame - trialstart_frame)/fps
        else:
            waittime_s = 0

        return waittime_s

    def CalculateRunningSpeed_runaverage(self, data, con, mouseID, r):
        # for each run get the run time (transition frame - start frame) and run distance (x values corresponding to these frames)

        date = int(data[con][mouseID]['Date'])

        # speed = distance/time
        # Calculate the average speed traveled for this run
        if 'TrialStart' in data[con][mouseID]['Side'].loc(axis=0)[r].index:
            runtime = data[con][mouseID]['Side'].loc(axis=0)[r,'Transition'].index[0] - data[con][mouseID]['Side'].loc(axis=0)[r,'RunStart'].index[0]
            distance = data[con][mouseID]['Side'].loc(axis=0)[r,'Transition'].loc(axis=0)['Nose','x'][0] - data[con][mouseID]['Side'].loc(axis=0)[r,'RunStart'].loc(axis=0)['Nose','x'][0]
            #distancecm =
            mousespeedav = distance/runtime
        else:
            raise ValueError('No runstart for run %s' %r)

        # subtract belt speed from totalspeed
        ##### TEMP ########
        if 'APAChar' in con:
            numruns = APACharRuns
        if 'PerceptionTest' in con:
            numruns = APAPerRuns
        if 'VMT' in con:
            numruns = APAVmtRuns

        logdf = pd.read_excel(r"C:\Users\Holly Morley\Dropbox (UCL)\Murray Lab\Holly\Aug22_APACharacteriseLong_log_recovered.xlsx", "Sheet1")

        if r < numruns[0]: # baseline
            b2speed = np.unique(logdf.loc(axis=0)[date, mouseID, 'Baseline'].loc(axis=1)['Belt2speed (cm/s)'].values)[0]
        elif np.logical_and(r >= numruns[0], r < numruns[0] + numruns[1]): # APA
            b2speed = np.unique(logdf.loc(axis=0)[date, mouseID, 'APA_characterise'].loc(axis=1)['Belt2speed (cm/s)'].values)[0]
        elif np.logical_and(r >= numruns[0] + numruns[1], r < numruns[0] + numruns[1] + numruns[2]): # washout
            b2speed = np.unique(logdf.loc(axis=0)[date, mouseID, 'Washout'].loc(axis=1)['Belt2speed (cm/s)'].values)[0]
        elif r >= numruns[0] + numruns[1] + numruns[2]:
            print('There are somehow too many runs here!!')

        realspeed = mousespeedav - b2speed

        return realspeed


    def setupForPlotting(self, conditions, means, measure_type, view, runphase, timelocked, n_apa1, means_all=False):
        '''

        :param conditions:
        :param measure_type:
        :param style:
        :param view:
        :param runphase:
        :param timelocked:
        :return:
        '''

        if measure_type != 'Wait Time':
            data = self.getMeasures(conditions,measure_type,view,timelocked=timelocked, n_apa1=n_apa1)
        else:
            data = self.getMeasures_wholerun(conditions, measure_type)

        if len(conditions) > 1:
            conditions_all = '\t'.join(conditions)
        else:
            conditions_all = conditions

        if means == False:
            ###### need to put in logic
            groupData = dict.fromkeys(conditions)
            for cidx, con in enumerate(data.keys()):
                groupData[con] = dict.fromkeys(runphase)
                for r in runphase:
                    if measure_type != 'Wait Time':
                        if 'VMT' in conditions_all and 'Char' not in conditions_all:
                            groupData[con][r] = pd.DataFrame(index=range(0, sum(APAVmtRuns)), columns=pd.MultiIndex.from_product([data[con].keys(),['mean', 'std']]))
                        else:
                            groupData[con][r] = pd.DataFrame(index=range(0, sum(APACharRuns)), columns=pd.MultiIndex.from_product([data[con].keys(),['mean', 'std']]))
                        for midx, mouseID in enumerate(data[con].keys()):
                            if 'VMT' in conditions_all and 'Char' not in conditions_all:
                                missingruns = sum(APAVmtRuns) - len(data[con][mouseID][r])
                            else:
                                missingruns = sum(APACharRuns) - len(data[con][mouseID][r])
                            if missingruns > 0:
                                for i in range(0, missingruns):
                                    data[con][mouseID][r] = np.concatenate((data[con][mouseID][r], np.array([[np.nan, np.nan]])), axis=0)
                            # groupData[con][r].loc(axis=1)[mouseID] = data[con][mouseID][r]
                            groupData[con][r].loc(axis=0)[0:41].loc(axis=1)[mouseID].update(data[con][mouseID][r])
                    else:
                        if 'VMT' in conditions_all and 'Char' not in conditions_all:
                            groupData[con][r] = pd.DataFrame(index=range(0, sum(APAVmtRuns)), columns=pd.MultiIndex.from_product([data[con].keys(),['mean']]))
                        else:
                            groupData[con][r] = pd.DataFrame(index=range(0, sum(APACharRuns)), columns=pd.MultiIndex.from_product([data[con].keys(),['mean']]))
                        for midx, mouseID in enumerate(data[con].keys()):
                            if 'VMT' in conditions_all and 'Char' not in conditions_all:
                                missingruns = sum(APAVmtRuns) - len(data[con][mouseID][r])
                            else:
                                missingruns = sum(APACharRuns) - len(data[con][mouseID][r])
                            if missingruns > 0:
                                for i in range(0, missingruns):
                                    data[con][mouseID][r] = np.concatenate((data[con][mouseID][r], np.array([[np.nan]])), axis=0)
                            groupData[con][r].loc(axis=1)[mouseID] = data[con][mouseID][r]

        if means == True:
            chunks = ['baseline', 'apa', 'apa_1', 'apa_2', 'washout']
            groupData = dict.fromkeys(conditions)
            for cidx, con in enumerate(data.keys()):
                groupData[con] = dict.fromkeys(runphase)
                for r in runphase:
                    groupData[con][r] = pd.DataFrame(index=chunks, columns=pd.MultiIndex.from_product([data[con].keys(), ['mean', 'std']]))
                    for midx, mouseID in enumerate(data[con].keys()):
                        for c in chunks:
                            groupData[con][r].loc(axis=1)[mouseID, 'mean'].loc(axis=0)[c] = data[con][mouseID]['%s_mean_std' %r][c][0]
                            groupData[con][r].loc(axis=1)[mouseID, 'std'].loc(axis=0)[c] = data[con][mouseID]['%s_mean_std' % r][c][1]

        return groupData

    def PlotByRun(self, conditions, measure_type, style, view, runphase, variance, meanstd='mean', n_apa1=10, legend=True, transparent=False, save=False, userlabels=False, timelocked=True, username=False):
        '''
       :param conditions:
       :param measure_type:
       :param style:
       :param view:
       :param runphase:
       :param variance:
       :param meanstd: always use mean (cv doesnt really work here as an indication of variance across a run, not phase)
       :param n_apa1:
       :param legend:
       :param transparent:
       :param save:
       :param userlabels:
       :param timelocked:
       :param username:
       :return:
       '''
        if len(conditions) > 1:
            conditions_all = '\t'.join(conditions)
        else:
            conditions_all = conditions

        means = False

        groupData = self.setupForPlotting(conditions, means, measure_type, view, runphase, timelocked, n_apa1)

        if len(conditions) > 1 and len(runphase) == 1:
            #colors = utils.Utils().get_cmap(len(conditions), 'tab20c')
            length = len(conditions)
            labels = 'conditions'
            print('Conditions are: %s' %conditions)
        elif len(runphase) > 1 and len(conditions) == 1:
            #colors = utils.Utils().get_cmap(len(runphase), 'tab20c')
            length = len(runphase)
            labels = 'runphase'
        elif len(runphase) == 1 and len(conditions) == 1:
            length = len(conditions)
            labels = 'conditions'
        else:
            raise ValueError('cant plot this many conditions')

        blues = utils.Utils().get_cmap(20, 'Blues')
        colors = utils.Utils().get_cmap(20, 'tab20')
        if 'APAChar_LowHigh_Day3' in conditions_all and 'APAChar_LowMid_Day3' in conditions_all  and len(conditions) ==2:
            sub_colors = [blues(13), colors(4)]
        elif 'APAChar_LowHigh_Day1' in conditions_all and 'APAChar_LowHigh_Day3' in conditions_all and len(conditions) ==2:
            sub_colors = [blues(18), blues(10)]
        elif 'APAChar_LowHigh_Day1' in conditions_all and 'APAChar_LowHigh_Day2' in conditions_all and 'APAChar_LowHigh_Day3' in conditions_all and len(conditions) ==3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all and len(conditions) ==2:
            sub_colors = [colors(6), colors(8)]
        elif 'PerceptionTest' in conditions_all and len(conditions) ==2:
            sub_colors = [colors(10), colors(11)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all and len(
                conditions) == 3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day3' in conditions_all and len(
                conditions) == 3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all \
                and 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and len(conditions) == 4:
            sub_colors = [blues(18), blues(15), blues(12), blues(9)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all \
                and 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day3' in conditions_all and len(conditions) == 6:
            sub_colors = [blues(19), blues(17), blues(15), blues(13), blues(11), blues(9)]

        plt.rcParams.update({
            "figure.facecolor": (1.0, 1.0, 1.0, 1.0),  # red   with alpha = 30%
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
            "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),  # blue  with alpha = 20%
            "font.size": 24,
            'axes.spines.right': False,
            'axes.spines.top':False
        })


        fig = plt.figure(num="%s_%s_%s_%s_%s_timelocked=%s" %(conditions, runphase, measure_type, view, style, timelocked), figsize=(11, 4))
        #ax =plt.subplot(121) # can give this ax more specific name
        ax = fig.add_axes([0.1, 0.1, 0.65, 0.75])
        ax.set_xlim(1, groupData[conditions[0]][runphase[0]].shape[0])
        #ax.set_xlabel('Run')

        linestyles = ['-', '--', ':', '-.']

        if style == 'individual':
            ccount = 0
            for cidx, con in enumerate(conditions):
                for r in runphase:
                    mcount = 0
                    for midx, mouseID in enumerate(groupData[con][runphase[0]].columns.get_level_values(level=0).unique()):
                        ax.plot(groupData[con][r].index + 1, groupData[con][r].loc(axis=1)[mouseID]['mean'], color=colors(mcount+1), linestyle=linestyles[ccount], label='%s, %s, %s' %(mouseID, con, runphase))
                        ax.fill_between(groupData[con][r].index + 1,
                                        groupData[con][r].loc(axis=1)[mouseID]['mean'] - groupData[con][r].loc(axis=1)[mouseID]['std'],
                                        groupData[con][r].loc(axis=1)[mouseID]['mean'] + groupData[con][r].loc(axis=1)[mouseID]['std'],
                                        interpolate=False, alpha=0.1, color=colors(mcount+1))
                        mcount +=1
                ccount +=1
            ax.legend()


        elif style == 'group':
            count = 0
            for cidx, con in enumerate(conditions):
                for r in runphase:
                    if meanstd == 'mean':
                        plotdata = groupData[con][r].xs('mean', axis=1, level=1)
                    elif meanstd == 'cv':
                        plotdata = groupData[con][r].xs('std', axis=1, level=1) / groupData[con][r].xs('mean', axis=1,level=1)
                    if labels == 'conditions':
                        barlabel = con
                        if userlabels == True:
                            barlabel = input('Type in new label for: %s' %con)
                    elif labels == 'runphase':
                        barlabel = r
                        if userlabels == True:
                            barlabel = input('Type in new label for: %s' %r)
                    ax.plot(groupData[con][r].index + 1, np.mean(plotdata, axis=1), label=barlabel, color=sub_colors[count])
                    if variance == 'std':
                        ax.fill_between(groupData[con][r].index + 1,
                                        np.mean(plotdata, axis=1) - np.std(plotdata, axis=1),
                                        np.mean(plotdata, axis=1) + np.std(plotdata, axis=1),
                                        interpolate=False, alpha=0.1, color=sub_colors[count])
                    if variance == 'sem':
                        ax.fill_between(groupData[con][r].index + 1,
                                        np.mean(groupData[con][r].xs('mean', axis=1, level=1), axis=1) - np.std(plotdata, axis=1) / np.sqrt(plotdata.count(axis=1)),
                                        np.mean(groupData[con][r].xs('mean', axis=1, level=1), axis=1) + np.std(plotdata, axis=1) / np.sqrt(plotdata.count(axis=1)),
                                        interpolate=False, alpha=0.1, color=sub_colors[count])

                    count += 1
            # if legend == True:
            #     ax.legend(bbox_to_anchor=(1, 1), loc='upper left')


        if userlabels == True:
            ylabel = input('Enter y axes label:')
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(measure_type)

        if legend == True:
            if userlabels == True:
                legendtitle = input('Enter legend title:')
                #ax.legend(title=legendtitle, bbox_to_anchor=(1, 1), loc='lower right')
                ax.legend(title=legendtitle, bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)
            else:
                ax.legend(bbox_to_anchor=(1, 1), loc='lower right')

        #plt.title(variance)


        # plot patches for experimental phases
        # plt.tick_params(
        #     axis='x',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False)


        if 'VMT' in conditions_all and 'Char' not in conditions_all:
            runnos = APAVmtRuns
        else:
            runnos = APACharRuns
        # patchwidth = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1333
        # ax.add_patch(plt.Rectangle((1, patchwidth), runnos[0], ax.get_ylim()[0]-patchwidth, facecolor='lightblue', clip_on=False, linewidth=0))
        # ax.add_patch(plt.Rectangle((runnos[0], patchwidth), runnos[1], ax.get_ylim()[0]-patchwidth, facecolor='lightskyblue', clip_on=False, linewidth=0))
        # ax.add_patch(plt.Rectangle((runnos[0]+runnos[1], patchwidth), runnos[2], ax.get_ylim()[0]-patchwidth, facecolor='dodgerblue', clip_on=False, linewidth=0))
        # ax.text(runnos[0]*0.45, patchwidth + (ax.get_ylim()[0]-patchwidth)*0.3, ' Baseline\nTrials = %s' %runnos[0], fontsize=10, zorder=5, color='k')
        # ax.text(runnos[0] + runnos[1]*0.45, patchwidth + (ax.get_ylim()[0]-patchwidth)*0.3, '     APA\nTrials = %s' %runnos[1], fontsize=10, zorder=5, color='k')
        # ax.text((runnos[0]+runnos[1]) + runnos[2]*0.45, patchwidth + (ax.get_ylim()[0]-patchwidth)*0.3, 'Washout\nTrials = %s' %runnos[2], fontsize=10, zorder=5, color='k')

        ax.set_xticks([runnos[0]/2, runnos[0] + runnos[1]/2, runnos[0] + runnos[1] + runnos[2]/2])
        ax.set_xticklabels(['Baseline', 'APA', 'Washout'])
        plt.autoscale(False)
        ax.vlines([runnos[0]+0.5, runnos[0]+runnos[1]+0.5], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='black')
        # ax.set_xticks([1.4, 2.4, 3.4])
        # ax.set_xticklabels(['APA\n(Phase 1)', 'APA\n(Phase 2)', 'Washout'])

        # save figure
        if save == True:
            if username == True:
                filename = input('Enter filename (describe the conditions included):')
                plt.savefig(r'%s\ByRun_%s_%s_%s_%s_%s_%s_%s_%s.png' % (
                plotting_destfolder, filename, runphase, measure_type, view, style, timelocked, variance, meanstd),
                            bbox_inches='tight', transparent=transparent, format='png')
            else:
                plt.savefig(r'%s\ByRun_%s_%s_%s_%s_%s_%s_%s_%s.png' %(plotting_destfolder, conditions, runphase, measure_type, view, style, timelocked, variance, meanstd), bbox_inches = 'tight', transparent=transparent, format='png')

            # pxto1cm = GetRuns.GetRunsimpo().findMarkers(data[con][mouseID]['Side'])['pxtocm']
            # pxto1mm = pxtocm*10


    def PlotByPhaseChunk(self, measure_type, view, conditions, runphase, n_apa1=10, meanbyrun='bysubject', meanstd='mean', style='group', variance='sem', legend=True, transparent=False, save=False, timelocked=True, userlabels=False, username=False):
        '''
        Plots measures as averages for APA (first and second half) and washout, all normalised to baseline
        :param meanbyrun:
            'bysubject' - !!!USE THIS!!! For each mouse get mean for the exp phase, then calculate means, var etc for each phase across mice
            'byrun' - average every run across mice, then group runs after found run averages - dont use this for comparing phases because not comparing var based on subject
            'oldbysubject' - get exp phase averages for each mouse and then average those
        :param measure_type: 'Body Length' ...
        :param view: 'side', 'front', 'overhead'
        :param conditions: experimental condition, see under analysis directory
        :param runphase: 'pre', 'apa' or 'post'
        :param n_apa1: how many trials to look at in the first split phase of APA trials. APA2 will be comprised of the remaining trials.
        :param meanstd: plotting differences in mean of measure or the variance in the measure
        :param style: 'group' or 'individual' nb currently only 'group' is finished
        :param save: save plot or not
        :param timelocked: apa and post runphase frames are timlocked to transition. If false, just all the frames the mouse was in that quadrant
        :param userlabels: set my own labels for legend
        :return:
        '''
        if len(conditions) > 1:
            conditions_all = '\t'.join(conditions)
        else:
            conditions_all = conditions


        # chunks = ['baseline', 'apa_1', 'apa_2', 'washout']
        # chunkslabels = ['Baseline', 'APA 1st half', 'APA 2nd half', 'Washout']

        if meanbyrun == 'byrun':
            means = False
        elif 'bysubject' in meanbyrun:
            means = True
        groupData = self.setupForPlotting(conditions, means, measure_type, view, runphase, timelocked, n_apa1)

        plt.rcParams.update({
            "figure.facecolor": (1.0, 0.0, 0.0, 0),  # red   with alpha = 30%
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
            "savefig.facecolor": (0.0, 0.0, 1.0, 0),  # blue  with alpha = 20%
            "font.size": 36
        })

        plt.figure(num="%s_%s_%s_%s_%s_%s_timelocked=%s" %(conditions, runphase, measure_type, view, style, meanstd, timelocked), figsize=(11, 16)) #figsize=(11, 4)
        ax =plt.subplot(111) # can give this ax more specific name

        if len(conditions) > 1 and len(runphase) == 1:
            #colors = utils.Utils().get_cmap(len(conditions), 'tab20c')
            length = len(conditions)
            labels = 'conditions'
            print('Conditions are: %s' %conditions)
        elif len(runphase) > 1 and len(conditions) == 1:
            #colors = utils.Utils().get_cmap(len(runphase), 'tab20c')
            length = len(runphase)
            labels = 'runphase'
        elif len(runphase) == 1 and len(conditions) == 1:
            length = len(conditions)
            labels = 'conditions'
        else:
            raise ValueError('cant plot this many conditions')

       # colors = utils.Utils().get_cmap(20, 'tab20c')
        blues = utils.Utils().get_cmap(20, 'Blues')
        colors = utils.Utils().get_cmap(20, 'tab20')
        if 'APAChar_LowHigh_Day3' in conditions_all and 'APAChar_LowMid_Day3' in conditions_all and len(conditions) ==2:
            sub_colors = [blues(13), colors(4)]
        elif 'APAChar_LowHigh_Day1' in conditions_all and 'APAChar_LowHigh_Day3' in conditions_all and len(conditions) ==2:
            sub_colors = [blues(18), blues(10)]
        elif 'APAChar_LowHigh_Day1' in conditions_all and 'APAChar_LowHigh_Day2' in conditions_all and 'APAChar_LowHigh_Day3' in conditions_all and len(conditions) ==3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all and len(conditions) ==2:
            sub_colors = [colors(6), colors(8)]
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all and 'APAChar_LowHigh' in conditions_all:
            sub_colors = [colors(6), colors(8), blues(13)]
        elif 'PerceptionTest' in conditions_all:
            sub_colors = [colors(10), colors(11)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all and len(
                conditions) == 3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day3' in conditions_all and len(
                conditions) == 3:
            sub_colors = [blues(19), blues(14), blues(9)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all \
                and 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and len(conditions) == 4:
            sub_colors = [blues(18), blues(15), blues(12), blues(9)]
        elif 'APAChar_LowHigh_Repeats_Wash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_Wash_Day3' in conditions_all \
                and 'APAChar_LowHigh_Repeats_NoWash_Day1' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day2' in conditions_all and 'APAChar_LowHigh_Repeats_NoWash_Day3' in conditions_all and len(conditions) == 6:
            sub_colors = [blues(19), blues(17), blues(15), blues(13), blues(11), blues(9)]

        if style == 'individual':
            raise ValueError('This plot is not set up')

        if meanstd == 'mean':
            value = meanstd
        elif meanstd == 'variance':
            value = 'std'
        elif meanstd == 'cv':
            value = meanstd

        if style == 'group':
            spacing = 0.8/length
            count = 0
            for cidx, con in enumerate(conditions):
                if meanbyrun == 'bysubject':
                    for r in runphase:
                        if value == 'mean' or value == 'std':
                            run_means_chunks = np.mean(groupData[con][r].xs(value, axis=1, level=1), axis=1)
                            if variance == 'std':
                                run_variance_chunks = np.std(groupData[con][r].xs(value, axis=1, level=1), axis=1)
                            elif variance == 'sem':
                                run_variance_chunks = np.std(groupData[con][r].xs(value, axis=1, level=1),
                                                             axis=1) / np.sqrt(
                                    len(groupData[con][r].xs(value, axis=1, level=1).columns))

                        elif value == 'cv':
                            cv = groupData[con][r].xs('std', axis=1, level=1)/groupData[con][r].xs('mean', axis=1, level=1)
                            run_means_chunks = np.mean(cv, axis=1)
                            if variance == 'std':
                                run_variance_chunks = np.std(cv, axis=1)
                            elif variance == 'sem':
                                run_variance_chunks = np.std(cv,axis=1) / np.sqrt(len(cv.columns))


                        means = [run_means_chunks['baseline'], run_means_chunks['apa_1'], run_means_chunks['apa_2'],
                                 run_means_chunks['washout']]
                        var = [run_variance_chunks['baseline'], run_variance_chunks['apa_1'],
                               run_variance_chunks['apa_2'],
                               run_variance_chunks['washout']]
                        if labels == 'conditions':
                            barlabel = con
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' % con)
                        elif labels == 'runphase':
                            barlabel = r
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' % r)
                        ax.bar([count * spacing + 1, count * spacing + 2, count * spacing + 3, count * spacing + 4],
                               means, yerr=var,
                               color=sub_colors[count], width=spacing, align='edge', label=barlabel)
                        print('bar should be plotted, value is %s' %value)
                        count += 1

                # DONT USE THIS, DONT THINK THIS IS THE RIGHT WAY STATISTICALLY!!!! AFFECTS THE VARIANCE
                elif meanbyrun == 'byrun':
                    print('#############################################\nIf chose meanstd=variance this will not be utilised here\n#############################################')
                    for r in runphase:
                        run_means = np.mean(groupData[con][r].xs('mean',axis=1,level=1),axis=1)
                        run_means_chunks = {
                            'baseline': np.mean(run_means[0:10]), ######################################### needs replacing with relevant run lengths##########################################
                            'apa': np.mean(run_means[10:30]),
                            'apa_1': np.mean(run_means[10:20]),
                            'apa_2': np.mean(run_means[20:30]),
                            'washout': np.mean(run_means[30:40])
                        }
                        if variance == 'std':
                            run_variance_chunks = {
                                'baseline': np.std(run_means[0:10]),
                                ######################################### needs replacing with relevant run lengths##########################################
                                'apa': np.std(run_means[10:30]),
                                'apa_1': np.std(run_means[10:20]),
                                'apa_2': np.std(run_means[20:30]),
                                'washout': np.std(run_means[30:40])
                            }
                        elif variance == 'sem':
                            run_variance_chunks = {
                                'baseline': np.std(run_means[0:10])/np.sqrt(len(run_means[0:10])),
                                ######################################### needs replacing with relevant run lengths##########################################
                                'apa': np.std(run_means[10:30])/np.sqrt(len(run_means[10:30])),
                                'apa_1': np.std(run_means[10:20])/np.sqrt(len(run_means[10:20])),
                                'apa_2': np.std(run_means[20:30])/np.sqrt(len(run_means[20:30])),
                                'washout': np.std(run_means[30:40])/np.sqrt(len(run_means[30:40]))
                            }
                        means = [run_means_chunks['baseline'], run_means_chunks['apa_1'], run_means_chunks['apa_2'], run_means_chunks['washout']]
                        var = [run_variance_chunks['baseline'], run_variance_chunks['apa_1'], run_variance_chunks['apa_2'], run_variance_chunks['washout']]
                        if labels == 'conditions':
                            barlabel = con
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' %con)
                        elif labels == 'runphase':
                            barlabel = r
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' %r)
                        ax.bar([count*spacing+1, count*spacing+2, count*spacing+3, count*spacing+4], means, yerr=var, color=sub_colors[count], width=spacing, align='edge', label=barlabel)
                        count += 1
                elif meanbyrun == 'oldbysubject':
                    for r in runphase:
                        normtobaseline = groupData[con][r]/groupData[con][r].loc(axis=0)['baseline']
                        normtobaseline = normtobaseline.drop(['apa', 'baseline'])
                        if meanstd == 'mean':
                            means = np.mean(normtobaseline.xs('mean', axis=1, level=1), axis=1)
                            var = np.std(normtobaseline.xs('mean', axis=1, level=1), axis=1)
                            ax.set_ylabel('%s \n(normalised to Baseline, +/- std)' %measure_type, fontsize=10)
                        if meanstd == 'std':
                            means = np.mean(normtobaseline.xs('std', axis=1, level=1), axis=1)
                            var = np.std(normtobaseline.xs('std', axis=1, level=1), axis=1)/np.sqrt(len(normtobaseline.columns.get_level_values(level=0).unique()))
                            ax.set_ylabel('Variance in %s \n(normalised to Baseline, +/- sem)' % measure_type, fontsize=10)
                        #spacing = 0.8/length
                        if labels == 'conditions':
                            barlabel = con
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' %con)
                        elif labels == 'runphase':
                            barlabel = r
                            if userlabels == True:
                                barlabel = input('Type in new label for: %s' %r)
                        ax.bar([count*spacing+1, count*spacing+2, count*spacing+3], means, yerr=var, color=sub_colors[count], width=spacing, align='edge', label=barlabel)
                        count += 1

        # if legend == True:
        #     ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_xticks([1.4, 2.4, 3.4, 4.4])
        ax.set_xticklabels(['Baseline', 'APA\nPhase\n1', 'APA\nPhase\n2', 'Washout'])
        ax.set_xlim([0.8, ax.get_xlim()[1]])
        if meanstd == 'mean' and measure_type == 'Back Height':
            ax.set_ylim([75,105])

        if userlabels == True:
            ylabel = input('Enter y axes label:')
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(measure_type)

        if legend == True:
            if userlabels == True:
                legendtitle = input('Enter legend title:')
                ax.legend(title=legendtitle, bbox_to_anchor=(1, 1), loc='lower right')
            else:
                ax.legend(bbox_to_anchor=(1, 1), loc='lower right')


        if save == True:
            if username == True:
                filename = input('Enter filename (describe the conditions included):')
                plt.savefig(r'%s\Chunk, %s, %s, %s, %s, %s, %s, value=%s.png' % (
                plotting_destfolder, filename, runphase, measure_type, view, style, timelocked, value),
                            bbox_inches='tight', transparent=transparent, format='png')
            else:
                plt.savefig(r'%s\Chunk, %s, %s, %s, %s, %s, %s, value=%s.png' %(plotting_destfolder, conditions, runphase, measure_type, view, style, timelocked, value), bbox_inches = 'tight', transparent=transparent, format='png')


    def chunkStats_expphase_day_washnowash(self,measure_type, view, conditions, runphase, n_apa1=10, timelocked=True, meanstd='mean'):
        warnings.filterwarnings("ignore", message="The default value of numeric_only in DataFrame.cov is deprecated.")
        groupData = self.setupForPlotting(conditions, means=True, measure_type=measure_type, view=view, runphase=runphase, timelocked=timelocked, n_apa1=n_apa1)

        # organise data into appropriate format
        columns = ['mouseID', 'Day', 'ExpPhase', 'Wash', 'Back_Height']
        df_for_stats = pd.DataFrame(columns=columns)
        for cidx, con in enumerate(conditions):
            for r in runphase:
                #run_means_chunks = np.mean(groupData[con][r].xs('mean', axis=1, level=1), axis=1)
                day = con.split('_')[-1][-1]
                wash = con.split('_')[-2]
                for m, mouseID in enumerate(groupData[con][r].columns.get_level_values(level=0).unique()):
                    if meanstd == 'mean':
                        mouse_data = groupData[con][r].loc(axis=1)[mouseID, 'mean']
                    elif meanstd == 'variance':
                        mouse_data = groupData[con][r].loc(axis=1)[mouseID, 'std']
                        print('WARNING: When looking at variance, think carefully about how calculating it - mean of variance calculated when averaging runs in phases per mouse (current approach) or take variance for each run for each mouse and average that??')
                    elif meanstd == 'cv':
                        mouse_data = groupData[con][r].loc(axis=1)[mouseID, 'std'] / groupData[con][r].loc(axis=1)[mouseID, 'mean']

                    current_entry = {
                        'mouseID': [mouseID] * 5,
                        'Day': [day] * 5,
                        'ExpPhase': ['Baseline', 'APA', 'APA1', 'APA2', 'Washout'],
                        'Wash': [wash] * 5,
                        'Back_Height': mouse_data.astype(float)
                    }
                    df_for_stats = pd.concat([df_for_stats, pd.DataFrame.from_dict(current_entry)], ignore_index=True)

        # Check for normality and homogeneity of variance for all variables
        for var in ['Day', 'ExpPhase']:
            for level in df_for_stats[var].unique():
                print(f'Shapiro-Wilk test for {var}={level}:')
                print(shapiro(df_for_stats[df_for_stats[var] == level]['Back_Height']))
                print()
            lev_stat, lev_p = levene(
                *[df_for_stats[df_for_stats[var] == level]['Back_Height'] for level in df_for_stats[var].unique()])
            print(f"Levene's test statistic for {var}: ", lev_stat)
            print(f"Levene's test p-value for {var}: ", lev_p)
            print()

        # # Create an AnovaRM object
        # aov = AnovaRM(df_for_stats, 'Back_Height', 'mouseID', within=['Day', 'ExpPhase'])
        # # Fit the model and print the summary
        # res = aov.fit()
        # print(res.summary())
        anova = pg.rm_anova(dv='Back_Height', within=['Day', 'ExpPhase'], subject='mouseID', data=df_for_stats)
        posthocs_day_exp = pg.pairwise_tests(dv='Back_Height', within=['Day', 'ExpPhase'], subject='mouseID', data=df_for_stats)
        posthocs_exp_day = pg.pairwise_tests(dv='Back_Height', within=['ExpPhase', 'Day'], subject='mouseID', data=df_for_stats)

        stats_dict = {
            'assumptions': {
                'levene': {
                    'p': lev_p,
                    't': lev_stat
                }
            }, ##### include all shapiro tests!
            'rm_anova': anova,
            'posthocs_day_exp': posthocs_day_exp,
            'posthocs_exp_day': posthocs_exp_day
        }

        return stats_dict



    def tempMain(self, measure_type):
        ### Back length
        # condition_names = ['APAChar_LowHigh_Day1', 'APAChar_LowMid_Day1', 'APAChar_LowHigh_Day3', 'APAChar_LowMid_Day3', 'APAChar_HighLow', 'PerceptionTest', 'VMT_LowHighac', 'VMT_LowHighpd', 'APAChar_LowHigh_Day2']
        # condition_combos = {
        #     'Low Day1': [condition_names[0], condition_names[1]],
        #     'Low Day3': [condition_names[2], condition_names[3]],
        #     'Low-High Repeats': [condition_names[0], condition_names[2]],
        #     'Low-High Repeats - All': [condition_names[0], condition_names[8], condition_names[2]],
        #     'High-Low': [condition_names[4]],
        #     'Perception Test': [condition_names[5]],
        #     'VMT': [condition_names[6], condition_names[7]],
        #     'Char vs VMT': [condition_names[0], condition_names[6], condition_names[7]]
        # }
        folder_names = [f for f in os.listdir(filtereddata_folder) if os.path.isdir(os.path.join(filtereddata_folder, f))]
        folder_names = sorted(folder_names)
        condition_names = {
            'APA': {
                'Day_comparison_nowash': {
                    'Low-High': ['%s_Repeats_NoWash_Day1' % folder_names[2], '%s_Repeats_NoWash_Day2' % folder_names[2], '%s_Repeats_NoWash_Day3' % folder_names[2]],
                    'Low-Mid': ['%s_Repeats_NoWash_Day1' % folder_names[3], '%s_Repeats_NoWash_Day2' % folder_names[3], '%s_Repeats_NoWash_Day3' % folder_names[3]]
                },
                'TransitionMagnitude_comparison': {
                    'Accelerating': [folder_names[2], folder_names[3]],  # low-high vs low-mid
                    'Decelerating': [folder_names[0], folder_names[5]] # high-low vs mid-low
                },
                'BaseSpeed_comparison': {
                    'Accelerating': [folder_names[3], folder_names[4]], # low-mid vs mid-high
                    'Decelerating': [folder_names[1], folder_names[5]] # high-mid vs mid-low
                }
            },
            'PerceptionTest': folder_names[6],
            'VMT': { ########### WAIT FOR FINAL FOLDERS TO DO THIS BIT ##############
                'Perceived': {
                    'SpeedComparison': [folder_names[7], folder_names[9]]
                },
                'Actual': {
                }
            }
        }

        view = ['Overhead', 'Side']
        runphase = ['apa', 'post']
        mean_std = ['mean', 'std']
        style = ['group', 'individual']

        # get a flattened list of condition_names keys so can iterate over
        list_conditions = utils.Utils().flatten_dict_keys(condition_names)

        if measure_type == 'Body Length':
            for cidx, c in enumerate(list_conditions.keys()):
                if 'Char vs VMT' not in c:
                    for v in view:
                        for r in runphase:
                            for s in style:
                                self.PlotByRun(conditions=list_conditions[c],measure_type='Body Length', view=v, runphase=[r], style=s,save=True)
            s='group'
            for cidx, c in enumerate(list_conditions.keys()):
                for v in view:
                    for r in runphase:
                        for m in mean_std:
                            self.PlotByPhaseChunk(conditions=list_conditions[c],measure_type='Body Length', view=v, runphase=[r], style=s,meanstd=m,save=True)

        if measure_type == 'Back Skew' or measure_type == 'Back Height':
            for cidx, c in enumerate(list_conditions.keys()):
                if 'Char vs VMT' not in c:
                    for r in runphase:
                        for s in style:
                            self.PlotByRun(conditions=list_conditions[c],measure_type=measure_type, view='Side', runphase=[r], style=s,save=True)
            s='group'
            for cidx, c in enumerate(list_conditions.keys()):
                for r in runphase:
                    for m in mean_std:
                        self.PlotByPhaseChunk(conditions=list_conditions[c],measure_type=measure_type, view='Side', runphase=[r], style=s,meanstd=m,save=True)



# # temp
# conditions=['APAChar_LowHigh_Extended_Day1','APAChar_LowHigh_Extended_Day2','APAChar_LowHigh_Extended_Day3','APAChar_LowHigh_Extended_Day4']
# data = Plot.Plot().getMeasures(conditions,measure_type='Back Height', view='Side',timelocked=True)
# test = pd.concat([data[conditions[0]][mouseID]['apa'], data[conditions[1]][mouseID]['apa'], data[conditions[2]][mouseID]['apa'], data[conditions[3]][mouseID]['apa']], ignore_index=True)
# Measures = []
# Var = []
# for midx, mouseID in enumerate(data[con].keys()):
#     measure_data = pd.concat(
#         [data[conditions[0]][mouseID]['apa'], data[conditions[1]][mouseID]['apa'], data[conditions[2]][mouseID]['apa'],
#          data[conditions[3]][mouseID]['apa']], ignore_index=True)
#     baseline = np.nanmean(measure_data.loc[0:9, 'mean'])
#     normalised_means = measure_data['mean'] - baseline
#     # plt.figure()
#     # plt.plot(measure_data.index.values, normalised_means)
#     Measures.append(normalised_means)
#     Var.append(measure_data['std'])
# biglist = pd.concat(Measures,axis=1)
# biglist_var = pd.concat(Var,axis=1)
# means = biglist.mean(axis=1).values
# var = biglist.sem(axis=1).values
# varmean = biglist_var.mean(axis=1).values
# upper = means + var
# lower = means - var
#
# sns.set(style='white')
# fig, ax = plt.subplots()
# ax.fill_between(biglist.index+1, upper, lower, alpha=0.2)
# ax.plot(biglist.index+1, means)
# ax.axvline(x=10)
# ax.axvline(x=110)
# ax.axvline(x=41,linestyle=':')
# ax.axvline(x=81,linestyle=':')
# ax.axvline(x=121,linestyle=':')
# plt.xlim(1,160)
# sns.despine(left=True,bottom=False)
# plt.ylabel('Relative change in back height (px)')
# # remove the xtick labels and set custom ticks at specific positions
# custom_ticks = [0, 10, 110, 160]
# ax.set_xticks(custom_ticks)
# ax.set_xticklabels(['' for tick in custom_ticks])
# # add text labels at specific positions
# label_positions = [5, 60, 135]
# label_names = ['Baseline', 'APA', 'Washout']
# for pos, name in zip(label_positions, label_names):
#     plt.text(pos, -0.1, name, ha='center', va='center', transform=ax.get_xaxis_transform())
# baseline_mean = np.mean(means[0:10])
