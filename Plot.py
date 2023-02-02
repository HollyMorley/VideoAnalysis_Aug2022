from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import Helpers.utils as utils
from Helpers.Config import *
from scipy.stats import skew
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
                files = utils.Utils().GetlistofH5files(directory=r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\%s" %(con), filtered=True)
            else:
                splitcon = con.split('_')
                conname = "_".join(splitcon[:-1])
                dayname = splitcon[-1]
                files = utils.Utils().GetlistofH5files(directory=r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\%s\%s" %(conname, dayname), filtered=True)
            #filesALL['%s' %con] =
            mouseIDALL = list()

            # sort lists of filenames so that in same order for reading
            files['Side'].sort()
            files['Front'].sort()
            files['Overhead'].sort()

            for f in range(0,len(files['Side'])):
                mouseID = os.path.basename(files['Side'][f]).split('_')[3]
                mouseIDALL.append(mouseID)
            data['%s' % con] = dict.fromkeys(mouseIDALL)
            for n, name in enumerate(mouseIDALL):
                data['%s' % con]['%s' %name] = dict.fromkeys(['Side', 'Front', 'Overhead'])
                data['%s' %con]['%s' %name] = {
                    'Side': pd.read_hdf(files['Side'][n]),
                    'Front': pd.read_hdf(files['Front'][n]),
                    'Overhead': pd.read_hdf(files['Overhead'][n])
                }
        return data

    def getMeasures(self, conditions, measure_type, view, timelocked):
        '''
        :param conditions: list of experimental conditions want to plot/analyse eg 'APAChar_HighLow', 'APAChar_LowHigh_Day1', 'APAVMT_LowHighac'. NB make sure to include the day if for a condition which has repeats
        :param measure: which measure you plan to calculate e.g. body length
        :return: dictionary holding specified measure at each run stage [pre, apa, post], both for every run and averaged across each run stage eg baseline.
        '''
        data = self.GetDFs(conditions)
        data_measure = dict.fromkeys(conditions)

        for cidx, con in enumerate(data.keys()):
            data_measure[con] = dict.fromkeys(data[con].keys())
            for midx, mouseID in enumerate(data[con].keys()):
                # first get the frame numbers for the x frames before and after transition point for every frame and experimental phase
                measureAllRuns_pre = list() # early in run - q1 and 2
                measureAllRuns_apa = list() # just before transition
                measureAllRuns_post = list() # just after transition
                for r in data[con][mouseID]['Side'].index.get_level_values(level='Run').unique():
                    if timelocked == True:
                        preidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q1', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')].index.get_level_values(level='FrameIdx')
                        apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3'][-APA_lengthruns:].index.get_level_values(level='FrameIdx')
                        postidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q4'][:after_lengthruns].index.get_level_values(level='FrameIdx')
                    else:
                        preidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][np.logical_or(data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q1', data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q2')].index.get_level_values(level='FrameIdx')
                        apaidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q3'].index.get_level_values(level='FrameIdx')
                        postidx = data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'][data[con][mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Quadrant'] == 'Q4'].index.get_level_values(level='FrameIdx')

                    premask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(preidx)
                    apamask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(apaidx)
                    postmask = data[con][mouseID]['Side'].loc(axis=0)[r].index.get_level_values(level='FrameIdx').isin(postidx)

                    # do calculations
                    if measure_type == 'Body Length':
                        measure = self.CalculateBodyLength(data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)
                    if measure_type == 'Back Skew':
                        measure = self.CalculateBack(calculation='skew',  data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)
                    if measure_type == 'Back Height':
                        measure = self.CalculateBack(calculation='height', data=data, con=con, mouseID=mouseID, r=r, premask=premask, apamask=apamask, postmask=postmask, view=view)


                    measureAllRuns_pre.append((measure['mean']['pre'], measure['std']['pre']))
                    measureAllRuns_apa.append((measure['mean']['apa'], measure['std']['apa']))
                    measureAllRuns_post.append((measure['mean']['post'], measure['std']['post']))

                measureAllRuns_pre = np.vstack(measureAllRuns_pre)
                measureAllRuns_apa = np.vstack(measureAllRuns_apa)
                measureAllRuns_post = np.vstack(measureAllRuns_post)

                # remove data from prep runs (different numbers depending on condition
                if np.logical_and('Char' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                    measureAllRuns_pre = measureAllRuns_pre[preruns_CharLow:]
                    measureAllRuns_apa = measureAllRuns_apa[preruns_CharLow:]
                    measureAllRuns_post = measureAllRuns_post[preruns_CharLow:]
                    print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' %(preruns_CharLow, mouseID, len(measureAllRuns_pre)))
                elif 'Char' in con: # for all other apa characterise conditions
                    measureAllRuns_pre = measureAllRuns_pre[preruns_CharMidHigh:]
                    measureAllRuns_apa = measureAllRuns_apa[preruns_CharMidHigh:]
                    measureAllRuns_post = measureAllRuns_post[preruns_CharMidHigh:]
                    print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' %(preruns_CharMidHigh, mouseID, len(measureAllRuns_pre)))
                elif np.logical_and('VMT' in con, np.logical_or('LowHigh' in con, 'LowMid' in con)):
                    if np.logical_and('LowHighpd' in con, np.logical_or.reduce(('1034980' in mouseID, '1034982' in mouseID, '1034983' in mouseID))):
                        measureAllRuns_pre = measureAllRuns_pre[preruns_CharLow:]
                        measureAllRuns_apa = measureAllRuns_apa[preruns_CharLow:]
                        measureAllRuns_post = measureAllRuns_post[preruns_CharLow:]
                        print('%s prep frames removed from start of experiment.\nNow total run number for mouse %s is: %s' % (preruns_CharLow, mouseID, len(measureAllRuns_pre)))
                    else:
                        print('No prep runs. Total run number for mouse %s is: %s' %(mouseID, len(measureAllRuns_pre)))
                elif 'PerceptionTest' in con:
                    measureAllRuns_pre = measureAllRuns_pre
                    measureAllRuns_apa = measureAllRuns_apa
                    measureAllRuns_post = measureAllRuns_post
                else:
                    raise ValueError('STOP: I havent categorised this file/condition yet in terms of its extra runs')

                #### calculate averages and std for measure in baseline, APA1, APA2, washout ###
                # Baseline runs
                measureBaseline_pre = [np.nanmean(measureAllRuns_pre[:10][:,0]), np.nanstd(measureAllRuns_pre[:10][:,0])]
                measureBaseline_apa = [np.nanmean(measureAllRuns_apa[:10][:,0]), np.nanstd(measureAllRuns_apa[:10][:,0])]
                measureBaseline_post = [np.nanmean(measureAllRuns_post[:10][:,0]), np.nanstd(measureAllRuns_post[:10][:,0])]

                # Washout runs
                measureWashout_pre = [np.nanmean(measureAllRuns_pre[-10:][:,0]), np.nanstd(measureAllRuns_pre[-10:][:,0])]
                measureWashout_apa = [np.nanmean(measureAllRuns_apa[-10:][:,0]), np.nanstd(measureAllRuns_apa[-10:][:,0])]
                measureWashout_post = [np.nanmean(measureAllRuns_post[-10:][:,0]), np.nanstd(measureAllRuns_post[-10:][:,0])]

                # APA runs (+ APA runs split into two halves)
                if 'Char' in con or 'Perception' in con:
                    numruns = APACharRuns
                elif 'VMT' in con:
                    numruns = APAVmtRuns
                measureAPA_pre = [np.nanmean(measureAllRuns_pre[numruns[0]:numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_pre[numruns[0]:numruns[0]+numruns[1]][:,0])]
                measureAPA_apa = [np.nanmean(measureAllRuns_apa[numruns[0]:numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_apa[numruns[0]:numruns[0]+numruns[1]][:,0])]
                measureAPA_post = [np.nanmean(measureAllRuns_post[numruns[0]:numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_post[numruns[0]:numruns[0]+numruns[1]][:,0])]

                measureAPA_1_pre = [np.nanmean(measureAllRuns_pre[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0]), np.nanstd(measureAllRuns_pre[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0])]
                measureAPA_1_apa = [np.nanmean(measureAllRuns_apa[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0]), np.nanstd(measureAllRuns_apa[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0])]
                measureAPA_1_post = [np.nanmean(measureAllRuns_post[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0]), np.nanstd(measureAllRuns_post[numruns[0]:numruns[0]+int(numruns[1]/2)][:,0])]

                measureAPA_2_pre = [np.nanmean(measureAllRuns_pre[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_pre[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0])]
                measureAPA_2_apa = [np.nanmean(measureAllRuns_apa[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_apa[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0])]
                measureAPA_2_post = [np.nanmean(measureAllRuns_post[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0]), np.nanstd(measureAllRuns_post[numruns[0]+int(numruns[1]/2):numruns[0]+numruns[1]][:,0])]

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
                    if measure_type == 'Wait Time':
                        measure = self.CalculateWaitTime(data=data, con=con, mouseID=mouseID, r=r)

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
        # speed = distance/time
        if 'TrialStart' in data[con][mouseID]['Side'].loc(axis=0)[r].index:
            runtime = data[con][mouseID]['Side'].loc(axis=0)[r,'Transition'].index[0] - data[con][mouseID]['Side'].loc(axis=0)[r,'RunStart'].index[0]
            distance = data[con][mouseID]['Side'].loc(axis=0)[r,'Transition'].loc(axis=0)['Nose','x'][0] - data[con][mouseID]['Side'].loc(axis=0)[r,'RunStart'].loc(axis=0)['Nose','x'][0]
            #distancecm =
            totalspeed = distance/runtime
        else:
            raise ValueError('No runstart for run %s' %r)

        # subtract belt speed from totalspeed
        ##### TEMP ########
        logdf = pd.read_excel(r"C:\Users\Holly Morley\Dropbox (UCL)\Murray Lab\Holly\Aug22_APACharacteriseLong_log_recovered.xlsx", "Sheet1")
        b2baselinespeed = np.unique(test.loc(axis=0)[20220824, 'FAA-1034976', 'Baseline'].loc(axis=1)['Belt2speed (cm/s)'].values)[0]
        b2apaspeed = np.unique(test.loc(axis=0)[20220824, 'FAA-1034976', 'APA_characterise'].loc(axis=1)['Belt2speed (cm/s)'].values)[0]

    def setupForPlotting(self, conditions, means, measure_type, view, runphase, timelocked, means_all=False):
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
            data = self.getMeasures(conditions,measure_type,view,timelocked=timelocked)
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
                            groupData[con][r].loc(axis=1)[mouseID] = data[con][mouseID][r]
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

    def PlotByRun(self, conditions, measure_type, style, view, runphase, legend=True, transparent=False, save=False, userlabels=False, timelocked=True):
        '''

        :param legendtitle: e.g. 'Experiment Day', 'Speed Transition', 'Run Phase'
        :param conditions:
        :param measure_type:
        :param style:
        :param view:
        :param runphase:
        :param timelocked:
        :return:
        '''
        if len(conditions) > 1:
            conditions_all = '\t'.join(conditions)
        else:
            conditions_all = conditions

        means = False

        groupData = self.setupForPlotting(conditions, means, measure_type, view, runphase, timelocked)

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

        # if 'apa' in runphase[0]:
        #     colors = utils.Utils().get_cmap(length+2, 'Blues')
        # if 'post' in runphase[0]:
        #     colors = utils.Utils().get_cmap(length + 2, 'Purples')
        # if 'pre' in runphase[0]:
        #     colors = utils.Utils().get_cmap(length + 2, 'Greens')
        # if 'all' in runphase[0]:
        #     colors = utils.Utils().get_cmap(length + 2, 'Greys')

        blues = utils.Utils().get_cmap(20, 'Blues')
        colors = utils.Utils().get_cmap(20, 'tab20')
        if 'APAChar_LowHigh_Day3' in conditions_all and 'APAChar_LowMid_Day3' in conditions_all:
            sub_colors = [blues(13), colors(4)]
        elif 'APAChar_LowHigh_Day1' in conditions_all and 'APAChar_LowHigh_Day3' in conditions_all:
            sub_colors = [blues(18), blues(10)]
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all:
            sub_colors = [colors(6), colors(8)]
        elif 'PerceptionTest' in conditions_all:
            sub_colors = [colors(10), colors(11)]

        plt.rcParams.update({
            "figure.facecolor": (1.0, 0.0, 0.0, 0),  # red   with alpha = 30%
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
            "savefig.facecolor": (0.0, 0.0, 1.0, 0),  # blue  with alpha = 20%
            "font.size": 24,
            'axes.spines.right': False,
            'axes.spines.top':False
        })


        plt.figure(num="%s_%s_%s_%s_%s_timelocked=%s" %(conditions, runphase, measure_type, view, style, timelocked), figsize=(11, 4))
        ax =plt.subplot(111) # can give this ax more specific name
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
                    if labels == 'conditions':
                        barlabel = con
                        if userlabels == True:
                            barlabel = input('Type in new label for: %s' %con)
                    elif labels == 'runphase':
                        barlabel = r
                        if userlabels == True:
                            barlabel = input('Type in new label for: %s' %r)
                    ax.plot(groupData[con][r].index + 1, np.mean(groupData[con][r].xs('mean', axis=1,level=1), axis=1), label=barlabel, color=sub_colors[count])
                    ax.fill_between(groupData[con][r].index + 1,
                                    np.mean(groupData[con][r].xs('mean', axis=1,level=1), axis=1) - np.std(groupData[con][r].xs('mean', axis=1,level=1), axis=1),
                                    np.mean(groupData[con][r].xs('mean', axis=1,level=1), axis=1) + np.std(groupData[con][r].xs('mean', axis=1,level=1), axis=1),
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
                ax.legend(title=legendtitle, bbox_to_anchor=(1, 1), loc='lower right')
            else:
                ax.legend(bbox_to_anchor=(1, 1), loc='lower right')


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
            plt.savefig(r'%s\ByRun, %s, %s, %s, %s, %s, %s.png' %(plotting_destfolder, conditions, runphase, measure_type, view, style, timelocked), bbox_inches = 'tight', transparent=transparent, format='png')

            # pxto1cm = GetRuns.GetRunsimpo().findMarkers(data[con][mouseID]['Side'])['pxtocm']
            # pxto1mm = pxtocm*10


    def PlotByPhaseChunk(self, meanbyrun, measure_type, view, conditions, runphase, meanstd='mean', style='group', legend=True, transparent=False, save=False, timelocked=True, userlabels=False):
        '''
        Plots measures as averages for APA (first and second half) and washout, all normalised to baseline
        :param measure_type: 'Body Length' ...
        :param view: 'side', 'front', 'overhead'
        :param conditions: experimental condition, see under analysis directory
        :param runphase: 'pre', 'apa' or 'post'
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

        if meanbyrun == True:
            means = False
        else:
            means = True

        groupData = self.setupForPlotting(conditions, means, measure_type, view, runphase, timelocked)

        plt.rcParams.update({
            "figure.facecolor": (1.0, 0.0, 0.0, 0),  # red   with alpha = 30%
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
            "savefig.facecolor": (0.0, 0.0, 1.0, 0),  # blue  with alpha = 20%
            "font.size": 36
        })

        plt.figure(num="%s_%s_%s_%s_%s_%s_timelocked=%s" %(conditions, runphase, measure_type, view, style, meanstd, timelocked), figsize=(11, 16)) #figsize=(11, 4)
        ax =plt.subplot(111) # can give this ax more specific name
        #ax.set_xlim(0, groupData[conditions[0]][runphase[0]].shape[0])
        #ax.set_ylabel(measure_type)

        #width = 0.1 # width of the bars
        #x = np.arange(len(chunks))

        # colormaps = ['blues', 'greens']

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
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all and len(conditions) ==2:
            sub_colors = [colors(6), colors(8)]
        elif 'VMT_LowHighac' in conditions_all and 'VMT_LowHighpd' in conditions_all and 'APAChar_LowHigh' in conditions_all:
            sub_colors = [colors(6), colors(8), blues(13)]
        elif 'PerceptionTest' in conditions_all:
            sub_colors = [colors(10), colors(11)]

        if style == 'individual':
            raise ValueError('This plot is not set up')
            # normtobasline = groupData[conditions[0]][runphase[0]] / groupData[conditions[0]][runphase[0]].loc(axis=0)['baseline']
            # normtobasline = normtobasline.drop(['apa', 'baseline'])
            # x = np.array([1,2,3])
            # count = 0
            # for midx, mouseID in enumerate(normtobasline.columns.get_level_values(level=0).unique()):
            #     ax.bar(x+(count*0.1), normtobasline.xs('mean', axis=1, level=1).loc(axis=1)[mouseID], width=0.08, yerr=normtobasline.xs('std', axis=1, level=1).loc(axis=1)[mouseID], align='edge')
            #     count += 1


        if style == 'group':
            spacing = 0.8/length
            count = 0
            for cidx, con in enumerate(conditions):
                if meanbyrun == True:
                    for r in runphase:
                        run_means = np.mean(groupData[con][r], axis=1)
                        run_means_chunks = {
                            'baseline': np.mean(run_means[0:10]), ######################################### needs replacing with relevant run lengths##########################################
                            'apa': np.mean(run_means[10:30]),
                            'apa_1': np.mean(run_means[10:20]),
                            'apa_2': np.mean(run_means[20:30]),
                            'washout': np.mean(run_means[30:40])
                        }
                        run_variance_chunks = {
                            'baseline': np.std(run_means[0:10]),
                            ######################################### needs replacing with relevant run lengths##########################################
                            'apa': np.std(run_means[10:30]),
                            'apa_1': np.std(run_means[10:20]),
                            'apa_2': np.std(run_means[20:30]),
                            'washout': np.std(run_means[30:40])
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
                else:
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

        if userlabels == True:
            legendtitle = input('Enter legend title:')
            plt.legend(title=legendtitle)

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
            plt.savefig(r'%s\Chunk, %s, %s, %s, %s, %s, %s, %s.png' %(plotting_destfolder, conditions, runphase, measure_type, view, style, timelocked, meanstd), transparent=transparent, format='png')

    def tempMain(self, measure_type):
        ### Back length
        condition_names = ['APAChar_LowHigh_Day1', 'APAChar_LowMid_Day1', 'APAChar_LowHigh_Day3', 'APAChar_LowMid_Day3', 'APAChar_HighLow', 'PerceptionTest', 'VMT_LowHighac', 'VMT_LowHighpd']
        condition_combos = {
            'Low Day1': [condition_names[0], condition_names[1]],
            'Low Day3': [condition_names[2], condition_names[3]],
            'Low-High Repeats': [condition_names[0], condition_names[2]],
            'High-Low': [condition_names[4]],
            'Perception Test': [condition_names[5]],
            'VMT': [condition_names[6], condition_names[7]],
            'Char vs VMT': [condition_names[0], condition_names[6], condition_names[7]]
        }

        view = ['Overhead', 'Side']
        runphase = ['apa', 'post']
        mean_std = ['mean', 'std']
        style = ['group', 'individual']

        if measure_type == 'Body Length':
            for cidx, c in enumerate(condition_combos.keys()):
                if 'Char vs VMT' not in c:
                    for v in view:
                        for r in runphase:
                            for s in style:
                                self.PlotByRun(conditions=condition_combos[c],measure_type='Body Length', view=v, runphase=[r], style=s,save=True)
            s='group'
            for cidx, c in enumerate(condition_combos.keys()):
                for v in view:
                    for r in runphase:
                        for m in mean_std:
                            self.PlotByPhaseChunk(conditions=condition_combos[c],measure_type='Body Length', view=v, runphase=[r], style=s,meanstd=m,save=True)

        if measure_type == 'Back Skew' or measure_type == 'Back Height':
            for cidx, c in enumerate(condition_combos.keys()):
                if 'Char vs VMT' not in c:
                    for r in runphase:
                        for s in style:
                            self.PlotByRun(conditions=condition_combos[c],measure_type=measure_type, view='Side', runphase=[r], style=s,save=True)
            s='group'
            for cidx, c in enumerate(condition_combos.keys()):
                for r in runphase:
                    for m in mean_std:
                        self.PlotByPhaseChunk(conditions=condition_combos[c],measure_type=measure_type, view='Side', runphase=[r], style=s,meanstd=m,save=True)


