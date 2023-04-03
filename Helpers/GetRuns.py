import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tables
import csv
import math
from tqdm import tqdm
import Helpers.utils as utils
from Helpers.Config import *
from pathlib import Path
from scipy.signal import find_peaks
from scipy import stats

tqdm.pandas()

class GetRuns:

    def __init__(self): # MouseData is input ie list of h5 files
        super().__init__()

    def DeleteRuns(self, exp, files, pre=0, baseline=None, apa=None, washout=None):
        # function to delete erronous extra runs from the **end** of each experimental phase
        # exp is either Char, Per or VMT
        # list files, eg [side filepath, front filepath, overhead filepath]
        files = utils.Utils().GetlistofH5files(files=files) # gets dictionary of side, front and overhead files

        if exp == 'Char':
            runs = APACharRuns
        elif exp == 'Per':
            runs = APAPerRuns
        elif exp == 'VMT':
            runs = APAVmtRuns
        else:
            raise ValueError('Entered exp argument incorrectly')

        if len(files.values()) != 3:
            raise ValueError('Wrong number of files. Check you have included the side, front and overhead views')

        check = input('Starting deletion of extra runs... \n%s pre runs detected, is this correct? y/n' %pre)
        if check == 'n':
            print('Quitting...')
        if check == 'y':
            dfs = {
                'side': pd.read_hdf(files['Side'][0]),
                'front': pd.read_hdf(files['Front'][0]),
                'overhead': pd.read_hdf(files['Overhead'][0])
            }
            #todel = list()

            if baseline is not None:
                del_runB = pre + runs[0]
                todelB = dfs['side'].index.get_level_values(level='Run').unique()[del_runB] # run number labels might not be consecutive so check for nth run label

                for vidx, view in enumerate(list(dfs.keys())):
                    dfs[view].drop(todelB, level='Run', inplace=True)

            if apa is not None:
                del_runA = pre + runs[0] + runs[1]
                todelA = dfs['side'].index.get_level_values(level='Run').unique()[del_runA] # run number labels might not be consecutive so check for nth run label

                for vidx, view in enumerate(list(dfs.keys())):
                    dfs[view].drop(todelA, level='Run', inplace=True)

            if washout is not None:
                del_runW = pre + runs[0] + runs[1] + runs[2]
                todelW = dfs['side'].index.get_level_values(level='Run').unique()[del_runW]  # run number labels might not be consecutive so check for nth run label

                for vidx, view in enumerate(list(dfs.keys())):
                    dfs[view].drop(todelW, level='Run', inplace=True)

            dfs['side'].to_hdf(files['Side'][0], key='RunsSide_cut', mode='w')
            dfs['front'].to_hdf(files['Front'][0], key='RunsFront_cut', mode='w')
            dfs['overhead'].to_hdf(files['Overhead'][0], key='RunsOverhead_cut', mode='w')

            print('The following files have been trimmed and are now %s runs long:\n%s\n%s\n%s' %(len(dfs['side'].index.get_level_values(level='Run').unique()), files['Side'], files['Front'], files['Overhead']))


    def Main(self, destfolder=(), files=None, directory=None, pcutoff=pcutoff):
        # if inputting file paths, make sure to put in [] even if just one
        # to get file names here, just input either files = [''] or directory = '' into function
        # destfolder should be raw format

        files = utils.Utils().GetlistofH5files(files, directory) # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(files) # before proceeding, check that mouse names are correctly labeled

            for j in range(0, len(files['Side'])): # all csv files from each cam are same length so use side for all
                try:
                    # Get (and clean up) dataframes for one mouse (/vid) for each view point
                    DataframeCoor_side = pd.read_hdf(files['Side'][j])
                    try:
                        DataframeCoor_side = DataframeCoor_side.loc(axis=1)[scorer_side].copy()
                    except:
                        DataframeCoor_side = DataframeCoor_side.loc(axis=1)[scorer_side_new].copy()

                    DataframeCoor_front = pd.read_hdf(files['Front'][j])
                    DataframeCoor_front = DataframeCoor_front.loc(axis=1)[scorer_front].copy()

                    DataframeCoor_overhead = pd.read_hdf(files['Overhead'][j])
                    DataframeCoor_overhead = DataframeCoor_overhead.loc(axis=1)[scorer_overhead].copy()
                    print("Starting analysis...")

                    data = self.filterData(DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff, filename=files['Side'][j])

                    if 'APAChar' in Path(files['Side'][j]).stem:
                        runnumbers = APACharRuns
                    elif 'APAPer' in Path(files['Side'][j]).stem:
                        runnumbers = APAPerRuns
                    elif 'APAVmt' in Path(files['Side'][j]).stem:
                        runnumbers = APAVmtRuns
                    else:
                        print('Experiment name in filename wrong for %s' %Path(files['Side'][j]).stem)

                    try:
                        baselinerun0 = data['Dataframes']['side'].loc(axis=0)[0].iloc[0].name[1]
                        APArun0 = data['Dataframes']['side'].loc(axis=0)[runnumbers[0]].iloc[0].name[1]
                        washoutrun0 = data['Dataframes']['side'].loc(axis=0)[runnumbers[0] + runnumbers[1]].iloc[0].name[1]

                        baselinerun0mins = math.modf((baselinerun0 / fps) / 60)[1]
                        baselinerun0secs = int(math.modf((baselinerun0 / fps) / 60)[0] * 60)
                        APArun0mins = math.modf((APArun0 / fps) / 60)[1]
                        APArun0secs = int(math.modf((APArun0 / fps) / 60)[0] * 60)
                        washoutrun0mins = math.modf((washoutrun0 / fps) / 60)[1]
                        washoutrun0secs = int(math.modf((washoutrun0 / fps) / 60)[0] * 60)
                    except:
                        print('Couldnt calculate runtimes')

                    # save reduced dataframe as a .h5 file for each mouse
                    destfolder = destfolder

                    newfilename_side = "%s_Runs.h5" %Path(files['Side'][j]).stem
                    newfilename_front = "%s_Runs.h5" % Path(files['Front'][j]).stem
                    newfilename_overhead = "%s_Runs.h5" % Path(files['Overhead'][j]).stem

                    data['Dataframes']['side'].to_hdf("%s\\%s" %(destfolder, newfilename_side), key='RunsSide', mode='a')
                    data['Dataframes']['front'].to_hdf("%s\\%s" % (destfolder, newfilename_front), key='RunsFront', mode='a')
                    data['Dataframes']['overhead'].to_hdf("%s\\%s" % (destfolder, newfilename_overhead), key='RunsOverhead', mode='a')

                    metadata_filename = "%s_%s_%s_%s_%s_metadata" %(Path(files['Side'][j]).stem.split('_')[0],
                                                      Path(files['Side'][j]).stem.split('_')[1],
                                                      Path(files['Side'][j]).stem.split('_')[2],
                                                      Path(files['Side'][j]).stem.split('_')[3],
                                                      Path(files['Side'][j]).stem.split('_')[4])


                    with open('%s\\%s.csv' %(destfolder, metadata_filename), mode='a', newline='') as metadata_file:
                        metadata_writer = csv.writer(metadata_file, delimiter=',')
                        metadata_writer.writerow(['Runs', data['Runs']])
                        metadata_writer.writerow(['Baseline 1st run', '%smins' %baselinerun0mins, '%ssecs' %baselinerun0secs])
                        metadata_writer.writerow(['APA 1st run', '%smins' %APArun0mins, '%ssecs' %APArun0secs])
                        metadata_writer.writerow(['Washout 1st run', '%smins' %washoutrun0mins, '%ssecs' %washoutrun0secs])
                        metadata_writer.writerow(['CameraMoved', data['CameraMoved']])
                        metadata_writer.writerow(['MarkerXPositions', data['MarkerXPositions']])
                        metadata_writer.writerow(['MarkerYWallPositions', data['MarkerYWallPositions']])
                        metadata_writer.writerow(['MarkerYBeltPositions', data['MarkerYBeltPositions']])
                        metadata_writer.writerow(['Run#', '#RunBacks', 'RunBackFrame#s'])
                    if len(data['RBs']) != 0:
                        rbdf = pd.DataFrame(data['RBs']).transpose()
                        rbdf.to_csv('%s\\%s.csv' % (destfolder, metadata_filename), mode='a', header=False)
                        # if len(data['RBs']) != 0:
                        #     for r in data['RBs']:
                        #         metadata_writer.writerow(['Run_%s' %r, ])
                        #         #this wont work as r is a dict not a name, needs a rethink
                        #
                        # else:
                        #     metadata_writer.writerow(['Runbacks'])


                    print("Reduced coordinate file saved for:\n%s\n%s\n%s" %(files['Side'][j], files['Front'][j], files['Overhead'][j]))
                except:
                    print('********* COULD NOT ANALYSE FILES: *********** \n%s\n%s\n%s' %(files['Side'][j], files['Front'][j],files['Overhead'][j]))
            print("Finished extracting runs for files: \n %s" %files)

        else:
            raise Exception('Missing 1 or some of side-front-overhead file triplets. Check all 3 files have been analysed')

    def findDoorOpCl(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff):
        ################################################################################################################
        ### Find (an overestimate of) frames where the door opens/closes
        # NB this casts a wide next so as to not miss anything. This will then be refined using the run stages information from findRunStages()
        ################################################################################################################

        doormask = np.logical_or.reduce((DataframeCoor_side.loc(axis=1)['Door', 'likelihood'] > 0.5,
                                         DataframeCoor_front.loc(axis=1)['Door', 'likelihood'] > 0.99,
                                         DataframeCoor_overhead.loc(axis=1)['Door', 'likelihood'] > 0.99))

        TrialStart_DataframeCoor_side = DataframeCoor_side[doormask]  #
        TrialStart_DataframeCoor_front = DataframeCoor_front[doormask]
        TrialStart_DataframeCoor_overhead = DataframeCoor_overhead[doormask]

        sidemask = TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'likelihood'] > 0.5
        frontmask = TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'likelihood'] > 0.99
        overheadmask = TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'likelihood'] > 0.99

        # find 1st derivative of x/y values (ie the 'speed' of the door's movement frame to frame)
        d1_side = TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask].diff()/TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask].index.to_series().diff()
        d1_front = TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask].diff()/TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask].index.to_series().diff()
        d1_overhead = TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask].diff()/TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask].index.to_series().diff()

        chunk = 75 #50 # min distance from other peaks

        # for every frame, find the sum of the last 1000 frames.
        side_d1rolling = d1_side.rolling(chunk).sum().shift(-chunk)
        front_d1rolling = d1_front.rolling(chunk).sum().shift(-chunk)
        overhead_d1rolling = d1_overhead.rolling(chunk).sum().shift(-chunk)

        # Find the both the +ve and -ve peaks in this array to show the frames where noisy/unchanging speed of the door changes to fast moving periods and vice versa
        highpeaks_side = find_peaks(side_d1rolling, height=1, prominence=1, distance=1000)[0]
        lowpeaks_side = find_peaks(-side_d1rolling, height=1, prominence=1, distance=1000)[0]
        highpeaks_front = find_peaks(front_d1rolling, height=1, prominence=1, distance=1000)[0]
        lowpeaks_front = find_peaks(-front_d1rolling, height=1, prominence=1, distance=1000)[0]
        highpeaks_overhead = find_peaks(overhead_d1rolling, height=1, prominence=1, distance=1000)[0]
        lowpeaks_overhead = find_peaks(-overhead_d1rolling, height=1, prominence=1, distance=1000)[0]

        # find difference in index values to show where door is not in frame (ie door is up)
        diffidx_side = TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'y'][sidemask].index.to_series().diff().shift(-1)
        diffidx_front = TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'y'][frontmask].index.to_series().diff().shift(-1)
        diffidx_overhead = TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask].index.to_series().diff().shift(-1)

        # find peaks in above to find frames where the door disappears
        diffidx_peaks_side = find_peaks(diffidx_side, height=1000)[0]
        # diffidx_peaks_front = find_peaks(diffidx_front, height=1000)[0]
        # diffidx_peaks_overhead = find_peaks(diffidx_overhead, height=1000)[0]

        # finds where the values corresponding to the above peaks fall either above or below 0.6 (data is normalised between 0 and 1) to exclude erronous data points
        closeidx_side = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[highpeaks_side][utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[highpeaks_side]<0.4]
        openidx_side = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[lowpeaks_side][utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[lowpeaks_side]>0.2]
        closeidx_front = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[highpeaks_front][utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[highpeaks_front]<0.5]
        openidx_front = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[lowpeaks_front][utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[lowpeaks_front]>0.6]
        closeidx_overhead = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[highpeaks_overhead][utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[highpeaks_overhead]<0.4]
        openidx_overhead = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[lowpeaks_overhead][utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[lowpeaks_overhead]>0.5]

        # finds frames which correspond to peaks found in door disappearing data
        openidx_framegap_side = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[diffidx_peaks_side]

        # if the final value is negative, this suggests this is a missed door open (cant pick up the peak and not closed
        if side_d1rolling.iloc[-chunk - 1] < side_d1rolling.quantile(.15):
            openidx_side.loc(axis=0)[side_d1rolling.index[-chunk - 1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'y'][sidemask]).loc(axis=0)[side_d1rolling.index[-chunk - 1]]
            #openidx_side.loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'y'][sidemask]).index[-1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'y'][sidemask]).loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'y'][sidemask]).index[-1]]
        if front_d1rolling.iloc[-chunk - 1] < front_d1rolling.quantile(.15):
            openidx_front.loc(axis=0)[front_d1rolling.index[-chunk - 1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'y'][frontmask]).loc(axis=0)[front_d1rolling.index[-chunk - 1]]
            #openidx_front.loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'y'][frontmask]).index[-1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'y'][frontmask]).loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'y'][frontmask]).index[-1]]
        if overhead_d1rolling.iloc[-chunk - 1] < overhead_d1rolling.quantile(.15):
            openidx_overhead.loc(axis=0)[overhead_d1rolling.index[-chunk - 1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask]).loc(axis=0)[overhead_d1rolling.index[-chunk - 1]]
            #openidx_overhead.loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask]).index[-1]] = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask]).loc(axis=0)[utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask]).index[-1]]

        doors = {
            'close_idx_side': closeidx_side,
            'open_idx_side': openidx_side,
            'close_idx_front': closeidx_front,
            'open_idx_front': openidx_front,
            'close_idx_overhead': closeidx_overhead,
            'open_idx_overhead': openidx_overhead,
            'open_idx_framegap_side': openidx_framegap_side
        }

        return doors

    def findRunStages(self,  DataframeCoor_side, pcutoff):
        ################################################################################################################
        ### Find frames that mark the RunStart, Transition, RunEnd, ReturnStart and ReturnEnd
        ################################################################################################################

        wall3_mode = stats.mode(np.int_(DataframeCoor_side.loc(axis=1)['Wall3', 'x']))[0][0]

        RunStartmask = np.logical_or(
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['StartPlatR','likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['HindpawToeR', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Wall3', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['Back12', 'x'] > DataframeCoor_side.loc(axis=1)['StartPlatR', 'x'], # Mouse in first block
                #DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Wall3', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] < wall3_mode, # Mouse in first block
                DataframeCoor_side.loc(axis=1)['HindpawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['StartPlatL', 'y'] # hindpaw down
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['StartPlatR','likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['HindpawToeL', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Wall3', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['Back12', 'x'] > DataframeCoor_side.loc(axis=1)['StartPlatR', 'x'], # Mouse in first block
                # DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Wall3', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] < wall3_mode,  # Mouse in first block
                DataframeCoor_side.loc(axis=1)['HindpawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['StartPlatL', 'y'] # hindpaw down
            ))
        )

        Transitionmask = np.logical_or.reduce((
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] - DataframeCoor_side.loc(axis=1)['Back12', 'x'] > 50,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values) - if y value bigger than transition (regardless of what side), suggests on belt
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(500).max().shift(-500) > DataframeCoor_side.loc(axis=1)['TransitionR', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] - DataframeCoor_side.loc(axis=1)['Back12', 'x'] > 50,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(500).max().shift(-500) > DataframeCoor_side.loc(axis=1)['TransitionR', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] - DataframeCoor_side.loc(axis=1)['Back12', 'x'] > 50,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(500).max().shift(-500) > DataframeCoor_side.loc(axis=1)['TransitionL', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,
                #DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] - DataframeCoor_side.loc(axis=1)['Back12', 'x'] > 50,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(500).max().shift(-500) > DataframeCoor_side.loc(axis=1)['TransitionL', 'x']
            ))
        ))

        RunEndmask = np.logical_and.reduce((
            DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
            DataframeCoor_side.loc(axis=1)['Back12', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'] # Mouse in last block
        ))

        ReturnStartmask = np.logical_and.reduce((
            DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Back6', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Back6', 'x'],# backward facing run
            DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'] # Mouse in last block
        ))

        ReturnEndmask = np.logical_and.reduce((
            DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
            DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Back12', 'x'],# backward facing run
            DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['StartPlatR', 'x'],# Mouse in last block
            DataframeCoor_side.loc(axis=1)['Back12', 'x'] < DataframeCoor_side.loc(axis=1)['Wall3', 'x']# Mouse in last block
        ))


        nextrungap = 900
        RunStartfirst = DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunStartmask][DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunStartmask].index.to_series().diff() > nextrungap]
        RunStartfirst = pd.concat([DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunStartmask].iloc[:1], RunStartfirst])
        Transitionfirst = DataframeCoor_side.loc(axis=1)['Nose', 'x'][Transitionmask][DataframeCoor_side.loc(axis=1)['Nose', 'x'][Transitionmask].index.to_series().diff() > nextrungap]
        Transitionfirst = pd.concat([DataframeCoor_side.loc(axis=1)['Nose', 'x'][Transitionmask].iloc[:1], Transitionfirst])
        RunEndlast = DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunEndmask][DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunEndmask].index.to_series().diff().shift(-1) > nextrungap]
        RunEndlast = pd.concat([RunEndlast, DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunEndmask].iloc[-1:]])
        ReturnStartfirst = DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnStartmask][DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnStartmask].index.to_series().diff() > nextrungap]
        ReturnStartfirst = pd.concat([DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnStartmask].iloc[:1], ReturnStartfirst])
        ReturnEndlast = DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnEndmask][DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnEndmask].index.to_series().diff().shift(-1) > nextrungap]
        ReturnEndlast = pd.concat([ReturnEndlast, DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnEndmask].iloc[-1:]])

        # getting rid of erronous run ends where nose has jumped when on a runback, e.g. due to hand nudging
        #RunEndlast = RunEndlast[np.logical_and(RunEndlast > RunEndlast.mean() - RunEndlast.std()*2, RunEndlast < RunEndlast.mean() + RunEndlast.std()*2)]
       # Transitionfirst = Transitionfirst[np.logical_and(Transitionfirst > Transitionfirst.mean() - Transitionfirst.std()*2, Transitionfirst < Transitionfirst.mean() + Transitionfirst.std()*2)]

        Runs = {
            'RunStart': RunStartfirst,
            'Transition': Transitionfirst,
            'RunEnd': RunEndlast,
            'ReturnStart': ReturnStartfirst,
            'ReturnEnd': ReturnEndlast
        }

        return Runs

    def filterData(self, data_side, data_front, data_overhead, pcutoff, filename):
        ################################################################################################################
        # Combine the door opening data with the run data to chunk data into runs
        ################################################################################################################
        DataframeCoor_side = data_side.copy(deep=True)
        DataframeCoor_front = data_front.copy(deep=True)
        DataframeCoor_overhead = data_overhead.copy(deep=True)

        runstages = self.findRunStages(DataframeCoor_side=DataframeCoor_side, pcutoff=pcutoff)
        doors = self.findDoorOpCl(DataframeCoor_side=DataframeCoor_side, DataframeCoor_front=DataframeCoor_front, DataframeCoor_overhead=DataframeCoor_overhead, pcutoff=pcutoff)

        if "HM_20220901_APAChar_FAA-1034979_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].append(pd.Series(data=runstages['Transition'].mean(), index=[206492]), ignore_index=False)
            runstages['Transition'] = runstages['Transition'].sort_index()
            runstages['RunEnd'] = runstages['RunEnd'].append(pd.Series(data=runstages['RunEnd'].mean(), index=[206500]), ignore_index=False)
            runstages['RunEnd'] = runstages['RunEnd'].sort_index()
        if "HM_20220824_APAChar_FAA-1034979_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping last baseline trial
            runstages['Transition'] = runstages['Transition'].drop(index=[89673])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[89716])
        if "HM_20220824_APAChar_FAA-1034983_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping last apa trial
            runstages['Transition'] = runstages['Transition'].drop(index=[214490])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[214534])
            # dropping last washout trial
            runstages['Transition'] = runstages['Transition'].drop(index=[314038])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[314058])
        if "HM_20220825_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping incorrect run where huma was pushing mouse
            runstages['Transition'] = runstages['Transition'].drop(index=[363367])
            # run end not found here as turned back quickly
        if "HM_20220902_APAChar_FAA-1034979_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping incorrect run where mouse almost stepped over despite it being a run back
            runstages['Transition'] = runstages['Transition'].drop(index=[95607])
        if "HM_20220902_APAChar_FAA-1034983_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping incorrect run where mouse almost stepped over despite it being a run back
            runstages['Transition'] = runstages['Transition'].drop(index=[62263, 98610, 153884, 213502, 215100, 277996, 280140])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[154049, 200310, 215323, 222107, 278464, 280648])
        if "HM_20220829_APAChar_FAA-1034978_MR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # dropping incorrect run where mouse almost stepped over despite it being a run back
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[114663, 146710])
            runstages['Transition'] = runstages['Transition'].drop(index=[146704])
        if "HM_20220829_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[351010])
        if "HM_20220827_APAChar_FAA-1034976_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[305571])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[305618])
        if "HM_20220827_APAChar_FAA-1034978_MR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[1824])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[1852])
        if "HM_20220905_APAChar_FAA-1034979_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[220202])
        if "HM_20220905_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[104561])
            runstages['Transition'] = runstages['Transition'].drop(index=[267790])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[64154])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[156728])
        if "HM_20230306_APACharRepeat_FAA-1035244_L_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[212837])
        if "HM_20230306_APACharRepeat_FAA-1035249_R_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[263385,200613,182649])
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[200649,182748])
        if "HM_20230306_APACharRepeat_FAA-1035245_R_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[298498,339900]) # mouse got stuck and couldn't finish last 2 trials (incl trial where got stuck).
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[298731,340102])
            runstages['Transition'] = runstages['Transition'].drop(index=[271271]) # mouse slipped under door when stationary
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[271302])
        if "HM_20230306_APACharRepeat_FAA-1035250_LR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[150479])  # extra run - mouse sat so jerry did another
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[150933])
        if "HM_20230309_APACharRepeat_FAA-1035298_LR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[77841,141427,211977])  # extra run - mouse sat so jerry did another
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[78278,212009])
        if "HM_20230309_APACharRepeat_FAA-1035299_None_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[52222,143313,218775,357521,378928,403650])  # lots of runbacks
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[143386, 218803,403716])
        if "HM_20230309_APACharRepeat_FAA-1035301_R_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[69734]) # belt off
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[69766])
        if "HM_20230309_APACharRepeat_FAA-1035302_LR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['Transition'] = runstages['Transition'].drop(index=[46271,217479]) # belt off
            runstages['RunEnd'] = runstages['RunEnd'].drop(index=[46430])



        # first check that the number of transitions is the same as run ends
        if len(runstages['Transition']) != len(runstages['RunEnd']):
            runend_outlier_mask = np.logical_and(runstages['RunEnd'] > runstages['RunEnd'].mean() - runstages['RunEnd'].std()*2, runstages['RunEnd'] < runstages['RunEnd'].mean() + runstages['RunEnd'].std()*2)
            if len(runstages['RunEnd']) != sum(runend_outlier_mask):
                runstages['RunEnd'] = runstages['RunEnd'][np.logical_and(runstages['RunEnd'] > runstages['RunEnd'].mean() - runstages['RunEnd'].std()*2, runstages['RunEnd'] < runstages['RunEnd'].mean() + runstages['RunEnd'].std()*2)]
                weirdidxs = runend_outlier_mask.index[runend_outlier_mask == False]
                print('!!!!!!!!Weird RunEnd value(/s) found and DELETED at frame(/s): %s!!!!!!!!!!' % weirdidxs)
            else:
                raise ValueError('There are a different number of transitions from run ends. Error in run stage detection somewhere, most likely due to obscured frames or an attempted transition being wrongly counted')

        # clean data by finding run backs - if there is a case where it goes runstart - ReturnEnd, with no RunEnd between, label that run a rb (put into another list) and delete from the RunStart list

        # To find the trial start frames, find the closest door opening values for each Transition (first) frame
        big_idxlist = pd.concat([doors['open_idx_side'], doors['open_idx_front'], doors['open_idx_overhead'],
                                 doors['open_idx_framegap_side']])
        dist = runstages['Transition'].index.values[:, np.newaxis] - big_idxlist.index.values
        dist = dist.astype(float)
        dist[dist < 0] = np.nan
        potentialClosestpos = np.nanargmin(dist, axis=1)
        closestFound, closestCounts = np.unique(potentialClosestpos, return_counts=True)
        potentialClosestpos_pd = pd.Series(potentialClosestpos)

        # Adding in wrongly missed run ends when e.g. a RB isn't caught as an end of trial
        if "HM_20220901_APAChar_FAA-1034983_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            # runstages['TrialEnd'] = runstages['TrialEnd'].append(pd.Series(data=1, index=[175584]), ignore_index=False)
            # runstages['TrialEnd'] = runstages['TrialEnd'].sort_index()
            runstages['RunEnd'] = runstages['RunEnd'].append(pd.Series(data=1, index=[175583]), ignore_index=False)
            runstages['RunEnd'] = runstages['RunEnd'].sort_index()

        # deal with instances where mouse ran back and forth *after* the trial had finished
        if closestFound.shape[0] != runstages['Transition'].index.values.shape[0]:
            mask = closestCounts > 1
            dups = closestFound[mask]
            to_delALL = list()
            for d in range(0, len(dups)):
                repeats_idxs = potentialClosestpos_pd.index[potentialClosestpos == dups[d]]
                to_del = repeats_idxs[1:]
                to_delALL.append(to_del)
            to_delALL = np.concatenate(to_delALL)

            potentialClosestpos_pd = potentialClosestpos_pd.drop(index=potentialClosestpos_pd.index[to_delALL])
            if len(runstages['Transition']) == len(runstages['RunEnd']):
                runstages['Transition'] = runstages['Transition'].drop(index=runstages['Transition'].index[to_delALL])
                runstages['RunEnd'] = runstages['RunEnd'].drop(index=runstages['RunEnd'].index[to_delALL])
            else:
                runstages['Transition'] = runstages['Transition'].drop(index=runstages['Transition'].index[to_delALL])
        runstages['TrialStart'] = big_idxlist.iloc[potentialClosestpos_pd]


        ############################ NEW AS OF 19-01-2023 ###########################################
        # changed looking for when nose disappears to when right ear disappears as seems more reliable. Also now look for closest frame whether before or after run end  (when the camera has shifted the tape blocks some of the view and so causes a discrepancy here.
        # choose frame after each RunEnd where ear is no longer in frame
        earnotpresent = DataframeCoor_side.loc(axis=1)['EarR', 'likelihood'][
            DataframeCoor_side.loc(axis=1)['EarR', 'likelihood'].rolling(
                100).mean() < pcutoff]  # finds where the mean of the *last* 100 frames is less than pcutoff
        eardisappear = earnotpresent[earnotpresent.index.to_series().diff() > 50]  # finds where there are large gaps between frames which have a rolling mean of over pcutoff

        # choose frame after each RunEnd where tail is no longer in frame
        tailnotpresent = DataframeCoor_side.loc(axis=1)['Tail1', 'likelihood'][
            DataframeCoor_side.loc(axis=1)['Tail1', 'likelihood'].rolling(
                100).mean() < pcutoff]  # finds where the mean of the *last* 100 frames is less than pcutoff
        taildisappear = tailnotpresent[tailnotpresent.index.to_series().diff() > 50]  # finds where there are large gaps between frames which have a rolling mean of over pcutoff

        eartaildisappear = pd.concat((eardisappear, taildisappear), axis=0)


        # find closest frame of where ear disappears to each run end value
        disteartail = eartaildisappear.index.values - runstages['RunEnd'].index.values[:, np.newaxis]
        disteartail = abs(disteartail.astype(float))
        potentialClosestpos_eartail = np.argmin(disteartail, axis=1)
        closestfoundeartail, closestCountseartail = np.unique(potentialClosestpos_eartail, return_counts=True)
        if closestfoundeartail.shape[0] != runstages['RunEnd'].index.values.shape[0]:
            raise ValueError('Seems to be a missing RunEnd value')
        runstages['TrialEnd'] = eartaildisappear.iloc[potentialClosestpos_eartail]

        # check that the trial end values are appropriate and importantly if they occur before run end that it is not too much before
        if min(runstages['TrialEnd'].index - runstages['RunEnd'].index) < -20:
            enddiff = runstages['TrialEnd'].index - runstages['RunEnd'].index
            rundiff = runstages['TrialEnd'].index[enddiff < -20]
            bigdiff = enddiff[enddiff > 100]
            rundiffno = np.asarray(runstages['TrialEnd'].index == rundiff[0]).nonzero()
            print('!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!\n>>>>>>TrialEnd occurs significantly before RunEnd - could be cutting off the run!! \nTrialEnd frame: %s, \nSize of difference: %s, \nRun # where this occurs: %s' %(rundiff, bigdiff, rundiffno))
        if max(runstages['TrialEnd'].index - runstages['RunEnd'].index) > 100:
            enddiff = runstages['TrialEnd'].index - runstages['RunEnd'].index
            bigdiff = enddiff[enddiff > 100]
            print('!!!WARNING!!!\n>>>>TrialEnd occurs a long time after RunEnd (Number frames: %s). Advisable to double check run detection is ok.' % bigdiff)

        # Throwing out erronous data manually (here extra TrialStarts, e.g. from door being moved out of trial start)
        if "HM_20220905_APAChar_FAA-1034978_MR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['TrialStart'] = runstages['TrialStart'].drop(index=180951)
        # if "HM_20220905_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
        #     runstages['TrialStart'] = runstages['TrialStart'].drop(index=266851)
        if "HM_20220901_APAChar_FAA-1034983_MLR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
            runstages['TrialStart'] = runstages['TrialStart'].drop(index=142714)

        # chunk up data for each camera between TrialStart and TrialEnd
        if len(runstages['TrialStart']) != len(runstages['TrialEnd']):
            raise ValueError('Different number of trial starts and ends.')

        print('Number of runs detected: %s' % len(runstages['TrialStart']))
        ### !!! here put some comparison of the lengths from the excel doc when ready for it

        runs = pd.DataFrame({'TrialStart': runstages['TrialStart'].index, 'TrialEnd': runstages['TrialEnd'].index})

        Run = np.full([len(DataframeCoor_side)], np.nan)
        DataframeCoor_side.loc(axis=1)['Run'] = Run
        DataframeCoor_front.loc(axis=1)['Run'] = Run
        DataframeCoor_overhead.loc(axis=1)['Run'] = Run

        FrameIdx = DataframeCoor_side.index
        DataframeCoor_side.loc(axis=1)['FrameIdx'] = FrameIdx
        DataframeCoor_front.loc(axis=1)['FrameIdx'] = FrameIdx
        DataframeCoor_overhead.loc(axis=1)['FrameIdx'] = FrameIdx

        for i in range(0, len(runs)):
            maskside = np.logical_and(DataframeCoor_side.index >= runs.loc(axis=1)['TrialStart'][i],
                                      DataframeCoor_side.index <= runs.loc(axis=1)['TrialEnd'][i])
            maskfront = np.logical_and(DataframeCoor_front.index >= runs.loc(axis=1)['TrialStart'][i],
                                       DataframeCoor_front.index <= runs.loc(axis=1)['TrialEnd'][i])
            maskoverhead = np.logical_and(DataframeCoor_overhead.index >= runs.loc(axis=1)['TrialStart'][i],
                                          DataframeCoor_overhead.index <= runs.loc(axis=1)['TrialEnd'][i])

            # self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
            DataframeCoor_side.loc[maskside, 'Run'] = i
            DataframeCoor_front.loc[maskfront, 'Run'] = i
            DataframeCoor_overhead.loc[maskoverhead, 'Run'] = i

        DataframeCoor_side = DataframeCoor_side[DataframeCoor_side.loc(axis=1)['Run'].notnull()]
        DataframeCoor_front = DataframeCoor_front[DataframeCoor_front.loc(axis=1)['Run'].notnull()]
        DataframeCoor_overhead = DataframeCoor_overhead[DataframeCoor_overhead.loc(axis=1)['Run'].notnull()]

        # DataframeCoor_side.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        # DataframeCoor_front.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        # DataframeCoor_overhead.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

        ###### should add in another column to show the run stages - RunStart, Tranisiton, RunEnd
        ### also find run backs (mid trial) and related to this repeat runs
        ### based on the above need to show where multiple run attempts are. presumably this is clear by, if a runback is present, only considering the second instance of the run (oe RunStart, Transition and RunEnd)

        # Tidy up run stages frames by deleting any after the first RunEnd (because these will be where the mouse has run about post trial)
        for rsidx, rs in enumerate(RunStages):
            err_mask = np.isin(runstages[rs].index, np.array(DataframeCoor_side.index))
            if len(runstages[rs]) != len(runstages[rs][err_mask == True]):
                numerr = len(runstages[rs]) - len(runstages[rs][err_mask == True])
                errframes = runstages[rs].index[err_mask == False]
                print('%s extra instances found for %s outside of trial\nFrame numbers: %s\nExtra runs deleted' % (
                numerr, rs, errframes))
                runstages[rs] = runstages[rs][err_mask == True]

        RunStage = np.full([len(DataframeCoor_side)], np.nan)
        DataframeCoor_side.loc(axis=1)['RunStage'] = RunStage
        DataframeCoor_front.loc(axis=1)['RunStage'] = RunStage
        DataframeCoor_overhead.loc(axis=1)['RunStage'] = RunStage

        for rsidx, rs in enumerate(RunStages):
            DataframeCoor_side.at[runstages[rs].index, 'RunStage'] = rs
            DataframeCoor_front.at[runstages[rs].index, 'RunStage'] = rs
            DataframeCoor_overhead.at[runstages[rs].index, 'RunStage'] = rs

        DataframeCoor_side.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        DataframeCoor_front.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        DataframeCoor_overhead.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

        ##### !!!! Readjust run numbers for missed runs mid experiment !!!! ######
        if 'HM_20220825_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_103000' in filename: # 1 missing APA, 2 missing washout (ignored as at end)
            print('One trial missing from APA phase. Shifting subsequent washout run labels...')
            self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=31, max=39, shifted=1)
        if 'HM_20230306_APACharRepeat_FAA-1035246_LR_side_1DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000' in filename: # 5 missing APA from end of APA phase (accidentally quit video)
            print('Five trials missing from APA phase. Shifting subsequent washout run labels...')
            self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=27, max=37, shifted=5)

        # Find where any runbacks are by looking in each run for multiple RunStarts
        runbackALL = list()
        runbacklog = {}
        for i in DataframeCoor_side.index.get_level_values('Run').unique():
            numRunStarts = sum(DataframeCoor_side.xs(axis=0, level='Run', key=i).loc(axis=1)['RunStage'] == 'RunStart')
            if numRunStarts > 1:
                numrunbacks = numRunStarts - 1
                print('%s run backs found for run: %s' % (numrunbacks, i))
                runstartrows = DataframeCoor_side.xs(axis=0, level='Run', key=i)[
                    DataframeCoor_side.xs(axis=0, level='Run', key=i).loc(axis=1)['RunStage'] == 'RunStart']
                runbackIdx = runstartrows.iloc[:-1].index
                runbackALL.append(runbackIdx.to_list())
                runbacklog['%s' %i] = [numrunbacks, runbackIdx]
            else:
                print('no runbacks for run: %s' % i)

        if len(runbackALL) != 0:
            runbackALL = np.concatenate(runbackALL).flatten()

        # Replace any erronous RunStarts as RunBacks
        for r in range(0, len(runbackALL)):
            rbmask = DataframeCoor_side.loc(axis=1)['RunStage'].index.get_level_values(level='FrameIdx') == runbackALL[r]
            DataframeCoor_side.loc[rbmask, 'RunStage'] = 'RunBack'

        # TEMP: Add in any missing RunStarts where code missed
        for r in DataframeCoor_side.index.get_level_values(level='Run').unique():
            if 'RunStart' not in DataframeCoor_side.loc(axis=0)[r].loc(axis=1)['RunStage'].values:
                print('run %s has no runstart, mouse likely pushed door open itself' % r)
                missingRSmask = DataframeCoor_side.loc(axis=0)[r].loc(axis=1)['RunStage'].index == \
                                DataframeCoor_side.loc(axis=0)[r].loc(axis=1)['RunStage'].index[1]
                RSidx = DataframeCoor_side.loc(axis=0)[r].loc(axis=1)['RunStage'].index[missingRSmask][0]
                DataframeCoor_side.loc[(r, RSidx), 'RunStage'] = 'RunStart'

        # Fill RunStage column so that each stage is continued until reach the next
        DataframeCoor_side = DataframeCoor_side.fillna(method='ffill')
        DataframeCoor_front = DataframeCoor_front.fillna(method='ffill')
        DataframeCoor_overhead = DataframeCoor_overhead.fillna(method='ffill')

        # Add RunStage to multi index
        DataframeCoor_side.set_index(['RunStage'], append=True, inplace=True)
        DataframeCoor_front.set_index(['RunStage'], append=True, inplace=True)
        DataframeCoor_overhead.set_index(['RunStage'], append=True, inplace=True)

        # Reorder multiindex
        DataframeCoor_side = DataframeCoor_side.reorder_levels(['Run', 'RunStage', 'FrameIdx'])
        DataframeCoor_front = DataframeCoor_front.reorder_levels(['Run', 'RunStage', 'FrameIdx'])
        DataframeCoor_overhead = DataframeCoor_overhead.reorder_levels(['Run', 'RunStage', 'FrameIdx'])

        # ##### !!!! Readjust run numbers for missed runs mid experiment !!!! ######
        # if 'HM_20220825_APAChar_FAA-1034980_MNone_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_103000' in filename: # 1 missing APA, 2 missing washout (ignored as at end)
        #     print('One trial missing from APA phase. Shifting subsequent washout run labels...')
        #     for r in reversed(range(31, 39)):
        #         DataframeCoor_side.rename(index={r: r + 1}, inplace=True)
        #         DataframeCoor_front.rename(index={r: r + 1}, inplace=True)
        #         DataframeCoor_overhead.rename(index={r: r + 1}, inplace=True)

        markers = self.findMarkers(DataframeCoor_side)['DualBeltMarkers']
        cam_moved = self.findMarkers(DataframeCoor_side)['CameraMoved']

        # based on physical landmarks in dualbelt, calculate the edges of the belt quadrants
        m0 = markers['x_edges'][0]
        m1 = (markers['x'][0] + markers['x'][1])/2
        m2 = (markers['x'][1] + markers['x'][2])/2
        # m3 = markers['x_edges'][2]
        # m4 = markers['x'][4]

        # Where mouse in RunStart, label q1,q2,q3. Where mouse in Transition + RunEnd, label q4
        Quadrant = np.full([len(DataframeCoor_side)], np.nan)
        DataframeCoor_side.loc(axis=1)['Quadrant'] = Quadrant
        DataframeCoor_front.loc(axis=1)['Quadrant'] = Quadrant
        DataframeCoor_overhead.loc(axis=1)['Quadrant'] = Quadrant

        # create mask where nose is present for each quadrant (across all frames)
        Q1mask = list()
        Q2mask = list()
        Q3mask = list()
        Q4mask = list()
        for r in range(0, len(DataframeCoor_side.index.get_level_values(level='Run').unique())):
            q1 = np.logical_and.reduce((DataframeCoor_side.loc(axis=1)['Nose', 'x'] > m0,
                                        DataframeCoor_side.loc(axis=1)['Nose', 'x'] < m1,
                                        DataframeCoor_side.index.get_level_values(level='RunStage') == 'RunStart',
                                        DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff))
            q2 = np.logical_and.reduce((DataframeCoor_side.loc(axis=1)['Nose', 'x'] > m1,
                                        DataframeCoor_side.loc(axis=1)['Nose', 'x'] < m2,
                                        DataframeCoor_side.index.get_level_values(level='RunStage') == 'RunStart',
                                        DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff))
            q3 = np.logical_and.reduce((DataframeCoor_side.loc(axis=1)['Nose', 'x'] > m2,
                                        #DataframeCoor_side.loc(axis=1)['Nose', 'x'] < m3,
                                        DataframeCoor_side.index.get_level_values(level='RunStage') == 'RunStart',
                                        DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff))
            q4 = np.logical_or((np.logical_and.reduce((#DataframeCoor_side.loc(axis=1)['Nose', 'x'] > m3,
                                        DataframeCoor_side.index.get_level_values(level='RunStage') == 'Transition',
                                        #DataframeCoor_side.loc(axis=1)['Nose', 'x'] < m4,
                                        DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff))),
                                (np.logical_and.reduce((  # DataframeCoor_side.loc(axis=1)['Nose', 'x'] > m3,
                                        DataframeCoor_side.index.get_level_values(level='RunStage') == 'RunEnd',
                                        # DataframeCoor_side.loc(axis=1)['Nose', 'x'] < m4,
                                        DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff)))
            )

        Q1mask.append(q1)
        Q2mask.append(q2)
        Q3mask.append(q3)
        Q4mask.append(q4)

        # then flatten
        Q1mask = np.concatenate(Q1mask)
        Q2mask = np.concatenate(Q2mask)
        Q3mask = np.concatenate(Q3mask)
        Q4mask = np.concatenate(Q4mask)
        Qmasks = [Q1mask, Q2mask, Q3mask, Q4mask]

        # need to get a mask for whole dataframe to set value
        sidemask = np.full((DataframeCoor_side.shape), False)
        frontmask = np.full((DataframeCoor_front.shape), False)
        overheadmask = np.full((DataframeCoor_overhead.shape), False)

        Qs = ['Q1', 'Q2', 'Q3', 'Q4']

        for qidx, q in enumerate(Qs):
            sidemask[:, -1] = Qmasks[qidx]
            frontmask[:, -1] = Qmasks[qidx]
            overheadmask[:, -1] = Qmasks[qidx]

            DataframeCoor_side = DataframeCoor_side.mask(sidemask, q)
            DataframeCoor_front = DataframeCoor_front.mask(frontmask, q)
            DataframeCoor_overhead = DataframeCoor_overhead.mask(overheadmask, q)

        Dataframes = {
            'side': DataframeCoor_side,
            'front': DataframeCoor_front,
            'overhead': DataframeCoor_overhead
        }

        numruns = len(runstages['TrialStart'])

        #Combine the following to return: dataframes, cam_moved, runbacklog, numruns, markers['x'], markers['y_wall'], markers['y_belt']
        results = {
            'Dataframes': Dataframes, # dict of side, front and overhead dfs
            'Runs': numruns, # int of total runs number
            'RBs': runbacklog, # dict where each key is run number, with data where [0] is number of rbs and [1] is list of frame numbers where rbs occurred for an individual run
            'CameraMoved': cam_moved,
            'MarkerXPositions': markers['x'],
            'MarkerYWallPositions': markers['y_wall'],
            'MarkerYBeltPositions': markers['y_belt']
        }

        return results

    def shiftRunNumbers(self, side, front, overhead, min, max, shifted):
        for r in reversed(range(min, max)):
            side.rename(index={r: r + shifted}, inplace=True)
            front.rename(index={r: r + shifted}, inplace=True)
            overhead.rename(index={r: r + shifted}, inplace=True)


    def findMarkers(self, df_side):
        # Function to find the x and y values of the physical landmarks on the travelator (and identify if the camera shifted)
        # get points for travelator landmarks
        Wallx = list()
        Wally = list()
        Beltx = list()
        Belty = list()

        for l in range(1, 6):
            # wallx = np.mean(df_side.loc(axis=1)['Wall%s' % l, 'x'][
            #                     df_side.loc(axis=1)['Wall%s' % l, 'likelihood'] > pcutoff])
            # beltx = np.mean(df_side.loc(axis=1)['Belt%s' % l, 'x'][
            #                     df_side.loc(axis=1)['Belt%s' % l, 'likelihood'] > pcutoff])
            # wally = np.mean(df_side.loc(axis=1)['Wall%s' % l, 'y'][
            #                     df_side.loc(axis=1)['Wall%s' % l, 'likelihood'] > pcutoff])
            # belty = np.mean(df_side.loc(axis=1)['Belt%s' % l, 'y'][
            #                     df_side.loc(axis=1)['Belt%s' % l, 'likelihood'] > pcutoff])

            # gets the most common integer. Even when confidence is low for wall 2 and 3, the coordinate is mostly correct so this captures that
            wallx = stats.mode(np.int_(df_side.loc(axis=1)['Wall%s' % l, 'x'].values))[0][0]
            beltx = stats.mode(np.int_(df_side.loc(axis=1)['Belt%s' % l, 'x'].values))[0][0]
            wally = stats.mode(np.int_(df_side.loc(axis=1)['Wall%s' % l, 'y'].values))[0][0]
            belty = stats.mode(np.int_(df_side.loc(axis=1)['Belt%s' % l, 'y'].values))[0][0]

            Wallx.append(wallx)
            Wally.append(wally)
            Beltx.append(beltx)
            Belty.append(belty)

        # Wallx[1] = Wallx[0] + (Wallx[2] - Wallx[0])*0.52 # replace dodgy wall2 value with middle between 1 and 3. 0.52 as stars are not equally apart on backing (ie Wall[1] is 52% of the way to Wall[2]
        # Wally[1] = Wally[0] + (Wally[2] - Wally[0])*0.5

        startplatRmeanx = np.mean(df_side.loc(axis=1)['StartPlatR', 'x'][
                                     df_side.loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff])
        startplatLmeanx = np.mean(df_side.loc(axis=1)['StartPlatL', 'x'][
                                     df_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff])
        transitionRmeanx = np.mean(df_side.loc(axis=1)['TransitionR', 'x'][
                                      df_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff])
        transitionLmeanx = np.mean(df_side.loc(axis=1)['TransitionL', 'x'][
                                      df_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff])

        startplatRmeany = np.mean(df_side.loc(axis=1)['StartPlatR', 'y'][
                                      df_side.loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff])
        startplatLmeany = np.mean(df_side.loc(axis=1)['StartPlatL', 'y'][
                                      df_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff])
        transitionRmeany = np.mean(df_side.loc(axis=1)['TransitionR', 'y'][
                                       df_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff])
        transitionLmeany = np.mean(df_side.loc(axis=1)['TransitionL', 'y'][
                                       df_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff])

        # creates lisdt of wall markers, where first in list is 1, and last is 5. For Platforms it goes in order of appearance from left to right
        landmarksx = {
            'Wall': Wallx,
            'Belt': Beltx,
            'Platforms': [startplatRmeanx, startplatLmeanx, transitionLmeanx, transitionRmeanx]
        }
        landmarksy = {
            'Wall': Wally,
            'Belt': Belty,
            'Platforms': [startplatRmeany, startplatLmeany, transitionLmeany, transitionRmeany]
        }

        wallbeltdiffx = abs(np.array(landmarksx['Wall']) - np.array(landmarksx['Belt']))
        walldiffLRx = np.diff(np.array(landmarksx['Wall']))
        beltdiffLRx = np.diff(np.array(landmarksx['Belt']))

        wallbeltdiffy = abs(np.array(landmarksy['Wall']) - np.array(landmarksy['Belt']))
        walldiffLRy = np.diff(np.array(landmarksy['Wall']))
        beltdiffLRy = np.diff(np.array(landmarksy['Belt']))

        cmtopx = (landmarksx['Wall'][2] - landmarksx['Wall'][0])/(distancescm[1]+distancescm[0])
        pxtocm = 1 / cmtopx

        distancescmtopx = np.array(distancescm) * int(cmtopx)

        # check if any discrepancies, if not make each marker value an average between belt and wall 'x' values
        if np.logical_or.reduce(((abs(distancescmtopx - walldiffLRx) > 2*cmtopx).any(), np.isnan(distancescmtopx - walldiffLRx).any(), walldiffLRy.any() > 10, np.isnan(walldiffLRy.any()))):
            raise ValueError('Either Wall0, Wall2, Wall3 or Wall4 is incorrect or missing. Need to check manually. (Wall1 is always bad so already accounted for')
        #elif np.logical_or.reduce(((abs(distancescmtopx - beltdiffLRx) > 1*cmtopx).any(), np.isnan(distancescmtopx - beltdiffLRx).any(), beltdiffLRy.any() > 10, np.isnan(beltdiffLRy.any()))):
        # elif np.logical_or.reduce((np.isnan(distancescmtopx - beltdiffLRx).any(), beltdiffLRy.any() > 10, np.isnan(beltdiffLRy.any()))):
        elif np.logical_or.reduce(((abs(distancescmtopx - beltdiffLRx) > 2*cmtopx).any(), np.isnan(distancescmtopx - beltdiffLRx).any(), beltdiffLRy.any() > 10, np.isnan(beltdiffLRy.any()))):
            if (np.array(landmarksx['Wall']) - np.array(landmarksx['Belt']))[1] > 2*cmtopx: # check if the specific issue is that belt2's value is very different
                x = (np.array(landmarksx['Wall']) + np.array(landmarksx['Belt'])) * 0.5
                x[1] = landmarksx['Wall'][1] # replace erronous belt2 value with wall's value as wall must be fine to have passed the above condition
                y_wall = (np.array(landmarksy['Wall']))
                y_belt = (np.array(landmarksy['Belt']))
            else:
                raise ValueError('Either Belt0, Belt1, Belt2, Belt3 or Belt4 is incorrect or missing. Need to check manually.')
        elif (wallbeltdiffx > 1*cmtopx).any():
            raise ValueError('Something wrong with a Belt or Wall x value')
        else:
            x = (np.array(landmarksx['Wall']) + np.array(landmarksx['Belt']))*0.5
            y_wall = (np.array(landmarksy['Wall']))
            y_belt = (np.array(landmarksy['Belt']))

        DualBeltMarkers = {
            'x': x,
            'x_edges': landmarksx['Platforms'],
            'y_edges': landmarksy['Platforms'],
            'y_wall': y_wall,
            'y_belt': y_belt
        }

        # check if ALL markers (except wall2) move across video, suggesting camera moved
        WallALL_RunsALL = list()
        for l in range(1, 6):
            Wall_RunsALL = list()
            for r in df_side.index.get_level_values('Run').unique():
                wall = np.mean(df_side.loc(axis=0)[r].loc(axis=1)['Wall%s' % l, 'x'][
                                   df_side.loc(axis=0)[r].loc(axis=1)['Wall%s' % l, 'likelihood'] > 0.999])
                Wall_RunsALL.append(wall)
            WallALL_RunsALL.append(Wall_RunsALL)

        cam_move = False
        if np.logical_and(
                sum(pd.Series(WallALL_RunsALL[4]).diff()[abs(pd.Series(WallALL_RunsALL[4]).diff()) > 1 * cmtopx] > 0) != sum(
                pd.Series(WallALL_RunsALL[4]).diff()[abs(pd.Series(WallALL_RunsALL[4]).diff()) > 1 * cmtopx] < 0),
                sum(pd.Series(WallALL_RunsALL[0]).diff()[abs(pd.Series(WallALL_RunsALL[0]).diff()) > 1 * cmtopx] > 0) != sum(
                    pd.Series(WallALL_RunsALL[0]).diff()[abs(pd.Series(WallALL_RunsALL[0]).diff()) > 1 * cmtopx] < 0)
        ):
            cam_move = True
            raise ValueError('The camera appears to have shifted!! Marker means for this video are not reliable')

        results = {
            'DualBeltMarkers': DualBeltMarkers,
            'CameraMoved': cam_move,
            'pxtocm': pxtocm,
            'cmtopx': cmtopx
        }

        return results