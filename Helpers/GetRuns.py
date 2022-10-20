import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
import Helpers.utils as utils
from Helpers.Config import *
from pathlib import Path
from scipy.signal import find_peaks

tqdm.pandas()

class GetRuns:

    def __init__(self): # MouseData is input ie list of h5 files
        super().__init__()


    def Main(self, destfolder=(), files=None, directory=None, pcutoff=pcutoff):
        # if inputting file paths, make sure to put in [] even if just one
        # to get file names here, just input either files = [''] or directory = '' into function
        # destfolder should be raw format

        files = utils.Utils().GetlistofH5files(files, directory) # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(files) # before proceeding, check that mouse names are correctly labeled

            for j in range(0, len(files['Side'])): # all csv files from each cam are same length so use side for all
                # Get (and clean up) dataframes for one mouse (/vid) for each view point
                DataframeCoor_side = pd.read_hdf(files['Side'][j])
                DataframeCoor_side = DataframeCoor_side.loc(axis=1)[scorer_side].copy()

                DataframeCoor_front = pd.read_hdf(files['Front'][j])
                DataframeCoor_front = DataframeCoor_front.loc(axis=1)[scorer_front].copy()

                DataframeCoor_overhead = pd.read_hdf(files['Overhead'][j])
                DataframeCoor_overhead = DataframeCoor_overhead.loc(axis=1)[scorer_overhead].copy()
                print("Starting analysis...")

                dfs = self.filterData(DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff)

                # save reduced dataframe as a .h5 file for each mouse
                destfolder = destfolder

                newfilename_side = "%s_Runs.h5" %Path(files['Side'][j]).stem
                newfilename_front = "%s_Runs.h5" % Path(files['Front'][j]).stem
                newfilename_overhead = "%s_Runs.h5" % Path(files['Overhead'][j]).stem

                DataframeCoor_side.to_hdf("%s\\%s" %(destfolder, newfilename_side), key='RunsSide', mode='a')
                DataframeCoor_front.to_hdf("%s\\%s" % (destfolder, newfilename_front), key='RunsFront', mode='a')
                newfilename_overhead.to_hdf("%s\\%s" % (destfolder, newfilename_overhead), key='RunsOverhead', mode='a')

                print("Reduced coordinate file saved for:\n%s\n%s\n%s" %(files['Start'][j], files['Front'][j], files['Overhead'][j]))
                del DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead # clear for next videos just in case it keeps any data

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

        RunStartmask = np.logical_or(
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['StartPlatR','likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['HindpawToeR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Wall3', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['Back12', 'x'] > DataframeCoor_side.loc(axis=1)['StartPlatR', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Wall3', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['HindpawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['StartPlatL', 'y'] # hindpaw down
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['StartPlatR','likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['HindpawToeL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Wall3', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['Back12', 'x'] > DataframeCoor_side.loc(axis=1)['StartPlatR', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] < DataframeCoor_side.loc(axis=1)['Wall3', 'x'], # Mouse in first block
                DataframeCoor_side.loc(axis=1)['HindpawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['StartPlatL', 'y'] # hindpaw down
            ))
        )

        Transitionmask = np.logical_or.reduce((
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'], # forward facing run
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(100).max().shift(-100) > DataframeCoor_side.loc(axis=1)['TransitionR', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionR', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(100).max().shift(-100) > DataframeCoor_side.loc(axis=1)['TransitionR', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeL', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(100).max().shift(-100) >  DataframeCoor_side.loc(axis=1)['TransitionL', 'x']
            )),
            np.logical_and.reduce((
                DataframeCoor_side.loc(axis=1)['TransitionL', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff,
                DataframeCoor_side.loc(axis=1)['Nose', 'x'] > DataframeCoor_side.loc(axis=1)['Back12', 'x'],# forward facing run
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'x'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'x'],
                DataframeCoor_side.loc(axis=1)['ForepawToeR', 'y'] > DataframeCoor_side.loc(axis=1)['TransitionL', 'y'].rolling(100).mean(),  # !!! this could take mean of none significant values)
                DataframeCoor_side.loc(axis=1)['Back12', 'x'].rolling(100).max().shift(-100) > DataframeCoor_side.loc(axis=1)['TransitionL', 'x']
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


        nextrungap = 1000
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

        Runs = {
            'RunStart': RunStartfirst,
            'Transition': Transitionfirst,
            'RunEnd': RunEndlast,
            'ReturnStart': ReturnStartfirst,
            'ReturnEnd': ReturnEndlast
        }

        return Runs

    def filterData(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff):
        ################################################################################################################
        # Combine the door opening data with the run data to chunk data into runs
        ################################################################################################################
        runstages = GetRuns.GetRuns().findRunStages(DataframeCoor_side=DataframeCoor_side, pcutoff=pcutoff)
        doors = GetRuns.GetRuns().findDoorOpCl(DataframeCoor_side=DataframeCoor_side,
                                               DataframeCoor_front=DataframeCoor_front,
                                               DataframeCoor_overhead=DataframeCoor_overhead, pcutoff=pcutoff)

        # first check that the number of transitions is the same as run ends
        if len(runstages['Transition']) != len(runstages['RunEnd']):
            raise ValueError('There are a different number of transitions from run ends. Error in run stage detection somewhere, most likely due to obscured frames or an attempted transition being wrongly counted')

        # clean data by finding run backs - if there is a case where it goes runstart - ReturnEnd, with no RunEnd between, label that run a rb (put into another list) and delete from the RunStart list

        # To find the trial start frames, find the closest door opening values for each Transition (first) frame
        big_idxlist = pd.concat([doors['open_idx_side'], doors['open_idx_front'], doors['open_idx_overhead'], doors['open_idx_framegap_side']])
        dist = runstages['Transition'].index.values[:, np.newaxis] - big_idxlist.index.values
        dist = dist.astype(float)
        dist[dist < 0] = np.nan
        potentialClosestpos = np.nanargmin(dist, axis=1)
        closestFound, closestCounts = np.unique(potentialClosestpos, return_counts=True)
        if closestFound.shape[0] != runstages['Transition'].index.values.shape[0]:
            raise ValueError('Seem to be missing a door opening. Duplicate frames found for 2 runs. (Or the extra run is not a real run, e.g. a runback) ')
            ############# instead of raising an error here, delete the second (/any subsequent) run from all the run stages lists as this is just the mouse running about after trial done. 
        TrialStart = big_idxlist.iloc[potentialClosestpos]


        # choose frame after each RunEnd where nose is no longer in frame
        nosenotpresent = DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'][DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'].rolling(100).mean() < pcutoff] # finds where the mean of the *last* 100 frames is less than pcutoff
        nosedisappear = nosenotpresent[nosenotpresent.index.to_series().diff() > 50] # finds where there are large gaps between frames which have a rolling mean of over pcutoff

        distnose = runstages['RunEnd'].index.values[:, np.newaxis] - nosedisappear.index.values
        distnose = distnose.astype(float)
        distnose[distnose < 0] = np.nan
        potentialClosestpos_nose = np.nanargmin(distnose, axis=1)
        closestfoundnose, closestCounts = np.unique(potentialClosestpos_nose, return_counts=True)
        if closestfoundnose.shape[0] != runstages['RunEnd'].index.values.shape[0]:
            raise ValueError('Seems to be a missing RunEnd value')
        TrialEnd = nosedisappear.iloc[potentialClosestpos_nose]

        # chunk up data for each camera between TrialStart and TrialEnd
        if len(TrialStart) != len(TrialEnd):
            raise ValueError('Different number of trial starts and ends.')

        print('Number of runs detected: %s' % len(TrialStart))
        ### !!! here put some comparison of the lengths from the excel doc when ready for it

        runs = pd.DataFrame({'TrialStart': TrialStart.index, 'TrialEnd': TrialEnd.index})

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

            #self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
            DataframeCoor_side.loc[maskside, 'Run'] = i
            DataframeCoor_front.loc[maskfront, 'Run'] = i
            DataframeCoor_overhead.loc[maskoverhead, 'Run'] = i

        DataframeCoor_side = DataframeCoor_side[DataframeCoor_side.loc(axis=1)['Run'].notnull()]
        DataframeCoor_front = DataframeCoor_front[DataframeCoor_front.loc(axis=1)['Run'].notnull()]
        DataframeCoor_overhead = DataframeCoor_overhead[DataframeCoor_overhead.loc(axis=1)['Run'].notnull()]

        DataframeCoor_side.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        DataframeCoor_front.set_index(['Run', 'FrameIdx'], append=False, inplace=True)
        DataframeCoor_overhead.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

        ###### should add in another column to show the run stages - RunStart, Tranisiton, RunEnd
        ### also find run backs (mid trial) and related to this repeat runs
        ### based on the above need to show where multiple run attempts are. presumably this is clear by, if a runback is present, only considering the second instance of the run (oe RunStart, Transition and RunEnd)

        Dataframes = {
            'side': DataframeCoor_side,
            'front': DataframeCoor_front,
            'overhead': DataframeCoor_overhead
        }

        return Dataframes
