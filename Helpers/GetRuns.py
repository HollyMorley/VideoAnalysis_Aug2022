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

            for j in range(0, len(files['Side'])):
                # Get (and clean up) dataframes for one mouse (/vid) for each view point
                DataframeCoor_side = pd.read_hdf(files['Side'][j])
                DataframeCoor_side = DataframeCoor_side.loc(axis=1)[scorer_side].copy()

                DataframeCoor_front = pd.read_hdf(files['Front'][j])
                DataframeCoor_front = DataframeCoor_front.loc(axis=1)[scorer_front].copy()

                DataframeCoor_overhead = pd.read_hdf(files['Overhead'][j])
                DataframeCoor_overhead = DataframeCoor_overhead.loc(axis=1)[scorer_overhead].copy()
                print("Starting analysis...")

                self.filterData(DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff)

                # save reduced dataframe as a .h5 file for each mouse
                destfolder = destfolder
                newfilename = "%s_RunsAll.h5" %Path(files[j]).stem
                self.ReducedDataframeCoor.to_hdf("%s\\%s" %(destfolder, newfilename), key='RunsAll', mode='a')
                print("Reduced coordinate file saved for %s" % files[j])
                del DataframeCoor

            print("Finished extracting runs for files: \n %s" %files)

        else:
            raise Exception('Missing 1 or some of side-front-overhead file triplets. Check all 3 files have been analysed')

    def filterData(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff):
        #pcutoff=0.99 ######## TEMP - pcutoff needs to be high for structural points otherwise too messy
        doormask = np.logical_or.reduce((DataframeCoor_side.loc(axis=1)['Door', 'likelihood'] > pcutoff,
                                          DataframeCoor_front.loc(axis=1)['Door', 'likelihood'] > 0.99,
                                          DataframeCoor_overhead.loc(axis=1)['Door', 'likelihood'] > 0.99))

        TrialStart_DataframeCoor_side = DataframeCoor_side[doormask] # should be 156773 rows for all 3 views
        TrialStart_DataframeCoor_front = DataframeCoor_front[doormask]
        TrialStart_DataframeCoor_overhead = DataframeCoor_overhead[doormask]

        sidemask = TrialStart_DataframeCoor_side.loc(axis=1)['Door', 'likelihood'] > pcutoff
        frontmask = TrialStart_DataframeCoor_front.loc(axis=1)['Door', 'likelihood'] > pcutoff
        overheadmask = TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'likelihood'] > pcutoff

        # find 1st derivative of x/y values (ie the 'speed' of the door's movement frame to frame)
        d1_side = TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask].diff()/TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask].index.to_series().diff()
        d1_front = TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask].diff()/TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask].index.to_series().diff()
        d1_overhead = TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask].diff()/TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask].index.to_series().diff()

        chunk = 75 #50 # min distance from other peaks

        # for every frame, find the sum of the last 1000 frames. Find the both the +ve and -ve peaks in this array to show the frames where noisy/unchanging speed of the door changes to fast moving periods and vice versa
        highpeaks_side = find_peaks(d1_side.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]
        lowpeaks_side = find_peaks(-d1_side.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]
        highpeaks_front = find_peaks(d1_front.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]
        lowpeaks_front = find_peaks(-d1_front.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]
        highpeaks_overhead = find_peaks(d1_overhead.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]
        lowpeaks_overhead = find_peaks(-d1_overhead.rolling(chunk).sum().shift(-chunk), height=1, prominence=1, distance=1000)[0]

        # finds where the values corresponding to the above peaks fall either above or below 0.6 (data is normalised between 0 and 1) to exclude erronous data points
        closeidx_side = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[highpeaks_side][utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[highpeaks_side]<0.4]
        openidx_side = utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[lowpeaks_side][utils.Utils.NormalizeData(TrialStart_DataframeCoor_side.loc(axis=1)['Door','y'][sidemask]).iloc(axis=0)[lowpeaks_side]>0.6]
        closeidx_front = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[highpeaks_front][utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[highpeaks_front]<0.4]
        openidx_front = utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[lowpeaks_front][utils.Utils.NormalizeData(TrialStart_DataframeCoor_front.loc(axis=1)['Door','y'][frontmask]).iloc(axis=0)[lowpeaks_front]>0.6]
        closeidx_overhead = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[highpeaks_overhead][utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[highpeaks_overhead]<0.4]
        openidx_overhead = utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[lowpeaks_overhead][utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]).iloc(axis=0)[lowpeaks_overhead]>0.6]

        # transitioncrossedmask = np.logical_and.reduce((
        #     DataframeCoor_overhead.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff,
        #     DataframeCoor_overhead.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
        #     DataframeCoor_overhead.loc(axis=1)['Back1', 'likelihood'] > pcutoff,
        #     DataframeCoor_overhead.loc(axis=1)['Back6', 'likelihood'] > pcutoff,
        #     DataframeCoor_overhead.loc(axis=1)['Back12', 'likelihood'] > pcutoff,
        #     DataframeCoor_overhead.loc(axis=1)['Back12', 'x'] > DataframeCoor_overhead.loc(axis=1)['TransitionR', 'x'],
        #     DataframeCoor_overhead.loc(axis=1)['Back1', 'x'] > DataframeCoor_overhead.loc(axis=1)['Back12', 'x']
        # ))

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

        beltpositions = ['RunStart', 'Transition', 'RunEnd', 'ReturnStart', 'ReturnEnd']

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

        # plt.plot(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask].index,
        #          utils.Utils.NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door', 'x'][overheadmask]),
        #          label="overhead")
        # plt.vlines(openidx_overhead.index, ymin=0, ymax=1, colors='red')
        # plt.plot(DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunStartmask].index, [0.5] * sum(RunStartmask), 'x',
        #          color=colors(0))
        # plt.plot(DataframeCoor_side.loc(axis=1)['Nose', 'x'][Transitionmask].index, [0.5] * sum(Transitionmask), 'x',
        #          color=colors(1))
        # plt.plot(DataframeCoor_side.loc(axis=1)['Nose', 'x'][RunEndmask].index, [0.5] * sum(RunEndmask), 'x',
        #          color=colors(2))
        # plt.plot(DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnStartmask].index, [0.5] * sum(ReturnStartmask), 'x',
        #          color=colors(3))
        # plt.plot(DataframeCoor_side.loc(axis=1)['Nose', 'x'][ReturnEndmask].index, [0.5] * sum(ReturnEndmask), 'x',
        #          color=colors(4))

        # plt.plot(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask].index, NormalizeData(TrialStart_DataframeCoor_overhead.loc(axis=1)['Door','x'][overheadmask]), label="overhead")
        # plt.plot(low_peak_data.index,low_peak_data,'x')
        # plt.plot(high_peak_data.index,high_peak_data,'x')
        # plt.vlines(openidx_front.index, ymin=0,ymax=1,colors='red')
        # plt.vlines(closeidx_front.index, ymin=0,ymax=1,colors='green')

        # handmask = np.logical_and(DataframeCoor_overhead.loc(axis=1)['Hand','likelihood'] > pcutoff, DataframeCoor_overhead.loc(axis=1)['Hand','x'] < 1920/4)
        # plt.plot(DataframeCoor_overhead.loc(axis=1)['Hand','x'][handmask].index, (DataframeCoor_overhead.loc(axis=1)['Hand','x'][handmask])/max(DataframeCoor_overhead.loc(axis=1)['Hand','x'][handmask]),'xg')


        ############################################## OLD ############################################################
        ### Find values where 'Nose' is in frame (creates 1D boolean array)
        RunIdxNose = FlatDataframeCoor.progress_apply(lambda x: self.getInFrame(x, pcutoff), axis=1)

        ### Filter original data by Nose index (RunIdxNose). All data where 'Nose' not in frame is chucked out.
        self.ReducedDataframeCoor = FlatDataframeCoor[RunIdxNose]
        self.ReducedDataframeCoor = self.ReducedDataframeCoor.copy()  # THIS IS NOT IDEAL BUT CANT FIND ANOTHER SOLUTION

        ### Find the beginning and ends of runs and chunk them into their respective runs
        errorMask = np.logical_and.reduce(( # creates a mask to chuck out any wrong tracking data. Logic is that if only tail1 is present it must be mislabelled
            self.ReducedDataframeCoor.loc(axis=1)['Platform', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Tail2', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['Tail3', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RForepaw', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RHindpaw', 'likelihood'] < pcutoff,
            self.ReducedDataframeCoor.loc(axis=1)['RAnkle', 'likelihood'] < pcutoff,
        ))
        self.ReducedDataframeCoor = self.ReducedDataframeCoor.drop(self.ReducedDataframeCoor[errorMask].index)

        chunkIdxend = np.logical_or.reduce((
                                    # ideal scenario: end is when in the next frame the tail base is on the left hand side of the frame, the tail base is very far right in the frame and the nose is not visible
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1800,
                                                        self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] < pcutoff)),
                                    # alt scenario 1: end is when in the next frame the tail base is on the left hand side of the frame and the average likelihood of Nose, Shoulder, Hump and Hip jumps from low to high in the next frame. In other words, if the above logic with the tail fails, see in which frame do all the (front end) body points go from being not present to present
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(-1) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(-1)) / 4 -
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 > 0.75
                                                            )),
                                    # alt scenario 2: same as above logic but looking at two frames ahead instead of one
                                    np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(-2) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] < -1500, # finds frame before the x coordinate of 'Tail1' jumps to a much lower number
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(-2) +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(-2)) / 4 -
                                                        (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                         self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 > 0.75
                                                            ))
                                    ))

        chunkIdxstart = np.logical_or.reduce((np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500, # finds frame where x coordinate of 'Nose' jumps from very high to low number
                                                            self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) > 1800
                                                             )),
                                      np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(1) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500,
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 -
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(1) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(1)) / 4 > 0.75
                                                            )),
                                      np.logical_and.reduce((self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'].shift(2) - self.ReducedDataframeCoor.loc(axis=1)['Tail1', 'x'] > 1500,
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'] +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood']) / 4 -
                                                          (self.ReducedDataframeCoor.loc(axis=1)['Nose', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Shoulder', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hump', 'likelihood'].shift(2) +
                                                           self.ReducedDataframeCoor.loc(axis=1)['Hip', 'likelihood'].shift(2)) / 4 > 0.75
                                                            ))
                                      ))

        chunks = pd.DataFrame({'Start': chunkIdxstart, 'End': chunkIdxend}, index= self.ReducedDataframeCoor.index)  # puts both series into one dataframe
        chunks.loc(axis=1)['Start'].values[0] = True  # adds very first value to dataframe
        chunks.loc(axis=1)['End'].values[-1] = True  # adds very last value to dataframe
        StartNos = chunks.index[chunks['Start']]
        StartNos = StartNos[abs(np.roll(StartNos, -1) - StartNos) > 1000]
        EndNos = chunks.index[chunks['End']]
        EndNos = EndNos[abs(np.roll(EndNos, -1) - EndNos) > 1000]

        # remove incorrect frames if any I am aware of:
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1822.9940185546875: # MR 30-11-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.6925048828125: # MR 01-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.5736083984375: # FL 03-12-2020
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.2921142578125: # FR 03-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1883.7017822265625: # MR 07-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose','x'] == 1887.3360595703125: # FR 11-12-2020
            StartNos = StartNos.delete(4)
            EndNos = EndNos.delete(4)
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1887.249755859375: # MR 11-12-2020
            StartNos = StartNos.delete([25,26])
            EndNos = EndNos.delete([25,26])
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.79248046875: #FL 14-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1901.6219482421875: # FLR 15-12-2020
            StartNos = StartNos[0:-1]
            EndNos = EndNos[0:-1]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1885.9906005859375: # FL 16-12-2020
            StartNos = StartNos.delete([11, 27])
            EndNos = EndNos.delete([11, 27])
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1883.7584228515625: # FR 16-12-2020
            StartNos = StartNos[0:-2]
            EndNos = EndNos[0:-2]
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1888.09228515625: # FLR 17-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1888.1256103515625: # FR 17-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.1680908203125: # FL 18-12-2020
            StartNos = StartNos.delete(25)
            EndNos = EndNos.delete(25)
        if FlatDataframeCoor.iloc[-1].loc['Nose', 'x'] == 1886.28564453125: # FR 18-12-2020
            StartNos = StartNos.delete([25,26])
            EndNos = EndNos.delete([25,26])

        frameNos = pd.DataFrame({'Start': StartNos, 'End': EndNos})  # creates dataframe of just the frame numbers for the starts and stops of runs
        print('Number of runs detected: %s' % len(frameNos))

        #Run = np.zeros([len(self.ReducedDataframeCoor)])
        Run = np.full([len(self.ReducedDataframeCoor)], np.nan)
        self.ReducedDataframeCoor.loc(axis=1)['Run'] = Run
        FrameIdx = self.ReducedDataframeCoor.index
        self.ReducedDataframeCoor.loc(axis=1)['FrameIdx'] = FrameIdx
        for i in range(0, len(frameNos)):
            mask = np.logical_and(self.ReducedDataframeCoor.index >= frameNos.loc(axis=1)['Start'][i],
                                  self.ReducedDataframeCoor.index <= frameNos.loc(axis=1)['End'][i])
            #self.ReducedDataframeCoor.loc(axis=1)['Run'][mask] = i
            self.ReducedDataframeCoor.loc[mask, 'Run'] = i
        self.ReducedDataframeCoor = self.ReducedDataframeCoor[self.ReducedDataframeCoor.loc(axis=1)['Run'].notnull()]
        self.ReducedDataframeCoor.set_index(['Run', 'FrameIdx'], append=False, inplace=True)

        self.ReducedDataframeCoor = self.findRunStart(self.ReducedDataframeCoor)

        return self.ReducedDataframeCoor