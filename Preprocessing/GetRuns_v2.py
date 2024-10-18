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
#from Helpers.Config import *
from Helpers.Config_23 import *
from pathlib import Path
from scipy.signal import find_peaks
from scipy import stats


class GetRuns:
    def __init__(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead):
        self.DataframeCoor_side = DataframeCoor_side
        self.DataframeCoor_front = DataframeCoor_front
        self.DataframeCoor_overhead = DataframeCoor_overhead

    def door_opn_cls(self):
        doormask = self.DataframeCoor_side.loc(axis=1)['Door', 'likelihood'] > 0.5
        door = self.DataframeCoor_side.loc(axis=1)['Door'][doormask]
        ymean = door['y'].mean()
        ystd = door['y'].std()
        door_close_mask = np.logical_and(door['y'] > ymean - ystd * 0.5, door['y'] < ymean + ystd * 0.5)
        door_close = door['y'][door_close_mask]
        close_chunks = utils.Utils().find_blocks(door_close.index, gap_threshold=500, block_min_size=1000)
        open_frames = close_chunks.T[1]
        close_frames = close_chunks.T[0]
        # append the final frame of the video to the close_frames list
        close_frames = np.append(close_frames, self.DataframeCoor_side.index[-1])
        close_frames = close_frames[1:]
        doors = pd.DataFrame({'open_frames': open_frames, 'close_frames': close_frames}, index=range(len(open_frames)))
        return doors

    def mouse_present(self):
        transition_x_mask = self.DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
        transition_x = self.DataframeCoor_side.loc(axis=1)['TransitionR', 'x'][transition_x_mask].mean()
        mouse_mask = np.logical_or(self.DataframeCoor_side.loc(axis=1)['Nose', 'likelihood'] > pcutoff, self.DataframeCoor_side.loc(axis=1)['Tail1', 'likelihood'] > pcutoff)
        nose = self.DataframeCoor_side.loc(axis=1)['Nose'][mouse_mask]
        tail = self.DataframeCoor_side.loc(axis=1)['Tail1'][mouse_mask]
        mouse_ran_mask = np.logical_and.reduce((nose['x'] > transition_x, tail['x'] > transition_x, nose['x'] > tail['x']))
        mouse_ran_index = nose[mouse_ran_mask].index
        return np.array(mouse_ran_index)

    def find_runs(self):
        doors = self.door_opn_cls()
        mouse_present = self.mouse_present()

        # check if the mouse is present between all the door open and close frames
        for i in range(len(doors)):
            door_open = doors['open_frames'][i]
            door_close = doors['close_frames'][i]
            mouse_present_mask = np.logical_and(mouse_present > door_open, mouse_present < door_close)
            mouse_present_frames = mouse_present[mouse_present_mask]
            if len(mouse_present_frames) == 0:
                doors.drop(i, inplace=True)
        return doors

    def check_post_runs(self, forward_chunks, forward_data, transition_x):
        post_transition_mask = np.logical_and.reduce((forward_data.loc(axis=1)['Tail1', 'x'] > transition_x,
                                                        forward_data.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                        forward_data.loc(axis=1)['Tail1', 'likelihood'] > pcutoff))
        post_transition_data = forward_data[post_transition_mask]
        first_transition = post_transition_data.index[0]
        # drop runs that occur after the first transition
        correct_chunks = [run for run in forward_chunks if run[0] < first_transition]
        return correct_chunks

    def find_runbacks(self, forward_chunks, mouse_data):
        runbacks = []
        true_run = []
        # for i, run in enumerate(forward_chunks[:-1]):
        i=0
        while i < len(forward_chunks)-1:
            run = forward_chunks[i]

            # check if mice run backwards between this run and the next
            run_data = mouse_data.loc(axis=0)[run[0]:forward_chunks[i+1][0]]
            runback_mask = np.logical_and.reduce((run_data.loc(axis=1)['Nose', 'x'] < run_data.loc(axis=1)['Tail1', 'x'],
                                                  #run_data.loc(axis=1)['Nose', 'x'] < run_data.loc(axis=1)['StartPlatR', 'x'],
                                                  run_data.loc(axis=1)['Nose', 'likelihood'] > pcutoff,
                                                  #run_data.loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff,
                                                  run_data.loc(axis=1)['Tail1', 'likelihood'] > pcutoff))
            runback_data = run_data[runback_mask]

            # check if mice step off the platform between this run and the next
            if len(runback_data) > 0:
                step_off_mask = np.logical_and.reduce((run_data.loc(axis=1)['Tail1', 'x'] < run_data.loc(axis=1)['StartPlatR', 'x'],
                                                        run_data.loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                                        run_data.loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff))
                step_off_data = run_data[step_off_mask]

                # if mice meet these conditions, add this run to the runbacks list
                if len(step_off_data) > 0:
                    runbacks.append(run)
                else:
                    true_run.append(run)
            else:
                if np.all(~(mouse_data.loc(axis=0)[run[1] + 1:forward_chunks[i+1][0] - 1].loc(axis=1)['Tail1','likelihood'] > pcutoff).values):
                    # this run is actually part of the next one and has been wrongly split
                    real_run_start = run[0]
                    real_run_end = forward_chunks[i + 1][1]
                    #forward_chunks[i] = [real_run_start, real_run_end]
                    run = (real_run_start, real_run_end)

                    # update the upcoming run to match the current run
                    forward_chunks[i + 1] = run

                elif np.all(~(mouse_data.loc(axis=0)[run[1] + 1:forward_chunks[i+1][0] - 1].loc(axis=1)['EarR','likelihood'] > pcutoff).values):
                    # this run is actually part of the next one and has been wrongly split
                    real_run_start = run[0]
                    real_run_end = forward_chunks[i + 1][1]
                    run = np.array([real_run_start, real_run_end], dtype=int)

                    # update the upcoming and current run to match the current run
                    forward_chunks[i] = run
                    forward_chunks[i + 1] = run
                else:
                    true_run.append(run)

            i += 1

        true_run.append(forward_chunks[-1])

        return runbacks, true_run


    def find_run_stages(self, frames, n=50):
        transition_x_mask = self.DataframeCoor_side.loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
        transition_x = self.DataframeCoor_side.loc(axis=1)['TransitionR', 'x'][transition_x_mask].mean()

        # get data between the door open and close frames
        data = self.DataframeCoor_side.loc(axis=0)[frames['open_frames']:frames['close_frames']]
        mouse_mask = np.logical_or(data.loc(axis=1)['EarR', 'likelihood'] > pcutoff, data.loc(axis=1)['Tail1', 'likelihood'] > pcutoff)
        mouse_data = data[mouse_mask]
        forward_mask = np.logical_and.reduce((mouse_data.loc(axis=1)['EarR', 'x'] > mouse_data.loc(axis=1)['Tail1', 'x'],
                                              mouse_data.loc(axis=1)['EarR', 'likelihood'] > pcutoff,
                                              mouse_data.loc(axis=1)['Tail1', 'likelihood'] > pcutoff))
        forward_data = mouse_data[forward_mask]
        forward_chunks = utils.Utils().find_blocks(forward_data.index, gap_threshold=10, block_min_size=15)
        if len(forward_chunks) > 1:
            forward_chunks = self.check_post_runs(forward_chunks, forward_data, transition_x)
            if len(forward_chunks) > 1:
                runbacks, forward_chunks = self.find_runbacks(forward_chunks, mouse_data)
                if len(forward_chunks) > 1:
                    # combine forward chunks into one if multiple are still left after checking for post run runs and runbacks
                    forward_chunks = [[forward_chunks[0][0], forward_chunks[-1][1]]]
            else:
                runbacks = []
        elif len(forward_chunks) == 0:
            raise ValueError('No forward chunks found for run %s' % frames.name)
        else:
            runbacks = []

        # find real run start by finding where either left or right forpaw crosses the start line just prior to the beginning of the last forward chunk
        start_x_mask = self.DataframeCoor_side.loc(axis=1)['StartPlatL', 'likelihood'] > pcutoff
        start_x = self.DataframeCoor_side.loc(axis=1)['StartPlatL', 'x'][start_x_mask].mean()
        runstart_buffered = mouse_data.loc(axis=0)[forward_chunks[-1][0] - 250:forward_chunks[-1][0]]
        L_data = runstart_buffered.loc(axis=1)['ForepawToeL'][np.logical_and(
            runstart_buffered.loc(axis=1)['ForepawToeL', 'likelihood'] > 0.99,
            runstart_buffered.loc(axis=1)['ForepawToeL', 'x'] > start_x)]
        R_data = runstart_buffered.loc(axis=1)['ForepawToeR'][np.logical_and(
            runstart_buffered.loc(axis=1)['ForepawToeR', 'likelihood'] > 0.99,
            runstart_buffered.loc(axis=1)['ForepawToeR', 'x'] > start_x)]
        r_chunks = utils.Utils().find_blocks(R_data.index, gap_threshold=1, block_min_size=5)
        l_chunks = utils.Utils().find_blocks(L_data.index, gap_threshold=1, block_min_size=5)

        # find the lowest index value between the left and right paw data
        if len(l_chunks) > 0 and len(r_chunks) > 0:
            if l_chunks[0][0] < r_chunks[0][0]:
                pawstart = l_chunks[0][0]
            else:
                pawstart = r_chunks[0][0]
        elif len(l_chunks) > 0:
            pawstart = l_chunks[0][0]
        elif len(r_chunks) > 0:
            pawstart = r_chunks[0][0]
        else:
            raise ValueError('No paw data found for run %s' % frames.name)
        #pawstart = paw_data.index[0]

        # add buffer of n frames to the end of the run
        start = pawstart
        end = forward_chunks[-1][1] + n
        run = mouse_data.loc(axis=0)[start:end]

        # create mask where either left or right front paw is past the transition
        paw_mask = np.logical_or(run.loc(axis=1)['ForepawToeR', 'likelihood'] > pcutoff, run.loc(axis=1)['ForepawToeL', 'likelihood'] > pcutoff)
        paw_data = run[paw_mask]
        paw_past_transition_mask = np.logical_or(paw_data.loc(axis=1)['ForepawToeR', 'x'] > transition_x, paw_data.loc(axis=1)['ForepawToeL', 'x'] > transition_x)
        paw_past_transition = paw_data[paw_past_transition_mask]
        first_paw_past_transition = paw_past_transition.index[0]

        # find run end
        runend_mask = np.logical_and.reduce((run.loc(axis=1)['Tail1', 'x'] > transition_x,
                                             run.loc(axis=1)['Tail1', 'likelihood'] > pcutoff,
                                             run.loc(axis=1)['Nose', 'likelihood'] < pcutoff))
        runend_data = run[runend_mask]

        # find indexes for each stage of the run
        if np.any(runbacks):
            runback_idxs = np.arange(runbacks[0][0], run.index[0])
            trialstart_idxs = np.arange(data.index[0], runbacks[0][0])
        else:
            runback_idxs = []
            trialstart_idxs = np.arange(data.index[0], run.index[0])
        runstart_idxs = np.arange(run.index[0], first_paw_past_transition)
        transition_idxs = np.arange(first_paw_past_transition, runend_data.index[0])
        runend_idxs = np.arange(runend_data.index[0], run.index[-1])
        trialend_idxs = np.arange(run.index[-1], data.index[-1])

        return trialstart_idxs, runback_idxs, runstart_idxs, transition_idxs, runend_idxs, trialend_idxs

    def filterData(self):#, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, pcutoff):
        runs = self.find_runs()
        TrialStart, RunBack, RunStart, Transition, RunEnd, TrialEnd, Run_idx, Run_ID = [], [], [], [], [], [], [], []
        for r in runs.index:
            frames = runs.loc(axis=0)[r]
            trialstart_idxs, runback_idxs, runstart_idxs, transition_idxs, runend_idxs, trialend_idxs = self.find_run_stages(frames)
            TrialStart.append(trialstart_idxs)
            RunBack.append(runback_idxs)
            RunStart.append(runstart_idxs)
            Transition.append(transition_idxs)
            RunEnd.append(runend_idxs)
            TrialEnd.append(trialend_idxs)
            Run_idx.append(np.arange(frames['open_frames'], trialend_idxs[-1] + 1))
            Run_ID.append(np.repeat(r, len(np.arange(frames['open_frames'], trialend_idxs[-1] + 1))))

        # concatenate all the indexes for each stage of the run
        TrialStart = np.concatenate(TrialStart)
        RunBack = np.concatenate(RunBack)
        RunStart = np.concatenate(RunStart)
        Transition = np.concatenate(Transition)
        RunEnd = np.concatenate(RunEnd)
        TrialEnd = np.concatenate(TrialEnd)
        Run_idx = np.concatenate(Run_idx)
        Run_ID = np.concatenate(Run_ID)

        # add columns to the dataframes to indicate the stage of the run
        for df in [self.DataframeCoor_side, self.DataframeCoor_front, self.DataframeCoor_overhead]:
            df['RunStages'] = 'None'
            df.loc[TrialStart, 'RunStages'] = 'TrialStart'
            df.loc[RunBack, 'RunStages'] = 'RunBack'
            df.loc[RunStart, 'RunStages'] = 'RunStart'
            df.loc[Transition, 'RunStages'] = 'Transition'
            df.loc[RunEnd, 'RunStages'] = 'RunEnd'
            df.loc[TrialEnd, 'RunStages'] = 'TrialEnd'

            df['Run'] = 'None'
            df.loc[Run_idx, 'Run'] = Run_ID

            # get current index as a column called 'FrameIdx'
            df['FrameIdx'] = df.index

            # make these columns into multiindex levels, where level 0 is the run number, level 1 is the stage of the run and level 2 is the frame number
            df.set_index(['Run', 'RunStages', 'FrameIdx'], inplace=True)

            df.drop('None', level='Run', inplace=True)



        print('sfaf')







class GetSingleExpData:
    def __init__(self, side_file, front_file, overhead_file):
        self.side_file = side_file
        self.front_file = front_file
        self.overhead_file = overhead_file

    def GetData(self):
        # Get (and clean up) dataframes for one mouse (/vid) for each view point
        DataframeCoor_side = pd.read_hdf(self.side_file)
        try:
            DataframeCoor_side = DataframeCoor_side.loc(axis=1)[vidstuff['scorers']['side']].copy()
        except:
            DataframeCoor_side = DataframeCoor_side.loc(axis=1)[vidstuff['scorers']['side_new']].copy()

        DataframeCoor_front = pd.read_hdf(self.front_file)
        DataframeCoor_front = DataframeCoor_front.loc(axis=1)[vidstuff['scorers']['front']].copy()

        DataframeCoor_overhead = pd.read_hdf(self.overhead_file)
        DataframeCoor_overhead = DataframeCoor_overhead.loc(axis=1)[vidstuff['scorers']['overhead']].copy()
        print("Starting analysis...")

        getruns = GetRuns(DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead)
        data = getruns.filterData()
        ## print how many runs were found

class GetALLRuns:
    def __init__(self, files=None, directory=None):
        self.files = files
        self.directory = directory


    def GetFiles(self):
        files = utils.Utils().GetlistofH5files(self.files, self.directory)  # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(files) # before proceeding, check that mouse names are correctly labeled

        for j in range(0, len(files['Side'])):  # all csv files from each cam are same length so use side for all
            getdata = GetSingleExpData(files['Side'][j], files['Front'][j], files['Overhead'][j])
            getdata.GetData()

        #### save data to csv

def main(directory):
    # Get all data
    GetALLRuns(directory=directory).GetFiles()

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    main(directory)