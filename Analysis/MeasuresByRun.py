import pandas as pd
import numpy as np

from Helpers.Config_23 import *

class CalculateMeasuresByRun():
    def __init__(self, XYZw, con, mouseID, r, stepping_limb):
        self.XYZw, self.con, self.mouseID, self.r, self.stepping_limb = XYZw, con, mouseID, r, stepping_limb

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[con][mouseID].loc(axis=0)[r]

    def wait_time(self):
        indexes = self.data_chunk.index.get_level_values('RunStage').unique()
        if 'TrialStart' in indexes:
            trial_start_idx = self.data_chunk.loc(axis=0)['TrialStart'].index[0]
            run_start_idx = self.data_chunk.loc(axis=0)['TrialStart'].index[-1]
            duration_idx = run_start_idx - trial_start_idx
            return duration_idx/fps
        else:
            return 0

    def num_rbs(self, gap_thresh=30):
        if np.any(self.data_chunk.index.get_level_values(level='RunStage') == 'RunBack'):
            rb_chunk = self.data_chunk[self.data_chunk.index.get_level_values(level='RunStage') == 'RunBack']
            nose_tail = rb_chunk.loc(axis=1)['Nose', 'x'] - rb_chunk.loc(axis=1)['Tail1', 'x']
            nose_tail_bkwd = nose_tail[nose_tail < 0]
            num = sum(np.diff(nose_tail_bkwd.index.get_level_values(level='FrameIdx')) > gap_thresh)
            return num + 1
        else:
            return 0

    def start_paw_pref(self): # todo update this when change scope of exp chunk in loco analysis
        xr = self.data_chunk.loc(axis=1)['ForepawToeR','x'].droplevel(level='RunStage')
        xl = self.data_chunk.loc(axis=1)['ForepawToeL','x'].droplevel(level='RunStage')
        xr_start = xr.index[xr > 0][0]
        xl_start = xl.index[xl > 0][0]
        if xr_start < xl_start:
            return micestuff['LR']['ForepawToeR']
        else:
            return micestuff['LR']['ForepawToeL']

    def trans_paw_pref(self):
        if self.stepping_limb == 'ForepawToeL':
            return micestuff['LR']['ForepawToeL']
        elif self.stepping_limb == 'ForepawToeR':
            return micestuff['LR']['ForepawToeR']

    def start_to_trans_paw_matching(self):
        if self.start_paw_pref() == self.trans_paw_pref():
            return True
        else:
            return False

    def run(self):
        column_names = ['wait_time','num_rbs','start_paw_pref','trans_paw_pref','start_to_trans_paw_matching']
        index_names = pd.MultiIndex.from_tuples([(self.mouseID,self.r)], names=['MouseID','Run'])
        df = pd.DataFrame(index=index_names, columns=column_names)
        data = np.array([self.wait_time(),self.num_rbs(),self.start_paw_pref(),self.trans_paw_pref(),self.start_to_trans_paw_matching()])
        df.loc[self.mouseID, self.r] = data
        return df