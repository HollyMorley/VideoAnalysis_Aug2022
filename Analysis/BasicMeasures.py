from Helpers.Config_23 import *
import numpy as np
from scipy.stats import skew, shapiro, levene


class calculate_body():
    def __init__(self, conditions):
        self.conditions = conditions

    def BodyLength(self, df, r, phase, frame_window):
        runphase = 'RunStart' if phase == 'apa' or phase == 'pre' else 'Transition'

        back1_mask = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back1', 'likelihood'] > pcutoff
        back12_mask = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back12', 'likelihood'] > pcutoff

        back1 = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back1', 'x'][back1_mask]
        back12 = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['Back12', 'x'][back12_mask]

        body_length_mean = np.mean(back1 - back12)
        body_length_std = np.std(back1 - back12)
        body_length_sem = np.std(back1 - back12)/np.sqrt(len(back1))
        body_length_cv = body_length_std / body_length_mean

        results = {
            'mean': body_length_mean,
            'std': body_length_std,
            'sem': body_length_sem,
            'cv': body_length_cv
        }

        return results

    def BackHeight(self, df, r, phase, frame_window, calculation):
        runphase = 'RunStart' if phase == 'apa' or phase == 'pre' else 'Transition'

        back_mask = (df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)[['Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12'], 'likelihood'] > pcutoff).all(axis=1)
        start_plat_R_mask = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['StartPlatR', 'likelihood'] > pcutoff
        transition_R_mask = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff

        start_plat_R_mean = np.mean(df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['StartPlatR', 'y'][start_plat_R_mask])
        transition_R_mean = np.mean(df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)['TransitionR', 'y'][transition_R_mask])

        belt_level = np.mean([start_plat_R_mean,transition_R_mean])
        back_y = df.loc(axis=0)[r, runphase, frame_window].loc(axis=1)[['Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12'], 'y'][back_mask]
        back_y_heights = belt_level - back_y

        if calculation == 'skew':
            result = skew(back_y_heights,axis=1)
        elif calculation == 'height':
            result =

        result_mean = np.mean(result)
