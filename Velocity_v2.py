import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import shapiro, levene
import pingouin as pg

import Helpers.BodyCentre as BodyCentre
import Helpers.GetRuns as GetRuns
from Helpers.Config_23 import *
import Helpers.utils as utils
import Plot

class Velocity:
    def __init__(self, data, con, mouseID, view): # MouseData is input ie list of h5 files
        self.data = data
        self.con = con
        self.mouseID = mouseID

    def getVelocity_specific_limb(self, limb, view, r, windowsize, markerstuff, xy):
        """
        Find velocity of mouse, based on available frames where tail base (side view) is visible
        :param limb:
        :param r:
        :param windowsize:
        :param markerstuff:
        :param xy:
        :return:
        """
        # Define relevant DLC tracking data
        Tail1_likli = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Tail1', 'likelihood'].values
        Tail2_likli = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].loc(axis=1)['Tail2', 'likelihood'].values
        limb_likli = self.data[self.con][self.mouseID][view].loc(axis=0)[r].loc(axis=1)[limb, 'likelihood'].values
        limb_xy = self.data[self.con][self.mouseID][view].loc(axis=0)[r].loc(axis=1)[limb, xy]
        if xy == 'x':
            limb_xy_ref = self.data[self.con][self.mouseID][view].loc(axis=0)[r].loc(axis=1)[limb, 'y']
        elif xy == 'y':
            limb_xy_ref = self.data[self.con][self.mouseID][view].loc(axis=0)[r].loc(axis=1)[limb, 'x']

        # create mask where all points of interest are above pcutoff (different depending on cam view)
        if limb == 'ForepawToeL':
            if view == 'Side':
                mask = np.logical_and(Tail1_likli > pcutoff, Tail2_likli > pcutoff)
            elif view == 'Front':
                mask = np.logical_and(limb_likli > pcutoff, Tail1_likli > pcutoff, Tail2_likli > pcutoff)
        else:
            mask = np.logical_and.reduce((limb_likli > pcutoff, Tail1_likli > pcutoff, Tail2_likli > pcutoff))

        # Find difference in x or y between frames for limb across a window (of size windowsize) on a rolling basis
        dx = limb_xy[mask].rolling(window=windowsize,center=True,min_periods=None).apply(lambda x: x[-1] - x[0])

        # Calculate difference in x or y across window at start and end of df where there were not enough values for window of size windowsize
        dxempty = np.where(limb_xy[mask].rolling(window=windowsize,center=True,min_periods=None).apply(lambda x: x[-1] - x[0]).isnull().values)[0]
        middle = np.where(np.diff(dxempty) > 1)[0][0]
        startpos = dxempty[0:middle + 1]
        endpos = dxempty[middle + 1:len(dxempty)]
        windowstart = np.array(range(0, windowsize, 2))
        windowend = np.flip(windowstart)[0:-1]
        Dx = dx.to_frame(name='dx')
        window = np.full([len(Dx)], windowsize)
        window[startpos] = windowstart
        window[endpos] = windowend
        Dx['window'] = window

        # calculate speed
        dxcm = Dx['dx'] * markerstuff['pxtocm']
        dt = (1 / fps) * windowsize
        v = dxcm / dt

        return v