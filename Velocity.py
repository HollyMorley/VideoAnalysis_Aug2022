import Helpers.utils as utils
import Plot
from Helpers.Config import *
import Helpers.BodyCentre as BodyCentre
import Helpers.GetRuns as GetRuns
import matplotlib.pyplot as plt
import numpy as np


def backspeed(data, con, mouseID, view, r):

    #######################################
    backlabels = ['Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12']
    allBackmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[backlabels, 'likelihood'] > pcutoff
    allBackx = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[backlabels, 'x']
    allBacky = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)[backlabels, 'y']
    allBackx_confx = allBackx.where(allBackmask.values)
    allBacky_confy = allBacky.where(allBackmask.values)

    endsmask = allBackx_confx.loc(axis=1)[['Back1','Back12']].notna().all(axis=1)

    backy = allBacky_confy[endsmask]
    backx = allBackx_confx[endsmask]

    bodycentre = BodyCentre.estimate_body_center(backy, backx)

    # standardised = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx') - v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
    # standardised_transition = v.loc(axis=0)['Transition'].index[0] - v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
    # centered_transition = standardised/standardised_transition

    # plt.figure()
    # plt.scatter(allBackx[endsmask].iloc[0], allBacky[endsmask].iloc[0])
    # plt.plot(allBackx_confx[endsmask].iloc[0], allBackx_confx[endsmask].iloc[0]*m[0] + b[0])


    # backmean = allBackx_confx.mean(axis=1)
    # plt.plot(backmean.index.get_level_values(level='FrameIdx'),v.values)




#### for plotting all runs (CHECK EXP PHASE)
def plotTailSpeedSingleMouse(data, con, mouseID, zeroed, view, expphase):
    #print('Warning: I am manually removing prep frames here!!')
    lastrun = data[con][mouseID][view].index.get_level_values(level='Run').unique()[-1]

    if expphase == 'Baseline':
        f = range(2,12)
    elif expphase == 'APA':
        f = range(12,32)
    elif expphase == 'Washout':
        end = lastrun + 1
        f = range(32,42)

    blues = utils.Utils().get_cmap((len(f)+1), 'PuBu')
    plt.figure()
    markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

    n = 4
    windowsize = round(fps / n)
    windowsize_s = 1000 / n

    for r in f:
        try:
            tailmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'likelihood'].values > pcutoff
            dx = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize).apply(lambda x: x[-1] - x[0])
            dxcm = dx * markerstuff['pxtocm']
            dt = (1 /fps) * windowsize
            v = dxcm / dt

            # standardised = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100:] - \
            #                v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
            #standardised_transition = v.loc(axis=0)['Transition'].index[0] - v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
            #centered_transition = standardised / standardised_transition
            transition_idx = v.loc(axis=0)['Transition'].index[0]
            centered_transition_f = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100:] / transition_idx
            transition_v = v.loc(axis=0)['Transition', transition_idx]
            centered_transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']][100:].values - transition_v

            if zeroed == True:
                plt.plot(centered_transition_f, centered_transition_v, color=blues(r-(f[0]-1)))
            else:
                plt.plot(centered_transition_f, v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']][100:].values, color=blues(r-(f[0]-1)))
        except:
            print('Cant plot, probably because no transition detected for run %s, mouse %s' %(r,mouseID))

    xmin, xmax = plt.gca().get_xlim()
    plt.xlim(xmin, xmax)
    plt.axvline(x=1, color='black', linestyle='--')
    plt.title('%s\n%s\nWindow size: %s ms' %(expphase,mouseID,windowsize_s))
    plt.ylabel('Velocity (cm/s)')
    plt.xticks([1], ['Transition'])

def plotAllMice(con, data, zeroed, phases=ExpPhases):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nWarning: I am manually removing prep frames here!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for m, mouseID in enumerate(data[con]):
        for e in phases:
            plotTailSpeedSingleMouse(data,con,mouseID, zeroed, view='Side', expphase=e)



# conditions = ['APAChar_LowHigh_Day1']
# data = Plot.Plot().GetDFs(conditions)
# plotAllMice(conditions)


