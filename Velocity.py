import Helpers.utils as utils
from Helpers.Config import *
import Helpers.BodyCentre as BodyCentre
import Helpers.GetRuns as GetRuns
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f):

    # if expphase == 'Baseline':
    #     f = range(2, 12)
    # elif expphase == 'APA':
    #     f = range(12, 32)
    # elif expphase == 'Washout':
    #     f = range(32, 42)
    #
    # #blues = utils.Utils().get_cmap((len(f) + 5), 'PuBu')
    # #plt.figure()
    # markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])
    #
    # windowsize = math.ceil((fps / n) / 2.) * 2
    # windowsize_s = 1000 / n
    runs = list()
    trans = list()
    for r in f:
        try:
            # find velocity of mouse, based on tail base
            tailmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'likelihood'].values > pcutoff
            dx = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize, center=True, min_periods=None).apply(lambda x: x[-1] - x[0])  # .shift(int(-windowsize/2))

            # create column with windowsize values, dependent on the available frames
            dxempty = np.where(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize, center=True, min_periods=None).apply(lambda x: x[-1] - x[0]).isnull().values)[0]
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

            dxcm = Dx['dx'] * markerstuff['pxtocm']
            dt = (1 / fps) * windowsize
            # dt = (1 /fps) * Dx['window']
            v = dxcm / dt

            # Check if missed any runbacks
            if sum(v.loc(axis=0)['RunStart'] < 0) > 10:
                # get indexes of where RunStart should be RunBack
                negmask = v.loc(axis=0)['RunStart'] < 0
                lastnegidx = v.loc(axis=0)['RunStart'][negmask].index[-1]
                firstidx = v.loc(axis=0)['RunStart'].index[0]

                # Change and reassign index
                RunStage = np.array(data[con][mouseID][view].index.get_level_values(level='RunStage'))
                FrameIdx = np.array(data[con][mouseID][view].index.get_level_values(level='FrameIdx'))
                Run = np.array(data[con][mouseID][view].index.get_level_values(level='Run'))

                data_subset = data[con][mouseID][view].loc(axis=0)[r, 'RunStart', range(firstidx, lastnegidx)]
                Mask = []
                for i in data[con][mouseID][view].index:
                    if i in data_subset.index:
                        Mask.append(True)
                    else:
                        Mask.append(False)
                RunStage[Mask] = 'RunBack'

                for v in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][v].loc(axis=1)['RunStage'] = RunStage
                    data[con][mouseID][v].loc(axis=1)['FrameIdx'] = FrameIdx
                    data[con][mouseID][v].loc(axis=1)['Run'] = Run

                    data[con][mouseID][v].set_index(['Run', 'RunStage', 'FrameIdx'], append=False, inplace=True)

                print('Missed runback for %s. Real run starts after frame %s (check this!!)' % (mouseID, lastnegidx))

            # find x position of tail base
            x = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask]
            x = x.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']]
            xcm = x * markerstuff['pxtocm']

            # find the frame where transition starts
            transition_idx = v.loc(axis=0)['Transition'].index[0]
            # Normalise the time/frame values to the transition (transition is 1)
            centered_transition_f = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(
                level='FrameIdx') / transition_idx
            # find the velocity at transition
            transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], transition_idx]
            # Normalise velocity values to that at transition
            centered_transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].values - transition_v.values

            if xaxis == 'x':
                xplot = xcm
            elif xaxis == 'time':
                xplot = centered_transition_f

            if zeroed == True:
                yplot = centered_transition_v
            else:
                yplot = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']]

            from statsmodels.nonparametric.smoothers_lowess import lowess
            # lowess = lowess(yplot[int(windowsize / 2):], xplot[:-int(windowsize / 2)], frac=.3)
            lowesslist = lowess(yplot, xplot, frac=.3)
            lowessdf = pd.DataFrame(lowesslist, index=yplot.index[yplot.notnull().values], columns=['x', 'lowess'])

            try:
                runs.append([lowessdf['x'], lowessdf['lowess']])
                trans.append([xplot.xs(transition_idx, level=1), lowessdf['lowess'].xs(transition_idx, level=1)])
            except: ################################ this doesnt work as would just fill from previous run....
                runs.append([np.nan,np.nan])
                trans.append([np.nan,np.nan])

            # plt.plot(xplot[:-int(windowsize / 2)], yplot[int(windowsize / 2):], color=blues(r-(f[0]-1)))
            #plt.plot(lowessdf['x'], lowessdf['lowess'], color=blues(r - (f[0] - 4)))
            #if xaxis == 'x':
            #    plt.scatter(xplot.xs(transition_idx, level=1), lowessdf['lowess'].xs(transition_idx, level=1),
            #                color=blues(r - (f[0] - 4)))

        except:
            runs.append([np.nan, np.nan])
            trans.append([np.nan, np.nan])
            print('Cant plot run %s, mouse %s' % (r, mouseID))

    vel = {
        'runs': np.array(runs,dtype=object),
        'trans': np.array(trans,dtype=object)
    }
    return vel



def plotTailSpeedSingleMouse(data, con, mouseID, zeroed, view, expphase, xaxis, n, exclStat):
    if expphase == 'Baseline':
        f = range(2, 12)
    elif expphase == 'APA':
        f = range(12, 32)
    elif expphase == 'Washout':
        f = range(32, 42)

    blues = utils.Utils().get_cmap((len(f) + 5), 'PuBu')
    plt.figure()
    markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

    windowsize = math.ceil((fps / n) / 2.) * 2
    windowsize_s = 1000 / n

    getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f)

    for r in f:
        try:
            # find velocity of mouse, based on tail base
            tailmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'likelihood'].values > pcutoff
            dx = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize, center=True, min_periods=None).apply(lambda x: x[-1] - x[0])  #.shift(int(-windowsize/2))

            # create column with windowsize values, dependent on the available frames
            dxempty = np.where(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize, center=True, min_periods=None).apply(lambda x: x[-1] - x[0]).isnull().values)[0]
            middle = np.where(np.diff(dxempty)>1)[0][0]
            startpos = dxempty[0:middle+1]
            endpos = dxempty[middle+1:len(dxempty)]
            windowstart = np.array(range(0,windowsize,2))
            windowend = np.flip(windowstart)[0:-1]
            Dx = dx.to_frame(name='dx')
            window = np.full([len(Dx)], windowsize)
            window[startpos] = windowstart
            window[endpos] = windowend
            Dx['window'] = window

            dxcm = Dx['dx'] * markerstuff['pxtocm']
            dt = (1 /fps) * windowsize
            #dt = (1 /fps) * Dx['window']
            v = dxcm / dt

            # Check if missed any runbacks
            if sum(v.loc(axis=0)['RunStart'] < 0) > 10:
                # get indexes of where RunStart should be RunBack
                negmask = v.loc(axis=0)['RunStart'] < 0
                lastnegidx = v.loc(axis=0)['RunStart'][negmask].index[-1]
                firstidx = v.loc(axis=0)['RunStart'].index[0]

                # Change and reassign index
                RunStage = np.array(data[con][mouseID][view].index.get_level_values(level='RunStage'))
                FrameIdx = np.array(data[con][mouseID][view].index.get_level_values(level='FrameIdx'))
                Run = np.array(data[con][mouseID][view].index.get_level_values(level='Run'))

                data_subset = data[con][mouseID][view].loc(axis=0)[r, 'RunStart', range(firstidx, lastnegidx)]
                Mask = []
                for i in data[con][mouseID][view].index:
                    if i in data_subset.index:
                        Mask.append(True)
                    else:
                        Mask.append(False)
                RunStage[Mask] = 'RunBack'

                for v in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][v].loc(axis=1)['RunStage'] = RunStage
                    data[con][mouseID][v].loc(axis=1)['FrameIdx'] = FrameIdx
                    data[con][mouseID][v].loc(axis=1)['Run'] = Run

                    data[con][mouseID][v].set_index(['Run', 'RunStage', 'FrameIdx'], append=False, inplace=True)

                print('Missed runback for %s. Real run starts after frame %s (check this!!)' %(mouseID, lastnegidx))


            # find x position of tail base
            x = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask]
            x = x.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']]
            xcm = x * markerstuff['pxtocm']

            # find the frame where transition starts
            transition_idx = v.loc(axis=0)['Transition'].index[0]
            # Normalise the time/frame values to the transition (transition is 1)
            centered_transition_f = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx') / transition_idx
            # find the velocity at transition
            transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd'], transition_idx]
            # Normalise velocity values to that at transition
            centered_transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].values - transition_v.values

            if xaxis == 'x':
                xplot = xcm
            elif xaxis == 'time':
                xplot = centered_transition_f

            if zeroed == True:
                yplot = centered_transition_v
            else:
                yplot = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']]

            from statsmodels.nonparametric.smoothers_lowess import lowess
            #lowess = lowess(yplot[int(windowsize / 2):], xplot[:-int(windowsize / 2)], frac=.3)
            lowesslist = lowess(yplot, xplot, frac=.3)
            lowessdf = pd.DataFrame(lowesslist,index=yplot.index[yplot.notnull().values],columns=['x','lowess'])

            #plt.plot(xplot[:-int(windowsize / 2)], yplot[int(windowsize / 2):], color=blues(r-(f[0]-1)))
            plt.plot(lowessdf['x'], lowessdf['lowess'], color=blues(r-(f[0]-4)))
            if xaxis == 'x':
                plt.scatter(xplot.xs(transition_idx,level=1), lowessdf['lowess'].xs(transition_idx,level=1), color=blues(r-(f[0]-4)))

        except:
            print('Cant plot run %s, mouse %s' %(r,mouseID))

    xmin, xmax = plt.gca().get_xlim()
    plt.xlim(xmin, xmax)
    if xaxis == 'x': ################################
        try:
            transmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
            transition_x = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'x'][transmask])
            transition_xcm = transition_x * markerstuff['pxtocm']
        except: # for when no run here
            transmask = data[con][mouseID][view].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
            transition_x = np.mean(data[con][mouseID][view].loc(axis=1)['TransitionR', 'x'][transmask])
            transition_xcm = transition_x * markerstuff['pxtocm']
        plt.axvline(x=transition_xcm, color='black', linestyle='--')
        plt.xlabel('Position on belt (cm)')
        plt.xlim(0,63)

    elif xaxis == 'time':
        plt.axvline(x=1, color='black', linestyle='--')
        plt.xticks([1], ['Transition'])
    plt.title('%s\n%s\nBin duration: %s ms, Bin size: %s frames' %(expphase,mouseID,windowsize_s, windowsize))
    plt.ylabel('Velocity (cm/s)')
    plt.ylim(0,100)




def plotAllMice(con, data, zeroed, n=5, xaxis='x', phases=ExpPhases, exclStat=False):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nWarning: I am manually removing prep frames here!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for m, mouseID in enumerate(data[con]):
        for e in phases:
            plotTailSpeedSingleMouse(data,con,mouseID, zeroed, view='Side', expphase=e, xaxis=xaxis, n=n, exclStat=exclStat)



# conditions = ['APAChar_LowHigh_Day1']
# data = Plot.Plot().GetDFs(conditions)
# plotAllMice(conditions)


# # Code to compare behind and ahead for a run
# r=31
# tailmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'likelihood'].values > pcutoff
# dx = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask].rolling(window=windowsize).apply(lambda x: x[-1] - x[0])
# dxcm = dx * markerstuff['pxtocm']
# dt = (1 /fps) * windowsize
# v = dxcm / dt
# x = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['Tail1', 'x'][tailmask]
# x = x.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']]
# # standardised = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100:] - \
# #                v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
# #standardised_transition = v.loc(axis=0)['Transition'].index[0] - v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx')[100]
# #centered_transition = standardised / standardised_transition
# transition_idx = v.loc(axis=0)['Transition'].index[0]
# centered_transition_f = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].index.get_level_values(level='FrameIdx') / transition_idx
# transition_v = v.loc(axis=0)['Transition', transition_idx]
# centered_transition_v = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].values - transition_v
# plt.figure()
# xaxis = 'x'
# rolling = 'Behind'
# if xaxis == 'x':
#     xplot = x
# elif xaxis == 'time':
#     xplot = centered_transition_f
# if zeroed == True:
#     yplot = centered_transition_v
# else:
#     yplot = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].values
# if rolling == 'Behind':
#     plt.plot(xplot, yplot, color='blue', label='Behind')
# elif rolling == 'Ahead':
#     plt.plot(xplot[:-windowsize], yplot[windowsize:], color='red', label='Ahead')
# rolling = 'Ahead'
# if xaxis == 'x':
#     xplot = x
# elif xaxis == 'time':
#     xplot = centered_transition_f
# if zeroed == True:
#     yplot = centered_transition_v
# else:
#     yplot = v.loc(axis=0)[['RunStart', 'Transition', 'RunEnd']].values
# if rolling == 'Behind':
#     plt.plot(xplot, yplot, color='blue', label='Behind')
# elif rolling == 'Ahead':
#     plt.plot(xplot[:-windowsize], yplot[windowsize:], color='red', label='Ahead')
# plt.axvline(x=x.loc(axis=0)['Transition'].values[0], color='black', linestyle='--')
# plt.legend()

