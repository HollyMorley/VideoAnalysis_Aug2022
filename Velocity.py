import Helpers.utils as utils
from Helpers.Config import *
import Helpers.BodyCentre as BodyCentre
import Helpers.GetRuns as GetRuns
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.stats import sem
import seaborn as sns
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
    runs_lowess = list()
    runs_raw = list()
    trans_lowess = list()
    trans_raw = list()
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
            if sum(v.loc(axis=0)['RunStart'] < -30) > 10:
                # get indexes of where RunStart should be RunBack
                negmask = v.loc(axis=0)['RunStart'] < -30
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

                for vw in ['Side', 'Front', 'Overhead']:
                    data[con][mouseID][vw].loc(axis=1)['RunStage'] = RunStage
                    data[con][mouseID][vw].loc(axis=1)['FrameIdx'] = FrameIdx
                    data[con][mouseID][vw].loc(axis=1)['Run'] = Run

                    data[con][mouseID][vw].set_index(['Run', 'RunStage', 'FrameIdx'], append=False, inplace=True)

                # update v array to not include the runback
                startidx = v.index.get_level_values(level='FrameIdx')[0]
                v.drop(index=range(startidx, lastnegidx), level='FrameIdx', inplace=True)

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
                runs_lowess.append([lowessdf['x'], lowessdf['lowess']])
                runs_raw.append([xplot,yplot])
                trans_lowess.append([xplot.xs(transition_idx, level=1).values[0],lowessdf['lowess'].xs(transition_idx, level=1).values[0]])
                trans_raw.append([xplot.xs(transition_idx, level=1).values[0], yplot.xs(transition_idx, level=1).values[0]])
                del lowessdf, xplot, yplot
            except: ################################ this doesnt work as would just fill from previous run....
                runs_lowess.append([np.nan,np.nan])
                runs_raw.append([np.nan,np.nan])
                trans_lowess.append([np.nan,np.nan])
                trans_raw.append([np.nan, np.nan])

        except:
            runs_lowess.append([np.nan, np.nan])
            runs_raw.append([np.nan, np.nan])
            trans_lowess.append([np.nan, np.nan])
            trans_raw.append([np.nan, np.nan])
            print('Cant plot run %s, mouse %s' % (r, mouseID))

    vel = {
        'runs_lowess': np.array(runs_lowess,dtype=object),
        'runs_raw': np.array(runs_raw,dtype=object),
        'trans_lowess': np.array(trans_lowess,dtype=float),
        'trans_raw': np.array(trans_raw, dtype=float)
    }
    return vel


def plotSpeed_SingleMouse_AllRunsInPhase(data, con, mouseID, zeroed, view, expphase, xaxis, n=30):
    if expphase == 'Baseline':
        f = range(2, 12)
        cmap = 'BuPu'
    elif expphase == 'APA':
        f = range(12, 32)
        cmap = 'PuBu'
    elif expphase == 'Washout':
        f = range(32, 42)
        cmap = 'BuGn'
    blues = utils.Utils().get_cmap((len(f) + 5), cmap)
    plt.figure()
    markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

    windowsize = math.ceil((fps / n) / 2.) * 2
    windowsize_s = 1000 / n

    vel = getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f)

    trans = []
    for i, r in enumerate(f):
        plt.plot(vel['runs_lowess'][i,0], vel['runs_lowess'][i,1], color=blues(r-(f[0]-4)))
        if xaxis == 'x':
            plt.scatter(vel['trans'][i,0],vel['trans'][i,1], color=blues(r-(f[0]-4)))
            try:
                transmask = data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
                transition_x = np.mean(data[con][mouseID][view].loc(axis=0)[r].loc(axis=1)['TransitionR', 'x'][transmask])
                transition_xcm = transition_x * markerstuff['pxtocm']
            except: # for when no run here
                pass
                # transmask = data[con][mouseID][view].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
                # transition_x = np.mean(data[con][mouseID][view].loc(axis=1)['TransitionR', 'x'][transmask])
                # transition_xcm = transition_x * markerstuff['pxtocm']
            trans.append(transition_xcm)


    if xaxis == 'x':
        trans_mean = np.mean(trans) # not sure there is any point doing it this way - could just calculate mean from data without specifying runs
        plt.axvline(x=trans_mean, color='black', linestyle='--')
        plt.xlabel('Position on belt (cm)')
        plt.xlim(0,63)

    elif xaxis == 'time':
        plt.axvline(x=1, color='black', linestyle='--')
        plt.xticks([1], ['Transition'])

    xmin, xmax = plt.gca().get_xlim()
    plt.xlim(xmin, xmax)
    plt.title('%s\n%s\nBin duration: %s ms, Bin size: %s frames' %(expphase,mouseID,windowsize_s, windowsize))
    plt.ylabel('Velocity (cm/s)')
    plt.ylim(0,100)


def plotSpeed_SingleMouse_Means(data, con, mouseID, zeroed, view, expphase=[], xaxis='x', n=30):
    # blues = utils.Utils().get_cmap((len(f) + 5), 'PuBu')
    plt.figure()
    markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

    windowsize = math.ceil((fps / n) / 2.) * 2
    windowsize_s = 1000 / n

    cmap = 'tab20c'
    colors = utils.Utils().get_cmap(20, cmap)
    # cmap = 'plasma'
    # colors = utils.Utils().get_cmap(5, cmap)

    for e in expphase:
        if e == 'Baseline':
            f = range(2, 12)
            cnum = 12
            #cnum = 0
            linestyle = '--'
        elif e == 'APA':
            f = range(12, 32)
            cnum = 0
            #cnum = 2
            linestyle = '-'
        elif e == 'APA1':
            f = range(12, 22)
            cnum = 1
            #cnum = 1
            linestyle = '-'
        elif e == 'APA2':
            f = range(22, 32)
            cnum = 0
            #cnum = 3
            linestyle = '-'
        elif e == 'Washout':
            f = range(32, 42)
            cnum = 8
            #cnum = 4
            linestyle = '-.'

        # get speeds for runs in phases as specified by f
        vel = getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f)
        try:
            # Create a common x array with the desired number of points
            x_min = np.nanmin([np.min(run[0]) for run in vel['runs_raw']])
            x_max = np.nanmax([np.max(run[0]) for run in vel['runs_raw']])
            num_points = 600  # number of points to interpolate to
            common_x = np.linspace(x_min, x_max, num_points)
            # Loop through the runs and interpolate each one onto the common x array
            interpolated_runs = []
            for i in range(len(vel['runs_raw'])):
                try:
                    na_mask = vel['runs_raw'][i, 1].notna()
                    x = vel['runs_raw'][i, 0][na_mask]
                    y = vel['runs_raw'][i, 1][na_mask]
                    interpolated_y = np.interp(common_x, x, y)
                    interpolated_runs.append(interpolated_y)
                except:
                    interpolated_runs.append([])
                    print('Cant plot run %s of phase %s' %(i,e))
                    na_mask = np.nan
                    x = np.nan
                    y = np.nan
                    interpolated_y = np.nan
                    #interpolated_runs = np.nan
            # Calculate average
            try:
                interp_mask = [isinstance(elem, np.ndarray) for elem in interpolated_runs]
                interpolated_runs_filtered = [elem for elem, m in zip(interpolated_runs, interp_mask) if m]
                average_interpolated_runs = np.nanmean(interpolated_runs_filtered, axis=0)
                #std_interpolated_runs = np.nanstd(interpolated_runs_filtered, axis=0)
                sem_interpolated_runs = sem(interpolated_runs_filtered, axis=0)
            except:
                print('Cant calculate mean of phase %s' %(e))
                interp_mask = np.nan
                interpolated_runs_filtered = np.nan
                average_interpolated_runs = np.nan
                #std_interpolated_runs = np.nan
                sem_interpolated_runs = np.nan

            # Plot the shaded region for the standard deviation
            # upper_bound = average_interpolated_runs + std_interpolated_runs
            # lower_bound = average_interpolated_runs - std_interpolated_runs
            upper_bound = average_interpolated_runs + sem_interpolated_runs
            lower_bound = average_interpolated_runs - sem_interpolated_runs
            plt.fill_between(common_x, upper_bound, lower_bound, color=colors(cnum), linestyle=linestyle, alpha=0.2)

            # Plot the mean line
            plt.plot(common_x, average_interpolated_runs, color=colors(cnum), linestyle=linestyle, label=e)

            ## Plot mean transition point
            # transition_x_mean = np.mean(vel['trans_raw'][:, 0])
            # transition_y_mean = np.mean(vel['trans_raw'][:, 1])
            # plt.scatter(transition_x_mean, transition_y_mean, color=colors(cnum))

        except:
            print('Cant plot phase %s, possibly no runs' %e)
            x_min = np.nan
            x_max = np.nan
            common_x = np.nan

        # Plot mean transition point
        # series_indices = [i for i in range(len(vel['trans_raw'][:, 0])) if
        #                   isinstance(vel['trans_raw'][:, 0][i], pd.Series)] # get where there are transition values only
        try:
            series_indices = np.argwhere(~np.isnan(vel['trans_raw'][:, 0]))
            transition_x_mean = np.mean(vel['trans_raw'][series_indices, 0])
            #transition_y_mean = np.mean(vel['trans_raw'][series_indices, 1])
            upper_closest = common_x[common_x - transition_x_mean > 0][0]
            lower_closest = common_x[common_x - transition_x_mean < 0][-1]
            if abs(transition_x_mean - upper_closest) < abs(transition_x_mean - lower_closest):
                transition_x_mean_est = upper_closest
            else:
                transition_x_mean_est = lower_closest
            xpos = np.where(common_x == transition_x_mean_est)[0][0]
            transition_y = average_interpolated_runs[xpos]
            plt.scatter(transition_x_mean, transition_y, color=colors(cnum))
        except:
            print('nothing to plot here...')

    # calculate transition position for experiment
    transmask = data[con][mouseID][view].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
    transition_x = np.mean(data[con][mouseID][view].loc(axis=1)['TransitionR', 'x'][transmask])
    transition_xcm = transition_x * markerstuff['pxtocm']
    trans_mean = np.mean(transition_xcm)
    plt.axvline(x=trans_mean, color='black', linestyle='--')
    plt.xlabel('Position on belt (cm)')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('%s\n%s\nBin duration: %s ms, Bin size: %s frames' %(mouseID, con, windowsize_s, windowsize))
    plt.legend()
    plt.ylim(0,100)




def plotAllMice(con, data, zeroed, n=5, xaxis='x', phases=ExpPhases, exclStat=False):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nWarning: I am manually removing prep frames here!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for m, mouseID in enumerate(data[con]):
        for e in phases:
            plotSpeed_SingleMouse_AllRunsInPhase(data,con,mouseID, zeroed, view='Side', expphase=e, xaxis=xaxis, n=n, exclStat=exclStat)



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

