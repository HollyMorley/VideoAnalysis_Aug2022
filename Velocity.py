import Plot
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
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import shapiro, levene
from pingouin import rm_anova, pairwise_ttests
from statsmodels.nonparametric.smoothers_lowess import lowess

class Velocity:
    def __init__(self): # MouseData is input ie list of h5 files
        super().__init__()

    def backspeed(self, data, con, mouseID, view, r):

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


    def getVelocityInfo(self, data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f):

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


    def plotSpeed_SingleMouse_AllRunsInPhase(self, data, con, mouseID, zeroed, view, expphase, xaxis, n=30):
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

        vel = self.getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f)

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


    def plotSpeed_SingleMouse_Means(self, data, con, mouseID, zeroed, view, expphase=[], xaxis='x', n=30, plotfig=False):
        if plotfig == True:
            sns.set(style='white')
            #sns.set_palette('icefire',20)
            colors = sns.color_palette(sns.diverging_palette(160,230,s=80,l=80,n=12,center='dark'))
            fig, ax = plt.subplots()

        markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID][view])

        windowsize = math.ceil((fps / n) / 2.) * 2
        windowsize_s = 1000 / n

        for e in expphase:
            if e == 'Baseline':
                f = range(2, 12)
                #cnum = 12
                cnum = 0
                linestyle = '--'
            elif e == 'APA':
                f = range(12, 32)
                #cnum = 0
                cnum = 1
                linestyle = '-'
            elif e == 'APA1':
                f = range(12, 22)
                #cnum = 1
                cnum = 2
                linestyle = '-'
            elif e == 'APA2':
                f = range(22, 32)
                #cnum = 0
                cnum = 10
                linestyle = '-'
            elif e == 'Washout':
                f = range(32, 42)
                #cnum = 8
                cnum = 11
                linestyle = '-.'

            # get speeds for runs in phases as specified by f
            vel = self.getVelocityInfo(data, con, mouseID, zeroed, view, xaxis, windowsize, markerstuff, f)
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
                upper_bound = average_interpolated_runs + sem_interpolated_runs
                lower_bound = average_interpolated_runs - sem_interpolated_runs
                if plotfig == True:
                    ax.fill_between(common_x, upper_bound, lower_bound, color=colors[cnum], linestyle=linestyle, alpha=0.2)
                    # Plot the mean line
                    ax.plot(common_x, average_interpolated_runs, color=colors[cnum], linestyle=linestyle, label=e)

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
                if plotfig == True:
                    ax.scatter(transition_x_mean, transition_y, color=colors[cnum])
            except:
                print('nothing to plot here...')

        # calculate transition position for experiment
        transmask = data[con][mouseID][view].loc(axis=1)['TransitionR', 'likelihood'] > pcutoff
        transition_x = np.mean(data[con][mouseID][view].loc(axis=1)['TransitionR', 'x'][transmask])
        transition_xcm = transition_x * markerstuff['pxtocm']
        trans_mean = np.mean(transition_xcm)

        if plotfig == True:
            ax.axvline(x=trans_mean, color='black', linestyle='--')

            # Add labels and title
            plt.xlabel('Position on belt (cm)')
            plt.ylabel('Velocity (cm/s)')
            plt.title('%s\n%s\nBin duration: %s ms, Bin size: %s frames' %(mouseID, con, windowsize_s, windowsize))
            plt.legend()
            plt.ylim(0,100)
        elif plotfig == False:
            return common_x, upper_bound, lower_bound, average_interpolated_runs, transition_x_mean_est, transition_y, trans_mean


    def plotSpeed_SingleMouse_Means_DayComp(self, data, mouseID, view, conditions=['APAChar_LowHigh_Repeats_Wash_Day1','APAChar_LowHigh_Repeats_Wash_Day2','APAChar_LowHigh_Repeats_Wash_Day3'], expphase=[]):
        sns.set(style='white')
        colors = sns.color_palette("ch:start=.2,rot=-.3", len(conditions)+1)
        fig, ax = plt.subplots()

        for e in expphase:
            if e == 'Baseline':
                f = range(2, 12)
            elif e == 'APA':
                f = range(12, 32)
            elif e == 'APA1':
                f = range(12, 22)
            elif e == 'APA2':
                f = range(22, 32)
            elif e == 'Washout':
                f = range(32, 42)

            # Get velocity data for single mouse in single condition (generally day1, 2 and 3)
            Veldata = []
            for con in conditions:
                veldata = self.plotSpeed_SingleMouse_Means(data=data, con=con, mouseID=mouseID, zeroed=False, view=view, expphase=[e], plotfig=False)
                Veldata.append(veldata)
            Veldata = np.array(Veldata, dtype=object)
            for d in reversed(range(0,len(Veldata))):
                common_x, upper_bound, lower_bound, average_interpolated_runs, transition_x_mean, transition_y, trans_mean = Veldata[d]
                # Plot shaded error region (se)
                ax.fill_between(common_x, upper_bound, lower_bound, color=colors[d+1], linestyle='-', alpha=0.2)
                # Plot the mean line
                ax.plot(common_x, average_interpolated_runs, color=colors[d+1], linestyle='-', label=conditions[d].split('_')[-1])
                # plot transition time point
                ax.scatter(transition_x_mean, transition_y, color=colors[d+1])
            ax.axvline(x=trans_mean, color='black', linestyle='--')

            # Add labels and title
            plt.xlabel('Position on belt (cm)')
            plt.ylabel('Velocity (cm/s)')
            plt.title("%s\n%s" %(mouseID, expphase[0]))
            plt.legend()
            plt.ylim(0, 100)
            fig.set_size_inches(10.44,  4.43, forward=True)
            plt.savefig(r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\April23_roughstuff\Repeats\Velocity\%s\%s_%s.png" %(expphase[0],mouseID,expphase[0]), bbox_inches='tight')



    def plotSpeed_AllMice_Means_DayComp(self, name, conditions=['APAChar_LowHigh_Repeats_Wash_Day1','APAChar_LowHigh_Repeats_Wash_Day2','APAChar_LowHigh_Repeats_Wash_Day3'], view='Side', expphase=[], plotfig=False):
        if plotfig == True:
            sns.set(style='white')
            colorrange = len(conditions) + 1
            colors = sns.color_palette("ch:start=.2,rot=-.3", colorrange)
            linestyles = ['-', '--', '-.', ':']

        data = Plot.Plot().GetDFs(conditions)

        for e in expphase:
            if plotfig == True:
                fig, ax = plt.subplots()
            if e == 'Baseline':
                f = range(2, 12)
                cnum = 0
            elif e == 'APA':
                f = range(12, 32)
                cnum = 1
            elif e == 'APA1':
                f = range(12, 22)
                cnum = 2
            elif e == 'APA2':
                f = range(22, 32)
                cnum = 10
            elif e == 'Washout':
                f = range(32, 42)
                cnum = 11

            Stats = []
            Veldata = []
            for con in conditions:
                allmice = []
                for m, mouseID in enumerate(data[con]):
                    try:
                        veldata = self.plotSpeed_SingleMouse_Means(data=data, con=con, mouseID=mouseID, zeroed=False, view=view, expphase=[e], plotfig=False)
                        veldata = list([veldata[0], veldata[3], veldata[4], veldata[5], veldata[6]])
                        allmice.append(veldata)
                    except:
                        allmice.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                        print('Missing data for mouse: %s, in condition: %s, for phase: %s' %(mouseID, con, e))
                allmice = np.array(allmice, dtype=object)

                try:
                    x_min = np.nanmin([np.min(mouse) for mouse in allmice[:, 0]])
                    x_max = np.nanmin([np.max(mouse) for mouse in allmice[:, 0]])
                    num_points = 600
                    common_x = np.linspace(x_min, x_max, num_points)

                    interpolated_runs = []
                    for i in range(len(allmice[:, 0])):
                        try:
                            na_mask = ~np.isnan(allmice[i, 0])
                            x = allmice[i, 0][na_mask]
                            y = allmice[i, 1][na_mask]
                            interpolated_y = np.interp(common_x, x, y)
                            interpolated_runs.append(interpolated_y)
                        except:
                            interpolated_runs.append([])
                            print('Cant plot mouse %s for phase %s' % (mouseID, e))
                            na_mask = np.nan
                            x = np.nan
                            y = np.nan
                            interpolated_y = np.nan

                    # Calculate average
                    try:
                        interp_mask = [isinstance(elem, np.ndarray) for elem in interpolated_runs]
                        interpolated_runs_filtered = [elem for elem, m in zip(interpolated_runs, interp_mask) if m]
                        average_interpolated_runs = np.nanmean(interpolated_runs_filtered, axis=0)
                        # std_interpolated_runs = np.nanstd(interpolated_runs_filtered, axis=0)
                        sem_interpolated_runs = sem(interpolated_runs_filtered, axis=0)
                    except:
                        print('Cant calculate mean of phase %s' % (e))
                        interp_mask = np.nan
                        interpolated_runs_filtered = np.nan
                        average_interpolated_runs = np.nan
                        # std_interpolated_runs = np.nan
                        sem_interpolated_runs = np.nan

                    # Plot the shaded region for the standard deviation
                    upper_bound = average_interpolated_runs + sem_interpolated_runs
                    lower_bound = average_interpolated_runs - sem_interpolated_runs

                except:
                    print('Cant plot phase %s, possibly no runs' % e)
                    x_min = np.nan
                    x_max = np.nan
                    common_x = np.nan

                try:
                    transition_x_mean = np.nanmean(allmice[:,2])
                    upper_closest = common_x[common_x - transition_x_mean > 0][0]
                    lower_closest = common_x[common_x - transition_x_mean < 0][-1]
                    if abs(transition_x_mean - upper_closest) < abs(transition_x_mean - lower_closest):
                        transition_x_mean_est = upper_closest
                    else:
                        transition_x_mean_est = lower_closest
                    xpos = np.where(common_x == transition_x_mean_est)[0][0]
                    transition_y = average_interpolated_runs[xpos]
                except:
                    print('cant plot transition frame for condition: %s, phase: %s' %(con, e))

                # calculate transition position for experiment
                trans_mean = np.mean(allmice[:, 4])

                Stats.append(np.array(interpolated_runs_filtered)[:,xpos]) # getting all values for the 'transition' point for each condition
                Veldata.append([common_x, upper_bound, lower_bound, average_interpolated_runs, transition_x_mean_est, transition_y, trans_mean])

            if plotfig == False:
                # Create a DataFrame with the data in long format
                df = pd.DataFrame(np.array(Stats).T, columns=conditions)
                df['mouse'] = data[con].keys()
                df = pd.melt(df, id_vars=['mouse'], var_name='condition', value_name='speed')

                print('*************************************************************************************\n'
                      'The following are results for the %s phase:\n'
                      '*************************************************************************************' %e)

                # Check normality assumption using Shapiro-Wilk test
                for cond in df['condition'].unique():
                    print(f"Shapiro-Wilk test for {cond}:")
                    sw_stat, sw_p = shapiro(df.loc[df['condition'] == cond, 'speed'])
                    print(f"Statistics={sw_stat:.4f}, p={sw_p:.4f}")
                    alpha = 0.05
                    if sw_p > alpha:
                        print(f"{cond} is normally distributed (fail to reject H0)")
                    else:
                        print(f"{cond} is not normally distributed (reject H0)")

                # Repeated measures ANOVA
                aovrm = AnovaRM(df, 'speed', 'mouse', within=['condition'])
                res = aovrm.fit()
                print(res)

                # Post hoc tests using Tukey's HSD
                from statsmodels.stats.multicomp import MultiComparison
                mc = MultiComparison(df['speed'], df['condition'])
                posthoc = mc.tukeyhsd()
                print(posthoc)


                # # Run the repeated measures ANOVA
                # rm = AnovaRM(statsdf, depvar='score', subject='subject', within=['condition'])
                # res = rm.fit()
                #
                # # Print the ANOVA table
                # print(res.summary())
                #
                # # Perform post-hoc tests
                # mc = MultiComparison(statsdf['score'], statsdf['condition'])
                # result = mc.tukeyhsd()


            if plotfig == True:
                for d in reversed(range(0, len(Veldata))):
                    common_x, upper_bound, lower_bound, average_interpolated_runs, transition_x_mean_est, transition_y, trans_mean = \
                        Veldata[d]
                    # Plot shaded error region (se)
                    ax.fill_between(common_x, upper_bound, lower_bound, color=colors[d + 1], linestyle=linestyles[d], alpha=0.2)
                    # Plot the mean line
                    ax.plot(common_x, average_interpolated_runs, color=colors[d + 1], linestyle=linestyles[d],
                            label=conditions[d].split('_')[-1])
                    # plot transition time point
                    ax.scatter(transition_x_mean_est, transition_y, color=colors[d + 1])
                Varray = np.array(Veldata, dtype=object)
                trans_mean_mean = np.nanmean(Varray[:,6])
                ax.axvline(x=trans_mean_mean, color='black', linestyle='--')

                # Add labels and title
                plt.xlabel('Position on belt (cm)')
                plt.ylabel('Velocity (cm/s)')
                plt.title("%s" % (e))
                plt.legend()
                plt.ylim(0, 100)
                fig.set_size_inches(10.44, 4.43, forward=True)
                plt.savefig(r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\April23_roughstuff\Repeats\Velocity\%s_%s.png" % (name, e), bbox_inches='tight')


        # for con in conditions:
        #     for m, mouseID in enumerate(data[con]):
        #         for e in expphase:
        #             veldata = self.plotSpeed_SingleMouse_Means(data[con][mouseID][view], con=con, mouseID=mouseID,zeroed=False,view=view,expphase=e,plotfig=False)


    def plotAllMice(self, con, data, zeroed, n=5, xaxis='x', phases=ExpPhases, exclStat=False):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nWarning: I am manually removing prep frames here!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        for m, mouseID in enumerate(data[con]):
            for e in phases:
                self.plotSpeed_SingleMouse_AllRunsInPhase(data,con,mouseID, zeroed, view='Side', expphase=e, xaxis=xaxis, n=n, exclStat=exclStat)



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

