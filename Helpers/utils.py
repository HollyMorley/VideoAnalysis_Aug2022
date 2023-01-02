# Some useful helper functions for video analysis

from glob import glob
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from pathlib import Path
from Helpers.Config import *
import sys
import numpy as np

class Utils:
    def __init__(self):
        super().__init__()

    def Getlistofvideofiles(self, view, directory, filetype=".avi"):
        # function to get a list of video files in a directory. Current use is for bulk DLC analysis of videos with different models in each directory
        # view can be either 'side', 'overhead' or 'front'
        #vidfiles = glob("%s\\*%s%s" % (directory, scorer, filetype))
        ignore = ["labeled", "test"]
        vidfiles = [f for f in glob("%s/*%s*%s" % (directory, view, filetype)) if not any(j in f for j in ignore)]
        print(vidfiles)
        return vidfiles


    def GetlistofH5files(self, files=None, directory=None, filtered=None): #### update and change name to Getlistofanalysedfiles
        if directory is not None and files is None:
            if filtered is None:
                datafiles_side = glob("%s\\*%s*%s.h5" % (directory, 'side', scorer_side))
                datafiles_front = glob("%s\\*%s*%s.h5" % (directory, 'front', scorer_front))
                datafiles_overhead = glob("%s\\*%s*%s.h5" % (directory, 'overhead', scorer_overhead))
            if filtered is True:
                datafiles_side = glob("%s\\*%s*%s_Runs.h5" % (directory, 'side', scorer_side))
                datafiles_front = glob("%s\\*%s*%s_Runs.h5" % (directory, 'front', scorer_front))
                datafiles_overhead = glob("%s\\*%s*%s_Runs.h5" % (directory, 'overhead', scorer_overhead))

        elif files is not None and directory is None:
            datafiles_side = []
            datafiles_front = []
            datafiles_overhead = []
            for i in range(0, len(files)):
                if 'front' in files[i]:
                    front = files[i]
                    datafiles_front.append(front)
                elif 'overhead' in files[i]:
                    overhead = files[i]
                    datafiles_overhead.append(overhead)
                elif 'side' in files[i]:
                    side = files[i]
                    datafiles_side.append(side)

        if bool(datafiles_side) or bool(datafiles_front) or bool(datafiles_overhead):
            print("Files to be analysed are:\n"
                  "Side: %d files\n"
                  "%s\n"
                  "Front: %d files\n"
                  "%s\n"
                  "Overhead: %d files\n"
                  "%s" % (
                  len(datafiles_side), datafiles_side, len(datafiles_front), datafiles_front, len(datafiles_overhead),
                  datafiles_overhead))

            datafiles = {
                'Side': datafiles_side,
                'Front': datafiles_front,
                'Overhead': datafiles_overhead
            }
            return datafiles
        else:
            raise Exception("No files found, check file format.\nHint: you should be using the .h5 files\nHint: If specifying just one file, put this is list format still")

    def checkFilenamesMouseID(self, files):
        # Checks if mouse ID corresponds to correct mouse name
        if type(files) is dict:
            files = sorted({x for v in files.values() for x in v})

        for m in range(0, len(mice_ID)):
            mousefiles = [s for s in files if mice_ID[m] in s]
            match = [f for f in mousefiles if mice_name[m] in f]
            if mousefiles != match:
                mislabeled = set(mousefiles) ^ set(match)
                print('The following file is labeled incorrectly:\n%s' % mislabeled)
                print('Code will now quit. Please correct this error and re-try!')
                sys.exit()
            else:
                print('All videos labeled correctly')

    def getFilepaths(self, data):
        filenameALL = list()
        skelfilenameALL = list()
        pathALL = list()
        for df in range(0, len(data)):
            filename = Path(data[df]).stem
            skelfilename = "%s_skeleton" %filename
            path = str(Path(data[df]).parent)
            filenameALL.append(filename)
            skelfilenameALL.append(skelfilename)
            pathALL.append(path)
        return filenameALL, skelfilenameALL, pathALL

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_cmap(self, n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    #def pxtocm(self, dataframe):

    def get_exp_details(self,file):
        exp = []
        runPhases = []
        splitrunPhases = []
        indexList = []
        splitindexList = []
        Acspeed = [] # actual speed cm/s
        Pdspeed = [] # perceived speed cm/s
        condition = []
        AcBaselineSpeed = []
        AcVMTSpeed = []
        AcWashoutSpeed = []
        PdBaselineSpeed = []
        PdVMTSpeed = []
        PdWashoutSpeed = []
        VMTcon = []
        VMTtype = []
        pltlabel = []

        if '20201130' in file:
            exp = 'APACharBaseline'
            runPhases = [list(range(0, 20))]
            indexList = ['BaselineRuns']
            Acspeed = 0
            Pdspeed = 0
            condition = 'Control'
        elif '20201201' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            Acspeed = 8
            Pdspeed = 8
            condition = 'FastToSlow'
        elif '20201202' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            Acspeed = 4
            Pdspeed = 4
            condition = 'SlowToFast'
        elif '20201203' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            condition = 'SlowToFast'
            Acspeed = 16
            Pdspeed = 16
        elif '20201204' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            condition = 'FastToSlow'
            Acspeed = 4
            Pdspeed = 4
        elif '20201207' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'SlowToFast'
            Acspeed = 8
            Pdspeed = 8
        elif '20201208' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'FastToSlow'
            Acspeed = 16
            Pdspeed = 16
        elif '20201209' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'FastToSlow'
            Acspeed = 32
            Pdspeed = 32
        elif '20201210' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'SlowToFast'
            Acspeed = 32
            Pdspeed = 32
        elif '20201211' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'PerceptionTest'
            Acspeed = 16
            Pdspeed = 100
        elif '20201214' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            AcBaselineSpeed = 4
            AcVMTSpeed = 4
            AcWashoutSpeed = 4
            PdBaselineSpeed = 4
            PdVMTSpeed = 16
            PdWashoutSpeed = 4
            VMTcon = 'Slow'
            VMTtype = 'Perceived change'
            pltlabel = 'Actual = 4cm/s, Perceived = 16cm/s'
        elif '20201215' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 32
            AcVMTSpeed = 32
            AcWashoutSpeed = 32
            PdBaselineSpeed = 32
            PdVMTSpeed = 4
            PdWashoutSpeed = 32
        elif '20201216' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 16
            AcVMTSpeed = 16
            AcWashoutSpeed = 16
            PdBaselineSpeed = 16
            PdVMTSpeed = 4
            PdWashoutSpeed = 16
            VMTcon = 'Fast'
            VMTtype = 'Perceived change'
            pltlabel = 'Actual = 16cm/s, Perceived = 4cm/s'
        elif '20201217' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 4
            AcVMTSpeed = 16
            AcWashoutSpeed = 4
            PdBaselineSpeed = 4
            PdVMTSpeed = 4
            PdWashoutSpeed = 4
            VMTcon = 'Slow'
            VMTtype = 'Actual change'
            pltlabel = 'Actual = 16cm/s, Perceived = 4cm/s'
        elif '20201218' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 16
            AcVMTSpeed = 4
            AcWashoutSpeed = 16
            PdBaselineSpeed = 16
            PdVMTSpeed = 16
            PdWashoutSpeed = 16
            VMTcon = 'Fast'
            VMTtype = 'Actual change'
            pltlabel = 'Actual = 4cm/s, Perceived = 16cm/s'
        else:
            print('Somethings gone wrong, cannot find this file')

        details = {
            'exp': exp,
            'runPhases': runPhases,
            'splitrunPhases': splitrunPhases,
            'indexList': indexList,
            'splitindexList': splitindexList,
            'Acspeed': Acspeed,
            'Pdspeed': Pdspeed,
            'condition': condition,
            'AcBaselineSpeed': AcBaselineSpeed,
            'AcVMTSpeed': AcVMTSpeed,
            'AcWashoutSpeed': AcWashoutSpeed,
            'PdBaselineSpeed': PdBaselineSpeed,
            'PdVMTSpeed': PdVMTSpeed,
            'PdWashoutSpeed': PdWashoutSpeed,
            'VMT condition': VMTcon,
            'VMT type': VMTtype,
            'plt label': VMTtype
        }

        return details

