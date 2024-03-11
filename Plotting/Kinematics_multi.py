from Helpers.utils import *
from Plotting import PlottingUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem


class GetData():
    def __init__(self, conditions):
        self.conditions = conditions

    def load_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'multi_kinematics_runXstride')
        for con in self.conditions:
            data[con] = PlottingUtils.remove_prepruns(data, con)
        return data

    def get_measure(self, measure):
        data = self.load_data()
        data_frames = []
        for condition, df in data.items():
            df['Condition'] = condition  # Adding a new column for condition info
            data_frames.append(df)
        result = pd.concat(data_frames)
        result.reset_index(inplace=True)
        result.set_index(['Condition', 'MouseID', 'Run', 'Stride', 'FrameIdx', 'Buffer'], inplace=True)
        return result.loc(axis=1)[measure]

    def get_stride_data(self):
        data = PlottingUtils.load_measures_files(self.conditions, 'StrideInfo')
        for con in self.conditions:
            data[con] = PlottingUtils.remove_prepruns(data, con)
        return data


class PlotKinematics(GetData):
    def __init__(self, conditions):
        super().__init__(conditions)
        #self.stride_info = self.get_stride_data()

    # def getData(self):
    #     data = self.get_measure('traj')
    #     return data

    def plot_trajectories(self, data, plottype, bodypart, coord, speed_correct, buffer_size=0.25, step_phase=None, full_stride=True, aligned=True):
        param_name = "bodypart:%s, coord:%s, step_phase:%s, full_stride:%s, speed_correct:%s, aligned:%s, buffer_size:%s" \
                     %(bodypart, coord, step_phase, full_stride, speed_correct, aligned, buffer_size)

        if plottype == 'SingleMouse_byStride_byPhase':
            for con in self.conditions:
                for mouseID in data[con].index.get_level_values(level='MouseID').unique():
                    fig, ax = plt.subplots(4,5, figsize=(15, 10))
                    for runs in expstuff['condition_exp_runs']['APACharRuns']['Short']:
                            traj_phase = data[con].xs(param_name, level='Params', axis=1).loc(axis=0)[mouseID,runs]
                            #for s in












def main():
    LowHigh_days_conditions = ['APAChar_LowHigh_Repeats_Wash_Day1', 'APAChar_LowHigh_Repeats_Wash_Day2',
                               'APAChar_LowHigh_Repeats_Wash_Day3']
    plotting = PlotKinematics(LowHigh_days_conditions)
    for m in measure:
        data = plotting.get_measure(m)
        plotting.plot_trajectories(data)


if __name__ == '__main__':
    main()
    print("Finished saving plots!! Hope they look good (don't freak out if they don't though!) :)")