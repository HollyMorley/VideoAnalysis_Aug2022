from Helpers.Config_23 import *
from Helpers import Structural_calculations
from Helpers import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

class SaveMappedFiles():
    """
    Gathers all experimental directories (or specified directories based on experiment types/conditions) and runs
    mapping code to save new real world coordinate files within their relevant directories.
    """
    def __init__(self, all, condition, exptype, wash, day):
        self.all, self.condition, self.exptype, self.wash, self.day = all, condition, exptype, wash, day

    def get_paths(self):
        if all == True:
            dir_selection = paths['filtereddata_folder']
            # get all subdirectories
            data_subdirs = []
            for root, subdirs, files in os.walk(dir_selection):
                # if not root.endswith(('temp_bin', 'consistent_speed_up')):
                if not subdirs:
                    if not root.endswith(('temp_bin', 'consistent_speed_up')):
                        data_subdirs.append(root)
                    else:
                        snip = root.split('\\')
                        new = '\\'.join(snip[:-1])
                        data_subdirs.append(new)
            data_subdirs = np.unique(data_subdirs)
        else:
            dir_selection = os.path.join(paths['filtereddata_folder'], self.condition, self.exptype, self.wash,
                                         self.day)
            # get all subdirectories
            data_subdirs = []
            for root, subdirs, files in os.walk(dir_selection):
                if not any(
                        sbdr in ['Day1', 'Day2', 'Day3', 'Wash', 'NoWash', 'Repeats', 'Extended', 'APAChar_LowHigh'] for
                        sbdr in subdirs):  # exclude parent dirs
                    if not root.endswith(('temp_bin', 'consistent_speed_up')):
                        data_subdirs.append(root)
        data_subdirs.sort()

        return data_subdirs

    def batch_save_mapped_files(self):
        dirs = self.get_paths()
        for d in tqdm(dirs):
            condition = '_'.join(list(filter(lambda x: len(x) > 0, d.split('\\')))[-4:])
            MapCon = MapSingleConditionFiles(condition)
            print('\n-----------------------------------------------------------------------------------\n'
                  'Calculating and saving real world coordinates for:\n'
                  '-----------------------------------------------------------------------------------\n %s' %d)
            MapCon.save_XYZw()


class MapSingleConditionFiles():
    """
    Compiles and saves new real world coordinate files for all mice files in a single condition. Files are stored as a
    dictionary of dataframes for every mouse and saved as pickle files
    """
    def __init__(self, conditions):
        self.conditions = conditions
        self.data = utils.Utils().GetDFs([conditions], reindexed_loco=True)

    def get_con_components(self):
        components = self.conditions.split('_')
        condition = '_'.join(components[:2])
        exptype = components[2]
        wash = components[3]
        day = components[4]
        return condition, exptype, wash, day

    def map_all_mice(self, con):
        mice = list(self.data[con].keys())
        mice_XYZw = dict.fromkeys(mice)
        for midx, mouseID in enumerate(mice):
            Map = MapSingleMouseFile(self.data, con, mouseID)
            XYZw = Map.get_real_xyz()
            mice_XYZw[mouseID] = XYZw
        return mice_XYZw

    # def map_all_conditions(self):
    #     con_XYZw = dict.fromkeys([self.conditions])
    #     for con in self.data.keys():
    #         XYZw = self.map_all_mice(con)
    #         con_XYZw[con] = XYZw
    #     return con_XYZw

    def save_XYZw(self):
        # XYZw = self.map_all_conditions()
        XYZw = self.map_all_mice(self.conditions)
        condition, exptype, wash, day = self.get_con_components()
        path = paths['filtereddata_folder']
        with open(r'%s\\%s\\%s\\%s\\%s\\allmice_%s_XYZw.pickle' %(path,condition, exptype, wash, day, self.conditions), 'wb') as f:
            pickle.dump(XYZw, f, protocol=pickle.HIGHEST_PROTOCOL)


class MapSingleMouseFile():
    """
    For a single mouse and condition performs mapping of 3x camera (side, front, overhead) coordinates into real 3D
    world coordinates (Xw,Yw,Zw). Data for an entire experiment is stored in a single dataframe.
    N.B. currently the data inputed is/should be my filtered dataframes, where only frames with runs are contained and
    each frame is labeled by locomotor step phase.
    """
    def __init__(self, data, con, mouseID):
        self.data, self.con, self.mouseID = data, con, mouseID
        self.maps = Structural_calculations.GetRealDistances(self.data, self.con, self.mouseID)
        # retrieve shortened dataframes for each camera for the current run
        self.df_s = self.data[con][mouseID]['Side']
        self.df_f = self.data[con][mouseID]['Front']
        self.df_o = self.data[con][mouseID]['Overhead']

    def get_h(self, yref):
        real_coords = self.maps.get_real_space_coordinates()
        cam_coords = self.maps.get_comb_camera_coords(yref)
        h = self.maps.get_homography_matrix(cam_coords, real_coords)
        return h

    def substitute_toe_y_for_ankle_y(self, dfy, labels):
        ankles = [col for col in labels if 'Ankle' in col]
        toes = [col for col in labels if 'Toe' in col]
        #dfy = dfy.drop(ankles, axis=1, level='bodyparts')
        for tidx, t in enumerate(toes):
            ankle = dfy[t].copy(deep=True)
            multi_col = pd.MultiIndex.from_product([[ankles[tidx]], ankle.columns], names=['bodyparts', 'coords'])
            ankle.columns = multi_col
            dfy = pd.concat([dfy, ankle], axis=1)
        return dfy

    def get_xy_coordinates(self, labels, dfx, dfy, yref):
        ydim = 'y' if yref == 'overhead' else 'x'
        if yref == 'front':
            dfy = self.substitute_toe_y_for_ankle_y(dfy, labels) ######## WARNING: THIS SUBSTITUTES X AND Y FOR FRONT CAM, THEREFORE IF I USE FRONT FOR Z MUST NOT BE DOWNSTREAM OF THIS
        mask = np.logical_and(dfx.loc(axis=1)[labels, 'likelihood'] > pcutoff,
                              dfy.loc(axis=1)[labels, 'likelihood'] > pcutoff).droplevel(axis=1, level='coords')
        x = dfx.loc(axis=1)[labels, 'x'][mask]
        y = dfy.loc(axis=1)[labels, ydim][mask]
        y.rename(columns={'x': 'y'}, level='coords', inplace=True)
        XYc = pd.concat([x, y], axis=1)
        return XYc

    def calculate_real_xy(self, XYc, h):
        columns = pd.MultiIndex.from_product([XYc.columns.get_level_values(level='bodyparts').unique(), ['x', 'y', 'z']], names=['bodyparts', 'coords'])
        XYZw = pd.DataFrame(index=XYc.index, columns=columns)
        bodyparts = XYc.columns.get_level_values(level='bodyparts').unique()

        for l in bodyparts:
            xy_3d = np.full((len(XYc), 3, 1), np.nan)
            xy_3d[:, :2, 0] = XYc.loc(axis=1)[l, ['x', 'y']].values
            xy_3d[:, 2, 0] = 1.0

            xw, yw = self.maps.get_transformed_coordinates(h, xy_3d)
            XYZw[l, 'x'] = xw
            XYZw[l, 'y'] = yw #condense the above
            XYZw[l, 'z'] = np.nan
        return XYZw

    def find_perspective_convergence(self, view):
        s, f, o = self.maps.assign_coordinates()
        # Define the endpoints of the two vertical lines
        if view == 'side':
            side_cam_coords = np.array([(s[key]['x'], s[key]['y']) for key in s.keys()])
            side_cam_coords = np.delete(side_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = side_cam_coords[[0, 3]]
            vertical_line2_endpoints = side_cam_coords[[1, 2]]
        elif view == 'front':
            front_cam_coords = np.array([(f[key]['x'], f[key]['y']) for key in f.keys()])
            front_cam_coords = np.delete(front_cam_coords, [2, 3], axis=0)
            vertical_line1_endpoints = front_cam_coords[[0, 1]]
            vertical_line2_endpoints = front_cam_coords[[3, 2]]
        else:
            raise ValueError('Incompatible view given')

        # Calculate the slope and intercept of each vertical line (mx + b form)
        slope1 = (vertical_line1_endpoints[1, 1] - vertical_line1_endpoints[0, 1]) / (
                    vertical_line1_endpoints[1, 0] - vertical_line1_endpoints[0, 0])
        intercept1 = vertical_line1_endpoints[0, 1] - slope1 * vertical_line1_endpoints[0, 0]
        slope2 = (vertical_line2_endpoints[1, 1] - vertical_line2_endpoints[0, 1]) / (
                    vertical_line2_endpoints[1, 0] - vertical_line2_endpoints[0, 0])
        intercept2 = vertical_line2_endpoints[0, 1] - slope2 * vertical_line2_endpoints[0, 0]
        # Calculate the x-coordinate of the intersection point
        intersection_x = (intercept2 - intercept1) / (slope1 - slope2)
        # Calculate the y-coordinate of the intersection point
        intersection_y = slope1 * intersection_x + intercept1

        return intersection_x, intersection_y

    def calculate_real_z(self, h, XYc, XYZw, offset):

        for l in XYZw.columns.get_level_values(level='bodyparts').unique():
            sidecam_y = self.df_s.loc(axis=1)[l,'y']
            zy_3d = np.full((len(XYc), 3, 1), np.nan)# Initialize an empty array to store results
            zy_3d[:, 0, 0] = sidecam_y.values + offset  # Compute the first column
            zy_3d[:, 1, 0] = XYc.loc(axis=1)[l, 'y'].values  # Assign values to the second column
            zy_3d[:, 2, 0] = 1.0  # Assign constant value to the third column

            zw, _ = self.maps.get_transformed_coordinates(h, zy_3d)
            XYZw[l, 'z'] = zw
        return XYZw

    def add_in_step_phase(self, XYZw):
        stepcycles = self.df_s.loc(axis=1)[label_list['sideXfront'], ['StepCycle', 'StepCycleFill']]
        new_XYZw = pd.concat([XYZw,stepcycles], axis=1)
        return new_XYZw

    def get_real_xyz(self):
        hf = self.get_h('front')
        ho = self.get_h('overhead')

        sideXfront_XYc = self.get_xy_coordinates(labels=label_list['sideXfront'], dfx=self.df_s, dfy=self.df_f, yref='front')
        sideXoverhead_XYc = self.get_xy_coordinates(labels=label_list['sideXoverhead'], dfx=self.df_s, dfy=self.df_o, yref='overhead')

        sideXfront_XYw = self.calculate_real_xy(XYc=sideXfront_XYc, h=hf)
        sideXoverhead_XYw = self.calculate_real_xy(XYc=sideXoverhead_XYc, h=ho)

        offset, _ = self.find_perspective_convergence('side')

        sideXfront_XYZw = self.calculate_real_z(h=hf, XYc=sideXfront_XYc, XYZw=sideXfront_XYw, offset=offset)
        sideXoverhead_XYZw = self.calculate_real_z(h=ho, XYc=sideXoverhead_XYc, XYZw=sideXoverhead_XYw, offset=offset)
        XYZw = pd.concat([sideXoverhead_XYZw, sideXfront_XYZw], axis=1)
        XYZw = self.add_in_step_phase(XYZw)
        return XYZw




