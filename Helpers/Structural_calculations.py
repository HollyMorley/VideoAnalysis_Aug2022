import numpy as np
import pandas as pd
import cv2
import warnings
from scipy import stats

class GetRealDistances:
    def __init__(self, data, con, mouseID):  # ?????????????????????????????????
        self.data = data
        self.con = con
        self.mouseID = mouseID

    def get_belt_coordinates(self, position, view):
        """
        Retrieve the pixel coordinates in x- and y-axis for left and right positions on either the start or transition position on the belt.
        :param data: dict of all dfs for each experiment
        :param con: condition
        :param mouseID: mouse name
        :param position: 'start' or 'trans'
        :return:
        """

        label_name = 'StartPlat' if position == 'start' else 'Transition'

        # y dimension (aka depth of the travellator)
        y_L = []
        y_R = []
        # x dimension (aka length of the travellator)
        x_L = []
        x_R = []

        # calculate mean of each coordinate across all runs
        for r in self.data[self.con][self.mouseID][view].index.get_level_values(level='Run').unique().astype(int):
            # retrieve y axis data
            y_L.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sL'%label_name, 'y']))
            y_R.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sR'%label_name, 'y']))

            # retrieve x axis data
            x_L.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sL'%label_name, 'x']))
            x_R.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sR'%label_name, 'x']))

        L_y_mean = np.mean(y_L)
        R_y_mean = np.mean(y_R)
        L_x_mean = np.mean(x_L)
        R_x_mean = np.mean(x_R)


        coords = {
            'L': {
                'x': L_x_mean,
                'y': L_y_mean
            },
            'R': {
                'x': R_x_mean,
                'y': R_y_mean
            }
        }

        return coords

    def assign_coordinates(self):
        """

        :return:
        """
        start_coords_front = self.get_belt_coordinates(position='start', view='Front')
        trans_coords_front = self.get_belt_coordinates(position='trans', view='Front')
        start_coords_side = self.get_belt_coordinates(position='start', view='Side')
        trans_coords_side = self.get_belt_coordinates(position='trans', view='Side')

        s = {
            '0': {'x': start_coords_side['R']['x'], 'y': start_coords_side['R']['y']},
            '1': {'x': start_coords_side['L']['x'], 'y': start_coords_side['L']['y']},
            '2': {'x': trans_coords_side['L']['x'], 'y': trans_coords_side['L']['y']},
            '3': {'x': trans_coords_side['R']['x'], 'y': trans_coords_side['R']['y']}
        }

        f = {
            '0': {'x': start_coords_front['R']['x'], 'y':  start_coords_front['R']['y']},
            '1': {'x': start_coords_front['L']['x'], 'y': start_coords_front['L']['y']},
            '2': {'x': trans_coords_front['L']['x'], 'y': trans_coords_front['L']['y']},
            '3': {'x': trans_coords_front['R']['x'], 'y':  trans_coords_front['R']['y']}
        }

        return s, f



    def calculate_pixel_size(self, left, right, real_size):
        """
        Calculate the real size of 1 pixel at either start of platform or the transition point
        :param coords: coordinates of each L and R point on either start or transition points
        :param position: Either 'StartPlat'
        :param real_size: real width (52) or length (?) of belt in mm
        :return:
        """
        belt_width_px = abs(left - right)

        px_mm = real_size / belt_width_px

        return px_mm

    def get_true_xy(self, src_coords, dst_coords, point_coords):
        h, status = cv2.findHomography(src_coords, dst_coords)
        point_coords = point_coords.reshape(3, 1)
        transformed_point_coords = np.dot(h, point_coords)
        transformed_point_coords = transformed_point_coords / transformed_point_coords[2]
        x = transformed_point_coords[0]
        y = transformed_point_coords[1]

        return x, y


    def interpolate_pixel_size_front_cam(self, real_size):
        """

        :param axis:  Axis of view point, where the other axis is the one which the area a pixel represents changes, e.g. in 'Side' or axis='x', the y-axis of the 3D representation represents the depth
        :param real_size:
        :return:
        """
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        s, f = self.assign_coordinates()

        near_px_size = self.calculate_pixel_size(left=f['3']['x'], right=f['2']['x'], real_size=real_size)
        far_px_size = self.calculate_pixel_size(left=f['0']['x'], right=f['1']['x'], real_size=real_size)

        displ_rect_coords = np.array([[s['0']['x'], s['0']['y']], [s['3']['x'], s['3']['y']], [s['2']['x'], s['2']['y']], [s['1']['x'], s['1']['y']]])
        displ_rect_coords[:, 1] = -displ_rect_coords[:, 1]
        corr_rect_coords = np.array([[s['0']['x'], s['0']['y']], [s['3']['x'],  s['0']['y']], [s['3']['x'], s['1']['y']], [s['0']['x'], s['1']['y']]])
        corr_rect_coords[:, 1] = -corr_rect_coords[:, 1]

        true_x, true_y = self.get_true_xy(displ_rect_coords, corr_rect_coords, )

        # slopeL, interceptL, r_valueL, p_valueL, std_errL = stats.linregress(
        #     [start_coords['L']['x'], trans_coords['L']['x']], [start_px_size, trans_px_size])


        # take y value from front
        # find corresponding x in corrected rect
        # find real x by transforming to true side rectangle




        #### do same for side view?????????
