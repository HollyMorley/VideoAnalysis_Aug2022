import numpy as np
import pandas as pd
import cv2
import warnings
from scipy import stats

from Helpers.Config_23 import *

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
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
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
        Arrange coordinates for front and side view in order: [[0, 0], [1, 0], [1, 1], [0, 1]]
        :return: dict for s and f with ordered coordinates
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

    # def estimate_pixel_size_front_cam(self, real_size, f, position=[]):
    #     """
    #     Estimate the real size in mm that each pixel represents throughout the depth of the front camera.
    #     Warning: Assumes change in pixel size is linear across depth
    #     :param real_size: real width of the travellator belt.
    #     :param position: List format. As a decimel, with 0 being at the start and 1 being at the transition.
    #     :return: array containing px sizes for each position in input
    #     """
    #     near_px_size = self.calculate_pixel_size(left=f['3']['x'], right=f['2']['x'], real_size=real_size)
    #     far_px_size = self.calculate_pixel_size(left=f['0']['x'], right=f['1']['x'], real_size=real_size)
    #
    #     # calculate what
    #     px_size_diff = far_px_size - near_px_size
    #
    #     position = np.array(position)
    #     est_px_size = far_px_size - px_size_diff * position
    #
    #     return est_px_size

    def get_transformed_xy_of_point(self, src_coords, dst_coords, point_coords):
        """
        Warps source 4-sided polygon into destination polygon.
        :param src_coords: source polygon coordinates (4-sided)
        :param dst_coords: destination polygon coordinates (4-sided)
        :param point_coords: x,y coordinates of point to be transformed. MUST be in shape (3,1)
        :return: Coordinates of specified point according the the warping of the whole polygon
        """
        h, status = cv2.findHomography(src_coords, dst_coords)
        #point_coords = point_coords.reshape(3, 1)
        transformed_point_coords = np.dot(h, point_coords)
        transformed_point_coords = transformed_point_coords / transformed_point_coords[2]
        x = transformed_point_coords[0]
        y = transformed_point_coords[1]

        return x, y

    def estimate_pixel_sizes_front_cam(self, real_size, f):
        near_px_size = self.calculate_pixel_size(left=f['3']['x'], right=f['2']['x'], real_size=real_size)
        far_px_size = self.calculate_pixel_size(left=f['0']['x'], right=f['1']['x'], real_size=real_size)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            [0, structural_stuff['belt_length_sideviewrange']], [far_px_size, near_px_size])
        pixel_sizes = []
        for n in np.arange(0, structural_stuff['belt_length_sideviewrange']):
            s = slope * n + intercept
            pixel_sizes.append([s] * structural_stuff['belt_width'])
        pixel_sizes = np.array(pixel_sizes).flatten()

        return pixel_sizes

    def map_px_size_to_front(self):

        standard_rect_coords = np.array([[0, 0], [structural_stuff['belt_length_sideviewrange'], 0], [structural_stuff['belt_length_sideviewrange'], structural_stuff['belt_width']], [0, structural_stuff['belt_width']]])
        grid = np.array([[[x], [y], [z]] for x in range(structural_stuff['belt_length_sideviewrange']) for y in
                range(structural_stuff['belt_width']) for z in [1]])

        s, f = self.assign_coordinates()

        side_rect_coords = np.array([[s['0']['x'], s['0']['y']], [s['3']['x'], s['3']['y']], [s['2']['x'], s['2']['y']],[s['1']['x'], s['1']['y'] + 2]])
        front_rect_coords = np.array([[f['0']['y'], f['0']['x']], [f['3']['y'], f['3']['x']], [f['2']['y'], f['2']['x']], [f['1']['y'], f['1']['x']]])

        slope, _, _, _, _ = stats.linregress([side_rect_coords[0][0], side_rect_coords[1][0]],
                                             [side_rect_coords[0][1], side_rect_coords[1][1]])
        angle = np.arctan(slope)
        side_rect_coords_rot = utils.Utils().Rotate2D(side_rect_coords, np.array([side_rect_coords[0][0], side_rect_coords[0][1]]), -angle)

        x_side, y_side = self.get_transformed_xy_of_point(standard_rect_coords,side_rect_coords_rot,grid)
        x_front, y_front = self.get_transformed_xy_of_point(standard_rect_coords,front_rect_coords,grid) # switched to rotate the view to same orientation

        grid_coordinates = np.column_stack((x_front.flatten(), y_front.flatten()))

        # convert front view back to standardised rectangle with new x coordinates from side view






        corr_rect_coords = np.array([[s['0']['x'], s['0']['y']], [s['3']['x'],  s['0']['y']], [s['3']['x'], s['1']['y']], [s['0']['x'], s['1']['y']]])
        corr_rect_coords[:, 1] = -corr_rect_coords[:, 1]

        #true_x, true_y = self.get_true_xy(displ_rect_coords, corr_rect_coords, )

        # slopeL, interceptL, r_valueL, p_valueL, std_errL = stats.linregress(
        #     [start_coords['L']['x'], trans_coords['L']['x']], [start_px_size, trans_px_size])


        # take y value from front
        # find corresponding x in corrected rect
        # find real x by transforming to true side rectangle




        #### do same for side view?????????
