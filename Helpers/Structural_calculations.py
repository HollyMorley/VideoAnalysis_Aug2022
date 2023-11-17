import numpy as np
import pandas as pd
import cv2
import warnings
import math
from scipy import stats
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

from Helpers.Config_23 import *
from Helpers import utils

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

    def estimate_end_coordinates_side(self, rect): #rect = side_rect_coords
        slope, _, _, _, _ = stats.linregress([rect[3][0], rect[2][0]],
                                             [rect[3][1], rect[2][1]])
        slope_top, intercept_top, _, _, _ = stats.linregress([rect[3][0], rect[2][0]],
                                                             [rect[3][1], rect[2][1]])
        slope_side, _, _, _, _ = stats.linregress([rect[3][0], rect[0][0]],
                                                  [rect[3][1], rect[0][1]])

        x_l = []
        y_r = []
        for r in self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='Run').unique().astype(int):
            x_l.append(np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['Belt5', 'x']))
            y_r.append(np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['Belt5', 'y']))

        # L side
        x_L = np.mean(x_l)
        y_L = slope_top * x_L + intercept_top

        # R side
        y_R = np.mean(y_r)

        # calculate missing x_R value
        side_angle = abs(np.rad2deg(np.arctan(slope_side)))
        angle = abs(np.rad2deg(np.arctan(slope)))
        internal_angle = 180 - (side_angle + angle)
        tilt_internal_angle = internal_angle - angle
        new_slope = math.tan((math.radians(tilt_internal_angle + 90)))

        x_R = x_L + (y_R / new_slope)

        coords = {
            'L': {
                'x': x_L,
                'y': y_L
            },
            'R': {
                'x': x_R,
                'y': y_R
            }
        }

        return coords

    def estimate_end_coordinates_front(self, rect): # rect= front_rect_coords
        x_L = rect[2][0] + 20
        x_R = rect[1][0] + 20

        slope_r, intercept_r, _, _, _ = stats.linregress([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]])
        slope_l, intercept_l, _, _, _ = stats.linregress([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]])

        # extrapolate these slopes to y_L and y_R
        y_L = slope_l*x_L + intercept_l
        y_R = slope_r*x_R + intercept_r

        coords = {
            'L': {
                'x': x_L,
                'y': y_L
            },
            'R': {
                'x': x_R,
                'y': y_R
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

        start_coords_side['L']['y'] = start_coords_side['L']['y'] + 2

        s = {
            '0': {'x': start_coords_side['R']['x'], 'y': start_coords_side['R']['y']},
            '1': {'x': trans_coords_side['R']['x'], 'y': trans_coords_side['R']['y']},
            '2': {'x': trans_coords_side['L']['x'], 'y': trans_coords_side['L']['y']},
            '3': {'x': start_coords_side['L']['x'], 'y': start_coords_side['L']['y']}
        }

        f = {
            '0': {'x': start_coords_front['R']['x'], 'y':  start_coords_front['R']['y']},
            '1': {'x': trans_coords_front['R']['x'], 'y':  trans_coords_front['R']['y']},
            '2': {'x': trans_coords_front['L']['x'], 'y': trans_coords_front['L']['y']},
            '3': {'x': start_coords_front['L']['x'], 'y': start_coords_front['L']['y']}
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

    def get_transformed_xy_of_point(self, src_coords, dst_coords, point_coords):
        """
        Warps source 4-sided polygon into destination polygon.
        :param src_coords: source polygon coordinates (4-sided)
        :param dst_coords: destination polygon coordinates (4-sided)
        :param point_coords: x,y coordinates of point to be transformed. MUST be in shape (3,1)
        :return: Coordinates of specified point according the the warping of the whole polygon
        """
        h, status = cv2.findHomography(src_coords, dst_coords)
        transformed_point_coords = np.dot(h, point_coords)
        transformed_point_coords = transformed_point_coords / transformed_point_coords[2]
        x = transformed_point_coords[0]
        y = transformed_point_coords[1]

        return x, y

    def estimate_pixel_sizes(self, view, fs):
        """
        Estimate the real size in mm that each pixel represents throughout the depth (and width) of the front camera.
        Warning: Assumes change in pixel size is linear across depth
        :param real_size: real width of the travellator belt.
        :param f: belt corner coordinates from front camera
        :return: array containing px sizes for each grid coordinate across belt view (which represents 1mm of real space)
        """
        near_px_size, far_px_size, depth, width = [], [], [], []

        if view == 'Front':
            depth = structural_stuff['belt_length_sideviewrange']
            width = structural_stuff['belt_width']
            near_px_size = self.calculate_pixel_size(left=fs['3']['x'], right=fs['2']['x'], real_size=width)
            far_px_size = self.calculate_pixel_size(left=fs['0']['x'], right=fs['1']['x'], real_size=width)
        elif view == 'Side':
            depth = structural_stuff['belt_width']
            width = structural_stuff['belt_length_sideviewrange']
            near_px_size = self.calculate_pixel_size(left=fs['0']['x'], right=fs['3']['x'], real_size=width)
            far_px_size = self.calculate_pixel_size(left=fs['1']['x'], right=fs['2']['x'], real_size=width)
        slope, intercept, r_value, p_value, std_err = stats.linregress([0, depth], [far_px_size, near_px_size])
        pixel_sizes = []
        for n in np.arange(0, depth):
            s = slope * n + intercept
            pixel_sizes.append([s] * width)
        pixel_sizes = np.array(pixel_sizes)
        if view == 'Side':
            pixel_sizes = np.flip(pixel_sizes).T
        pixel_sizes = pixel_sizes.flatten()

        return pixel_sizes

    def get_warped_grid_coordinates(self, view):
        """
        Get warped grid coordinates for either front or side view of belt
        :param view: 'front' or 'side'
        :return: x and y coordinates
        """
        standard_rect_coords = np.array([[0, 0],
                                         [structural_stuff['belt_length_sideviewrange'] - structural_stuff['belt_length_sideviewend'], 0],
                                         [structural_stuff['belt_length_sideviewrange'], 0],
                                         [structural_stuff['belt_length_sideviewrange'], structural_stuff['belt_width']],
                                         [structural_stuff['belt_length_sideviewrange'] - structural_stuff['belt_length_sideviewend'], structural_stuff['belt_width']],
                                         [0, structural_stuff['belt_width']]])
        grid = np.array([[[x], [y], [z]] for x in range(structural_stuff['belt_length_sideviewrange']) for y in
                range(structural_stuff['belt_width']) for z in [1]])

        s, f = self.assign_coordinates()

        # side_rect_coords = np.array([[s['0']['x'], s['0']['y']], [s['1']['x'], s['1']['y']+ 2], [s['2']['x'], s['2']['y']], [s['3']['x'], s['3']['y']]])
        # front_rect_coords = np.array([[f['0']['y'], f['0']['x']], [f['1']['y'], f['1']['x']], [f['2']['y'], f['2']['x']], [f['3']['y'], f['3']['x']]])
        side_rect_coords = np.array([(point['x'], point['y']) for point in s.values()])
        front_rect_coords = np.array([(point['y'], point['x']) for point in f.values()])

        side_end = self.estimate_end_coordinates_side(side_rect_coords)
        front_end = self.estimate_end_coordinates_front(front_rect_coords)

        side_rect_coords = np.insert(side_rect_coords,2,np.array([[side_end['R']['x'], side_end['R']['y']], [side_end['L']['x'], side_end['L']['y']]]), axis=0)
        front_rect_coords = np.insert(front_rect_coords,2,np.array([[front_end['R']['x'], front_end['R']['y']], [front_end['L']['x'], front_end['L']['y']]]), axis=0)

        x, y = [], []
        if view == 'Side':
            # x, y = self.get_transformed_xy_of_point(standard_rect_coords,side_rect_coords_rot,grid)
            x, y = self.get_transformed_xy_of_point(standard_rect_coords, side_rect_coords, grid)
        elif view == 'Front':
            x, y = self.get_transformed_xy_of_point(standard_rect_coords,front_rect_coords,grid) # switched to rotate the view to same orientation

        return x, y

    def map_pixel_sizes_to_belt(self, view):
        """
        Use triangulation to construct map of how pixel size changes across the flat camera field
        :return:
        """
        x, _ = self.get_warped_grid_coordinates(view='Side')
        _, y = self.get_warped_grid_coordinates(view='Front')

        s, f = self.assign_coordinates()
        fs = []
        if view == 'Front':
            fs = f
        elif view == 'Side':
            fs = s

        pixel_sizes = self.estimate_pixel_sizes(view=view, fs=fs)

        grid_coordinates = np.column_stack((x.flatten(), y.flatten()))
        triang = Triangulation(grid_coordinates[:, 0], grid_coordinates[:, 1])
        interpolated_pixel_sizes = griddata(grid_coordinates, pixel_sizes, (triang.x, triang.y), method='nearest')

        return triang, interpolated_pixel_sizes

    def find_interpolated_pixel_size(self, x, y, pixel_sizes, triang):
        """
        Uses previously computed map of pixel size to find true pixel size at any point in field of view
        :param x: target x coordinate
        :param y: target y coordinate
        :param pixel_sizes: array of pixel sizes in rows through length (x) of the belt
        :param triang: triangulation of the pixel size data
        :return: closest found pixel size
        """
        # Find the index of the closest point in the triangulation
        index = np.argmin((triang.x - x) ** 2 + (triang.y - y) ** 2)

        # Get the interpolated pixel size for the closest point
        interpolated_pixel_size = pixel_sizes[index]

        return interpolated_pixel_size



