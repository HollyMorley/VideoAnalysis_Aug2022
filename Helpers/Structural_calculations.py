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
            index_levels = self.data[self.con][self.mouseID][view].loc(axis=0)[r].index.get_level_values('RunStage').unique()
            if 'TrialStart' in index_levels:
                phase = 'TrialStart'
            else:
                phase = 'RunStart'
                if view == 'Front' and position == 'start': # because front and start is always processed first
                    print('No TrialStart phase found for run %s, using RunStart instead' % r)
            # retrieve y axis data
            y_L.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, phase].loc(axis=1)['%sL'%label_name, 'y']))
            y_R.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, phase].loc(axis=1)['%sR'%label_name, 'y']))

            # retrieve x axis data
            x_L.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, phase].loc(axis=1)['%sL'%label_name, 'x']))
            x_R.append(
                np.mean(self.data[self.con][self.mouseID][view].loc(axis=0)[r, phase].loc(axis=1)['%sR'%label_name, 'x']))

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

    def estimate_end_coordinates_side(self, start, trans): #rect = side_rect_coords
        slope, intercept, _, _, _ = stats.linregress([start['L']['x'], trans['L']['x']],
                                             [start['L']['y'], trans['L']['y']])
        slope_side, _, _, _, _ = stats.linregress([start['L']['x'], start['R']['x']],
                                                  [start['L']['y'], start['R']['y']])

        x_l = []
        y_r = []
        for r in self.data[self.con][self.mouseID]['Side'].index.get_level_values(level='Run').unique().astype(int):
            index_levels = self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r].index.get_level_values(
                'RunStage').unique()
            if 'TrialStart' in index_levels:
                phase = 'TrialStart'
            else:
                phase = 'RunStart'
            x_l.append(np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, phase].loc(axis=1)['Belt5', 'x']))
            y_r.append(np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, phase].loc(axis=1)['Belt5', 'y']))

        # L side
        x_L = np.mean(x_l)
        y_L = slope * x_L + intercept

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

    def estimate_end_coordinates_front(self, start, trans): # rect= front_rect_coords
        x_L = trans['L']['y'] + 20
        x_R = trans['R']['y'] + 20

        slope_r, intercept_r, _, _, _ = stats.linregress([start['R']['y'], trans['R']['y']],
                                                         [start['R']['x'], trans['R']['x']])
        slope_l, intercept_l, _, _, _ = stats.linregress([start['L']['y'], trans['L']['y']],
                                                         [start['L']['x'], trans['L']['x']])
        # extrapolate these slopes to y_L and y_R
        y_L = slope_l*x_L + intercept_l
        y_R = slope_r*x_R + intercept_r

        coords = {
            'L': {
                'x': y_L,
                'y': x_L
            },
            'R': {
                'x': y_R,
                'y': x_R
            }
        }
        return coords

    def estimate_end_coordinates_overhead(self, start, trans):
        x_L = trans['L']['x'] + 200
        x_R = trans['R']['x'] + 200

        slope_r, intercept_r, _, _, _ = stats.linregress([start['R']['x'], trans['R']['x']],
                                                         [start['R']['y'], trans['R']['y']])
        slope_l, intercept_l, _, _, _ = stats.linregress([start['L']['x'], trans['L']['x']],
                                                         [start['L']['y'], trans['L']['y']])

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
        start_coords_overhead = self.get_belt_coordinates(position='start', view='Overhead')
        trans_coords_overhead = self.get_belt_coordinates(position='trans', view='Overhead')

        start_coords_side['L']['y'] = start_coords_side['L']['y'] + 2

        end_coords_front = self.estimate_end_coordinates_front(start_coords_front, trans_coords_front)
        end_coords_side = self.estimate_end_coordinates_side(start_coords_side, trans_coords_side)
        end_coords_overhead = self.estimate_end_coordinates_overhead(start_coords_overhead, trans_coords_overhead)

        s = {
            '0': {'x': start_coords_side['R']['x'], 'y': start_coords_side['R']['y']},
            '1': {'x': trans_coords_side['R']['x'], 'y': trans_coords_side['R']['y']},
            '2': {'x': end_coords_side['R']['x'], 'y': end_coords_side['R']['y']},
            '3': {'x': end_coords_side['L']['x'], 'y': end_coords_side['L']['y']},
            '4': {'x': trans_coords_side['L']['x'], 'y': trans_coords_side['L']['y']},
            '5': {'x': start_coords_side['L']['x'], 'y': start_coords_side['L']['y']}
        }

        f = {
            '0': {'x': start_coords_front['R']['x'], 'y':  start_coords_front['R']['y']},
            '1': {'x': trans_coords_front['R']['x'], 'y':  trans_coords_front['R']['y']},
            '2': {'x': end_coords_front['R']['x'], 'y': end_coords_front['R']['y']},
            '3': {'x': end_coords_front['L']['x'], 'y': end_coords_front['L']['y']},
            '4': {'x': trans_coords_front['L']['x'], 'y': trans_coords_front['L']['y']},
            '5': {'x': start_coords_front['L']['x'], 'y': start_coords_front['L']['y']}
        }

        o = {
            '0': {'x': start_coords_overhead['R']['x'], 'y': start_coords_overhead['R']['y']},
            '1': {'x': trans_coords_overhead['R']['x'], 'y': trans_coords_overhead['R']['y']},
            '2': {'x': end_coords_overhead['R']['x'], 'y': end_coords_overhead['R']['y']},
            '3': {'x': end_coords_overhead['L']['x'], 'y': end_coords_overhead['L']['y']},
            '4': {'x': trans_coords_overhead['L']['x'], 'y': trans_coords_overhead['L']['y']},
            '5': {'x': start_coords_overhead['L']['x'], 'y': start_coords_overhead['L']['y']}
        }

        return s, f, o

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
        :param src_coords: sourcea polygon coordintes (4-sided)
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
            far_px_size = self.calculate_pixel_size(left=fs['5']['x'], right=fs['0']['x'], real_size=width)
        elif view == 'Side':
            depth = structural_stuff['belt_width']
            width = structural_stuff['belt_length_sideviewrange']
            near_px_size = self.calculate_pixel_size(left=fs['0']['x'], right=fs['2']['x'], real_size=width)
            far_px_size = self.calculate_pixel_size(left=fs['5']['x'], right=fs['3']['x'], real_size=width)
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

    def get_warped_grid_coordinates(self, view, four_pts=True, real_size_grid=True, length=20, depth=5):
        """
        Get warped grid coordinates for either front or side view of belt
        :param view: 'front', 'side' or 'overhead'
        :return: x and y coordinates
        """
        standard_rect_coords = np.array([[0, 0],
                                         [structural_stuff['belt_length_sideviewrange'] - structural_stuff['belt_length_sideviewend'], 0],
                                         [structural_stuff['belt_length_sideviewrange'], 0],
                                         [structural_stuff['belt_length_sideviewrange'], structural_stuff['belt_width']],
                                         [structural_stuff['belt_length_sideviewrange'] - structural_stuff['belt_length_sideviewend'], structural_stuff['belt_width']],
                                         [0, structural_stuff['belt_width']]])
        if four_pts == True:
            standard_rect_coords = np.delete(standard_rect_coords, [2, 3], axis=0)

        if real_size_grid == True:
            grid = np.array([[[x], [y], [z]] for x in range(structural_stuff['belt_length_sideviewrange']) for y in
                    range(structural_stuff['belt_width']) for z in [1]])
        else:
            grid = np.array([[[x], [y], [z]] for x in range(0, structural_stuff['belt_length_sideviewrange'] - structural_stuff['belt_length_sideviewend'], length) for y in
                             range(0, structural_stuff['belt_width'], depth) for z in [1]])

        s, f, o = self.assign_coordinates()

        x, y = [], []
        if view == 'Side':
            side_rect_coords = np.array([(point['x'], point['y']) for point in s.values()])
            if four_pts == True:
                side_rect_coords = np.delete(side_rect_coords, [2, 3], axis=0)
            x, y = self.get_transformed_xy_of_point(standard_rect_coords, side_rect_coords, grid)
        elif view == 'Front':
            front_rect_coords = np.array([(point['y'], point['x']) for point in f.values()])  # switched to rotate the view to same orientation
            if four_pts == True:
                front_rect_coords = np.delete(front_rect_coords, [2, 3], axis=0)
            x, y = self.get_transformed_xy_of_point(standard_rect_coords,front_rect_coords,grid)
        elif view == 'Overhead':
            overhead_rect_coords = np.array([(point['x'], point['y']) for point in o.values()])
            if four_pts == True:
                overhead_rect_coords = np.delete(overhead_rect_coords, [2, 3], axis=0)
            x, y = self.get_transformed_xy_of_point(standard_rect_coords, overhead_rect_coords, grid)

        return x, y

    def map_pixel_sizes_to_belt(self, view, yref='Front'):
        """
        Use triangulation to construct map of how pixel size changes across the flat camera field
        :param view: view of primary analysis where depth is relevant and potentially confounding, e.g. if you are measuring tail trajectory in x plane, view is side.
        If you are measuring nose trajectory in y plane, view is front.
        :param yref: Reference for depth of primary view. Which view (front or overhead) depends on which labelled points are of interest.
        Front has highest resolution but not all labels are visible, e.g. back and tail
        :return: triangulated map and array of possible pixel sizes
        """

        x, _ = self.get_warped_grid_coordinates(view='Side')
        if yref == 'Front':
            _, y = self.get_warped_grid_coordinates(view='Front')
        elif yref == 'Overhead':
            _, y = self.get_warped_grid_coordinates(view='Overhead')

        s, f, _ = self.assign_coordinates()
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
        :param x: target x coordinate/s (as array)
        :param y: target y coordinate/s (as array)
        :param pixel_sizes: array of pixel sizes in rows through length (x) of the belt
        :param triang: triangulation of the pixel size data
        :return: closest found pixel size
        """
        # interpolated_pixel_size = pixel_sizes[index]
        coordinates = np.column_stack((triang.x, triang.y))
        # Create a 2D array of (x, y) coordinates from x_values and y_values
        query_coordinates = np.column_stack((x, y))
        # Calculate the squared distance for all points
        distances_squared = np.sum((coordinates[:, np.newaxis, :] - query_coordinates) ** 2, axis=2)
        # Find the index of the minimum distance for each point in x_values and y_values
        indices = np.argmin(distances_squared, axis=0)
        # Get the interpolated pixel size for each closest point
        interpolated_pixel_size = pixel_sizes[indices]

        return interpolated_pixel_size


    ################ new #################

    def get_real_space_coordinates(self):
        real_coords = np.array([[0, 0],
                                [structural_stuff['belt_length_sideviewrange'] - structural_stuff[
                                    'belt_length_sideviewend'], 0],
                                # [structural_stuff['belt_length_sideviewrange'], 0],
                                # [structural_stuff['belt_length_sideviewrange'], structural_stuff['belt_width']],
                                [structural_stuff['belt_length_sideviewrange'] - structural_stuff[
                                    'belt_length_sideviewend'], structural_stuff['belt_width']],
                                [0, structural_stuff['belt_width']]])
        return real_coords

    def get_cam_coords(self, view):
        s, f, o = self.assign_coordinates()
        cam_map = {'side': s, 'front': f, 'overhead': o}
        cam = cam_map.get(view)
        if cam is None:
            raise ValueError(f"View '{view}' is not recognized.")
        cam_coords = np.array([(cam[key]['x'], cam[key]['y']) for key in cam.keys()])
        cam_coords = np.delete(cam_coords, [2, 3], axis=0)
        return cam_coords

    def get_comb_camera_coords(self, yref):
        s, f, o = self.assign_coordinates()
        if yref == 'front':
            combined_cam_coords = np.array([(s[key]['x'], f[key]['x']) for key in s.keys()])
            combined_cam_coords = np.delete(combined_cam_coords, [2, 3], axis=0)
            return combined_cam_coords
        elif yref == 'overhead':
            combined_cam_coords = np.array([(s[key]['x'], o[key]['y']) for key in s.keys()])
            combined_cam_coords = np.delete(combined_cam_coords, [2, 3], axis=0)
            return combined_cam_coords
        else:
            raise ValueError("yref must be 'front' or 'overhead'")

    def get_homography_matrix(self, src_coords, dst_coords):
        h, status = cv2.findHomography(src_coords, dst_coords)
        return h

    def get_perspective_transform(self, src_coords, dst_coords):
        h = cv2.getPerspectiveTransform(src_coords.astype(np.float32), dst_coords.astype(np.float32))
        return h

    def get_transformed_coordinates(self, h, coords):
        transformed_coords = np.dot(h, coords)
        transformed_coords = transformed_coords / transformed_coords[2]
        x = transformed_coords[0]
        y = transformed_coords[1]
        return x, y





# class create_real_space_dataframes:
#     def convert_px_to_mm_WHOLE_DF(self, conditions):
#         data = utils.Utils().GetDFs(conditions, reindexed_loco=False)
#


