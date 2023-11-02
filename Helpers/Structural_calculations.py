import numpy as np
from scipy import stats

class GetRealDistances:
    def __init__(self, data, con, mouseID):  # ?????????????????????????????????
        self.data = data
        self.con = con
        self.mouseID = mouseID

    def get_belt_coordinates(self, position):
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
        yax_L = []
        yax_R = []
        # x dimension (aka length of the travellator)
        xax_L = []
        xax_R = []

        # calculate mean of each coordinate across all runs
        for r in self.data[self.con][self.mouseID]['Front'].index.get_level_values(level='Run').unique().astype(int):
            # retrieve y axis data
            yax_L.append(
                np.mean(self.data[self.con][self.mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sL'%label_name, 'x']))
            yax_R.append(
                np.mean(self.data[self.con][self.mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sR'%label_name, 'x']))

            # retrieve x axis data
            xax_L.append(
                np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sL'%label_name, 'x']))
            xax_R.append(
                np.mean(self.data[self.con][self.mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['%sR'%label_name, 'x']))

        L_yax_mean = np.mean(yax_L)
        R_yax_mean = np.mean(yax_R)
        L_xax_mean = np.mean(xax_L)
        R_xax_mean = np.mean(xax_R)


        coords = {
            'L': {
                'x': L_xax_mean,
                'y': L_yax_mean
            },
            'R': {
                'x': R_xax_mean,
                'y': R_yax_mean
            }
        }

        return coords

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

    def interpolate_pixel_size(self, axis, real_size):
        """

        :param axis:  Axis of view point, where the other axis is the one which the area a pixel represents changes, e.g. in 'Side' or axis='x', the y-axis of the 3D representation represents the depth
        :param real_size:
        :return:
        """
        start_coords = self.get_belt_coordinates(position='start')
        trans_coords = self.get_belt_coordinates(position='trans')

        if axis == 'y':

            near_px_size = self.calculate_pixel_size(left=trans_coords['R']['y'], right=trans_coords['L']['y'], real_size=real_size)
            far_px_size = self.calculate_pixel_size(left=start_coords['R']['y'], right=start_coords['L']['y'], real_size=real_size)
        elif axis == 'x':
            near_px_size = self.calculate_pixel_size(left=start_coords['R']['x'], right=trans_coords['R']['x'], real_size=real_size)
            far_px_size = self.calculate_pixel_size(left=start_coords['L']['x'], right=trans_coords['L']['x'], real_size=real_size)

        slopeL, interceptL, r_valueL, p_valueL, std_errL = stats.linregress(
            [start_coords['L']['x'], trans_coords['L']['x']], [start_px_size, trans_px_size])
