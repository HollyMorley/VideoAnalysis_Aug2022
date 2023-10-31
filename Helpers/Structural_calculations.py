import numpy as np

class GetRealDistances:
    def __init__(self, r):
        self.r = r

    def get_belt_coordinates(self, data, con, mouseID, position):
        # for main use with Locomotion file
        # y dimension (aka depth of the travellator)
        start_yax_L = []
        start_yax_R = []
        trans_yax_L = []
        trans_yax_R = []

        # x dimension (aka length of the travellator)
        start_xax_L = []
        start_xax_R = []
        trans_xax_L = []
        trans_xax_R = []

        for r in data[con][mouseID]['Front'].index.get_level_values(level='Run').unique().astype(int):
            # y axis
            start_yax_L.append(
                np.mean(data[con][mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['StartPlatL', 'x']))
            start_yax_R.append(
                np.mean(data[con][mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['StartPlatR', 'x']))

            trans_yax_L.append(
                np.mean(data[con][mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['TransitionL', 'x']))
            trans_yax_R.append(
                np.mean(data[con][mouseID]['Front'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['TransitionR', 'x']))

            # x axis
            start_xax_L.append(
                np.mean(data[con][mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['StartPlatL', 'x']))
            start_xax_R.append(
                np.mean(data[con][mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['StartPlatR', 'x']))

            trans_xax_L.append(
                np.mean(data[con][mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['TransitionL', 'x']))
            trans_xax_R.append(
                np.mean(data[con][mouseID]['Side'].loc(axis=0)[r, 'TrialStart'].loc(axis=1)['TransitionR', 'x']))

        # start_yax_px_dist = np.mean(start_yax_L) - np.mean(start_yax_R)
        # trans_yax_px_dist = np.mean(trans_yax_L) - np.mean(trans_yax_R)

        start_L_yax_mean = np.mean(start_yax_L)
        start_R_yax_mean = np.mean(start_yax_R)
        trans_L_yax_mean = np.mean(trans_yax_L)
        trans_R_yax_mean = np.mean(trans_yax_R)

        start_L_xax_mean = np.mean(start_xax_L)
        start_R_xax_mean = np.mean(start_xax_R)
        trans_L_xax_mean = np.mean(trans_xax_L)
        trans_R_xax_mean = np.mean(trans_xax_R)

        coords = {
            'a': [start_L_xax_mean, start_L_yax_mean],
            'b': [start_R_xax_mean, start_R_yax_mean],
            'c': [trans_R_xax_mean, trans_R_yax_mean],
            'd': [trans_L_xax_mean, trans_L_yax_mean]
        }

        return coords

    def calculate_pixel_size(self, coords, position):
        """
        Calculate the real size of pixel at either start of platform or the transition point
        :param coords:
        :param position: Either 'StartPlat'
        :return:
        """
        belt_width = 52  # mm