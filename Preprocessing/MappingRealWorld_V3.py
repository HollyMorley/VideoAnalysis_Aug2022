from Helpers.utils_3d_reconstruction import CameraData, BeltPoints
from Helpers.CalibrateCams import BasicCalibration
from Helpers import utils
from Helpers.Config_23 import *
from Helpers import MultiCamLabelling_config as opt_config
from Helpers import OptimizeCalibration
import os, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pycalib.calib import triangulate
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

class MapExperiment:
    def __init__(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, belt_coords, snapshot_paths):
        self.DataframeCoors = {'side': DataframeCoor_side, 'front': DataframeCoor_front, 'overhead': DataframeCoor_overhead}

        self.cameras = CameraData(snapshot_paths)
        self.cameras_specs = self.cameras.specs
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        self.belt_pts = BeltPoints(belt_coords)
        self.belt_coords_CCS = self.belt_pts.coords_CCS
        self.belt_coords_WCS = self.belt_pts.coords_WCS

    def plot_CamCoorSys(self):
        return self.belt_pts.plot_CCS(self.cameras)

    def plot_WorldCoorSys(self):
        return self.belt_pts.plot_WCS()

    def estimate_pose(self):
        cameras_extrinsics = self.cameras.compute_cameras_extrinsics(self.belt_coords_WCS, self.belt_coords_CCS)
        self.print_reprojection_errors(cameras_extrinsics)
        return cameras_extrinsics

    def estimate_pose_with_guess(self):
        cameras_extrinsics_ini_guess = self.cameras.compute_cameras_extrinsics(
            self.belt_coords_WCS,
            self.belt_coords_CCS,
            use_extrinsics_ini_guess=True
        )
        self.print_reprojection_errors(cameras_extrinsics_ini_guess, with_guess=True)
        return cameras_extrinsics_ini_guess

    def print_reprojection_errors(self, cameras_extrinsics, with_guess=False):
        if with_guess:
            print('Reprojection errors (w/ initial guess):')
        else:
            print('Reprojection errors:')
        for cam, data in cameras_extrinsics.items():
            print(f'{cam}: {data["repr_err"]}')

    def plot_cam_locations_and_pose(self, cameras_extrinsics):
        fig, ax = self.belt_pts.plot_WCS()
        for cam in self.cameras.specs:
            vec_WCS_to_CCS, rot_cam_opencv = self.get_camera_vectors(cameras_extrinsics, cam)
            self.add_camera_pose(ax, cam, vec_WCS_to_CCS, rot_cam_opencv)
        return fig, ax

    def get_camera_vectors(self, cameras_extrinsics, cam):
        cob_cam_opencv = cameras_extrinsics[cam]['rotm'].T
        vec_WCS_to_CCS = -cob_cam_opencv @ cameras_extrinsics[cam]['tvec']
        return vec_WCS_to_CCS, cameras_extrinsics[cam]['rotm']

    def add_camera_pose(self, ax, cam, vec_WCS_to_CCS, rot_cam_opencv):
        ax.scatter(*vec_WCS_to_CCS, s=50, c="b", marker=".", linewidth=0.5, alpha=1)
        ax.text(*vec_WCS_to_CCS.flatten(), s=cam, c="b")
        for row, color in zip(rot_cam_opencv, ["r", "g", "b"]):
            ax.quiver(
                *vec_WCS_to_CCS.flatten(), *row, color=color,
                length=500, arrow_length_ratio=0, normalize=True, linewidth=2
            )
        ax.axis("equal")


class GetSingleExpData:
    def __init__(self, side_file, front_file, overhead_file):
        self.files = {'side': side_file, 'front': front_file, 'overhead': overhead_file}

        # retrieve raw data, format, and match frames across the 3 camera views
        self.DataframeCoors = self.load_and_clean_all_data()

    def load_and_clean_all_data(self):
        dataframes = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            dataframes[view] = self.get_and_format_data(self.files[view], view)

        aligned_videos = self.align_dfs(dataframes)
        return aligned_videos

    def get_and_format_data(self, file, view):
        df = pd.read_hdf(file)
        try:
            df = df.loc(axis=1)[vidstuff['scorers'][view]].copy()
        except:
            df = df.loc(axis=1)[vidstuff['scorers'][f'{view}_new']].copy()
        return df

    def load_timestamps(self, view):
        timestamp_path = utils.Utils().Get_timestamps_from_analyse_file(self.files[view], view)
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    def zero_timestamps(self, timestamps):
        timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, side_timestamps, other_timestamps):
        mask = other_timestamps['Timestamp'].diff() < 4.045e+6
        other_timestamps_single_frame = other_timestamps[mask]
        side_timestamps_single_frame = side_timestamps[mask]
        diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

        # find the best fit line for the lower half of the data by straightning the line
        model = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values.reshape(-1, 1), diff.values)
        slope = model.coef_[0]
        intercept = model.intercept_
        straightened_diff = diff - (slope * side_timestamps_single_frame['Timestamp'] + intercept)
        correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

        model_true = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values[correct_diff_idx].reshape(-1, 1), diff.values[correct_diff_idx])
        slope_true = model_true.coef_[0]
        intercept_true = model_true.intercept_
        adjusted_timestamps = other_timestamps['Timestamp'] - (slope_true * other_timestamps['Timestamp'] + intercept_true)
        return adjusted_timestamps

    def adjust_frames(self):
        timestamps = {'side': [], 'front': [], 'overhead': []}
        timestamps_adj = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            timestamps[view] = self.zero_timestamps(self.load_timestamps(view))
            if view != 'side':
                timestamps_adj[view] = self.adjust_timestamps(timestamps['side'], timestamps[view])
        timestamps_adj['side'] = timestamps['side']['Timestamp'].astype(float)
        return timestamps_adj

    def match_frames(self):
        timestamps = self.adjust_frames()

        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        dfs = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            timestamps[view] = timestamps[view].sort_values().reset_index(drop=True)
            dfs[view] = pd.DataFrame({'Timestamp': timestamps[view], 'Frame_number_%s' %view: range(len(timestamps[view]))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(dfs['side'], dfs['front'], on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, dfs['overhead'], on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        return matched_frames

    def align_dfs(self, dfs):
        matched_frames = self.match_frames()

        # find first index row where all frames are in positive time
        start = None
        for idx, it in enumerate(matched_frames):
            if np.all(np.array(it) > 0):
                start = idx
                break
        matched_frames = matched_frames[start:]

        frames = {'side': [], 'front': [], 'overhead': []}
        aligned_dfs = {'side': [], 'front': [], 'overhead': []}
        for vidx, view in enumerate(vidstuff['cams']):
            frames[view] = [frame[vidx] for frame in matched_frames]
            aligned_dfs[view] = dfs[view].iloc[frames[view]].reset_index(drop=False).rename(columns={'index': 'original_index'})

        return aligned_dfs


    ###################################################################################################################
    # Map the camera views to the real world coordinates for each video file
    ###################################################################################################################
    def map(self):
        calibration_coords = self.get_belt_coords()

        calib_obj = BasicCalibration(calibration_coords)
        cameras_extrinsics = calib_obj.estimate_cams_pose()
        cameras_intrinsics = calib_obj.cameras_intrinsics

        belt_points_WCS = calib_obj.belt_coords_WCS
        belt_points_CCS = calib_obj.belt_coords_CCS

        # optimise the cameras extrinsics
        optimise = OptimizeCalibration.optimize(calibration_coords, cameras_extrinsics, cameras_intrinsics, self)
        new_calibration_data = optimise.optimise_calibration()

        coords, repr_err = self.get_realworld_coords(new_calibration_data['extrinsics'], new_calibration_data['intrinsics'])


    def get_belt_coords(self):
        masks = {'side': [], 'front': [], 'overhead': []}
        belt_coords = {'side': [], 'front': [], 'overhead': []}
        means = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            masks[view] = np.all(self.DataframeCoors[view].loc(axis=1)[
                                        ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR',
                                            'TransitionL'], 'likelihood'] > 0.99, axis=1)
            belt_coords[view] = self.DataframeCoors[view].loc(axis=1)[
                ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR', 'TransitionL'], ['x', 'y']][masks[view]]
            means[view] = belt_coords[view].mean(axis=0)

        # concatenate the mean values of the belt coordinates from the 3 camera views with the camera names as columns
        belt_coords = pd.concat([means['side'], means['front'], means['overhead']], axis=1)
        belt_coords.columns = vidstuff['cams']

        # add door coordinates
        door_coords = self.get_door_coords()
        belt_coords.reset_index(inplace=True, drop=False)
        coords = pd.concat([belt_coords, door_coords], axis=0).reset_index(drop=True)

        return coords

    def get_door_coords(self):
        masks = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            masks[view] = self.DataframeCoors[view].loc(axis=1)['Door', 'likelihood'] > pcutoff

        mask = masks['side'] & masks['front'] & masks['overhead']

        sem_factor = 100
        door_present = {'side': [], 'front': [], 'overhead': []}
        door_closed_masks = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            door_present[view] = self.DataframeCoors[view].loc(axis=1)['Door', ['x', 'y']][mask]
            door_closed_masks[view] = np.logical_and(
                door_present[view]['Door', 'y'] < door_present[view]['Door', 'y'].mean() + door_present[view]['Door', 'y'].sem() * sem_factor,
                door_present[view]['Door', 'y'] > door_present[view]['Door', 'y'].mean() - door_present[view]['Door', 'y'].sem() * sem_factor)

        closed_mask = door_closed_masks['side'] & door_closed_masks['front'] & door_closed_masks['overhead']

        means = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            means[view] = door_present[view][closed_mask].mean(axis=0)

        # concatenate the mean values of the door coordinates from the 3 camera views with the camera names as columns
        door_coords = pd.concat([means['side'], means['front'], means['overhead']], axis=1)
        door_coords.columns = ['side', 'front', 'overhead']
        door_coords.reset_index(inplace=True, drop=False)

        return door_coords

    ####################################################################################################################
    # Triangulate the 3D coordinates of the common body parts in the same order for all 3 camera views
    ####################################################################################################################

    def get_realworld_coords(self, cameras_extrinsics, cameras_intrinsics):
        # Initial setup and data preparation
        labels, side_coords, front_coords, overhead_coords = self.prepare_data()

        # Get camera parameters
        K, R_gt, t_gt, P_gt, Nc = self.get_camera_params(cameras_extrinsics, cameras_intrinsics)

        # Set up dataframes for results
        real_world_coords_allparts, repr_error_allparts, repr_allparts = self.setup_dataframes(labels)

        # Process each body part
        for bidx, body_part in enumerate(labels['all']):
            print(f"Triangulating {body_part}...")
            self.process_body_part(
                bidx, body_part,
                side_coords, front_coords, overhead_coords,
                cameras_extrinsics, cameras_intrinsics,
                P_gt, Nc, real_world_coords_allparts,
                repr_error_allparts, repr_allparts
            )
        print('done')
        return real_world_coords_allparts, repr_error_allparts

    def prepare_data(self):
        labels = self.find_common_bodyparts()
        side_coords, front_coords, overhead_coords = self.get_common_camera_arrays(labels)
        return labels, side_coords, front_coords, overhead_coords

    def find_common_bodyparts(self):
        bodyparts = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            bodyparts[view] = self.get_unique_bodyparts(self.DataframeCoors[view])

        common_bodyparts = self.get_intersected_bodyparts(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'])
        all_bodyparts = self.get_all_bodyparts(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'], common_bodyparts)
        labels = self.create_labels_dict(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'], all_bodyparts, common_bodyparts)
        return labels

    def get_unique_bodyparts(self, dataframe):
        return dataframe.columns.get_level_values('bodyparts').unique()

    def get_intersected_bodyparts(self, side_bodyparts, front_bodyparts, overhead_bodyparts):
        sidexfront = np.intersect1d(side_bodyparts, front_bodyparts)
        sidexoverhead = np.intersect1d(side_bodyparts, overhead_bodyparts)
        frontxoverhead = np.intersect1d(front_bodyparts, overhead_bodyparts)
        return list(set(sidexfront) & set(sidexoverhead) & set(frontxoverhead))

    def get_all_bodyparts(self, side_bodyparts, front_bodyparts, overhead_bodyparts, common_bodyparts):
        sidexfront = np.intersect1d(side_bodyparts, front_bodyparts)
        sidexoverhead = np.intersect1d(side_bodyparts, overhead_bodyparts)
        frontxoverhead = np.intersect1d(front_bodyparts, overhead_bodyparts)
        return list(set(sidexfront) | set(sidexoverhead) | set(frontxoverhead))

    def create_labels_dict(self, side_bodyparts, front_bodyparts, overhead_bodyparts, all_bodyparts, common_bodyparts):
        side = list(set(side_bodyparts) & set(all_bodyparts))
        front = list(set(front_bodyparts) & set(all_bodyparts))
        overhead = list(set(overhead_bodyparts) & set(all_bodyparts))
        if set(all_bodyparts) == set(label_list_World):
            all_bodyparts = label_list_World
        else:
            raise ValueError('The labels in all_bodyparts are not the same as label_list_World')
        return {
            'all': all_bodyparts, 'allcommon': common_bodyparts, 'side': side,
            'front': front, 'overhead': overhead
        }

    def get_common_camera_arrays(self, labels):
        """
        Get the 3D coordinates of the common body parts in the same order for all 3 camera views
        :param labels: dictionary of lists of common body parts for each camera view
        :return: 3 numpy arrays with shape (num_rows, num_labels, 3) for side, front and overhead camera views
        """
        data = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            data[view] = self.DataframeCoors[view].to_numpy()

        num_rows = self.DataframeCoors['side'].shape[0]

        # Mapping of current labels to their column indices in the common label list
        label_to_index = {'side': {}, 'front': {}, 'overhead': {}}
        coords = {'side': [], 'front': [], 'overhead': []}

        for view in vidstuff['cams']:
            for idx, label in enumerate(labels[view]):
                pos = labels['all'].index(label)
                label_to_index[view][label] = pos
            # create empty array with shape (3, num_labels, num_rows) filled with NaNs
            coords[view] = np.full((num_rows, len(labels['all']), 3), np.nan, dtype=data[view].dtype)

        # Fill in the data for existing labels in their new positions for each camera view
        for idx, label in enumerate(labels['all']):
            for view in vidstuff['cams']:
                if label in labels[view]:
                    pos = label_to_index[view][label]
                    original_pos_mask = self.DataframeCoors[view].columns.get_loc(label)
                    original_pos = np.where(original_pos_mask)[0]
                    coords[view][:, pos, :] = data[view][:, original_pos]

        return coords['side'], coords['front'], coords['overhead']

    def get_camera_params(self, cameras_extrinsics, cameras_intrinsics):
        # Camera intrinsics
        K = [cameras_intrinsics[cam] for cam in cameras_intrinsics]

        # Camera poses: cameras are at the vertices of a hexagon
        R_gt = [cameras_extrinsics[cam]['rotm'] for cam in cameras_extrinsics]
        t_gt = [cameras_extrinsics[cam]['tvec'] for cam in cameras_extrinsics]
        P_gt = [np.dot(K[i], np.hstack((R_gt[i], t_gt[i]))) for i in range(len(K))]
        Nc = len(K)

        return K, R_gt, t_gt, P_gt, Nc

    def setup_dataframes(self, labels):
        multi_column = pd.MultiIndex.from_product([labels['all'], ['x', 'y', 'z']])
        real_world_coords_allparts = pd.DataFrame(columns=multi_column)
        multi_column_err = pd.MultiIndex.from_product([labels['all'], ['side', 'front', 'overhead'], ['x', 'y']])
        repr_error_allparts = pd.DataFrame(index=range(7170, 7900), columns=multi_column_err)  # todo remove hardcoding
        repr_allparts = pd.DataFrame(index=range(7170, 7900), columns=multi_column_err)  # todo remove hardcoding
        return real_world_coords_allparts, repr_error_allparts, repr_allparts

    def process_body_part(self, bidx, body_part, side_coords, front_coords, overhead_coords, cameras_extrinsics,
                          cameras_intrinsics, P_gt, Nc, real_world_coords_allparts, repr_error_allparts, repr_allparts):
        coords_2d_all, likelihoods = self.get_coords_and_likelihoods(bidx, side_coords, front_coords, overhead_coords)

        coords_2d, likelihoods, P_gt_bp, Nc_bp, empty_cameras = self.find_empty_cameras(coords_2d_all, likelihoods,
                                                                                        P_gt, Nc)

        real_world_coords, side_repr, front_repr, overhead_repr, side_err, front_err, overhead_err = self.triangulate_points(
            coords_2d, likelihoods, Nc_bp, P_gt_bp, cameras_extrinsics, cameras_intrinsics, coords_2d_all)

        self.store_results(
            body_part, real_world_coords, side_repr, front_repr, overhead_repr, side_err, front_err, overhead_err,
            real_world_coords_allparts, repr_error_allparts, repr_allparts
        )

    def find_empty_cameras(self, coords_2d, likelihoods, P_gt, Nc):
        empty_cameras = np.where(np.all(np.all(np.isnan(coords_2d), axis=2), axis=1))
        if len(empty_cameras) > 0:
            coords_2d = np.delete(coords_2d, empty_cameras, axis=0)
            likelihoods = np.delete(likelihoods, empty_cameras, axis=0)
            Nc_bp = len(coords_2d)
            P_gt_bp = np.delete(P_gt, empty_cameras, axis=0)
        else:
            Nc_bp = Nc
            P_gt_bp = P_gt
        return coords_2d, likelihoods, P_gt_bp, Nc_bp, empty_cameras

    def get_coords_and_likelihoods(self, bidx, side_coords, front_coords, overhead_coords):
        coords_2d_all = np.array([side_coords[:, bidx, :2], front_coords[:, bidx, :2], overhead_coords[:, bidx, :2]])
        likelihoods = np.array([side_coords[:, bidx, 2], front_coords[:, bidx, 2], overhead_coords[:, bidx, 2]])

        coords_2d_all = coords_2d_all[:, 7170:7900, :]  # todo remove hardcoding
        likelihoods = likelihoods[:, 7170:7900]  # todo remove hardcoding

        return coords_2d_all, likelihoods

    def triangulate_points(self, coords_2d, likelihoods, Nc_bp, P_gt_bp, cameras_extrinsics, cameras_intrinsics,
                           coords_2d_all):
        real_world_coords = []
        side_err = []
        front_err = []
        overhead_err = []
        side_repr = []
        front_repr = []
        overhead_repr = []

        Np = len(coords_2d[0])
        for point_idx in range(Np):
            if Nc_bp < 3:
                # where there are just 2 cameras with available data, if both have greater than pcutoff likelihood, triangulate
                conf = np.all(likelihoods[:, point_idx] > pcutoff)
                wcs = self.triangulate(coords_2d, point_idx, Nc_bp, P_gt_bp) if conf else np.array(
                    [np.nan, np.nan, np.nan, np.nan])
            else:
                conf = np.where(likelihoods[:, point_idx] > pcutoff)[0]
                wcs = self.handle_three_cameras(conf, coords_2d, point_idx, Nc_bp, P_gt_bp)

            real_world_coords.append(wcs)
            self.project_back_to_cameras(wcs, cameras_extrinsics, cameras_intrinsics, coords_2d_all, point_idx,
                                         side_err,
                                         front_err, overhead_err, side_repr, front_repr, overhead_repr)

        final = self.finalize_coords(real_world_coords, side_repr, front_repr, overhead_repr, side_err, front_err,
                                     overhead_err)
        return final

    def triangulate(self, coords_2d, point_idx, Nc_bp, P_gt_bp):
        # Gather points from each camera for the current point index
        points_from_all_cameras = [coords_2d[camera_idx, point_idx, :] for camera_idx in range(Nc_bp)]
        # Reshape to meet the input requirement of the triangulate function
        x_2d = np.array(points_from_all_cameras)
        # Assuming triangulate function accepts x_2d of shape (Nc, 2) and returns (x, y, z) coordinates
        w = triangulate(x_2d, P_gt_bp)  # Make sure P_gt and triangulate function are prepared to handle this
        return w

    def handle_three_cameras(self, conf, coords_2d, point_idx, Nc_bp, P_gt_bp):
        if len(conf) <= 1:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        elif len(conf) == 2:
            return self.triangulate(coords_2d[conf], point_idx, 2, P_gt_bp[conf])
        else:
            return self.triangulate(coords_2d, point_idx, Nc_bp, P_gt_bp)

    def project_back_to_cameras(self, wcs, cameras_extrinsics, cameras_intrinsics, coords_2d_all, point_idx, side_err,
                                front_err, overhead_err, side_repr, front_repr, overhead_repr):
        if np.all(wcs != np.nan):
            # if np.all(~np.isnan(wcs)):
            wcs[2] = wcs[2]  # + 30 # TODO REMOVE!!!!!!!!
            for c, cam in enumerate(['side', 'front', 'overhead']):
                CCS_repr, _ = cv2.projectPoints(
                    wcs[:3],
                    cv2.Rodrigues(cameras_extrinsics[cam]['rotm'])[0],
                    cameras_extrinsics[cam]['tvec'],
                    cameras_intrinsics[cam],
                    np.array([]),
                )
                repr_error = np.array([[np.nan, np.nan]]) if np.all(
                    np.isnan(coords_2d_all[c, point_idx])) else np.linalg.norm(CCS_repr - coords_2d_all[c, point_idx],
                                                                               axis=1)
                repr = np.array([[np.nan, np.nan]]) if np.all(np.isnan(coords_2d_all[c, point_idx])) else CCS_repr

                if cam == 'side':
                    side_err.append(repr_error)
                    side_repr.append(repr)
                elif cam == 'front':
                    front_err.append(repr_error)
                    front_repr.append(repr)
                elif cam == 'overhead':
                    overhead_err.append(repr_error)
                    overhead_repr.append(repr)

    def finalize_coords(self, real_world_coords, side_repr, front_repr, overhead_repr, side_err, front_err,
                        overhead_err):
        real_world_coords = np.array(real_world_coords).T
        side_repr_arr = np.array(side_repr).T
        front_repr_arr = np.array(front_repr).T
        overhead_repr_arr = np.array(overhead_repr).T
        side_err_arr = np.array(side_err).T
        front_err_arr = np.array(front_err).T
        overhead_err_arr = np.array(overhead_err).T

        if real_world_coords.shape[0] == 4:
            real_world_coords = real_world_coords[:3, :] / real_world_coords[3, :]
        real_world_coords = real_world_coords.T

        return real_world_coords, side_repr_arr, front_repr_arr, overhead_repr_arr, side_err_arr, front_err_arr, overhead_err_arr

    def store_results(self, body_part, real_world_coords, side_repr, front_repr, overhead_repr, side_err, front_err,
                      overhead_err, real_world_coords_allparts, repr_error_allparts, repr_allparts):
        if len(real_world_coords.shape) == 1:
            real_world_coords_allparts[body_part, 'x'] = np.nan
            real_world_coords_allparts[body_part, 'y'] = np.nan
            real_world_coords_allparts[body_part, 'z'] = np.nan
        else:
            real_world_coords_allparts[body_part, 'x'] = real_world_coords[:, 0]
            real_world_coords_allparts[body_part, 'y'] = real_world_coords[:, 1]
            real_world_coords_allparts[body_part, 'z'] = real_world_coords[:, 2]

        repr_allparts[body_part, 'side', 'x'] = np.squeeze(side_repr[0, :])
        repr_allparts[body_part, 'side', 'y'] = np.squeeze(side_repr[1, :])
        repr_allparts[body_part, 'front', 'x'] = np.squeeze(front_repr[0, :])
        repr_allparts[body_part, 'front', 'y'] = np.squeeze(front_repr[1, :])
        repr_allparts[body_part, 'overhead', 'x'] = np.squeeze(overhead_repr[0, :])
        repr_allparts[body_part, 'overhead', 'y'] = np.squeeze(overhead_repr[1, :])

        repr_error_allparts[body_part, 'side', 'x'] = side_err[0, :][0]
        repr_error_allparts[body_part, 'side', 'y'] = side_err[1, :][0]
        repr_error_allparts[body_part, 'front', 'x'] = front_err[0, :][0]
        repr_error_allparts[body_part, 'front', 'y'] = front_err[1, :][0]
        repr_error_allparts[body_part, 'overhead', 'x'] = overhead_err[0, :][0]
        repr_error_allparts[body_part, 'overhead', 'y'] = overhead_err[1, :][0]

    def plot_2d_prep_frame(self, view, frame_number):
        # Determine the correct video file based on the view
        if view == 'side':
            video_file = self.side_file
        elif view == 'front':
            video_file = self.front_file
        elif view == 'overhead':
            video_file = self.overhead_file
        else:
            raise ValueError('Invalid view')

        exp_day = video_file.split("\\")[-2] if view != 'front' else video_file.split("\\")[-3]
        if '_Pre_' in video_file:
            video_file_tag = '_'.join(video_file.split("\\")[-1].split("_")[0:6])
        else:
            video_file_tag = '_'.join(video_file.split("\\")[-1].split("_")[0:5])

        video_dir = os.path.join(paths['video_folder'], exp_day)
        video_files = [f for f in os.listdir(video_dir) if f.startswith(video_file_tag) and f.endswith('.avi')]

        # Select the video file corresponding to the specified view
        video_file = next((f for f in video_files if view in f), None)
        if not video_file:
            raise FileNotFoundError(f"No video file found for view {view}")

        video_path = os.path.join(video_dir, video_file)

        # Capture the specific frame from the video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Error reading frame {frame_number} from {video_path}")

        return frame

    def plot_2d_skeleton_overlay_on_frame_fromOriginal_singleView(self, frame_number, labels, view):
        # plot the video frame
        frame = self.plot_2d_prep_frame(view, frame_number)

        data_file = self.DataframeCoors[view]

        # # minus 10 from the y of overhead view
        # if view == 'overhead':
        #     data_file.loc(axis=1)['StartPlatL', 'y'] = data_file.loc(axis=1)['StartlatL', 'y'] - 7

        # Plot the 2D skeleton overlay on the frame
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for i, body_part in enumerate(labels['all']):
            if body_part in labels[view]:
                conf = data_file.loc[frame_number, (body_part, 'likelihood')]
                if conf > pcutoff:
                    x = data_file.loc[frame_number, (body_part, 'x')]
                    y = data_file.loc[frame_number, (body_part, 'y')]
                    if np.isfinite(x) and np.isfinite(y):
                        ax.scatter(x, y, color=cmap(i / len(labels['all'])), s=6, label=body_part, zorder=100)
                        # ax.text(x, y, body_part, fontsize=8, color='r')
        # draw lines for skeleton connections underneath the scatter points
        # for start, end in micestuff['skeleton']:
        #     sx = data_file.loc[frame_number, (start, 'x')]
        #     sy = data_file.loc[frame_number, (start, 'y')]
        #     ex = data_file.loc[frame_number, (end, 'x')]
        #     ey = data_file.loc[frame_number, (end, 'y')]
        #     ax.plot([sx, ex], [sy, ey], 'grey', linewidth=0.7, zorder=0)

        ax.axis('off')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        plt.title("Original")
        plt.show()
        return fig, ax

    def plot_2d_skeleton_overlay_on_frame_fromOriginal_allViews(self, frame_number, labels):
        for view in ['side', 'front', 'overhead']:
            self.plot_2d_skeleton_overlay_on_frame_fromOriginal_singleView(frame_number, labels, view)

    def plot_2d_skeleton_overlay_on_frame_fromReprojection_singleView(self, repr_allparts, frame_number, labels, view):
        # plot the video frame
        frame = self.plot_2d_prep_frame(view, frame_number)

        # get the relevant columns from repr_allparts for the view
        repr = repr_allparts.xs(view, level=1, axis=1)

        # Plot the 2D skeleton overlay on the frame
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for i, body_part in enumerate(labels['all']):
            if body_part in labels[view]:
                x = repr.loc[frame_number, (body_part, 'x')]
                y = repr.loc[frame_number, (body_part, 'y')]
                if np.isfinite(x) and np.isfinite(y):
                    ax.scatter(x, y, color=cmap(i / len(labels['all'])), s=6, label=body_part, zorder=100)
                    # ax.text(x, y, body_part, fontsize=8, color='r')
        belt_coords_hardcoded = self.TEMP_belt_coords()
        belt_coords_hardcoded = belt_coords_hardcoded.set_index(['bodyparts', 'coords'])
        for belt_part in belt_coords_hardcoded.index.get_level_values('bodyparts').unique():
            x = belt_coords_hardcoded.loc(axis=0)[belt_part, 'x'][view]
            y = belt_coords_hardcoded.loc(axis=0)[belt_part, 'y'][view]
            # make scatter with cross marker
            ax.scatter(x, y, color='red', marker='x', s=6, label=belt_part, zorder=100)
        # draw lines for skeleton connections underneath the scatter points
        for start, end in micestuff['skeleton']:
            sx, sy = repr.loc[frame_number, (start, 'x')], repr.loc[frame_number, (start, 'y')]
            ex, ey = repr.loc[frame_number, (end, 'x')], repr.loc[frame_number, (end, 'y')]
            ax.plot([sx, ex], [sy, ey], 'grey', linewidth=0.7, zorder=0)
        ax.axis('off')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        plt.title("Reprojection")
        plt.show()
        return fig, ax

    def plot_2d_skeleton_overlay_on_frame_fromReprojection_allViews(self, repr_allparts, frame_number, labels):
        for view in ['side', 'front', 'overhead']:
            self.plot_2d_skeleton_overlay_on_frame_fromReprojection_singleView(repr_allparts, frame_number, labels,
                                                                               view)

    def plot_3d_mouse(self, real_world_coords_allparts, labels, frame):
        # plt 3d scatter of all body parts at frame 500 and colour them based on the body part using viridis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis')
        coords = real_world_coords_allparts.loc[frame]
        for i, body_part in enumerate(labels['all']):
            # scatter with small marker size
            ax.scatter(coords[body_part, 'x'], coords[body_part, 'y'], coords[body_part, 'z'],
                       label=body_part, color=cmap(i / len(labels['all'])), s=10)
        # Draw lines for each connection
        for start, end in micestuff['skeleton']:
            sx, sy, sz = coords[(start, 'x')], coords[(start, 'y')], coords[(start, 'z')]
            ex, ey, ez = coords[(end, 'x')], coords[(end, 'y')], coords[(end, 'z')]
            ax.plot([sx, ex], [sy, ey], [sz, ez], 'gray')  # Draw line in gray
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        # set axes as equal scales so each tick on each axis represents the same space
        ax.axis('equal')
        # ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 for x:y:z
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def plot_3d_video(self, real_world_coords_allparts, labels):
        temp = real_world_coords_allparts.loc[400:]
        temp = temp.reset_index(drop=True)
        n_frames = len(temp)  # number of frames

        # Example setup
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10, azim=0)
        # Setting up a colormap
        n_labels = len(labels['all'])
        viridis = plt.get_cmap('viridis')  # Alternatively, you can use: viridis = plt.cm.get_cmap('viridis')
        colors = viridis(np.linspace(0, 1, n_labels))
        # colors = colormaps['viridis'](np.linspace(0, 1, n_labels))
        # Define belts (fixed in position, change visibility instead of adding/removing)
        belt1_verts = [[(0, 0, 0), (470, 0, 0), (470, 53.5, 0), (0, 53.5, 0)]]
        belt2_verts = [[(471, 0, 0), (600, 0, 0), (600, 53.5, 0), (471, 53.5, 0)]]
        belt1 = Poly3DCollection(belt1_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        belt2 = Poly3DCollection(belt2_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        ax.add_collection3d(belt1)
        ax.add_collection3d(belt2)
        # Initialize lines for parts and skeleton lines
        lines = {part: ax.plot([], [], [], 'o-', ms=2, label=part, color=colors[i])[0] for i, part in
                 enumerate(labels['all'])}
        skeleton_lines = {pair: ax.plot([], [], [], 'black', linewidth=0.5)[0] for pair in micestuff['skeleton']}

        def init():
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 53.5)
            ax.set_zlim(0, 53.5)
            ax.set_box_aspect([600 / 53.5, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            belt1.set_visible(True)
            belt2.set_visible(True)
            return [belt1, belt2] + list(lines.values()) + list(skeleton_lines.values())

        def update(frame):
            # Adjust visibility or other properties if needed
            belt1.set_visible(True)
            belt2.set_visible(True)
            for part, line in lines.items():
                x = [temp.loc[frame, (part, 'x')]]
                y = [temp.loc[frame, (part, 'y')]]
                z = [temp.loc[frame, (part, 'z')]]
                line.set_data(x, y)
                line.set_3d_properties(z)
            for (start, end), s_line in skeleton_lines.items():
                xs = [temp.loc[frame, (start, 'x')],
                      temp.loc[frame, (end, 'x')]]
                ys = [temp.loc[frame, (start, 'y')],
                      temp.loc[frame, (end, 'y')]]
                zs = [temp.loc[frame, (start, 'z')],
                      temp.loc[frame, (end, 'z')]]
                s_line.set_data(xs, ys)
                s_line.set_3d_properties(zs)
            ax.view_init(elev=10, azim=frame * 360 / n_frames)
            return [belt1, belt2] + list(lines.values()) + list(skeleton_lines.values())

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
        ani.save('walking_mouse_2.mp4', writer='ffmpeg', fps=30, dpi=300)
        plt.close(fig)  # Close the figure to avoid displaying it inline if running in a notebook


class GetALLRuns:
    def __init__(self, files=None, directory=None):
        self.files = files
        self.directory = directory

    def GetFiles(self):
        files = utils.Utils().GetlistofH5files(self.files,
                                               self.directory)  # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(
                files)  # before proceeding, check that mouse names are correctly labeled

        for j in range(0, len(files['Side'])):  # all csv files from each cam are same length so use side for all
            getdata = GetSingleExpData(files['Side'][j], files['Front'][j], files['Overhead'][j])
            getdata.map()

        #### save data


def main(directory):
    # Get all data
    GetALLRuns(directory=directory).GetFiles()


if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    main(directory)



