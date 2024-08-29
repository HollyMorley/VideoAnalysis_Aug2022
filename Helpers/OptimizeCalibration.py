
import Helpers.MultiCamLabelling_config as opt_config
from Helpers.Config_23 import *
from Helpers.CalibrateCams import BasicCalibration
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from pycalib.calib import triangulate
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class optimize:
    def __init__(self, calibration_coords, extrinsics, intrinsics, parent_instance):
        self.calibration_coords = calibration_coords
        self.P = None
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.parent = parent_instance
        self.iteration_count = 0

    def optimise_calibration(self):
        reference_points = ['Nose', 'EarL', 'EarR', 'ForepawToeR', 'ForepawToeL', 'HindpawToeR',
                            'HindpawKnuckleR', 'Back1', 'Back3', 'Back6', 'Back9', 'Tail1', 'Tail6', 'Tail12',
                            'StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR', 'TransitionL']
        reference_data = self.get_data_for_optimisation(reference_points)

        initial_total_error, initial_errors = self.compute_reprojection_error(reference_points, reference_data)
        print(f"Initial total reprojection error for {reference_points}: \n{initial_total_error}")

        initial_flat_points = self.flatten_calibration_points()
        args = (reference_points, reference_data)

        bounds = [(initial_flat_points[i] - 3.0, initial_flat_points[i] + 3.0) for i in range(len(initial_flat_points))]

        print("Optimizing calibration points...")
        result = minimize(self.objective_function, initial_flat_points, args=args, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 10000, 'ftol': 1e-5, 'gtol': 1e-5, 'disp': False}) # 100000, 1e-15, 1e-15

        optimized_points = self.reshape_calibration_points(result.x)

        for label, views in optimized_points.items():
            for view, point in views.items():
                self.calibration_coords[label][view] = point

        calibration_data = self.recalculate_camera_parameters()

        new_total_error, new_errors = self.compute_reprojection_error(reference_points, reference_data)
        print(f"New total reprojection error for {reference_points}: \n{new_total_error}")

        return calibration_data

    def flatten_calibration_points(self):
        """
        Flattens the calibration coordinates into a 1D numpy array for optimization.
        """
        flat_points = self.calibration_coords[['side', 'front', 'overhead']].values.flatten()
        return np.array(flat_points, dtype=float)

    def objective_function(self, flat_points, *args):
        reference_points = args[0]  # Extract reference points from args
        frame_indices = args[1]  # Extract frame indices from args
        calibration_points = self.reshape_calibration_points(flat_points)
        temp_extrinsics = self.estimate_extrinsics(calibration_points)

        total_error, _ = self.compute_reprojection_error(reference_points, frame_indices, temp_extrinsics,
                                                         weighted=True)
        print("Total error: %s" %total_error)
        return total_error

    def estimate_extrinsics(self, calibration_points):
        """
        Estimates the camera extrinsics using the reshaped calibration coordinates DataFrame.
        """
        calibration_coordinates = calibration_points[['bodyparts', 'coords', 'side', 'front', 'overhead']]

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        return cameras_extrinsics

    def reshape_calibration_points(self, flat_points):
        """
        Reshapes a 1D numpy array back into the calibration coordinates DataFrame format.
        """
        reshaped_coords = self.calibration_coords.copy()
        reshaped_coords['side'] = flat_points[0::3]
        reshaped_coords['front'] = flat_points[1::3]
        reshaped_coords['overhead'] = flat_points[2::3]
        return reshaped_coords

    def recalculate_camera_parameters(self):
        """
        Recalculates camera parameters based on the updated calibration coordinates.
        """
        calibration_coordinates = self.calibration_coords[['bodyparts', 'coords', 'side', 'front', 'overhead']]

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        cameras_intrinsics = calib.cameras_intrinsics
        belt_points_WCS = calib.belt_coords_WCS
        belt_points_CCS = calib.belt_coords_CCS

        calibration_data = {
            'extrinsics': cameras_extrinsics,
            'intrinsics': cameras_intrinsics,
            'belt points WCS': belt_points_WCS,
            'belt points CCS': belt_points_CCS
        }
        return calibration_data

    def compute_frame_error(self, frame_index, labels, reference_data, extrinsics, weighted):
        cams = ["side", "front", "overhead"]
        frame_total_error = 0
        frame_errors = {label: {"side": 0, "front": 0, "overhead": 0} for label in labels}

        for label in labels:
            point_3d = self.triangulate(reference_data, label, extrinsics, frame_index)
            if point_3d is not None:
                point_3d = point_3d[:3]
                projections = self.project_to_view(point_3d, extrinsics)

                for view in cams:
                    if ~reference_data[view].loc[frame_index, label].isna().any():
                        original_x, original_y = reference_data[view].loc[frame_index, label].values
                        if view in projections:
                            projected_x, projected_y = projections[view]
                            error = np.sqrt((projected_x - original_x) ** 2 + (projected_y - original_y) ** 2)
                            if weighted:
                                weight = opt_config.REFERENCE_LABEL_WEIGHTS.get(label, 1.0)
                                error *= weight
                            frame_errors[label][view] = error
                            frame_total_error += error
        return frame_total_error, frame_errors

    def compute_reprojection_error(self, labels, reference_data, extrinsics=None, weighted=False):
        errors = {label: {"side": 0, "front": 0, "overhead": 0} for label in labels}
        cams = ["side", "front", "overhead"]
        total_error = 0

        for frame_index in reference_data['side'].index:
            for label in labels:
                point_3d = self.triangulate(reference_data, label, extrinsics,
                                            frame_index)  # todo test formatting of x and y
                if point_3d is not None:
                    point_3d = point_3d[:3]
                    projections = self.project_to_view(point_3d, extrinsics)

                    for view, frame in zip(cams, [frame_index]):
                        if ~reference_data[view].loc[frame, label].isna().any():
                            original_x, original_y = reference_data[view].loc[frame, label].values
                            if view in projections:
                                projected_x, projected_y = projections[view]
                                error = np.sqrt(
                                    (projected_x - original_x) ** 2 + (projected_y - original_y) ** 2)
                                if weighted:
                                    weight = opt_config.REFERENCE_LABEL_WEIGHTS.get(label, 1.0)
                                    error *= weight
                                errors[label][view] = error
                                total_error += error
        return total_error, errors

    def project_to_view(self, point_3d, extrinsics=None):
        projections = {}
        for view in ["side", "front", "overhead"]:
            if extrinsics is None:
                extrinsics = self.extrinsics
            if extrinsics[view] is not None:
                CCS_repr, _ = cv2.projectPoints(
                    point_3d,
                    cv2.Rodrigues(extrinsics[view]['rotm'])[0],
                    extrinsics[view]['tvec'],
                    self.intrinsics[view],
                    np.array([]),
                )
                projections[view] = CCS_repr[0].flatten()
        return projections

    def triangulate(self, reference_data, label, extrinsics=None, frame=None):
        P = []
        coords = []

        frame_mapping = {'side': frame, 'front': frame, 'overhead': frame}
        for view, frame in frame_mapping.items():
            if ~reference_data[view].loc[frame, label].isna().any():
                P.append(self.get_p(view, extrinsics=extrinsics, return_value=True))
                coords.append(reference_data[view].loc[frame, label].values)

        if len(P) < 2 or len(coords) < 2:
            return None

        P = np.array(P)
        coords = np.array(coords)

        point_3d = triangulate(coords, P)
        return point_3d

    def get_p(self, view, extrinsics=None, return_value=False):
        if extrinsics is None:
            extrinsics = self.extrinsics

        # Camera intrinsics
        K = self.intrinsics[view]

        # Camera extrinsics
        R = extrinsics[view]['rotm']
        t = extrinsics[view]['tvec']

        # Ensure t is a column vector
        if t.ndim == 1:
            t = t[:, np.newaxis]

        # Form the projection matrix
        P = np.dot(K, np.hstack((R, t)))

        if return_value:
            return P
        else:
            self.P = P

    def get_data_for_optimisation(self, reference_points):
        self.DataframeCoor_side = self.parent.DataframeCoor_side
        self.DataframeCoor_front = self.parent.DataframeCoor_front
        self.DataframeCoor_overhead = self.parent.DataframeCoor_overhead

        visibility_mask = {
            'side': self.DataframeCoor_side.loc(axis=1)[reference_points, 'likelihood'] > pcutoff,
            'front': self.DataframeCoor_front.loc(axis=1)[reference_points, 'likelihood'] > pcutoff,
            'overhead': self.DataframeCoor_overhead.loc(axis=1)[reference_points, 'likelihood'] > pcutoff
        }
        # Initialize combined visibility DataFrame with all True
        combined_visibility = pd.DataFrame(True, index=visibility_mask[next(iter(visibility_mask))].index,
                                           columns=visibility_mask[next(iter(visibility_mask))].columns.levels[0])

        # Combine visibility across all views
        for view_mask in visibility_mask.values():
            combined_visibility = combined_visibility & view_mask

        data = {'side': self.DataframeCoor_side, 'front': self.DataframeCoor_front,
                'overhead': self.DataframeCoor_overhead}

        visible_counts = combined_visibility.sum(axis=1)
        ordered_frames = visible_counts.sort_values(ascending=False)
        # mask of more than 6 visible points
        high_visibility_frames = ordered_frames[visible_counts > 9]
        visible_index = high_visibility_frames.index[:int(len(high_visibility_frames.index) * 0.5)]
        data_visible = {view: data[view].loc(axis=0)[visible_index] for view in data.keys()}

        # order dfs by side's x, select random 10 frames, then order by side's y, select random 10 frames
        nose_mask = self.create_combined_visibility_mask(data_visible, 'Nose', 0.95)
        tail12_mask = self.create_combined_visibility_mask(data_visible, 'Tail12', 0.95)
        hindpaw_toeR_mask = self.create_combined_visibility_mask(data_visible, 'HindpawToeR', 0.95)
        earR_mask = self.create_combined_visibility_mask(data_visible, 'EarR', 0.95)

        indexes = []
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Nose', 'x'][nose_mask].sort_values().index,
                                     [0.1, 0.5, 0.7, 0.99, 0.999]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Nose', 'y'][nose_mask].sort_values().index,
                                     [0.1, 0.5, 0.7, 0.9, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Tail12', 'y'][tail12_mask].sort_values().index,
                                     [0.1, 0.2, 0.7, 0.75, 0.8, 0.85, 0.95, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['front'].loc(axis=1)['Nose', 'x'][nose_mask].sort_values().index,
                                     [0.01, 0.3, 0.5, 0.7, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['front'].loc(axis=1)['Tail12', 'x'][tail12_mask].sort_values().index,
                                     [0.01, 0.3, 0.99]))
        indexes.append(self.get_index_snapshots(
            data_visible['front'].loc(axis=1)['HindpawToeR', 'x'][hindpaw_toeR_mask].sort_values().index,
            [0.01, 0.3, 0.7, 0.99]))
        indexes.append(self.get_index_snapshots(
            data_visible['front'].loc(axis=1)['HindpawToeR', 'y'][hindpaw_toeR_mask].sort_values().index,
            [0.01, 0.3, 0.7, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['EarR', 'y'][earR_mask].sort_values().index,
                                     [0.2, 0.7, 0.9]))

        # check for door positions too
        door_mask = np.logical_and.reduce((data['side'].loc(axis=1)['Door', 'likelihood'] > pcutoff,
                                           data['front'].loc(axis=1)['Door', 'likelihood'] > pcutoff,
                                           data['overhead'].loc(axis=1)['Door', 'likelihood'] > pcutoff))
        indexes.append(self.get_index_snapshots(data['side'].loc(axis=1)['Door', 'y'][door_mask].sort_values().index,
                                                [0.001, 0.005, 0.01, 0.05, 0.9, 0.95]))

        # remove any duplicates
        flattened_list = [item for sublist in indexes for item in sublist]
        unique_items = set(flattened_list)
        unique_list = list(unique_items)

        reference_data = {view: data[view].loc(axis=0)[unique_list].loc(axis=1)[reference_points + ['Door']] for view in
                          data.keys()}
        for view in reference_data.keys():
            mask = reference_data[view].xs(axis=1, level=1, key='likelihood') > 0.95
            reference_data[view] = reference_data[view][mask]
            reference_data[view].drop('likelihood', axis=1, level=1, inplace=True)

        return reference_data

    def show_side_view_frames(self, frames):
        """
        Displays all frames extracted for the side view.
        """
        # frames = reference_data['side'].index
        video_path = r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1.avi"
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found at {video_path}")
        else:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)

            # Iterate over the frames
            for frame_number in frames:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                # Check if the frame is captured correctly
                if not ret:
                    print(f"Error: Could not read frame {frame_number} from the video.")
                    continue  # Skip to the next frame if the current frame is not read correctly

                # Display the frame using Matplotlib
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f'Side View - Frame {frame_number}')
                plt.axis('off')
                plt.show()

            # Release the video capture object
            cap.release()

    def create_combined_visibility_mask(self, data_visible, bodypart, pcutoff):
        side_mask = data_visible['side'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff
        front_mask = data_visible['front'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff
        overhead_mask = data_visible['overhead'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff

        # Check visibility in at least two views (side & front, side & overhead, overhead & front) or all three
        combined_mask = (side_mask & front_mask) | (side_mask & overhead_mask) | (overhead_mask & front_mask)
        return combined_mask

    def get_index_snapshots(self, frames, positions):
        total_frames = len(frames)
        indexes = []
        for p in positions:
            idx = frames[int(total_frames * p)]
            indexes.append(idx)
        return indexes