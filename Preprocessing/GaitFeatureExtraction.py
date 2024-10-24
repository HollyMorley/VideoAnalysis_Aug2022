import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time
from tqdm import tqdm  # Added for progress bar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from Helpers.Config_23 import *

class FeatureExtractor:
    def __init__(self, data, fps):
        """
        Initialize the FeatureExtractor with data as a DataFrame.
        """
        self.data = data
        self.fps = fps  # Frames per second of the video/data
        self.features_df = None

    def extract_features(self, frames_to_process):
        # Define joints and paws
        joints = ['Toe', 'Knuckle', 'Ankle', 'Knee']
        paws = {
            'ForepawR': {'Toe': 'ForepawToeR', 'Knuckle': 'ForepawKnuckleR', 'Ankle': 'ForepawAnkleR',
                         'Knee': 'ForepawKneeR'},
            'ForepawL': {'Toe': 'ForepawToeL', 'Knuckle': 'ForepawKnuckleL', 'Ankle': 'ForepawAnkleL',
                         'Knee': 'ForepawKneeL'},
            'HindpawR': {'Toe': 'HindpawToeR', 'Knuckle': 'HindpawKnuckleR', 'Ankle': 'HindpawAnkleR',
                         'Knee': 'HindpawKneeR'},
            'HindpawL': {'Toe': 'HindpawToeL', 'Knuckle': 'HindpawKnuckleL', 'Ankle': 'HindpawAnkleL',
                         'Knee': 'HindpawKneeL'}
        }
        coords = ['x', 'z']
        time_offsets = [-5, 0, 5]
        delta_t = 1 / self.fps  # Time difference between consecutive frames

        features_list = []
        indices = self.data.index

        for frame in frames_to_process:
            frame_features = {'Frame': frame}
            for offset in time_offsets:
                t = frame + offset
                if t in indices:
                    # For velocity calculation at time t + offset
                    t_minus = t - 1
                    t_plus = t + 1
                    for paw_name, paw_joints in paws.items():
                        for joint in joints:
                            joint_label = paw_joints.get(joint)
                            if joint_label is None:
                                continue  # Skip if joint is not available

                            for coord in coords:
                                # Position features at time t + offset
                                pos = self.data.loc[t, (joint_label, coord)]
                                feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
                                frame_features[feature_name] = pos

                                # Velocity features at time t + offset
                                if t_minus in indices and t_plus in indices:
                                    pos_minus = self.data.loc[t_minus, (joint_label, coord)]
                                    pos_plus = self.data.loc[t_plus, (joint_label, coord)]
                                    velocity = (pos_plus - pos_minus) / (2 * delta_t)
                                    velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                    frame_features[velocity_feature_name] = velocity
                                else:
                                    # Assign NaN if neighboring frames are not available
                                    velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                    frame_features[velocity_feature_name] = np.nan

                        # Angle features at time t + offset
                        for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
                            joint1_label = paw_joints.get(angle_joints[0])
                            joint2_label = paw_joints.get(angle_joints[1])
                            joint3_label = paw_joints.get(angle_joints[2])

                            if joint1_label and joint2_label and joint3_label:
                                coord1 = self.data.loc[t, (joint1_label, ['x', 'z'])].values.astype(float)
                                coord2 = self.data.loc[t, (joint2_label, ['x', 'z'])].values.astype(float)
                                coord3 = self.data.loc[t, (joint3_label, ['x', 'z'])].values.astype(float)

                                angle = self.calculate_angle(coord1, coord2, coord3)
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = angle
                            else:
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = np.nan
                else:
                    # Assign NaN for all features at time t + offset if t + offset is out of bounds
                    for paw_name, paw_joints in paws.items():
                        for joint in joints:
                            if paw_joints.get(joint) is None:
                                continue
                            for coord in coords:
                                feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
                                frame_features[feature_name] = np.nan
                                velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                frame_features[velocity_feature_name] = np.nan
                            for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = np.nan

            features_list.append(frame_features)

        # Convert list of dicts to DataFrame
        self.features_df = pd.DataFrame(features_list).set_index('Frame')

    def calculate_angle(self, point1, point2, point3):
        # Calculate angle at point2 between point1 and point3
        vector1 = point1 - point2
        vector2 = point3 - point2
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            return np.nan
        cos_theta = np.dot(vector1, vector2) / norm_product
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)


class RunClassifier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.limb_data = self.load_limb_data()
        self.coordinate_data = self.load_coordinate_data()
        smoothed_data_path = os.path.join(self.base_dir, 'smoothed_data.h5')
        if os.path.exists(smoothed_data_path):
            print(f"Loading smoothed data from {smoothed_data_path}")
            self.smoothed_coordinate_data = pd.read_hdf(smoothed_data_path, key='smoothed_data')
        else:
            print("Smoothed data not found. Computing smoothed data...")
            self.smoothed_coordinate_data = self.smooth_data()
            # Save the smoothed data
            self.smoothed_coordinate_data.to_hdf(smoothed_data_path, key='smoothed_data')
            print(f"Smoothed data saved to {smoothed_data_path}")
        self.features_df = None  # Will store the final features

    def load_limb_data(self):
        data_path = os.path.join(self.base_dir, "limb_labels.csv")
        data = pd.read_csv(data_path)
        return data

    def load_coordinate_data(self):
        subdirs = self.limb_data['Subdirectory'].unique()
        subdirs = [subdir.replace('_side_1', '') for subdir in subdirs]

        coords = []
        for subdir in subdirs:
            filename = subdir + '_mapped3D.h5'
            filepath = os.path.join(self.base_dir, "data", filename)
            if os.path.exists(filepath):
                crd_data = pd.read_hdf(filepath, key='real_world_coords')
                limb_crds = crd_data[['ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
                                      'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
                                      'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
                                      'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL']].copy()
                # Remove 'y' coordinates
                limb_crds = limb_crds.drop('y', axis=1, level=1)
                # Add filename as an index, preserving the original index as FrameIdx
                limb_crds.loc[:, 'Filename'] = filename
                limb_crds.loc[:, 'FrameIdx'] = limb_crds.index
                limb_crds.set_index(['Filename', 'FrameIdx'], inplace=True)
                # Name columns with bodyparts and coordinates
                limb_crds.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in limb_crds.columns],
                                                              names=['bodyparts', 'coords'])
                coords.append(limb_crds)
            else:
                print(f"Error: File {filename} does not exist.")
        if len(coords) > 0:
            flat_coords = pd.concat(coords)
            return flat_coords
        else:
            raise ValueError("Error: No coordinate files found.")

    def smooth_data(self):
        # Start the timer
        start_time = time.time()
        print("Starting data smoothing...")

        # Get the unique limbparts and coords
        limbparts = self.coordinate_data.columns.get_level_values('bodyparts').unique()
        coords = self.coordinate_data.columns.get_level_values('coords').unique()

        # Create an empty list to hold smoothed data for each group
        smoothed_data_list = []

        # Reset index to have 'Filename' and 'FrameIdx' as columns
        coordinate_data_reset = self.coordinate_data.reset_index()

        total_files = coordinate_data_reset['Filename'].nunique()
        file_counter = 0

        # Process each 'Filename' group separately
        for filename, group in coordinate_data_reset.groupby('Filename'):
            file_counter += 1
            print(f"Processing file {file_counter}/{total_files}: {filename}")

            # Start timing for this file
            file_start_time = time.time()

            # Set 'FrameIdx' as the index
            group = group.set_index('FrameIdx')
            # Drop 'Filename' as it's now constant for this group
            group = group.drop(columns='Filename', level=0)

            total_limbparts = len(limbparts)
            limbpart_counter = 0

            # Interpolate and smooth each limbpart and coord
            for limbpart in limbparts:
                limbpart_counter += 1
                #print(f"  Processing limbpart {limbpart_counter}/{total_limbparts}: {limbpart}")

                for coord in coords:
                    limbpart_coords = group[(limbpart, coord)].copy()

                    # Check if the Series is not all NaNs
                    if limbpart_coords.notnull().any():
                        # Interpolate missing values
                        interpolated = limbpart_coords.interpolate(
                            method='spline',
                            order=3,
                            limit=20,
                            limit_direction='both'
                        )

                        # Apply Gaussian smoothing
                        smoothed = gaussian_filter1d(interpolated.values, sigma=2)

                        # Assign back to group
                        group.loc[:, (limbpart, coord)] = smoothed
                    else:
                        # If all values are NaN, keep them as NaN
                        group.loc[:, (limbpart, coord)] = np.nan

            # End timing for this file
            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            print(f"Finished processing {filename} in {file_duration:.2f} seconds.")

            # Add 'Filename' back as a column
            group['Filename'] = filename
            # Reset index to include 'FrameIdx'
            group.reset_index(inplace=True)
            # Set index to ['Filename', 'FrameIdx']
            group.set_index(['Filename', 'FrameIdx'], inplace=True)
            # Append the group to the list
            smoothed_data_list.append(group)

        # Concatenate all smoothed groups
        smoothed_data = pd.concat(smoothed_data_list)
        # Ensure that columns are in the same order as self.coordinate_data.columns
        smoothed_data = smoothed_data.loc[:, self.coordinate_data.columns]

        # End the timer
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Data smoothing completed in {total_duration:.2f} seconds.")

        return smoothed_data

    def collect_features(self):
        print("Starting feature extraction...")
        # Create an empty list to store features from all files
        features_list = []

        # Ensure that 'Subdirectory' in limb_data matches 'Filename' in smoothed_coordinate_data
        # Map 'Subdirectory' to 'Filename'
        self.limb_data['Filename'] = self.limb_data['Subdirectory'].apply(
            lambda x: x.replace('_side_1', '') + '_mapped3D.h5')

        # Process each unique 'Filename' with progress bar
        filenames = self.limb_data['Filename'].unique()
        for filename in tqdm(filenames, desc="Collecting features"):
            # Get frames to process for this filename
            limb_data_subset = self.limb_data[self.limb_data['Filename'] == filename].copy()
            frames_to_process = limb_data_subset['Frame'].unique()
            frames_to_process = frames_to_process.astype(int).tolist()

            # Get the data for this filename
            try:
                data = self.smoothed_coordinate_data.xs(filename, level='Filename')
            except KeyError:
                print(f"Warning: Data for {filename} not found in smoothed_coordinate_data.")
                continue  # Skip this filename if data not found

            # Create FeatureExtractor with data and actual fps
            feature_extractor = FeatureExtractor(data=data, fps=247)
            # Extract features
            feature_extractor.extract_features(frames_to_process)
            # Get the features DataFrame
            features_df = feature_extractor.features_df
            # Add 'Filename' as a column
            features_df['Filename'] = filename
            # Merge with limb_data to get labels or additional info
            # Ensure 'Frame' is an integer
            features_df.reset_index(inplace=True)
            features_df['Frame'] = features_df['Frame'].astype(int)
            limb_data_subset['Frame'] = limb_data_subset['Frame'].astype(int)
            # Merge on ['Filename', 'Frame']
            merged_df = pd.merge(features_df, limb_data_subset, on=['Filename', 'Frame'], how='left')
            # remove 'Subdirectory' column # todo ?????
            # if 'Subdirectory' in merged_df.columns:
            #     merged_df = merged_df.drop(columns='Subdirectory')
            # Append to the list
            features_list.append(merged_df)

        # Concatenate all features
        if features_list:
            self.features_df = pd.concat(features_list, ignore_index=True)
            # Set index to ['Filename', 'Frame']
            self.features_df.set_index(['Filename', 'Frame'], inplace=True)
            print("Feature extraction completed.")
        else:
            print("No features were extracted.")

    def save_features(self, output_path):
        if self.features_df is not None:
            # Drop the 'Subdirectory' column if it exists
            if 'Subdirectory' in self.features_df.columns:
                self.features_df = self.features_df.drop(columns='Subdirectory')
            # Save to CSV (MultiIndex will be saved as columns)
            self.features_df.to_csv(output_path)
            print(f"Features saved to {output_path}")
        else:
            print("No features to save.")


def main():
    base_directory = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff"
    classifier = RunClassifier(base_directory)
    classifier.collect_features()
    # Save features to a CSV file
    output_features_path = os.path.join(base_directory, 'extracted_features.csv')
    classifier.save_features(output_features_path)


if __name__ == '__main__':
    main()
