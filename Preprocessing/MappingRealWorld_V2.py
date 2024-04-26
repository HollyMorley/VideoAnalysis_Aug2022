from Helpers.utils_3d_reconstruction import CameraData, BeltPoints
from Helpers import utils
from Helpers.Config_23 import *

import pandas as pd
import numpy as np
import os
import cv2
import warnings
from scipy.spatial.transform import Rotation
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
from scipy.spatial.transform import Rotation as R


class MapExperiment:
    def __init__(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, belt_coords, snapshot_paths):
        self.DataframeCoor_side = DataframeCoor_side
        self.DataframeCoor_front = DataframeCoor_front
        self.DataframeCoor_overhead = DataframeCoor_overhead

        self.cameras = CameraData(snapshot_paths)

        self.cameras_specs = self.cameras.specs
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        self.belt_pts = BeltPoints(belt_coords)
        self.belt_coords_CCS = self.belt_pts.coords_CCS
        self.belt_coords_WCS = self.belt_pts.coords_WCS

    def plot_CamCoorSys(self):
        fig, ax = self.belt_pts.plot_CCS(self.cameras)
        return fig, ax

    def plot_WorldCoorSys(self):
        fig, ax = self.belt_pts.plot_WCS()
        return fig, ax

    def estimate_pose(self):
        cameras_extrinsics = self.cameras.compute_cameras_extrinsics(self.belt_coords_WCS, self.belt_coords_CCS)
        print('Reprojection errors:\n%s' %[(cam, cameras_extrinsics[cam]['repr_err']) for cam in cameras_extrinsics])
        return cameras_extrinsics

    def estimate_pose_with_guess(self):
        cameras_extrinsics_ini_guess = self.cameras.compute_cameras_extrinsics(
            self.belt_coords_WCS,
            self.belt_coords_CCS,
            use_extrinsics_ini_guess=True
        )
        print('Reprojection errors (w/ initial guess):\n%s' %[(cam, cameras_extrinsics_ini_guess[cam]['repr_err'])
                                                              for cam in cameras_extrinsics_ini_guess])
        return cameras_extrinsics_ini_guess

    def plot_cam_locations_and_pose(self, cameras_extrinsics):
        fig, ax = self.belt_pts.plot_WCS()

        for cam in self.cameras.specs:

            rot_cam_my_def = cameras_extrinsics[cam]['rotm'].T

            rot_cam_opencv = cameras_extrinsics[cam]['rotm']
            cob_cam_opencv = cameras_extrinsics[cam]['rotm'].T  # cob=change of basis

            vec_WCS_to_CCS = - cob_cam_opencv @ cameras_extrinsics[cam]['tvec']

            # add scatter
            ax.scatter(
                vec_WCS_to_CCS[0],
                vec_WCS_to_CCS[1],
                vec_WCS_to_CCS[2],
                s=50,
                c="b",
                marker=".",
                linewidth=0.5,
                alpha=1,
            )

            # add text
            ax.text(
                vec_WCS_to_CCS[0, 0],
                vec_WCS_to_CCS[1, 0],
                vec_WCS_to_CCS[2, 0],
                s=cam,
                c="b",
            )

            # add pose
            for row, color in zip(rot_cam_opencv, ["r", "g", "b"]):
                ax.quiver(
                    vec_WCS_to_CCS[0],
                    vec_WCS_to_CCS[1],
                    vec_WCS_to_CCS[2],
                    row[0],
                    row[1],
                    row[2],
                    color=color,
                    length=500,
                    arrow_length_ratio=0,
                    normalize=True,
                    linewidth=2,
                )
                ax.axis("equal")
        return fig, ax

    def get_CamObj(self, cameras_extrinsics, cameras_intrinsics, cameras_specs):
        camera_models = {'side': None, 'front': None, 'overhead': None}
        for cam_name in cameras_intrinsics.keys():
            r = R.from_matrix(cameras_extrinsics[cam_name]['rotm'])
            quaternion = r.as_quat()  # returns (x, y, z, w)
            width = cameras_specs[cam_name]["x_size_px"]
            height = cameras_specs[cam_name]["y_size_px"]
            K = cameras_intrinsics[cam_name] # Intrinsic matrix
            distortion = np.zeros((5, 1))  # Assuming no distortion
            rect = None  # Assuming no rectification is needed
            rotm = cameras_extrinsics[cam_name]['rotm']
            t = cameras_extrinsics[cam_name]['tvec']
            camera_center = -np.dot(rotm.T, tvec) # Compute the camera center

            # # Define the 180-degree rotation matrix around the X-axis
            # R_x_180 = np.array([
            #     [1, 0, 0],
            #     [0, -1, 0],
            #     [0, 0, -1]
            # ])
            #
            # if cam_name == 'side' or cam_name == 'front':
            #     # For the side camera, adjust the rotation matrix
            #     new_rotm = np.dot(R_x_180, rotm)
            # else:
            #     new_rotm = rotm
            #
            # # Compute the new [R|t] matrix for the updated rotation
            # R_t = np.hstack([new_rotm, t.reshape(-1, 1)])
            #
            #
            # # Compute the new projection matrix P
            # P_new = np.dot(K, R_t)

            # Define rotation matrices
            # Rotate Y-axis by 90 degrees clockwise
            rotate_y_90 = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])

            # Flip Z-axis (180-degree rotation around X-axis)
            flip_z = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])

            # Combine the transformations
            combined_rotation = np.dot(flip_z, rotate_y_90)

            # Adjust the rotation matrix
            adjusted_rotm = np.dot(combined_rotation, rotm)

            # calculate P_adjusted
            translation_vector = cameras_extrinsics[cam_name]['tvec']
            adjusted_R_t = np.hstack([adjusted_rotm, translation_vector])
            P_adjusted = np.dot(K, adjusted_R_t)


            # # Define a rotation matrix that rotates points -90 degrees about the X-axis
            # rot_x_neg_90 = np.array([
            #     [1, 0, 0],
            #     [0, 0, -1],
            #     [0, 1, 0]
            # ])
            #
            # # For each camera, adjust the rotation matrix
            # rotm = cameras_extrinsics[cam_name]['rotm']
            # adjusted_rotm = np.dot(rot_x_neg_90, rotm)
            #
            # # Extracting only the first 3 elements of the last column (3x1 vector)
            # translation_vector = cameras_extrinsics[cam_name]['full'][:3, 3].reshape(3, 1)
            #
            # # Now rebuild the full extrinsic matrix [R|t] with the adjusted rotation matrix
            # adjusted_R_t = np.hstack([adjusted_rotm, translation_vector])
            #
            # # Re-calculate the projection matrix with the adjusted extrinsic matrix
            # P_adjusted = np.dot(K, adjusted_R_t)
            # # Flip the sign of the third column (z-component)
            # P_adjusted[:, 2] = -P_adjusted[:, 2]
            #
            # # Optionally, if adjusting the translation associated with the depth is necessary:
            # P_adjusted[:, 3] = -P_adjusted[:,
            #                     3]  # Be cautious with flipping the translation vector sign if not necessary.
            # ############################## TEMPORARY FIX ##############################
            # # Manually set P[2,2] to exactly 1.0
            # P_adjusted[2, 2] = 1.0
            # # Set the last column of P_adjusted to zeros
            # P_adjusted[:, 3] = 0
            # ##########################################################################

            # Initialize CameraModel
            camera_model = CameraModel(
                name=cam_name,
                width=width,
                height=height,
                _rquat=quaternion,
                _camcenter=camera_center,
                P=P_adjusted,  # Assuming P is used here as K*[R|t]
                K=K,
                distortion=distortion,
                rect=rect
            )

        return camera_models

            # rot_90 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
            #
            # K = cameras_intrinsics[cam_name]  # Intrinsic matrix
            # R = cameras_extrinsics[cam_name]['rotm']  # Rotation matrix
            # #R = cameras_extrinsics[cam_name]['full'][:3, :3]  # Rotation matrix
            # t = cameras_extrinsics[cam_name]['tvec']  # Translation vector
            # # Construct [R|t]
            # extrinsic_matrix = np.hstack((R, t))
            # # Calculate projection matrix P = K * [R|t]
            # # C = np.eye(4)
            # # C[:3, :3] = K @ R
            # # C[:3, 3] = K @ R @ t.T[0]
            # # P = C[:3, :]
            # P = np.dot(K, extrinsic_matrix)
            # P /= P[2, 2]
            # # Extract rotation vector
            # rvec = cameras_extrinsics[cam_name]['rvec'].flatten()
            # # Convert rotation vector to rotation quaternion
            # r = Rotation.from_rotvec(rvec)
            # # Calculate camera center
            # camcenter = -np.dot(R.T, t)[:, 0]
            # # Extract image dimensions
            # image_width = cameras_intrinsics[cam_name][0][-1]*2
            # image_height = cameras_intrinsics[cam_name][1][-1]*2
            # # Initialize CameraModel
            # camera_models[cam_name] = CameraModel(
            #     name=cam_name,
            #     width=image_width,
            #     height=image_height,
            #     _rquat=r.as_quat(),
            #     _camcenter=camcenter,
            #     P=P,
            #     K=K,
            #     distortion=np.zeros(5),
            #     rect=None
            # )

    def triangulate_points(self, pts1, pts2, camera_name1, camera_name2, cameras_extrinsics):
        ############################################################################
        ################################# OLD CODE #################################
        ############################################################################
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        # Get extrinsic parameters
        rvec1 = cameras_extrinsics[camera_name1]['rvec']
        tvec1 = cameras_extrinsics[camera_name1]['tvec']
        rvec2 = cameras_extrinsics[camera_name2]['rvec']
        tvec2 = cameras_extrinsics[camera_name2]['tvec']

        # Get intrinsic parameters
        camera_matrix1 = self.cameras_intrinsics[camera_name1]
        camera_matrix2 = self.cameras_intrinsics[camera_name2]

        # Perform triangulation
        points_3d_homogeneous = cv2.triangulatePoints(
            np.hstack((cv2.Rodrigues(rvec1)[0], tvec1)), np.hstack((cv2.Rodrigues(rvec2)[0], tvec2)),
            pts1, pts2
        )
        points_3d = points_3d_homogeneous / points_3d_homogeneous[3]
        points_3d_euclidean = points_3d[:3] / points_3d[3]
        return points_3d_euclidean



class GetSingleExpData:
    def __init__(self, side_file, front_file, overhead_file):
        self.side_file = side_file
        self.front_file = front_file
        self.overhead_file = overhead_file

        # Get (and clean up) dataframes for one mouse (/vid) for each view point
        DataframeCoor_side = pd.read_hdf(self.side_file)
        try:
            self.DataframeCoor_side = DataframeCoor_side.loc(axis=1)[vidstuff['scorers']['side']].copy()
        except:
            self.DataframeCoor_side = DataframeCoor_side.loc(axis=1)[vidstuff['scorers']['side_new']].copy()

        DataframeCoor_front = pd.read_hdf(self.front_file)
        self.DataframeCoor_front = DataframeCoor_front.loc(axis=1)[vidstuff['scorers']['front']].copy()

        DataframeCoor_overhead = pd.read_hdf(self.overhead_file)
        self.DataframeCoor_overhead = DataframeCoor_overhead.loc(axis=1)[vidstuff['scorers']['overhead']].copy()


    def find_common_bodyparts(self):
        # for each camera pair, find the common body parts
        side_bodyparts = self.DataframeCoor_side.columns.get_level_values('bodyparts').unique()
        front_bodyparts = self.DataframeCoor_front.columns.get_level_values('bodyparts').unique()
        overhead_bodyparts = self.DataframeCoor_overhead.columns.get_level_values('bodyparts').unique()

        common_bodyparts_sidexfront = list(np.intersect1d(side_bodyparts, front_bodyparts))
        common_bodyparts_sidexoverhead = list(np.intersect1d(side_bodyparts, overhead_bodyparts))
        common_bodyparts_frontxoverhead = list(np.intersect1d(front_bodyparts, overhead_bodyparts))

        # find common body parts between all 3 camera views
        common_bodyparts = list(set(common_bodyparts_sidexfront) & set(common_bodyparts_sidexoverhead) & set(
            common_bodyparts_frontxoverhead))

        # find all bodyparts that are in at least 2 camera views
        all_bodyparts = list(set(common_bodyparts_sidexfront) | set(common_bodyparts_sidexoverhead) | set(
            common_bodyparts_frontxoverhead))

        # find the values in each camera that are in all_bodyparts
        side = list(set(side_bodyparts) & set(all_bodyparts))
        front = list(set(front_bodyparts) & set(all_bodyparts))
        overhead = list(set(overhead_bodyparts) & set(all_bodyparts))

        # remove Hand from all lists where it is present
        lists_to_check = [all_bodyparts, common_bodyparts, common_bodyparts_sidexfront,
                          common_bodyparts_sidexoverhead, common_bodyparts_frontxoverhead,
                          side, front, overhead]

        for lst in lists_to_check:
            if 'Hand' in lst:
                lst.remove('Hand')
            # sort the lists if anything inside
            if len(lst) > 0:
                lst.sort()

        labels = {'all': all_bodyparts,
                'allcommon': common_bodyparts,
                'sidexfront': common_bodyparts_sidexfront,
                'sidexoverhead': common_bodyparts_sidexoverhead,
                'frontxoverhead': common_bodyparts_frontxoverhead,
                'side': side,
                'front': front,
                'overhead': overhead}
        return labels

    def triangulate(self, mapping_obj, cameras_extrinsics, cameras_intrinsics, cameras_specs):
        # Get dictionary of camera objects
        cams = mapping_obj.get_CamObj(cameras_extrinsics, cameras_intrinsics, cameras_specs)


        ############################################################################################################################
        ############################### sort labels from all 3 cameras into a common array structure ###############################
        #todo may change the below a lot but keeping as a placeholder for now
        # suppress future warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        labels = self.find_common_bodyparts()
        side_all = self.DataframeCoor_side.loc(axis=1)[labels['side'],:]
        front_all = self.DataframeCoor_front.loc(axis=1)[labels['front'],:]
        overhead_all = self.DataframeCoor_overhead.loc(axis=1)[labels['overhead'],:]

        # find the bodyparts in labels['all'] that are not in labels['side'], labels['front'] or labels['overhead']
        missing_side = list(set(labels['all']) - set(labels['side']))
        missing_front = list(set(labels['all']) - set(labels['front']))
        missing_overhead = list(set(labels['all']) - set(labels['overhead']))

        missing_side_df = pd.DataFrame(index=side_all.index, columns=pd.MultiIndex.from_product([missing_side,['x', 'y', 'likelihood']]))
        missing_front_df = pd.DataFrame(index=front_all.index, columns=pd.MultiIndex.from_product([missing_front,['x', 'y', 'likelihood']]))
        missing_overhead_df = pd.DataFrame(index=overhead_all.index, columns=pd.MultiIndex.from_product([missing_overhead,['x', 'y', 'likelihood']]))

        side_all = pd.concat([side_all, missing_side_df], axis=1)
        front_all = pd.concat([front_all, missing_front_df], axis=1)
        overhead_all = pd.concat([overhead_all, missing_overhead_df], axis=1)

        # sort the order of the columns in the dataframes
        side_all = side_all.reindex(labels['all'], axis=1, level='bodyparts')
        front_all = front_all.reindex(labels['all'], axis=1, level='bodyparts')
        overhead_all = overhead_all.reindex(labels['all'], axis=1, level='bodyparts')

        side_coords = np.transpose(side_all.values.reshape(-1, len(side_all.columns) // 3, 3), (2,0,1))
        front_coords = np.transpose(front_all.values.reshape(-1, len(front_all.columns) // 3, 3), (2,0,1))
        overhead_coords = np.transpose(overhead_all.values.reshape(-1, len(overhead_all.columns) // 3, 3), (2,0,1))

        for bidx, body_part in enumerate(labels['all']):
            side_coord = np.array([side_coords[0,:,bidx], side_coords[1,:,bidx]])
            front_coord = np.array([front_coords[0,:,bidx], front_coords[1,:,bidx]])
            overhead_coord = np.array([overhead_coords[0,:,bidx], overhead_coords[1,:,bidx]])

            # Triangulate points between camera 1 and camera 2
            points_3d_side_front = mapping_obj.triangulate_points(side_coord, front_coord, 'side', 'front', cameras_extrinsics, cameras_intrinsics)

            # Triangulate points between camera 1 and camera 3
            points_3d_side_overhead = mapping_obj.triangulate_points(side_coord, overhead_coord, 'side', 'overhead', cameras_extrinsics, cameras_intrinsics)

            # Triangulate points between camera 2 and camera 3
            points_3d_front_overhead = mapping_obj.triangulate_points(front_coord, overhead_coord, 'front', 'overhead', cameras_extrinsics, cameras_intrinsics)

            points_3d_merged = (points_3d_side_front + points_3d_side_overhead + points_3d_front_overhead) / 3.0



    def map(self):
        belt_coords = self.get_belt_coords()
        snapshot_paths = self.save_video_frames()

        mapping_obj = MapExperiment(self.DataframeCoor_side, self.DataframeCoor_front, self.DataframeCoor_overhead, belt_coords, snapshot_paths)
        cameras_extrinsics = mapping_obj.estimate_pose()
        cameras_extrinsics_guess = mapping_obj.estimate_pose_with_guess()
        cameras_intrinsics = mapping_obj.cameras_intrinsics
        cameras_specs = mapping_obj.cameras_specs


        # plotting to check if the mapping is correct
        ccs_fig, ccs_ax = mapping_obj.plot_CamCoorSys()
        wcs_fig, wcs_ax = mapping_obj.plot_WorldCoorSys()
        loc_and_pose_fig, lp_ax = mapping_obj.plot_cam_locations_and_pose(cameras_extrinsics)

        triang = self.triangulate(mapping_obj, cameras_extrinsics, cameras_intrinsics, cameras_specs)

        # Save the plots
        path = '\\'.join(self.side_file.split("\\")[:-1])
        if '_Pre_' in self.side_file:
            file_tag = '_'.join(self.side_file.split("\\")[-1].split("_")[0:6])
        else:
            file_tag = '_'.join(self.side_file.split("\\")[-1].split("_")[0:5])

        ccs_fig.savefig(os.path.join(path, f"{file_tag}_CCS.png"))
        wcs_fig.savefig(os.path.join(path, f"{file_tag}_WCS.png"))
        loc_and_pose_fig.savefig(os.path.join(path, f"{file_tag}_Loc_and_Pose.png"))




    def get_belt_coords(self):
        side_mask = np.all(self.DataframeCoor_side.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], 'likelihood'] > 0.99, axis=1)
        front_mask = np.all(self.DataframeCoor_front.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], 'likelihood'] > 0.99, axis=1)
        overhead_mask = np.all(self.DataframeCoor_overhead.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], 'likelihood'] > 0.99, axis=1)

        belt_coords_side = self.DataframeCoor_side.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], ['x', 'y']][side_mask]
        belt_coords_front = self.DataframeCoor_front.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], ['x', 'y']][front_mask]
        belt_coords_overhead = self.DataframeCoor_overhead.loc(axis=1)[['StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL'], ['x', 'y']][overhead_mask]

        side_mean = belt_coords_side.mean(axis=0)
        front_mean = belt_coords_front.mean(axis=0)
        overhead_mean = belt_coords_overhead.mean(axis=0)

        # concatenate the mean values of the belt coordinates from the 3 camera views with the camera names as columns
        belt_coords = pd.concat([side_mean, front_mean, overhead_mean], axis=1)
        belt_coords.columns = ['side', 'front', 'overhead']
        belt_coords.reset_index(inplace=True, drop=False)

        return belt_coords

    def save_video_frames(self):
        # save frame from side front and overhead camera views
        exp_day = self.side_file.split("\\")[-2]
        if '_Pre_' in self.side_file:
            video_file_tag = '_'.join(self.side_file.split("\\")[-1].split("_")[0:6])
        else:
            video_file_tag = '_'.join(self.side_file.split("\\")[-1].split("_")[0:5])
        video_dir = os.path.join(paths['video_folder'], exp_day)
        # find files beginning with video_file_tag adn ending avi under video_dir
        video_files = [f for f in os.listdir(video_dir) if
                        f.startswith(video_file_tag) and f.endswith('.avi')]
        if len(video_files) == 3:
            # get frame where all corners are visible but mouse is not present
            corner_front_mask = np.all(self.DataframeCoor_front.loc(axis=1)[['StartPlatL', 'StartPlatR', 'TransitionL', 'TransitionR'], 'likelihood'] > pcutoff, axis=1)
            corner_side_mask = np.all(self.DataframeCoor_side.loc(axis=1)[['StartPlatL', 'StartPlatR', 'TransitionL', 'TransitionR'], 'likelihood'] > pcutoff, axis=1)
            corner_mask = corner_front_mask & corner_side_mask
            mouse_mask = np.all(self.DataframeCoor_side.loc(axis=1)[['Nose', 'EarR', 'Tail1', 'Tail12', 'Door'], 'likelihood'] < pcutoff, axis=1)
            mask = corner_mask & mouse_mask
            frame_num = self.DataframeCoor_side[mask].index[0]

            # Open video files
            cap_dict = {}
            for video_file in video_files:
                cap_dict[video_file] = cv2.VideoCapture(os.path.join(video_dir, video_file))

            # Read and save frames
            Path = {'side': None, 'front': None, 'overhead': None}
            for video_file, cap in cap_dict.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    # find which of side, front and overhead is in the video_file name and save cam_name as that value
                    cam_name = [cam for cam in ['side', 'front', 'overhead'] if cam in video_file][0]
                    path = os.path.join(paths['data_folder'], exp_day, f"{video_file.split('.')[0]}_{frame_num}.jpg")
                    cv2.imwrite(path, frame)
                    Path[cam_name] = path
                else:
                    print(f"Error reading frame for {video_file}")

            # Release video capture objects
            for cap in cap_dict.values():
                cap.release()

            return Path
        else:
            raise ValueError(f"No video files found for {video_file_tag}")



class GetALLRuns:
    def __init__(self, files=None, directory=None):
        self.files = files
        self.directory = directory

    def GetFiles(self):
        files = utils.Utils().GetlistofH5files(self.files, self.directory)  # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(files) # before proceeding, check that mouse names are correctly labeled

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



