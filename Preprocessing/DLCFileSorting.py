import pandas as pd
import os
import shutil

from Helpers.Config_23 import *

dlc_dir = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round2"  # Adjust if needed
dlc_dest = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round3_Oct24"

MouseIDs = micestuff['mice_IDs']

# Function to create directories if they don't exist
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory: {path}')


# Function to find and copy files recursively based on exp_cats and MouseIDs
def copy_files_recursive(src_dir, dest_dir, current_dict, current_path, MouseIDs, overwrite):
    if isinstance(current_dict, dict):
        # Check if we are at the level where 'A' and 'B' keys are present
        if 'A' in current_dict and 'B' in current_dict:
            # Process for both 'A' and 'B' groups
            for mouse_group in ['A', 'B']:
                if mouse_group in MouseIDs:
                    for date in current_dict[mouse_group]:
                        print(f'Processing: {current_path}, {mouse_group}, {date}')
                        copy_files_for_mouse_group(src_dir, dest_dir, current_path, mouse_group, date, MouseIDs,
                                                   overwrite)
        else:
            # Continue recursion if we haven't reached 'A'/'B' level yet
            for key, value in current_dict.items():
                new_path = os.path.join(current_path, key)
                copy_files_recursive(src_dir, dest_dir, value, new_path, MouseIDs, overwrite)
    else:
        # If current_dict is not a dict, something is wrong, as we should not reach here
        raise ValueError("Unexpected structure in experiment categories.")


# Function to find and copy files for a specific mouse group and date, stitching if needed
def copy_files_for_mouse_group(src_dir, dest_dir, current_path, mouse_group, date, MouseIDs, overwrite):
    final_dest_dir = os.path.join(dest_dir, current_path)  # Keep experiment structure, but no A/B in folder path
    ensure_dir_exists(final_dest_dir)

    file_found = False
    for mouse_id in MouseIDs[mouse_group]:
        current_src_dir = os.path.join(src_dir, date)

        # Gather all files for this mouse_id on this date, ignoring directories (e.g., calibration)
        relevant_files = [file_name for file_name in os.listdir(current_src_dir)
                          if file_name.endswith('.h5') and f"FAA-{mouse_id}" in file_name and os.path.isfile(
                os.path.join(current_src_dir, file_name))]

        # Organize them by segments ('', '_2', '_3', etc.)
        relevant_files.sort()  # This will ensure '' comes first, followed by '_2', '_3'

        if not relevant_files:
            print(f"No relevant h5 files found for {mouse_id} on {date}")
            continue

        if len(relevant_files) > 3:
            # Initialize DataFrame for stitched data
            stitched_dfs = {'side': pd.DataFrame(), 'front': pd.DataFrame(), 'overhead': pd.DataFrame()}

            for cam in stitched_dfs.keys():
                cam_files = [file_name for file_name in relevant_files if cam in file_name]
                # order the files in the correct order by moving the last file to the first position
                cam_files.insert(0, cam_files.pop())

                dest_file = os.path.join(final_dest_dir, cam_files[0])

                if overwrite or not os.path.exists(dest_file):
                    # stitch all available files for the current camera
                    cam_dfs = []
                    for file_name in cam_files:
                        src_file = os.path.join(current_src_dir, file_name)
                        df = pd.read_hdf(src_file)
                        cam_dfs.append(df)
                    stitched_dfs[cam] = pd.concat(cam_dfs, ignore_index=True)

                    # Save the stitched data to the destination directory

                    print(f'Stitching and saving {cam_files} to {dest_file}')
                    stitched_dfs[cam].to_hdf(dest_file, key='df_with_missing', mode='w')
                    file_found = True
                else:
                    print(f'Skipped stitching {cam_files} (already exists in {final_dest_dir})')
                    file_found = True
        else:
            # Only 3 files (one each for side, front and overhead), no need to stitch. Just copy the 3 files.
            for file_name in relevant_files:
                src_file = os.path.join(current_src_dir, file_name)
                dest_file = os.path.join(final_dest_dir, file_name)
                if overwrite or not os.path.exists(dest_file):
                    print(f'Copying {src_file} to {dest_file}')
                    shutil.copyfile(src_file, dest_file)
                    file_found = True
                else:
                    print(f'Skipped copying {file_name} (already exists in {final_dest_dir})')
                    file_found = True
    if not file_found:
        print(f"No files found for {mouse_group} on {date}")


def manual_changes():
    # Define the paths for destination (where modifications occur) and source (where new files are copied from)
    highlow_dir = os.path.join(dlc_dest, 'APAChar_HighLow', 'Extended')
    source_dir = dlc_dir  # This is the directory where the un-copied files are

    # Define mouse ID for the changes
    mouse_id = '1035243'

    # Define manual date modifications
    changes = [
        {'action': 'delete', 'mouse_id': mouse_id, 'date': '20230325', 'day': 'Day1'},
        {'action': 'delete', 'mouse_id': mouse_id, 'date': '20230326', 'day': 'Day2'},
        {'action': 'move', 'mouse_id': mouse_id, 'date': '20230327', 'src_day': 'Day3', 'dst_day': 'Day1'},
        {'action': 'move', 'mouse_id': mouse_id, 'date': '20230328', 'src_day': 'Day4', 'dst_day': 'Day2'},
        {'action': 'copy', 'mouse_id': mouse_id, 'date': '20230329', 'src_day': '20230329', 'dst_day': 'Day3'},
        {'action': 'copy', 'mouse_id': mouse_id, 'date': '20230330', 'src_day': '20230330', 'dst_day': 'Day4'}
    ]

    for change in changes:
        if change['action'] == 'delete':
            # Delete relevant files
            for file in os.listdir(highlow_dir):
                if f"{change['mouse_id']}_{change['date']}" in file:
                    file_path = os.path.join(highlow_dir, change['day'], file)
                    if os.path.exists(file_path):
                        print(f"Deleting {file_path}")
                        os.remove(file_path)

        elif change['action'] == 'move':
            # Move files from one destination day to another within dlc_dest
            src_day = os.path.join(highlow_dir, change['src_day'])
            dst_day = os.path.join(highlow_dir, change['dst_day'])
            for file in os.listdir(src_day):
                if f"{change['mouse_id']}_{change['date']}" in file:
                    src_file = os.path.join(src_day, file)
                    dst_file = os.path.join(dst_day, file)
                    print(f"Moving {src_file} to {dst_file}")
                    shutil.move(src_file, dst_file)

        elif change['action'] == 'copy':
            # Copy files from dlc_dir (source) to the appropriate destination day
            src_day = os.path.join(source_dir, change['src_day'])  # Source is dlc_dir with the date
            dst_day = os.path.join(highlow_dir, change['dst_day'])  # Destination is within dlc_dest

            # Ensure destination directory exists
            ensure_dir_exists(dst_day)

            for file in os.listdir(src_day):
                if f"{change['mouse_id']}_{change['date']}" in file:
                    src_file = os.path.join(src_day, file)
                    dst_file = os.path.join(dst_day, file)
                    print(f"Copying {src_file} to {dst_file}")
                    shutil.copy(src_file, dst_file)


def final_checks():
    # List all directories in the source folder (dlc_dir)
    all_folders = os.listdir(dlc_dir)

    # Flatten the exp_cats dictionary to get all the dates
    included_dates = []
    for category, subcats in exp_cats.items():
        for subcat, phases in subcats.items():
            for phase, days in phases.items():
                if isinstance(days, dict):
                    for day, mice_dates in days.items():
                        for mouse_group, dates in mice_dates.items():
                            included_dates.extend(dates)

    # Exclude manually handled days
    manual_dates = ['20230329', '20230330']

    # Check which folders are not included in exp_cats or manually handled
    excluded_folders = [folder for folder in all_folders if folder not in included_dates and folder not in manual_dates]

    # Print the excluded folders
    if excluded_folders:
        print(f"The following folders are not included in the experiment categories or manual changes: {excluded_folders}")
    else:
        print("All folders are accounted for in exp_cats or manual changes.")


# Example usage
overwrite = False  # Set this to True if you want to overwrite files
copy_files_recursive(dlc_dir, dlc_dest, exp_cats, '', MouseIDs, overwrite)

# Perform manual changes
manual_changes()

# Final check for any folders left out
final_checks()
