import os
import pandas as pd
import inspect
from typing import Optional, List, Dict, Tuple
import itertools
import datetime

from Analysis.Tools.config import (
    global_settings, condition_specific_settings, instance_settings
)
from Helpers.Config_23 import *


def initialize_experiment(condition, exp, day, compare_condition, settings_to_log,
                          base_save_dir_no_c, condition_specific_settings) -> Tuple:
    # Collect stride data and create directories.
    stride_data, stride_data_compare = collect_stride_data(condition, exp, day, compare_condition)
    base_save_dir, base_save_dir_condition = set_up_save_dir(
        condition, exp, base_save_dir_no_c
    )
    script_name = os.path.basename(inspect.getfile(inspect.currentframe()))
    log_settings(settings_to_log, base_save_dir, script_name)

    # collect feature data from each mouse and stride
    feature_data = {}
    feature_data_compare = {}
    for stride in global_settings["stride_numbers"]:
        for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
            # Load and preprocess data for each mouse.
            filtered_data_df = load_and_preprocess_data(mouse_id, stride, condition, exp, day, measures_list_manual_reduction)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day, measures_list_manual_reduction)
            feature_data_compare[(stride, mouse_id)] = filtered_data_comparison_df

    # Edit the feature data based on my manual feature reduction
    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_df = process_features(feature_data_df)

    feature_data_compare = pd.concat(feature_data_compare, axis=0)
    feature_data_compare = process_features(feature_data_compare)

    return feature_data_df, feature_data_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition


def process_features(df):
    # Replace back heights with their mean.
    back_cols = [col for col in df.columns if col.startswith('back_height|')]
    df['back_height_mean'] = df[back_cols].mean(axis=1)
    df.drop(columns=back_cols, inplace=True)

    # Replace tail heights with their mean.
    tail_cols = [col for col in df.columns if col.startswith('tail_height|')]
    df['tail_height_mean'] = df[tail_cols].mean(axis=1)
    df.drop(columns=tail_cols, inplace=True)

    # Replace double, triple, and quadruple support with an average support value.
    double_name = [col for col in df.columns if col.startswith('double_support')]
    triple_name = [col for col in df.columns if col.startswith('triple_support')]
    quadruple_name = [col for col in df.columns if col.startswith('quadruple_support')]
    average_support_val = (2 * df[double_name].values + 3 * df[triple_name].values + 4 * df[
        quadruple_name].values) / 100
    df['average_support_val'] = average_support_val
    df.drop(columns=double_name + triple_name + quadruple_name, inplace=True)
    return df

def load_stride_data(stride_data_path):
    stride_data = pd.read_hdf(stride_data_path, key='stride_info')
    return stride_data

def collect_stride_data(condition, exp, day, compare_condition):
    stride_data_path = None
    stride_data_path_compare = None
    if exp == 'Extended':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_StrideInfo.h5")
        if compare_condition != 'None':
            stride_data_path_compare = os.path.join(paths['filtereddata_folder'], f"{compare_condition}\\{exp}\\MEASURES_StrideInfo.h5")
    elif exp == 'Repeats':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
        if compare_condition != 'None':
            stride_data_path_compare = os.path.join(paths['filtereddata_folder'], f"{compare_condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
    stride_data = load_stride_data(stride_data_path)
    stride_data_compare = load_stride_data(stride_data_path_compare)
    return stride_data, stride_data_compare

def set_up_save_dir(condition, exp, base_save_dir_no_c):
    pcs_total= global_settings['pcs_to_show']
    pcs_using = global_settings['pcs_to_use']
    strides = str(global_settings['stride_numbers']).replace(' ', '').replace(',','').replace('[','').replace("'", "").replace(']','')
    phase_comp = str(global_settings['phases']).replace(' ', '').replace(',','').replace('[','').replace("'", "").replace(']','')

    base_save_dir = base_save_dir_no_c + f'_{strides}_{phase_comp}' + f'-PCStot={pcs_total}-PCSuse={pcs_using}'
    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')
    return base_save_dir, base_save_dir_condition

def log_settings(settings, log_dir, script_name):
    """
    Save the provided settings (a dict) to a timestamped log file.
    Also include the name of the running script and the current date.
    """
    # check if a log file already exists, if so delete it
    # to do above
    os.makedirs(log_dir, exist_ok=True)

    # Delete pre-existing log files starting with 'settings_log_'
    for filename in os.listdir(log_dir):
        if filename.startswith("settings_log_") and filename.endswith(".txt"):
            os.remove(os.path.join(log_dir, filename))

    # Get the current datetime as string
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build the content of the log file.
    log_content = f"Script: {script_name}\nDate: {datetime.datetime.now()}\n\nSettings:\n"
    for key, value in settings.items():
        log_content += f"{key}: {value}\n"

    # Define the log file name.
    log_file = os.path.join(log_dir, f"settings_log_{now_str}.txt")
    with open(log_file, "w") as f:
        f.write(log_content)
    print(f"Settings logged to {log_file}")

def load_and_preprocess_data(mouse_id, stride_number, condition, exp, day, measures):
    """
    Load data for the specified mouse and preprocess it by selecting desired features and
    interpolating for missing values
    """
    if exp == 'Extended':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_single_kinematics_runXstride.h5")
    elif exp == 'Repeats':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\Wash\\Exp\\{day}\\MEASURES_single_kinematics_runXstride.h5")
    else:
        raise ValueError(f"Unknown experiment type: {exp}")
    data_allmice = pd.read_hdf(filepath, key='single_kinematics')

    try:
        data = data_allmice.loc[mouse_id]
    except KeyError:
        raise ValueError(f"Mouse ID {mouse_id} not found in the dataset.")

    # # Build desired columns using the simplified build_desired_columns function
    # measures = measures_list_feature_reduction

    col_names = []
    for feature in measures.keys():
        if any(measures[feature]):
            if feature != 'signed_angle':
                for param in itertools.product(*measures[feature].values()):
                    param_names = list(measures[feature].keys())
                    formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                    col_names.append((feature, formatted_params))
            else:
                for combo in measures['signed_angle'].keys():
                    col_names.append((feature, combo))
        else:
            col_names.append((feature, 'no_param'))

    col_names_trimmed = []
    for c in col_names:
        if np.logical_and('full_stride:True' in c[1], 'step_phase:None' not in c[1]):
            pass
        elif np.logical_and('full_stride:False' in c[1], 'step_phase:None' in c[1]):
            pass
        else:
            col_names_trimmed.append(c)

    selected_columns = col_names_trimmed


    filtered_data = data.loc[:, selected_columns]

    separator = '|'
    # Collapse MultiIndex columns to single-level strings including group info.
    filtered_data.columns = [
        f"{measure}{separator}{params}" if params != 'no_param' else f"{measure}"
        for measure, params in filtered_data.columns
    ]

    try:
        filtered_data = filtered_data.xs(stride_number, level='Stride', axis=0)
    except KeyError:
        raise ValueError(f"Stride number {stride_number} not found in the data.")

    # interpolate missing values
    filtered_data_interp = filtered_data.interpolate(method='linear', limit_direction='both')

    return filtered_data_interp

def normalize_df(df):
    normalize_mean = []
    normalize_std = []
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return df, normalize_mean, normalize_std

def normalize_Xdr(Xdr):
    normalize_mean = []
    normalize_std = []
    for row in range(Xdr.shape[0]):
        mean = np.mean(Xdr[row, :])
        std = np.std(Xdr[row, :])
        Xdr[row, :] = (Xdr[row, :] - mean) / std # normalise each pcs's data
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return Xdr, normalize_mean, normalize_std

def get_mask_p1_p2(data, p1, p2):
    mask_p1 = data.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][p1])
    mask_p2 = data.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][p2])
    return mask_p1, mask_p2

def get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2):
    mask_phase1, mask_phase2 = get_mask_p1_p2(scaled_data_df, phase1, phase2)

    if not mask_phase1.any():
        raise ValueError(f"No runs found for phase '{phase1}'.")
    if not mask_phase2.any():
        raise ValueError(f"No runs found for phase '{phase2}'.")

    run_numbers_phase1 = scaled_data_df.index[mask_phase1]
    run_numbers_phase2 = scaled_data_df.index[mask_phase2]
    run_numbers = list(run_numbers_phase1) + list(run_numbers_phase2)

    # Determine stepping limbs.
    stepping_limbs = [determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                      for run in run_numbers]

    return run_numbers, stepping_limbs, mask_phase1, mask_phase2

def determine_stepping_limbs(stride_data, mouse_id, run, stride_number):
    """
    Determine the stepping limb (ForepawL or ForepawR) for a given MouseID, Run, and Stride.

    Parameters:
        stride_data (pd.DataFrame): Stride data DataFrame.
        mouse_id (str): Identifier for the mouse.
        run (str/int): Run identifier.
        stride_number (int): Stride number.

    Returns:
        str: 'ForepawL' or 'ForepawR' indicating the stepping limb.
    """
    paws = stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any().index[
        stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any()]
    if len(paws) > 1:
        raise ValueError(f"Multiple paws found for Mouse {mouse_id}, Run {run}.")
    else:
        return paws[0]

def select_runs_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2):
    try:
        scaled_data_df = feature_data.loc(axis=0)[stride_number, mouse_id]
        # Get runs and stepping limbs for each phase.
        run_numbers, stepping_limbs, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id,
                                                                         stride_number, phase1, phase2)

        # Select only runs from the two phases in feature data
        selected_mask = mask_phase1 | mask_phase2
        selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

        return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2
    except KeyError:
        raise ValueError(f"Stride: {stride_number} and mouse: {mouse_id} not found in the data.")


def create_mouse_save_directory(base_dir, mouse_id, stride_number, phase1, phase2):
    """
    Create a directory path based on the settings to save plots.

    Parameters:
        base_dir (str): Base directory where plots will be saved.
        mouse_id (str): Identifier for the mouse.
        stride_number (int): Stride number used in analysis.
        phase1 (str): First experimental phase.
        phase2 (str): Second experimental phase.

    Returns:
        str: Path to the directory where plots will be saved.
    """
    # Construct the directory name
    dir_name = f"Mouse_{mouse_id}_Stride_{stride_number}_Compare_{phase1}_vs_{phase2}"
    save_path = os.path.join(base_dir, dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    return save_path
