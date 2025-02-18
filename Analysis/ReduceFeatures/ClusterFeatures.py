import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random

from Helpers.Config_23 import *
import Analysis.ReduceFeatures.utils_feature_reduction as utils

# ----------------------------
# Configuration Section
# ----------------------------

mouse_ids = [
    '1035243', '1035244', '1035245', '1035246',
    '1035249', '1035250', '1035297', '1035298',
    '1035299', '1035301', '1035302'
]  # List of mouse IDs to analyze
stride_numbers = [-1]  # List of stride numbers to filter data
phases = ['APA2', 'Wash2']  # List of phases to compare
exp = 'Extended'
day = None
# The two conditions want to process:
condition = 'APAChar_LowHigh'

base_save_dir = os.path.join(paths['plotting_destfolder'], 'FeatureReduction', 'Round11-20250217-ClusterFeatures')

overwrite_FeatureSelection = False

n_iterations_selection = 100
nFolds_selection = 5

# ----------------------------
# Function Definitions
# ----------------------------
sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)


def load_and_preprocess_data(mouse_id, stride_number, condition, exp, day):
    """
    Load data for the specified mouse and preprocess it by selecting desired features,
    imputing missing values, and standardizing.
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

    # Build desired columns using the simplified build_desired_columns function
    measures = measures_list_feature_reduction

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

    filtered_data_imputed = filtered_data.fillna(filtered_data.mean())

    if filtered_data_imputed.isnull().sum().sum() > 0:
        print("Warning: There are still missing values after imputation.")
    else:
        print("All missing values have been handled.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data_imputed)
    scaled_data_df = pd.DataFrame(scaled_data, index=filtered_data_imputed.index,
                                  columns=filtered_data_imputed.columns)
    return scaled_data_df

def select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2):
    # Load and preprocess data.
    scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp,
                                              day)  # Load data for the specified mouse and preprocess it by selecting desired features, imputing missing values, and standardizing.
    print('Data loaded and preprocessed.')

    # Get runs and stepping limbs for each phase.
    run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id,
                                                                     stride_number, phase1, phase2)

    # Select only runs from the two phases in feature data
    selected_mask = mask_phase1 | mask_phase2
    selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

    return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2

def process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, condition, exp, day,
                             base_save_dir_condition):
    """
    Process a single phase comparison for a given mouse. If selected_features is provided,
    that global feature set is used; otherwise local feature selection is performed.
    """
    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    save_path = "\\\\?\\" + save_path
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2)

    print(f"Selected {selected_scaled_data_df.shape[0]} features for analysis.")

def main():
    stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_StrideInfo.h5")
    stride_data = utils.load_stride_data(stride_data_path)

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, condition, exp, day,
                             base_save_dir)


if __name__ == "__main__":
    main()