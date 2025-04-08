import os
import itertools
import inspect
import random
import seaborn as sns
import pandas as pd
import pickle
from typing import Optional, List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt


'''
Run PCA on all features (no reduction) and both conditions.
Then fit regression model separately for each condition.
Plot weights of PCs
Hopefully have 2 large PCs that separate conditions (ie different directions)
Check which features load highest on these PCs
Check is there overlap with features selected in FeatRed+PCA-Lh_Reg-LhHl.py
Can then use these features to predict condition
If PCs look similar across models, can say i use same feature sets for LH and HL

'''

# ----------------------------
# Library Imports
# ----------------------------
from Analysis.Tools.PredictPhase import (
    cluster_features_main, global_feature_selection_main, global_pca_main, process_mice_main, find_outliers)
from Analysis.Tools.PCA import (compute_global_pca_for_phase)
from Analysis.Tools import utils_feature_reduction as utils
from Helpers.Config_23 import *
from Analysis.Tools.config import (
    global_settings, condition_specific_settings, instance_settings
)
base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round23-9mice_descriptives--Lh+HL-PCA-LhHlsep-reg')

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

# -----------------------------------------------------
# Initialization Helper
# -----------------------------------------------------
def initialize_experiment(condition, exp, day, compare_condition, settings_to_log,
                          base_save_dir_no_c, condition_specific_settings) -> Tuple:
    # Collect stride data and create directories.
    stride_data, stride_data_compare = utils.collect_stride_data(condition, exp, day, compare_condition)
    base_save_dir, base_save_dir_condition = utils.set_up_save_dir(
        condition, exp, condition_specific_settings[condition]['c'], base_save_dir_no_c
    )
    script_name = os.path.basename(inspect.getfile(inspect.currentframe()))
    utils.log_settings(settings_to_log, base_save_dir, script_name)

    # collect feature data from each mouse and stride
    feature_data = {}
    feature_data_compare = {}
    for stride in global_settings["stride_numbers"]:
        for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
            # Load and preprocess data for each mouse.
            filtered_data_df = utils.load_and_preprocess_data(mouse_id, stride, condition, exp, day, measures_list_feature_reduction)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = utils.load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day, measures_list_feature_reduction)
            feature_data_compare[(stride, mouse_id)] = filtered_data_comparison_df

    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_compare = pd.concat(feature_data_compare, axis=0)

    return feature_data_df, feature_data_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition


# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------

def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):

    # Initialize experiment (data collection, directories, logging). - NOT SCALED YET!!
    feature_data_notscaled, feature_data_compare_notscaled, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)
    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")

    if global_settings.get("overwrite_data_collection", True) or not os.path.exists(global_data_path):
        """
            # -------- Find Outliers --------
            (Both conditions)
            - Run PCA on all data 
            - Get PCs for each mouse
            - Find outlier runs across mice based on PCA   
            - NB ****** NOT SCALED YET!!!! ******
        """
        print("Finding outliers...")
        feature_data_notscaled = find_outliers(feature_data_notscaled, condition, exp, day, stride_data, phases, stride_numbers, base_save_dir_condition)
        feature_data_compare_notscaled = find_outliers(feature_data_compare_notscaled, compare_condition, exp, day, stride_data_compare, phases, stride_numbers, base_save_dir_condition)

        """
            # -------- Scale Data --------
            (Both conditions)
            - Z-score scale data for each mouse and stride
        """
        print("Scaling data...")
        feature_data = feature_data_notscaled.copy()
        feature_data_compare = feature_data_compare_notscaled.copy()
        Normalize = {}
        idx = pd.IndexSlice

        feature_names = feature_data.columns

        for (stride, mouse_id), data in feature_data.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = utils.normalize_df(data)
            feature_data.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize[(stride, mouse_id)] = norm_df
        Normalize_compare = {}
        for (stride, mouse_id), data in feature_data_compare.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = utils.normalize_df(data)
            feature_data_compare.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize_compare[(stride, mouse_id)] = norm_df

        """
            # -------- Feature Clustering --------
            (Combined conditions)
            For each stride number and phase pair, find K with cross-validation and then cluster features. Mapping is saved and 
            used to plot feature clustering and a chart describing the content of each cluster.
        """
        print("Clustering features...")
        cluster_mappings = cluster_features_main(feature_data, feature_data_compare, phases, stride_numbers, condition, compare_condition, stride_data, stride_data_compare, base_save_dir_condition, combine_conditions=True)

        """
            # -------- PCA --------
            (Combined conditions separately)  
        """
        global_pca_results = global_pca_main(feature_data, feature_data_compare, None, phases, stride_numbers, condition, compare_condition, stride_data, stride_data_compare, select_feats=False, combine_conditions=True)

        """
                    # -------- Process Each Mouse --------
                    (Both conditions)
                    For each mouse, process the phase comparison and collect aggregated info.
                        - Use the global feature selection results to process each mouse.
                        - Run regression and predict runs for each mouse.
                    !!! NB: Regression is performed on both conditions, but only the main condition is used for feature selection and PCA.
                """
        print("Processing mice...")
        results = process_mice_main(mouse_ids, phases, stride_numbers, condition, exp, day,
                                    stride_data, base_save_dir_condition, feature_data,
                                    global_fs_results, global_pca_results, global_stride_fs_results, cluster_mappings)
        (aggregated_predictions, aggregated_feature_weights, aggregated_raw_features, aggregated_raw_features_all,
         aggregated_cluster_loadings, multi_stride_data, even_ws, odd_ws,
         phase1_pc, phase2_pc, normalize_mean_pc, normalize_std_pc) = results

        results_compare = process_mice_main(mouse_ids, phases, stride_numbers, compare_condition, exp, day,
                                            stride_data_compare, base_save_dir_condition, feature_data_compare,
                                            global_fs_results, global_pca_results, global_stride_fs_results,
                                            cluster_mappings)
        (aggregated_predictions_compare, aggregated_feature_weights_compare, aggregated_raw_features_compare,
         aggregated_raw_features_all_compare,
         aggregated_cluster_loadings_compare, multi_stride_data_compare, even_ws_compare, odd_ws_compare,
         phase1_pc_compare, phase2_pc_compare, normalize_mean_pc_compare, normalize_std_pc_compare) = results_compare


# ----------------------------
# Execute Main Function
# ----------------------------

if __name__ == "__main__":
    # add flattened LowHigh etc settings to global_settings for log
    global_settings["LowHigh_c"] = condition_specific_settings['APAChar_LowHigh']['c']
    global_settings["HighLow_c"] = condition_specific_settings['APAChar_HighLow']['c']
    global_settings["LowHigh_mice"] = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
    global_settings["HighLow_mice"] = condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']

    # Combine the settings in a single dict to log.
    settings_to_log = {
        "global_settings": global_settings,
        "instance_settings": instance_settings
    }

    # Run each instance.
    for inst in instance_settings:
        main(
            global_settings["mouse_ids"],
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log
        )