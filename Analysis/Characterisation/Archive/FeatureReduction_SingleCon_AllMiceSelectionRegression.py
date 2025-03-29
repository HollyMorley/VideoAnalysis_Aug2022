import os
import itertools
import datetime
import inspect
import random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

'''
Similar to Analysis/Characterisation/FeatureReduction_SingleCon.py, but only with allmice conditions set to True.
Pipeline:
  - For each phase-comparison and stride, perform a leave-one-out (LOO) evaluation: train on all mice except one,
    evaluate that mouse and store its accuracy.
  - Plot the accuracy for each mouse (for each condition) to visually inspect potential outliers.
  - Then, run the aggregated regression on all mice as before.
'''

# ----------------------------
# Library Imports (unchanged)
# ----------------------------
from Analysis.Tools.PCA import (
    perform_pca, plot_pca, plot_scree, cross_validate_pca,
    plot_average_variance_explained_across_folds, compute_global_pca_for_phase
)
from Analysis.Tools.LogisticRegression import (run_regression, fit_regression_model, plot_LOO_regression_accuracies, predict_runs)
from Analysis.Tools.FeatureSelection import (global_feature_selection)
from Analysis.Tools.ClusterFeatures import (
    plot_feature_clustering, plot_feature_clustering_3d, plot_feature_clusters_chart,
    find_feature_clusters, plot_corr_matrix_sorted_by_cluster, cluster_features_with_pca_cv, cluster_features_with_ica_cv
)
from Analysis.Tools import utils_feature_reduction as utils
from Helpers.Config_23 import *
from Analysis.Tools.config import (
    base_save_dir_no_c, global_settings, condition_specific_settings, instance_settings
)

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
    return stride_data, stride_data_compare, base_save_dir, base_save_dir_condition

# -----------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------
from sklearn.metrics import balanced_accuracy_score


def evaluate_regression(w, norm_mean, norm_std, loadings_df,
                        test_selected_df, test_mask1, test_mask2, mouse, phase1, phase2, stride_number, condition) -> float:
    """
    Evaluate the regression model on test data using the same approach as in your
    LogisticRegression.py. This function fits the regression model on the training data
    (using fit_regression_model) and then applies the model to the test data to compute
    the balanced accuracy.
    """
    _, y_pred = predict_runs(loadings_df, test_selected_df.T, norm_mean, norm_std, w, '', mouse, phase1, phase2, stride_number, condition, plot_pred=False)
    y_test = np.concatenate([np.ones(np.sum(test_mask1)), np.zeros(np.sum(test_mask2))])

    # Calculate balanced accuracy between the predicted and true labels.
    y_pred_acc = y_pred.copy()
    y_pred_acc[y_pred_acc > 0] = 1
    y_pred_acc[y_pred_acc < 0] = 0
    acc = balanced_accuracy_score(y_test.T, y_pred_acc.T)
    return acc, y_pred


# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------
def main(stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):
    # Initialize experiment (data collection, directories, logging).
    stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(
        condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings
    )

    # -----------------------------------------------------
    # Feature Clustering (remains unchanged)
    # -----------------------------------------------------
    cluster_mappings = {}
    if global_settings.get("cluster_all_strides", True):
        for phase1, phase2 in itertools.combinations(phases, 2):
            if global_settings['cluster_method'] == 'kmeans':
                cluster_mapping, feature_matrix = find_feature_clusters(
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    "all", condition, exp, day, stride_data, phase1, phase2,
                    base_save_dir_condition, method='kmeans'
                )
            for stride_number in stride_numbers:
                cluster_mappings[(phase1, phase2, stride_number)] = cluster_mapping

            plot_feature_clustering(feature_matrix, cluster_mapping, phase1, phase2, "all", base_save_dir_condition)
            plot_feature_clustering_3d(feature_matrix, cluster_mapping, phase1, phase2, "all", base_save_dir_condition)
            sorted_features = sorted(feature_matrix.index, key=lambda f: cluster_mapping.get(f, -1))
            plot_feature_clusters_chart(cluster_mapping, sorted_features, phase1, phase2, "all",
                                        base_save_dir_condition)
            plot_corr_matrix_sorted_by_cluster(
                feature_matrix, sorted_features, cluster_mapping, base_save_dir_condition,
                filename=f'CorrMatrix_{phase1}_{phase2}_all'
            )
    else:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                if global_settings['cluster_method'] == 'kmeans':
                    cluster_mapping, feature_matrix = find_feature_clusters(
                        condition_specific_settings[condition]['global_fs_mouse_ids'],
                        stride_number, condition, exp, day, stride_data, phase1, phase2,
                        base_save_dir_condition, method='kmeans'
                    )
                cluster_mappings[(phase1, phase2, stride_number)] = cluster_mapping

                plot_feature_clustering(feature_matrix, cluster_mapping, phase1, phase2, stride_number,
                                        base_save_dir_condition)
                plot_feature_clustering_3d(feature_matrix, cluster_mapping, phase1, phase2, stride_number,
                                           base_save_dir_condition)
                sorted_features = sorted(feature_matrix.index, key=lambda f: cluster_mapping.get(f, -1))
                plot_feature_clusters_chart(cluster_mapping, sorted_features, phase1, phase2, stride_number,
                                            base_save_dir_condition)
                plot_corr_matrix_sorted_by_cluster(feature_matrix, sorted_features, cluster_mapping,
                                                   base_save_dir_condition,
                                                   filename=f'CorrMatrix_{phase1}_{phase2}_stride{stride_number}')

    # -----------------------------------------------------
    # Global Feature Selection and Global PCA
    # -----------------------------------------------------
    global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
    os.makedirs(global_fs_dir, exist_ok=True)

    global_fs_results = {}
    global_pca_results = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
            selected_features, fs_df = global_feature_selection(
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                stride_number, phase1, phase2, condition, exp, day,
                stride_data, save_dir=global_fs_dir,
                c=condition_specific_settings[condition]['c'],
                nFolds=global_settings["nFolds_selection"],
                n_iterations=global_settings["n_iterations_selection"],
                overwrite=global_settings["overwrite_FeatureSelection"],
                method=global_settings["method"]
            )

            pca, loadings_df = compute_global_pca_for_phase(
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                stride_number, phase1, phase2, condition, exp, day, stride_data, selected_features
            )

            global_fs_results[(phase1, phase2, stride_number)] = (selected_features, fs_df)
            global_pca_results[(phase1, phase2, stride_number)] = (pca, loadings_df)

    # -----------------------------------------------------
    # Leave-One-Out Evaluation (per phase and stride)
    # -----------------------------------------------------
    loo_results = {}  # key: (phase1, phase2, stride_number) -> dict { mouse: accuracy }

    print("Starting leave-one-out evaluation (LOO) for each phase/stride combination.")
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            mouse_accuracies = {}  # store accuracy for each mouse
            predictions = {}  # store predictions for each mouse
            for mouse in global_settings["mouse_ids"]:
                # Aggregate training data from all mice except the current one.
                train_reduced_data = []
                train_reduced_selected = []
                train_mask_phase1 = []
                train_mask_phase2 = []
                for m in condition_specific_settings[condition]['global_fs_mouse_ids']:
                    if m == mouse:
                        continue
                    scaled_data_df, selected_scaled_data_df, _, _, mask_phase1, mask_phase2 = \
                        utils.select_runs_data(m, stride_number, condition, exp, day, stride_data, phase1, phase2)
                    tup = global_fs_results.get((phase1, phase2, stride_number), None)
                    if tup is not None:
                        selected_features, fs_df = tup
                    else:
                        selected_features = list(selected_scaled_data_df.index)
                    # Reduce data using the selected features.
                    train_reduced_data.append(scaled_data_df[selected_features])
                    train_reduced_selected.append(selected_scaled_data_df.loc(axis=0)[selected_features])
                    train_mask_phase1.append(mask_phase1)
                    train_mask_phase2.append(mask_phase2)

                # Concatenate training data.
                agg_train_data = pd.concat(train_reduced_data)
                agg_train_selected = pd.concat(train_reduced_selected, axis=1)
                agg_train_mask1 = np.concatenate(train_mask_phase1)
                agg_train_mask2 = np.concatenate(train_mask_phase2)

                # Get global PCA results.
                pca, loadings_df = global_pca_results[(phase1, phase2, stride_number)]

                # Define a temporary save directory.
                loo_save_dir = os.path.join(base_save_dir_condition, "LeaveOneOut",
                                            f"{mouse}_{phase1}_{phase2}_{stride_number}")
                os.makedirs(loo_save_dir, exist_ok=True)

                # Fit regression on the training (aggregated) data.
                _, _, w, norm_mean, norm_std = run_regression(
                    loadings_df, agg_train_data, agg_train_selected,
                    agg_train_mask1, agg_train_mask2,
                    f"agg_without_{mouse}", phase1, phase2, stride_number,
                    loo_save_dir, condition, plot_pred=False, plot_weights=False
                )

                # Prepare the test (left-out) data.
                scaled_data_df_test, selected_scaled_data_df_test, _, _, test_mask1, test_mask2 = \
                    utils.select_runs_data(mouse, stride_number, condition, exp, day, stride_data, phase1, phase2)
                test_reduced_selected = selected_scaled_data_df_test.loc(axis=0)[selected_features]

                # Evaluate the model on the left-out mouse.
                test_accuracy, y_pred = evaluate_regression(
                    w, norm_mean, norm_std, loadings_df,
                    test_reduced_selected, test_mask1, test_mask2, mouse, phase1, phase2, stride_number, condition
                )
                mouse_accuracies[mouse] = test_accuracy
                predictions[mouse] = y_pred
                print(f"LOO for mouse {mouse}, {phase1} vs {phase2}, stride {stride_number}: accuracy = {test_accuracy:.2f}")
            loo_results[(phase1, phase2, stride_number)] = mouse_accuracies
            plot_LOO_regression_accuracies(mouse_accuracies, phase1, phase2, stride_number, base_save_dir_condition)


    # -----------------------------------------------------
    # Final Aggregated Regression on All Mice (per condition)
    # -----------------------------------------------------
    final_regression_results = {}
    final_save_dir = os.path.join(base_save_dir_condition, "AggregatedRegression_Final")
    os.makedirs(final_save_dir, exist_ok=True)

    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            final_train_data = []
            final_train_selected = []
            final_mask_phase1 = []
            final_mask_phase2 = []
            for m in global_settings["mouse_ids"]:
                scaled_data_df, selected_scaled_data_df, _, _, mask_phase1, mask_phase2 = \
                    utils.select_runs_data(m, stride_number, condition, exp, day, stride_data, phase1, phase2)
                tup = global_fs_results.get((phase1, phase2, stride_number), None)
                if tup is not None:
                    selected_features, fs_df = tup
                else:
                    selected_features = list(selected_scaled_data_df.index)
                final_train_data.append(scaled_data_df[selected_features])
                final_train_selected.append(selected_scaled_data_df[selected_features])
                final_mask_phase1.append(mask_phase1)
                final_mask_phase2.append(mask_phase2)

            agg_final_data = pd.concat(final_train_data)
            agg_final_selected = pd.concat(final_train_selected)
            agg_final_mask1 = np.concatenate(final_mask_phase1)
            agg_final_mask2 = np.concatenate(final_mask_phase2)

            pca, loadings_df = global_pca_results[(phase1, phase2, stride_number)]
            smoothed_scaled_pred, feature_weights, _, _, _ = run_regression(
                loadings_df, agg_final_data, agg_final_selected,
                agg_final_mask1, agg_final_mask2,
                "all_remaining", phase1, phase2, stride_number,
                final_save_dir, condition
            )
            final_regression_results[(phase1, phase2, stride_number)] = (smoothed_scaled_pred, feature_weights)
            print(f"Final aggregated regression complete for {phase1} vs {phase2}, stride {stride_number}.")

    # -----------------------------------------------------
    # Aggregated Plots and Further Processing
    # -----------------------------------------------------
    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    os.makedirs(aggregated_save_dir, exist_ok=True)

    # Example aggregated plotting calls.
    for (phase1, phase2, stride_number), (pred, weights) in final_regression_results.items():
        # Plot aggregated run predictions.
        utils.plot_aggregated_run_predictions(
            [pred], aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition
        )
        # Plot aggregated feature weights.
        utils.plot_aggregated_feature_weights_byFeature(
            weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition
        )
        # Optionally, include further plotting routines as needed.
    print("Global analysis complete.")

# ----------------------------
# Execute Main Function
# ----------------------------
if __name__ == "__main__":
    # Add flattened settings for logging.
    global_settings["LowHigh_c"] = condition_specific_settings['APAChar_LowHigh']['c']
    global_settings["HighLow_c"] = condition_specific_settings['APAChar_HighLow']['c']
    global_settings["LowHigh_mice"] = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
    global_settings["HighLow_mice"] = condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']

    settings_to_log = {
        "global_settings": global_settings,
        "instance_settings": instance_settings
    }

    for inst in instance_settings:
        main(
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log
        )
