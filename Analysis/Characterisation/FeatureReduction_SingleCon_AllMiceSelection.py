import os
import itertools
import inspect
import random
import seaborn as sns
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt


'''
Similar to Analysis/Characterisation/FeatureReduction_SingleCon.py, but only with allmice conditions set to True.
Pipeline:

'''

# ----------------------------
# Library Imports
# ----------------------------
from Analysis.Tools.PCA import (
    plot_pca, plot_scree, compute_global_pca_for_phase)

from Analysis.Tools.LogisticRegression import (run_regression, run_linear_regression_derivative)
from Analysis.Tools.FeatureSelection import (global_feature_selection)
from Analysis.Tools.ClusterFeatures import (
    plot_feature_clustering, plot_feature_clustering_3d, plot_feature_clusters_chart,
    find_feature_clusters, plot_corr_matrix_sorted_by_cluster
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
# Processing Functions
# -----------------------------------------------------
def process_mouse_phase_comparison(mouse_id: str, stride_number: int, phase1: str, phase2: str,
                                   stride_data, stride_data_compare, condition: str, exp: str, day,
                                   base_save_dir_condition: str,
                                   selected_features: Optional[List[str]] = None,
                                   fs_df: Optional[dict] = None,
                                   global_pca: Optional[Tuple] = None,
                                   compare_condition: Optional[str] = None,
                                   cluster_mapping: Optional[Dict] = None) -> (
        Tuple)[utils.PredictionData, utils.FeatureWeights, any, Optional[Dict[int, any]], any, any, any, any]:
    """
    Process a single phase comparison for a given mouse.
    Returns:
      - PredictionData (with mouse_id, x_vals, smoothed_scaled_pred, group_id),
      - feature_weights,
      - reduced_feature_data_df,
      - cluster_loadings.
    """

    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    # Get the data for this mouse and phase comparison.
    scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2)

    # Optionally, reduce the data to only the selected features.
    if global_settings["select_features"] == True:
        print("Using globally selected features for feature reduction.")
    else:
        print("Using all features for feature reduction.")
        selected_features = list(selected_scaled_data_df.index)

    if fs_df is not None:
        selected_features_accuracies = fs_df.loc[selected_features]
        print(f"Length of significant features: {len(selected_features)}")

        if global_settings["plot_raw_features"]:
            # Plot significant features.
            if not os.path.exists(os.path.join(save_path, 'feature_significances')):
                os.makedirs(os.path.join(save_path, 'feature_significances'))
            utils.plot_significant_features(selected_features_accuracies, save_path, selected_features)

            # Plot non-significant features.
            nonselected_features = fs_df[~fs_df['significant']].index
            nonselected_features_accuracies = fs_df.loc[nonselected_features]
            if not os.path.exists(os.path.join(save_path, 'feature_nonsignificances')):
                os.makedirs(os.path.join(save_path, 'feature_nonsignificances'))
            utils.plot_nonsignificant_features(nonselected_features_accuracies, save_path, nonselected_features)
    else:
        # Detailed per-feature results are not available (i.e. when using RFECV or RF).
        print("No detailed per-feature selection results available for plotting; skipping per-feature plots.")

    # Now reduce the data to only the significant features
    reduced_feature_data_df = scaled_data_df.loc(axis=1)[selected_features]
    reduced_feature_selected_data_df = selected_scaled_data_df.loc[selected_features]

    # ---------------- PCA Analysis ----------------

    # Use the global PCA transformation.
    pca, loadings_df = global_pca # features x PCs
    # Project this mouse's data using the global PCA.
    pcs = pca.transform(reduced_feature_data_df) # runs x PCs
    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]
    labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
    labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
    labels = np.concatenate([labels_phase1, labels_phase2])
    pcs_combined = np.vstack([pcs_phase1, pcs_phase2])
    # Plot using the global PCA.
    plot_pca(pca, pcs_combined, labels, phase1, phase2, stride_number, stepping_limbs, run_numbers, mouse_id, save_path)
    plot_scree(pca, phase1, phase2, stride_number, save_path)

    # ---------------- Prediction - Regression-Based Feature Contributions ----------------
    smoothed_scaled_pred, feature_weights, _, _, _ = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, stride_number, save_path, condition)

    # ---------------- Within vs between mice comparison - setup ----------------
    # fit regression model using even and odd Xdr values separately
    reduced_feature_selected_data_df_even = reduced_feature_selected_data_df.T.iloc[::2].T
    reduced_feature_selected_data_df_odd = reduced_feature_selected_data_df.T.iloc[1::2].T
    smoothed_scaled_pred_even, feature_weights_even, _, _, _ = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df_even, mask_phase1[::2], mask_phase2[::2], mouse_id, phase1, phase2, stride_number, save_path, condition, plot_pred=False, plot_weights=False)
    smoothed_scaled_pred_odd, feature_weights_odd, _, _, _ = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df_odd, mask_phase1[1::2], mask_phase2[1::2], mouse_id, phase1, phase2, stride_number, save_path, condition, plot_pred=False, plot_weights=False)

    # get weights in PC space for even and odd mice
    even_weights = np.dot(feature_weights_even, loadings_df)
    odd_weights = np.dot(feature_weights_odd, loadings_df)

    # ---------------- Map selected features to respective clusters ----------------
    feature_cluster_assignments = {feat: cluster_mapping.get(feat, -1) for feat in selected_features}

    # Sum the regression loadings by cluster.
    cluster_loadings = {}
    for feat, weight in feature_weights.items():
        cluster = feature_cluster_assignments.get(feat)
        if cluster is not None and cluster != -1:
            cluster_loadings[cluster] = cluster_loadings.get(cluster, 0) + weight
    print(f"Mouse {mouse_id} - Cluster loadings: {cluster_loadings}")


    # Return aggregated prediction as a PredictionData instance.
    pred_data = utils.PredictionData(mouse_id=mouse_id,
                               x_vals=list(reduced_feature_data_df.index),
                               smoothed_scaled_pred=smoothed_scaled_pred)
    ftr_wghts = utils.FeatureWeights(mouse_id=mouse_id,
                                    feature_weights=feature_weights)

    return pred_data, ftr_wghts, reduced_feature_data_df, cluster_loadings, even_weights, odd_weights, pcs_phase1, pcs_phase2


# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------

def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):
    # Initialize experiment (data collection, directories, logging).
    stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)

    """
        # -------- Feature Clustering --------
        For each stride number and phase pair, find K with cross-validation and then cluster features. Mapping is saved and 
        used to plot feature clustering and a chart describing the content of each cluster.
    """
    cluster_mappings = {}
    if global_settings.get("cluster_all_strides", True):
        # Cluster across all available strides
        for phase1, phase2 in itertools.combinations(phases, 2):
            if global_settings['cluster_method'] == 'kmeans':
                cluster_mapping, feature_matrix = find_feature_clusters(
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    "all", condition, exp, day, stride_data, phase1, phase2,
                    base_save_dir_condition, method='kmeans'
                )
            # Store same mapping for all strides.
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
                    cluster_mapping, feature_matrix = find_feature_clusters(condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                                        stride_number, condition, exp, day, stride_data, phase1, phase2,
                                                                        base_save_dir_condition, method='kmeans')
                cluster_mappings[(phase1, phase2, stride_number)] = cluster_mapping

                plot_feature_clustering(feature_matrix, cluster_mapping, phase1, phase2, stride_number, base_save_dir_condition)
                plot_feature_clustering_3d(feature_matrix, cluster_mapping, phase1, phase2, stride_number,
                                           base_save_dir_condition)
                sorted_features = sorted(feature_matrix.index, key=lambda f: cluster_mapping.get(f, -1))
                plot_feature_clusters_chart(cluster_mapping, sorted_features, phase1, phase2, stride_number, base_save_dir_condition)
                plot_corr_matrix_sorted_by_cluster(feature_matrix, sorted_features, cluster_mapping, base_save_dir_condition, filename=f'CorrMatrix_{phase1}_{phase2}_stride{stride_number}')

    """
        # -------- Global Feature Selection --------
        Perform feature selection on data from all mice combined    
    """
    # Directory for global feature selection.
    global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
    os.makedirs(global_fs_dir, exist_ok=True)

    global_fs_results = {}
    global_pca_results = {}
    global_stride_fs_results = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
            # Perform global feature selection.
            selected_features, fs_df = global_feature_selection(condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                                stride_number, phase1, phase2, condition,
                                                                exp, day,
                                                                stride_data, save_dir=global_fs_dir,
                                                                c=condition_specific_settings[condition]['c'],
                                                                nFolds=global_settings["nFolds_selection"],
                                                                n_iterations=global_settings["n_iterations_selection"],
                                                                overwrite=global_settings["overwrite_FeatureSelection"],
                                                                method=global_settings["method"])

            # Compute global PCA using the selected features.
            pca, loadings_df = compute_global_pca_for_phase(condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                            stride_number, phase1, phase2, condition,
                                                            exp, day, stride_data, selected_features)

            global_fs_results[(phase1, phase2, stride_number)] = (selected_features, fs_df)
            global_pca_results[(phase1, phase2, stride_number)] = (pca, loadings_df)
        # Combine strides
        stride_features = []
        for s in stride_numbers:
            features = global_fs_results[(phase1, phase2, s)][0]
            stride_features.extend(features)
        # remove duplicates
        stride_features = list(set(stride_features))
        global_stride_fs_results[(phase1, phase2)] = stride_features

    """
        # -------- Process Each Mouse --------
        For each mouse, process the phase comparison and collect aggregated info.
            - Use the global feature selection results to process each mouse.
            - Run regression and predict runs for each mouse.
    """
    # Initialize an aggregation dictionary keyed by (phase1, phase2)
    aggregated_predictions = {}
    aggregated_feature_weights = {}
    aggregated_raw_features = {}
    aggregated_cluster_loadings = {}
    multi_stride_data = {}
    even_ws = {}
    odd_ws = {}
    phase1_pc = {}
    phase2_pc = {}

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                try:
                    tup = global_fs_results.get((phase1, phase2, stride_number), None)
                    if tup is not None:
                        selected_features, fs_df = tup
                        global_pca = global_pca_results.get((phase1, phase2, stride_number), None)
                    else:
                        selected_features, fs_df, global_pca = None, None, None

                    # Retrieve the stored cluster mapping for this (phase1, phase2, stride_number)
                    current_cluster_mapping = cluster_mappings.get((phase1, phase2, stride_number), {})

                    # Process phase comparison and collect aggregated info
                    agg_info, ftr_wghts, raw_features, cluster_loadings, evenW, oddW, pcs_p1, pcs_p2 = process_mouse_phase_comparison(
                        mouse_id=mouse_id, stride_number=stride_number,
                        phase1=phase1, phase2=phase2,
                        stride_data=stride_data, stride_data_compare=stride_data_compare,
                        condition=condition, exp=exp, day=day,
                        base_save_dir_condition=base_save_dir_condition,
                        selected_features=selected_features, fs_df=fs_df,
                        global_pca=global_pca,
                        compare_condition=compare_condition,
                        cluster_mapping=current_cluster_mapping
                    )

                    # Store the prediction in the multi_stride_data dictionary.
                    key = (phase1, phase2)
                    if key not in multi_stride_data:
                        multi_stride_data[key] = {}
                    multi_stride_data[key].setdefault(stride_number, []).append(agg_info)

                    # Determine if agg_info is a tuple or already a PredictionData:
                    if isinstance(agg_info, tuple):
                        prediction = agg_info[0]
                    else:
                        prediction = agg_info

                    aggregated_predictions.setdefault((phase1, phase2, stride_number), []).append(prediction)
                    aggregated_feature_weights[(mouse_id, phase1, phase2, stride_number)] = ftr_wghts
                    aggregated_raw_features[(mouse_id, phase1, phase2, stride_number)] = raw_features
                    aggregated_cluster_loadings.setdefault((phase1, phase2, stride_number), {})[
                        mouse_id] = cluster_loadings
                    even_ws[(mouse_id, phase1, phase2, stride_number)] = evenW
                    odd_ws[(mouse_id, phase1, phase2, stride_number)] = oddW
                    phase1_pc[(mouse_id, phase1, phase2, stride_number)] = pcs_p1
                    phase2_pc[(mouse_id, phase1, phase2, stride_number)] = pcs_p2

                except ValueError as e:
                    print(f"Error processing {mouse_id}, stride {stride_number}, {phase1} vs {phase2}: {e}")

    """
        # -------- Aggregated Plots and Further Processing --------
        After processing all mice, create aggregated plots for each phase pair.
    """
    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    os.makedirs(aggregated_save_dir, exist_ok=True)

    # Plot aggregated cluster loadings.
    utils.plot_cluster_loadings_lines(aggregated_cluster_loadings, aggregated_save_dir)

    # Single stride data plots
    for (phase1, phase2, stride_number), agg_data in aggregated_predictions.items():
        utils.plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_aggregated_feature_weights(aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_regression_loadings_PC_space_across_mice(global_pca_results, aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_even_odd_PCs_across_mice(even_ws, odd_ws, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.rank_within_vs_between_differences(even_ws, odd_ws, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)

        for p in [phase1, phase2]:
            pc = phase1_pc if p == phase1 else phase2_pc # todo check this works with more phase combinations
            # filter by (phase1, phase2, stride_number), keeping format of (mouse_id, phase1, phase2, stride_number)
            pc = {k: v for k, v in pc.items() if k[1] == phase1 and k[2] == phase2 and k[3] == stride_number}

            # get available mice from predictions
            mice = []
            for a in agg_data:
                mice.append(a.mouse_id)
            con = condition.split('_')[0]
            phase_runs = expstuff['condition_exp_runs'][con][exp][p]
            mouse_runs = {}
            for m in mice:
                mouse_x_vals = [pred.x_vals for pred in agg_data if pred.mouse_id == m][0]
                runs_mask = np.isin(phase_runs, mouse_x_vals)
                mouse_runs[m] = phase_runs[runs_mask]

            utils.compare_within_between_variability(pc, mouse_runs, aggregated_save_dir, p, stride_data, phase1, phase2, stride_number, exp, condition_label=condition)

    # Multi stride data plots
    for (phase1, phase2), stride_dict in multi_stride_data.items():
        for normalize in [True, False]:
            for smooth in [True, False]:
                utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=smooth, normalize=normalize)
                utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=smooth, normalize=normalize)
        summary_curves_dict, summary_derivatives_dict, plateau_dict, learning_rate_dict = utils.fit_exponential_to_prediction(stride_dict, aggregated_save_dir, phase1, phase2, condition, exp)


    """
    Fitting derivative curve to APA phase data
    """
    derivative_save_dir = os.path.join(base_save_dir, "Derivative")
    derivative_save_dir_condition = os.path.join(base_save_dir_condition, "Derivative")
    global_pca_derivative_results = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            y_reg_dict = summary_derivatives_dict[(phase1, phase2, stride_number)]['individual']
            available_mice = y_reg_dict.keys()

            allmice_data_selected = {}
            allmice_data = {}
            all_mice_data_broad = {}
            for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
                # ----------------- Get data and features -----------------
                scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.select_runs_data(
                    mouse_id, stride_number, condition, exp, day, stride_data, 'APA1', 'APA2')
                if mouse_id in available_mice:
                    allmice_data_selected[mouse_id] = selected_scaled_data_df
                    allmice_data[mouse_id] = scaled_data_df
                all_mice_data_broad[mouse_id] = scaled_data_df

            features = list(all_mice_data_broad[mouse_id].T.index) # todo maybe eventually select the features

            # ---------------- Compute global PCA ---------------------
            pca, loadings_df = compute_global_pca_for_phase(
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                stride_number, 'APA1', 'APA2', condition,
                exp, day, stride_data, features)
            global_pca_derivative_results[('APA1', 'APA2', stride_number)] = (pca, loadings_df)

            # Retrieve the stored cluster mapping for this (phase1, phase2, stride_number)
            current_cluster_mapping = cluster_mappings.get((phase1, phase2, stride_number), {})

            X_all = pd.concat(allmice_data_selected.values(), axis=1).T # todo check axis
            Xdr_all = np.dot(loadings_df.T, X_all.T)
            Xdr_all, normalize_mean_all, normalize_std_all = utils.normalize(Xdr_all)

            y_reg_zeroed = []
            y_reg_dict_zeroed = {}
            for m in available_mice:
                expected_runs = expstuff['condition_exp_runs'][condition.split('_')[0]][exp]['APA']
                available_runs = allmice_data_selected[m].T.index
                run_mask = np.isin(expected_runs, available_runs)
                # subtract mean
                mean = np.mean(y_reg_dict[m])
                y = y_reg_dict[m] - mean
                y = y[run_mask]
                y_reg_dict_zeroed[m] = [y, available_runs]
                y_reg_zeroed.append(y)
            y_reg_all = np.concatenate(y_reg_zeroed)

            model = LinearRegression(fit_intercept=False)
            model.fit(Xdr_all.T, y_reg_all)
            w = model.coef_

            y_pred = np.dot(w, Xdr_all)
            # y_check_acc = y_pred.copy()
            # y_check_acc[y_check_acc > 0] = 1
            # y_check_acc[y_check_acc <= 0] = 0
            # bal_acc = balanced_accuracy(y_reg_all.T, y_check_acc.T)

            # predict for each mouse
            predictions = {}
            X_vals = {}
            interpolated_curves = []

            common_x = np.arange(0, 160, 1)

            plt.figure(figsize=(10, 8))
            for m in condition_specific_settings[condition]['global_fs_mouse_ids']:
                try:
                    all_trials_dr = np.dot(loadings_df.T, all_mice_data_broad[m].T)
                    all_trials_dr = ((all_trials_dr.T - normalize_mean_all) / normalize_std_all).T
                    run_pred_scaled = np.dot(w, all_trials_dr)

                    kernel_size = 5
                    padded_run_pred_scaled = np.pad(run_pred_scaled, pad_width=kernel_size, mode='reflect')
                    smoothed_scaled_pred = medfilt(padded_run_pred_scaled, kernel_size=kernel_size)
                    smoothed_scaled_pred = smoothed_scaled_pred[kernel_size:-kernel_size]

                    x_vals = list(all_mice_data_broad[m].index)

                    # Plot run prediction
                    # utils.plot_run_prediction(all_mice_data_broad[m], run_pred_scaled, smoothed_scaled_pred,
                    #                           save_path, mouse_id, phase1, phase2, stride_number,
                    #                           scale_suffix="scaled", dataset_suffix=condition_name)
                    predictions[m] = smoothed_scaled_pred
                    X_vals[m] = x_vals

                    max_abs = max(abs(smoothed_scaled_pred.min()), abs(smoothed_scaled_pred.max()))
                    normalized_curve = smoothed_scaled_pred / max_abs if max_abs != 0 else smoothed_scaled_pred

                    interp_curve = np.interp(common_x, x_vals, normalized_curve)
                    interpolated_curves.append(interp_curve)

                    plt.plot(common_x, interp_curve, label=f'Mouse {m}', alpha=0.3)

                except:
                    # skip if mouse has no data
                    continue

            all_curves_array = np.vstack(interpolated_curves)
            mean_curve = np.mean(all_curves_array, axis=0)
            plt.plot(common_x, mean_curve, color='black', linewidth=2, label='Mean Curve')

            plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
            plt.title(
                f'Aggregated Scaled Run Predictions using derivative of APA curve, stride {stride_number}')
            plt.xlabel('Run Number')
            plt.ylabel('Normalized Prediction (Smoothed)')
            plt.ylim(-1, 1)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.grid(False)
            plt.gca().yaxis.grid(True)
            plt.tight_layout()

            save_path_full = os.path.join(derivative_save_dir_condition,
                                          f"Aggregated_Run_Predictions_derivativeAPAcurve_stride{stride_number}.png")
            os.makedirs(derivative_save_dir_condition, exist_ok=True)
            plt.savefig(save_path_full, dpi=300)
            plt.close()




    print("Analysis complete.")

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
