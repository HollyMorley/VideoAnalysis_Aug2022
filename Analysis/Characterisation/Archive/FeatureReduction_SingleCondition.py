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

# ----------------------------
# Library Imports (unchanged)
# ----------------------------
from Analysis.Tools.PCA import (
    perform_pca, plot_pca, plot_scree, cross_validate_pca,
    plot_average_variance_explained_across_folds, compute_global_pca_for_phase)

from Analysis.Tools.LogisticRegression import (run_regression)
from Analysis.Tools.FeatureSelection import (global_feature_selection)
from Analysis.Tools.ClusterFeatures import (
    plot_feature_clustering, plot_feature_clustering_3d, plot_feature_clusters_chart,
    find_feature_clusters, plot_corr_matrix_sorted_by_cluster, cluster_features_with_pca_cv, cluster_features_with_ica_cv
)
from Analysis.Tools.FindMousePools import (
    compute_mouse_similarity, get_loading_matrix_from_nested, pool_mice_by_similarity
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
    utils.log_settings(settings_to_log, base_save_dir)
    return stride_data, stride_data_compare, base_save_dir, base_save_dir_condition

# -----------------------------------------------------
# Processing Functions
# -----------------------------------------------------
def process_phase_comparison(mouse_id: str, stride_number: int, phase1: str, phase2: str,
                             stride_data, stride_data_compare, condition: str, exp: str, day,
                             base_save_dir_condition: str,
                             selected_features: Optional[List[str]] = None,
                             fs_df: Optional[dict] = None,
                             global_pca: Optional[Tuple] = None,
                             compare_condition: Optional[str] = None,
                             cluster_mapping: Optional[Dict] = None) -> Tuple[utils.PredictionData, utils.FeatureWeights, any, Optional[Dict[int, any]]]:
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
    #save_path = "\\\\?\\" + save_path
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2)

    if selected_features is None and global_settings["select_features"] == True:
        y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])
        # For local selection you might choose not to save to file (or provide a unique save_file path).
        selected_features, _ = utils.unified_feature_selection(
            feature_data_df=selected_scaled_data_df,
            y=y_reg,
            c=condition_specific_settings[condition]['c'],
            method=global_settings["method"],  # or 'rf' or 'regression'
            cv=global_settings["nFolds_selection"],
            n_iterations=global_settings["n_iterations_selection"],
            save_file=None,
            overwrite_FeatureSelection=global_settings["overwrite_FeatureSelection"]
        )
        print("Selected features (local):", selected_features)
    else:
        if global_settings["select_features"] == True:
            print("Using globally selected features for feature reduction.")
        else:
            print("Using all features for feature reduction.")
            selected_features = list(selected_scaled_data_df.index)
        fs_df = fs_df


    if fs_df is not None:
        # Detailed regression results are available; proceed with plotting.
        selected_features_accuracies = fs_df.loc[selected_features]
        print(f"Length of significant features: {len(selected_features)}")

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

    if global_pca is not None:
        # Use the global PCA transformation.
        pca, loadings_df = global_pca
        # Project this mouse's data using the global PCA.
        pcs = pca.transform(reduced_feature_data_df)
        pcs_phase1 = pcs[mask_phase1]
        pcs_phase2 = pcs[mask_phase2]
        labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
        labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
        labels = np.concatenate([labels_phase1, labels_phase2])
        pcs_combined = np.vstack([pcs_phase1, pcs_phase2])
        # Plot using the global PCA.
        plot_pca(pca, pcs_combined, labels, phase1, phase2, stride_number, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, phase1, phase2, stride_number, save_path)
    else:
        # Local PCA (existing behavior).
        fold_variances = cross_validate_pca(reduced_feature_data_df, save_path, n_folds=10)
        plot_average_variance_explained_across_folds(fold_variances, phase1, phase2, stride_number)
        #n_components = 10 if allmice else 4
        n_components = 10 if len(selected_features) > 10 else len(selected_features)
        pca, pcs, loadings_df = perform_pca(reduced_feature_data_df, n_components=n_components)
        pcs_combined, labels, pcs_phase1, pcs_phase2 = utils.get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2)
        plot_pca(pca, pcs_combined, labels, phase1, phase2, stride_number, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, phase1, phase2, stride_number, save_path)


    # ---------------- Prediction - Regression-Based Feature Contributions ----------------
    smoothed_scaled_pred, feature_weights, _ = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, stride_number, save_path, condition)

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

    return pred_data, ftr_wghts, reduced_feature_data_df, cluster_loadings


def process_mouse_group(group_id: int, mouse_ids: List[str], stride_number: int, phase1: str, phase2: str,
                        condition: str, exp: str, day, stride_data, base_save_dir_condition: str,
                        c, nFolds, n_iterations, overwrite, method) -> Tuple[List[utils.PredictionData], List[str], Dict[Tuple[str, str, str, int], utils.FeatureWeights]]:
    print(f"Processing Group {group_id} with mice {mouse_ids} for {phase1} vs {phase2}, stride {stride_number}")

    # Run group-level feature selection and PCA.
    print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
    # Perform global feature selection.
    group_save_dir = os.path.join(base_save_dir_condition, f'Group_{group_id}_GlobalFS')
    selected_features, fs_df = global_feature_selection(mouse_ids, stride_number, phase1, phase2, condition, exp, day,
                                                        stride_data, save_dir=group_save_dir, c=c, nFolds=nFolds,
                                                        n_iterations=n_iterations, overwrite=overwrite, method=method)

    # Compute global PCA using the selected features.
    group_pca, loadings_df = compute_global_pca_for_phase(mouse_ids, stride_number, phase1, phase2, condition, exp, day,
                                                    stride_data, selected_features)

    # Now, for each mouse in the group, re-run the prediction steps using the group PCA.
    group_predictions = []
    group_weights: Dict[Tuple[str, str, str, int], utils.FeatureWeights] = {}
    for mouse_id in mouse_ids:
        pred_data, ftr_wghts, _, _ = process_phase_comparison(
            mouse_id=mouse_id, stride_number=stride_number,
            phase1=phase1, phase2=phase2,
            stride_data=stride_data, stride_data_compare=None,
            condition=condition, exp=exp, day=day,
            base_save_dir_condition=base_save_dir_condition,
            selected_features=selected_features, fs_df=fs_df,
            global_pca=(group_pca, loadings_df),
            compare_condition='None',
            cluster_mapping={}
        )

        # Set the group id in the data class
        pred_data.group_id = group_id
        ftr_wghts.group_id = group_id
        group_predictions.append(pred_data)
        group_weights[(mouse_id, phase1, phase2, stride_number)] = ftr_wghts


    return group_predictions, selected_features, group_weights

def run_grouped_mice(aggregated_predictions: Dict, aggregated_cluster_loadings: Dict, stride_data,
                     condition: str, exp: str, day, base_save_dir_condition: str,
                     aggregated_save_dir: str):
    mouse_groups = {}
    grouped_predictions = {}
    grouped_feature_weights = {}
    grouped_multi_strides = {}
    groups_by_mouse = {}

    for (phase1, phase2, stride_number), agg_data in aggregated_predictions.items():
        loading_df = get_loading_matrix_from_nested(aggregated_cluster_loadings, (phase1, phase2, stride_number))
        loading_dict = loading_df.to_dict(orient="index")
        sim_df, loading_matrix = compute_mouse_similarity(loading_dict)
        groups = pool_mice_by_similarity(sim_df, threshold=global_settings["mouse_pool_thresh"])
        mouse_groups[(phase1, phase2, stride_number)] = groups
        for group_id, group_mouse_ids in groups.items():
            stride_data_group = stride_data.loc[group_mouse_ids]
            group_preds, _, group_wghts = process_mouse_group(
                group_id=group_id,
                mouse_ids=group_mouse_ids,
                stride_number=stride_number,
                phase1=phase1,
                phase2=phase2,
                condition=condition,
                exp=exp,
                day=day,
                stride_data=stride_data_group,
                base_save_dir_condition=base_save_dir_condition,
                c=condition_specific_settings[condition]['c'],
                nFolds=global_settings["nFolds_selection"],
                n_iterations=global_settings["n_iterations_selection"],
                overwrite=global_settings["overwrite_FeatureSelection"],
                method=global_settings["method"]
            )
            # Aggregate predictions per (phase1, phase2, stride_number)
            key_full = (phase1, phase2, stride_number)
            grouped_predictions.setdefault(key_full, []).extend(group_preds)

            # Merge feature weights; note that if the same mouse appears in multiple groups for a given stride,
            # you may wish to combine them rather than overwrite.
            for k, v in group_wghts.items():
                grouped_feature_weights[k] = v  # adjust merging strategy if needed

            # Also, add these predictions to the multi-stride structure (keyed only by (phase1, phase2))
            key_phase = (phase1, phase2)
            if key_phase not in grouped_multi_strides:
                grouped_multi_strides[key_phase] = {}
            grouped_multi_strides[key_phase].setdefault(stride_number, []).extend(group_preds)

            # Also record groups by mouse (if a mouse appears in multiple groups, the last group id will be used)
            for m in group_mouse_ids:
                groups_by_mouse[m] = group_id

    # Now that we have aggregated all grouped predictions across all strides, call the plotting functions:
    for (phase1, phase2, stride_number), preds in grouped_predictions.items():
        utils.plot_aggregated_run_predictions_by_group(preds, aggregated_save_dir, phase1, phase2,
                                                       stride_number, condition_label=condition)

    utils.plot_aggregated_feature_weights_by_group(grouped_feature_weights, groups_by_mouse,
                                                   aggregated_save_dir, phase1, phase2, stride_number,
                                                   condition_label=condition)

    # Loop over each phase pair in grouped_multi_strides and plot multi-stride predictions
    for (phase1, phase2), stride_dict in grouped_multi_strides.items():
        utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir,
                                            condition_label=condition, smooth=False, normalize=True)
        utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir,
                                            condition_label=condition, smooth=True, normalize=True)
        utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir,
                                                       condition_label=condition, smooth=False, normalize=True)
        utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir,
                                                         condition_label=condition, smooth=True, normalize=True)


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
        # -------- ** OPTIONAL ** Global Feature Selection --------
        Optionally, perform feature selection on data from all mice combined - when allmice is True.   
    """
    if global_settings["allmice"]:
        # Directory for global feature selection.
        global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
        os.makedirs(global_fs_dir, exist_ok=True)

        global_fs_results = {}
        global_pca_results = {}
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

    """
        # -------- Process Each Mouse --------
        For each mouse, process the phase comparison and collect aggregated info.
        - If allmice is False, 
            - Find selected features per mouse. 
            - Find similar groups of mice based on individually selected features and then select features again from each group.
            - Run PCA for each group
        - If allmice is True,
            - Use the global feature selection results to process each mouse.
        - Finally, run regression and predict runs for each mouse.
    """

    # Initialize an aggregation dictionary keyed by (phase1, phase2)
    aggregated_predictions = {}
    aggregated_feature_weights = {}
    aggregated_raw_features = {}
    aggregated_cluster_loadings = {}
    multi_stride_data = {}

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                try:
                    # Retrieve global feature selection results if allmice is True.
                    if global_settings["allmice"]:
                        tup = global_fs_results.get((phase1, phase2, stride_number), None)
                        if tup is not None:
                            selected_features, fs_df = tup
                            global_pca = global_pca_results.get((phase1, phase2, stride_number), None)
                        else:
                            selected_features, fs_df, global_pca = None, None, None
                    # Otherwise, leave blank to perform local feature selection per mouse
                    else:
                        selected_features, fs_df, global_pca = None, None, None

                    # Retrieve the stored cluster mapping for this (phase1, phase2, stride_number)
                    current_cluster_mapping = cluster_mappings.get((phase1, phase2, stride_number), {})

                    # Process phase comparison and collect aggregated info
                    agg_info, ftr_wghts, raw_features, cluster_loadings = process_phase_comparison(
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
                    aggregated_cluster_loadings.setdefault((phase1, phase2, stride_number), {})[mouse_id] = cluster_loadings
                except ValueError as e:
                    print(f"Error processing {mouse_id}, stride {stride_number}, {phase1} vs {phase2}: {e}")

    """
        # -------- Aggregated Plots and Further Processing --------
        After processing all mice, create aggregated plots for each phase pair.
    """
    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    os.makedirs(aggregated_save_dir, exist_ok=True)

    # Plot aggregated cluster loadings. todo Should work for both allmice=True and False.
    utils.plot_cluster_loadings_lines(aggregated_cluster_loadings, aggregated_save_dir)

    #### -------- ** OPTION 1 ** Don't pool mice and run aggregated plots across all mice --------
    if not global_settings["pool_mice"]:
        for (phase1, phase2, stride_number), agg_data in aggregated_predictions.items():
            utils.plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
            utils.plot_aggregated_feature_weights_byFeature(aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
            utils.plot_regression_loadings_PC_space_across_mice(global_pca_results, aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)

        for (phase1, phase2), stride_dict in multi_stride_data.items():
            utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=False)
            utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=True, smooth_window=21)

            utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=False)
            utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=True, smooth_window=21)



        # # -------------------------- Cluster Regression Weights Across Mice -------------------------- # todo maybe just get rid of this
        # unique_phase_pairs = set((p1, p2, s) for (_, p1, p2, s) in aggregated_feature_weights.keys())
        # for phase_pair in unique_phase_pairs:
        #     cluster_df, kmeans_model = utils.cluster_regression_weights_across_mice(
        #         aggregated_feature_weights, phase_pair, aggregated_save_dir, n_clusters=3, aggregate_strides=True
        #     )
        #     if cluster_df is not None:
        #         print(f"Global clustering for phase pair {phase_pair}:")
        #         print(cluster_df)

        # -------------------------- Process compare condition predictions --------------------------
        # todo need to add feature_data and feature_data_compare
        utils.process_compare_condition(mouseIDs_base=condition_specific_settings[condition]['global_fs_mouse_ids'],
                                        mouseIDs_compare=condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
                                        condition=condition, compare_condition=compare_condition, exp=exp, day=day, stride_data=stride_data,
                                        stride_data_compare=stride_data_compare, phases=phases, stride_numbers=stride_numbers,
                                        base_save_dir_condition=base_save_dir_condition, aggregated_save_dir=aggregated_save_dir,
                                        global_fs_results=global_fs_results, global_pca_results=global_pca_results)

    #### -------- ** OPTION 2 ** Pool Mice by Similarity and then re-run feature selection, PCA and prediction --------
    elif global_settings["pool_mice"]:
        run_grouped_mice(aggregated_predictions, aggregated_cluster_loadings, stride_data, condition, exp, day, base_save_dir_condition, aggregated_save_dir)

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
