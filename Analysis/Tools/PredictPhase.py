import os
import itertools
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.linear_model import LinearRegression

# ----------------------------
# Library Imports
# ----------------------------
from Analysis.Tools.PCA import (
    plot_pca, plot_scree, compute_global_pca_for_phase, compute_pca_allStrides)
from Analysis.Tools.LogisticRegression import (run_regression)
from Analysis.Tools.FeatureSelection import (global_feature_selection)
from Analysis.Tools.ClusterFeatures import (
    plot_feature_clustering, plot_feature_clustering_3d, plot_feature_clusters_chart,
    find_feature_clusters, plot_corr_matrix_sorted_by_cluster, plot_corr_matrix_sorted_manually
)
from Analysis.Tools import utils_feature_reduction as utils
from Helpers.Config_23 import *
from Analysis.Tools.config import (
    base_save_dir_no_c, global_settings, condition_specific_settings, instance_settings
)


# ----------------------------
# Function Definitions
# ----------------------------

def cluster_features_main(feature_data, feature_data_compare, phases, stride_numbers, condition, compare_condition, stride_data, stride_data_compare, base_save_dir_condition, combine_conditions):
    cluster_mappings = {}
    if global_settings.get("cluster_all_strides", True):
        # Cluster across all available strides
        for phase1, phase2 in itertools.combinations(phases, 2):
            if global_settings['cluster_method'] == 'kmeans':
                cluster_mapping, feature_matrix = find_feature_clusters(
                    feature_data,
                    feature_data_compare,
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
                    "all",
                    stride_data,
                    stride_data_compare,
                    phase1,
                    phase2,
                    base_save_dir_condition,
                    combine_conditions,
                    method='kmeans'
                )
            # Store same mapping for all strides.
            for stride_number in stride_numbers:
                cluster_mappings[(phase1, phase2, stride_number)] = cluster_mapping

            plot_feature_clustering(feature_matrix, cluster_mapping, phase1, phase2, "all", base_save_dir_condition)
            plot_feature_clustering_3d(feature_matrix, cluster_mapping, phase1, phase2, "all", base_save_dir_condition)
            sorted_features = sorted(feature_matrix.index, key=lambda f: cluster_mapping.get(f, -1))
            plot_feature_clusters_chart(cluster_mapping, sorted_features, phase1, phase2, "all",
                                        base_save_dir_condition)
            plot_corr_matrix_sorted_by_cluster(feature_matrix, sorted_features, cluster_mapping,
                                               base_save_dir_condition,
                                               filename=f'CorrMatrix_{phase1}_{phase2}_all')
            plot_corr_matrix_sorted_manually(
                feature_matrix, base_save_dir_condition,
                filename=f'CorrMatrix_manualclustering_{phase1}_{phase2}_all'
            )
    else:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                if global_settings['cluster_method'] == 'kmeans':
                    cluster_mapping, feature_matrix = find_feature_clusters(
                        feature_data,
                        feature_data_compare,
                        condition_specific_settings[condition]['global_fs_mouse_ids'],
                        condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
                        stride_number,
                        stride_data,
                        stride_data_compare,
                        phase1,
                        phase2,
                        base_save_dir_condition,
                        combine_conditions,
                        method='kmeans')
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
                plot_corr_matrix_sorted_manually(feature_matrix, base_save_dir_condition,
                                                 filename=f'CorrMatrix_manualclustering_{phase1}_{phase2}_stride{stride_number}')
    return cluster_mappings

def global_feature_selection_main(feature_data: pd.DataFrame, phases: List[str], stride_numbers: List[int], condition: str, exp: str, day: Optional[str],
                      stride_data: pd.DataFrame, base_save_dir: str):
    # Directory for global feature selection.
    global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
    os.makedirs(global_fs_dir, exist_ok=True)

    global_fs_results = {}
    global_pca_results = {}
    global_stride_fs_results = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            if global_settings["select_features"] == True:
                print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
                # Perform global feature selection.
                selected_features, _ = global_feature_selection(
                    feature_data,
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    stride_number, phase1, phase2, condition,
                    exp, day,
                    stride_data, save_dir=global_fs_dir,
                    c=condition_specific_settings[condition]['c'],
                    nFolds=global_settings["nFolds_selection"],
                    n_iterations=global_settings["n_iterations_selection"],
                    overwrite=global_settings["overwrite_FeatureSelection"],
                    method=global_settings["method"])
                global_fs_results[(phase1, phase2, stride_number)] = selected_features

                if not global_settings["combine_stride_features"]:
                    if not global_settings["pca_CombineAllStrides"]:
                        # Compute global PCA using the selected features.
                        pca, loadings_df = compute_global_pca_for_phase(
                            feature_data, None,
                            condition_specific_settings[condition]['global_fs_mouse_ids'], None,
                            stride_number, phase1, phase2, stride_data, None, selected_features,
                            combine_conditions=False, n_components=global_settings["pcs_to_show"])

                        global_pca_results[(phase1, phase2, stride_number)] = (pca, loadings_df)
                    else:
                        pca, loadings_df = compute_pca_allStrides(
                            feature_data, None,
                            condition_specific_settings[condition]['global_fs_mouse_ids'], None,
                            stride_numbers, phase1, phase2, stride_data, None,
                            selected_features, combine_conditions=False, n_components=global_settings["pcs_to_show"])
                        global_pca_results[(phase1, phase2, "all_strides")] = (pca, loadings_df)
            else:
                # If not selecting features, just use all features.
                selected_features = list(feature_data.loc(axis=0)[stride_number].columns)
                global_fs_results[(phase1, phase2, stride_number)] = selected_features

                if not global_settings["combine_stride_features"]:
                    if not global_settings["pca_CombineAllStrides"]:
                        # Compute global PCA using the selected features.
                        pca, loadings_df = compute_global_pca_for_phase(
                            feature_data, None,
                            condition_specific_settings[condition]['global_fs_mouse_ids'], None,
                            stride_number, phase1, phase2, stride_data, None, selected_features,
                            combine_conditions=False, n_components=global_settings["pcs_to_show"])

                        global_pca_results[(phase1, phase2, stride_number)] = (pca, loadings_df)
                    else:
                        pca, loadings_df = compute_pca_allStrides(
                            feature_data, None,
                            condition_specific_settings[condition]['global_fs_mouse_ids'], None,
                            stride_numbers, phase1, phase2, stride_data, None,
                            selected_features, combine_conditions=False, n_components=global_settings["pcs_to_show"])
                        global_pca_results[(phase1, phase2, "all_strides")] = (pca, loadings_df)

        # Combine strides
        stride_features = []
        for s in stride_numbers:
            features = global_fs_results[(phase1, phase2, s)]
            stride_features.extend(features)
        # remove duplicates
        stride_features = list(set(stride_features))
        global_stride_fs_results[(phase1, phase2)] = stride_features

        return global_fs_results, global_pca_results, global_stride_fs_results

def global_pca_main(feature_data, feature_data_compare, global_stride_fs_results, phases, stride_numbers,
                    condition, compare_condition, stride_data, stride_data_compare, base_save_dir_condition,
                    select_feats=True, combine_conditions=False, combine_strides=False):
    global_pca_results = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        if combine_strides:
            # When aggregating across all strides:
            if global_settings["select_features"]:
                selected_features = global_stride_fs_results.get((phase1, phase2),
                                                                 list(feature_data.columns)) if select_feats else list(
                    feature_data.columns)
            else:
                selected_features = list(feature_data.columns)

            pca, pca_loadings = compute_pca_allStrides(
                feature_data, feature_data_compare,
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
                stride_numbers, phase1, phase2, stride_data, stride_data_compare,
                selected_features, combine_conditions, n_components=global_settings["pcs_to_show"]
            )
            global_pca_results[(phase1, phase2, "all_strides")] = (pca, pca_loadings)
            plot_scree(pca, phase1, phase2, "all_strides", condition, base_save_dir_condition)

        else:
            for stride_number in stride_numbers:
                if global_settings["select_features"] == True:
                    if select_feats == True:
                        selected_features = global_stride_fs_results.get((phase1, phase2), None)
                    else:
                        selected_features = list(feature_data.columns)
                else:
                    selected_features = list(feature_data.columns)

                pca, pca_loadings = compute_global_pca_for_phase(
                    feature_data, feature_data_compare,
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
                    stride_number, phase1, phase2,stride_data, stride_data_compare, selected_features,
                    combine_conditions, n_components=global_settings["pcs_to_show"])
                global_pca_results[(phase1, phase2, stride_number)] = (pca, pca_loadings)
                plot_scree(pca, phase1, phase2, stride_number, condition, base_save_dir_condition)
    return global_pca_results

def process_mice_main(mouse_ids: List[str], phases: List[str], stride_numbers: List[int], condition: str,
                        exp: str, day: Optional[str],
                        stride_data: pd.DataFrame, base_save_dir_condition: str,
                        feature_data: pd.DataFrame,
                        global_fs_results: Dict[Tuple[str, str, int], List[str]],
                        global_pca_results: Dict[Tuple[str, str, any], Tuple[LinearRegression, pd.DataFrame]],
                        global_stride_fs_results: Dict[Tuple[str, str], List[str]],
                        cluster_mappings: Dict[Tuple[str, str, any], Dict[str, int]]):

    # Initialize an aggregation dictionary keyed by (phase1, phase2)
    aggregated_predictions = {}
    aggregated_feature_weights = {}
    aggregated_contributions = {}
    aggregated_raw_features = {}
    aggregated_raw_features_all = {}
    aggregated_cluster_loadings = {}
    multi_stride_data = {}
    even_ws = {}
    odd_ws = {}
    phase1_pc = {}
    phase2_pc = {}
    normalize_mean_pc = {}
    normalize_std_pc = {}

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                try:
                    if global_settings["select_features"] == True:
                        if global_settings["combine_stride_features"]:
                            selected_features = global_stride_fs_results.get((phase1, phase2), None)
                        else:
                            selected_features = global_fs_results.get((phase1, phase2, stride_number), None)
                    else:
                        selected_features = list(feature_data.loc(axis=0)[stride_number, :].columns.unique())
                    if global_settings["pca_CombineAllStrides"]:
                        global_pca = global_pca_results.get((phase1, phase2, 'all_strides'), None)
                    else:
                        global_pca = global_pca_results.get((phase1, phase2, stride_number), None)

                    # Retrieve the stored cluster mapping for this (phase1, phase2, stride_number)
                    if cluster_mappings is None:
                        current_cluster_mapping = None
                    else:
                        current_cluster_mapping = cluster_mappings.get((phase1, phase2, stride_number), {})

                    # Process phase comparison and collect aggregated info
                    results = process_mouse_phase_comparison(
                        mouse_id=mouse_id, stride_number=stride_number,
                        phase1=phase1, phase2=phase2,
                        stride_data=stride_data,
                        condition=condition, exp=exp, day=day,
                        base_save_dir_condition=base_save_dir_condition,
                        feature_data=feature_data,
                        selected_features=selected_features,
                        global_pca=global_pca,
                        cluster_mapping=current_cluster_mapping,
                        cluster_present=bool(current_cluster_mapping)
                    )
                    agg_info, ftr_wghts, contribution, raw_features, raw_features_all, cluster_loadings, evenW, oddW, pcs_p1, pcs_p2, normalize_mean, normalize_std = results
                    # agg_info, ftr_wghts, raw_features, raw_features_all, cluster_loadings, evenW, oddW, pcs_p1, pcs_p2 = results

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
                    aggregated_contributions[(mouse_id, phase1, phase2, stride_number)] = contribution
                    aggregated_raw_features[(mouse_id, phase1, phase2, stride_number)] = raw_features
                    aggregated_raw_features_all[(mouse_id, phase1, phase2, stride_number)] = raw_features_all
                    aggregated_cluster_loadings.setdefault((phase1, phase2, stride_number), {})[
                        mouse_id] = cluster_loadings
                    even_ws[(mouse_id, phase1, phase2, stride_number)] = evenW
                    odd_ws[(mouse_id, phase1, phase2, stride_number)] = oddW
                    phase1_pc[(mouse_id, phase1, phase2, stride_number)] = pcs_p1
                    phase2_pc[(mouse_id, phase1, phase2, stride_number)] = pcs_p2
                    normalize_mean_pc[(mouse_id, phase1, phase2, stride_number)] = normalize_mean
                    normalize_std_pc[(mouse_id, phase1, phase2, stride_number)] = normalize_std

                except ValueError as e:
                    print(f"Error processing {mouse_id}, stride {stride_number}, {phase1} vs {phase2}: {e}")
    return aggregated_predictions, aggregated_feature_weights, aggregated_contributions, aggregated_raw_features, aggregated_raw_features_all, aggregated_cluster_loadings, multi_stride_data, even_ws, odd_ws, phase1_pc, phase2_pc, normalize_mean_pc, normalize_std_pc


def get_mouse_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2, selected_features):
    # Get the data for this mouse and phase comparison.
    scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.select_runs_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2)

    # Optionally, reduce the data to only the selected features.
    if global_settings["select_features"] == True:
        print("Using globally selected features for feature reduction.")
    else:
        print("Using all features for feature reduction.")
        selected_features = list(selected_scaled_data_df.index)

    # Now reduce the data to only the significant features
    reduced_feature_data_df = scaled_data_df.loc(axis=1)[selected_features]
    reduced_feature_selected_data_df = selected_scaled_data_df.loc[selected_features]

    return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2, reduced_feature_data_df, reduced_feature_selected_data_df

def process_mouse_phase_comparison(mouse_id: str, stride_number: int, phase1: str, phase2: str,
                                   stride_data, condition: str, exp: str, day,
                                   base_save_dir_condition: str,
                                   feature_data: pd.DataFrame,
                                   selected_features: Optional[List[str]] = None,
                                   global_pca: Optional[Tuple] = None,
                                   cluster_mapping: Optional[Dict] = None,
                                   cluster_present: Optional[bool] = True) -> (
        Tuple)[utils.PredictionData, utils.FeatureWeights, utils.ContributionData, pd.DataFrame, pd.DataFrame, Optional[Dict[int, any]], any, any, any, any, any, any]:
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

    # Get the data for this mouse and phase comparison. # todo: I am calculating data again here!!!!
    (scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2,
     reduced_feature_data_df, reduced_feature_selected_data_df) = get_mouse_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2, selected_features)

    # ---------------- PCA Plotting (loaded from global) ----------------

    # Use the global PCA transformation.
    pca, loadings_df = global_pca # features x PCs
    # Project this mouse's data using the global PCA.
    pcs = pca.transform(reduced_feature_data_df) # runs x PCs
    # Trim to number of PCs to use.
    pcs = pcs[:, :global_settings["pcs_to_use"]]
    pcs_df = pd.DataFrame(pcs, index=reduced_feature_data_df.index, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]
    runs_phase1 = reduced_feature_data_df.index[mask_phase1]
    runs_phase2 = reduced_feature_data_df.index[mask_phase2]
    labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
    labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
    labels = np.concatenate([labels_phase1, labels_phase2])
    pcs_p1p2 = np.vstack([pcs_phase1, pcs_phase2])
    pcs_p1p2_df = pd.DataFrame(pcs_p1p2, index=np.concatenate([runs_phase1, runs_phase2]), columns=[f'PC{i+1}' for i in range(pcs_p1p2.shape[1])])
    # Plot using the global PCA.
    plot_pca(pca, pcs_p1p2, labels, phase1, phase2, stride_number, stepping_limbs, run_numbers, mouse_id, condition, save_path)

    # ---------------- Prediction - Regression-Based Feature Contributions ----------------
    # smoothed_scaled_pred, feature_weights, _, normalize_mean_pc, normalize_std_pc = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, stride_number, save_path, condition)
    smoothed_scaled_pred, feature_weights, _, normalize_mean_pc, normalize_std_pc, single_f, unique_f, single_pc, unique_pc = run_regression(loadings_df, pcs_p1p2_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, stride_number, save_path, condition)

    # ---------------- Within vs between mice comparison - setup ----------------
    # fit regression model using even and odd Xdr values separately
    combined_mask = mask_phase1 | mask_phase2
    combined_mask_even = combined_mask[::2]
    combined_mask_odd = combined_mask[1::2]

    reduced_feature_selected_data_df_even = reduced_feature_data_df[::2][combined_mask_even].T
    reduced_feature_selected_data_df_odd = reduced_feature_data_df[1::2][combined_mask_odd].T

    pcs_df_even = pcs_df[::2][combined_mask_even].T
    pcs_df_odd = pcs_df[1::2][combined_mask_odd].T

    smoothed_scaled_pred_even, feature_weights_even, _, _, _, _, _, _, _ = run_regression(loadings_df, pcs_df_even.T, reduced_feature_data_df, reduced_feature_selected_data_df_even, mask_phase1[::2], mask_phase2[::2], mouse_id, phase1, phase2, stride_number, save_path, condition, plot_pred=False, plot_weights=False)
    smoothed_scaled_pred_odd, feature_weights_odd, _, _, _, _, _, _, _ = run_regression(loadings_df, pcs_df_odd.T, reduced_feature_data_df, reduced_feature_selected_data_df_odd, mask_phase1[1::2], mask_phase2[1::2], mouse_id, phase1, phase2, stride_number, save_path, condition, plot_pred=False, plot_weights=False)

    # get weights in PC space for even and odd mice
    even_weights = np.dot(feature_weights_even, loadings_df)
    odd_weights = np.dot(feature_weights_odd, loadings_df)

    # ---------------- Map selected features to respective clusters ----------------
    if cluster_present:
        feature_cluster_assignments = {feat: cluster_mapping.get(feat, -1) for feat in selected_features}

        # Sum the regression loadings by cluster.
        cluster_loadings = {}
        for feat, weight in feature_weights.items():
            cluster = feature_cluster_assignments.get(feat)
            if cluster is not None and cluster != -1:
                cluster_loadings[cluster] = cluster_loadings.get(cluster, 0) + weight
        print(f"Mouse {mouse_id} - Cluster loadings: {cluster_loadings}")
    else:
        cluster_loadings = None


    # Return aggregated prediction as a PredictionData instance.
    pred_data = utils.PredictionData(mouse_id=mouse_id,
                               x_vals=list(reduced_feature_data_df.index),
                               smoothed_scaled_pred=smoothed_scaled_pred)
    ftr_wghts = utils.FeatureWeights(mouse_id=mouse_id,
                                    feature_weights=feature_weights)
    contribution = utils.ContributionData(mouse_id=mouse_id,
                                          unique_feature_contribution=unique_f,
                                          unique_pc_contribution=unique_pc,
                                          single_feature_contribution=single_f,
                                            single_pc_contribution=single_pc)


    return pred_data, ftr_wghts, contribution, reduced_feature_data_df, scaled_data_df, cluster_loadings, even_weights, odd_weights, pcs_phase1, pcs_phase2, normalize_mean_pc, normalize_std_pc

def get_mouse_pcs_by_run(mouse_id: str, stride_number: int, phase1: str, phase2: str,
                                   stride_data, feature_data: pd.DataFrame,
                                   base_save_dir_condition: str,
                                   global_pca: Optional[Tuple] = None,
                                   selected_features: Optional[List[str]] = None):
    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    # Get the data for this mouse and phase comparison.
    (_, _, _, _, mask_phase1, mask_phase2, reduced_feature_data_df, _) = get_mouse_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2, selected_features)

    # ---------------- PCA Plotting (loaded from global) ----------------

    # Use the global PCA transformation.
    pca, loadings_df = global_pca  # features x PCs
    # Project this mouse's data using the global PCA.
    pcs = pca.transform(reduced_feature_data_df)  # runs x PCs
    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]
    return pcs_phase1, pcs_phase2

def find_pcs_outliers_to_remove(pcs_runs_dict, mouse_runs, phase, stride_number):
    outlier_runs, threshold, global_stats = utils.find_outlier_runs_global_std(pcs_runs_dict, phase, stride_number)
    real_runs_to_remove = {}
    for mouse, outlier_indices in outlier_runs.items():
        outlier_indices = list(outlier_indices)
        runs = list(mouse_runs[mouse])
        real_runs = [runs[i] for i in sorted(list(outlier_indices)) if i < len(runs)]
        real_runs_to_remove[mouse] = real_runs
    return real_runs_to_remove, outlier_runs

def find_outliers(feature_data, feature_data_compare, condition, condition_compare, exp, day, stride_data, stride_data_compare, phases, stride_numbers, base_save_dir_condition):
    all_feats_pca = global_pca_main(feature_data, feature_data_compare, None, phases, stride_numbers,
                                    condition, condition_compare, stride_data, stride_data_compare,
                                    base_save_dir_condition, combine_strides=global_settings["pca_CombineAllStrides"],select_feats=False)
    pcs_p1 = {}
    pcs_p2 = {}
    for mouse in condition_specific_settings[condition]['global_fs_mouse_ids']:
        for stride in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                if global_settings["pca_CombineAllStrides"]:
                    stride_pca = all_feats_pca[(phase1, phase2, "all_strides")]
                else:
                    stride_pca = all_feats_pca[(phase1, phase2, stride)]
                features = list(feature_data.loc(axis=0)[stride, mouse].columns)
                pcs_phase1, pcs_phase2 = get_mouse_pcs_by_run(mouse, stride, phase1, phase2, stride_data, feature_data, base_save_dir_condition, stride_pca, features)
                pcs_p1[(mouse, phase1, phase2, stride)] = pcs_phase1
                pcs_p2[(mouse, phase1, phase2, stride)] = pcs_phase2

    to_drop = set()
    for phase1, phase2 in itertools.combinations(phases, 2):
        print(f"\n--- Outlier removal for {phase1} vs {phase2} ---")
        #phase1, phase2 = phases
        print(f"\nLength of feature data before outlier removal: {len(feature_data)}")
        for stride in stride_numbers:
            print("\n----- Finding outliers for stride", stride)
            all_runs_to_remove = {}
            for p in [phase1, phase2]:
                pc = pcs_p1 if p == phase1 else pcs_p2
                # filter by (phase1, phase2, stride_number), keeping format of (mouse_id, phase1, phase2, stride_number)
                pc = {k: v for k, v in pc.items() if k[1] == phase1 and k[2] == phase2 and k[3] == stride}

                # get available mice from feature_data
                mouse_runs = {}
                for m in condition_specific_settings[condition]['global_fs_mouse_ids']:
                    phase_runs = expstuff['condition_exp_runs'][condition.split('_')[0]][exp][p]
                    # get available mouse runs from within phase runs
                    mouse_phase_runs = [run for run in phase_runs if run in feature_data.loc(axis=0)[stride, m].index]
                    mouse_runs[m] = mouse_phase_runs
                runs_to_remove, outlier_runs_indices = find_pcs_outliers_to_remove(pc, mouse_runs, p, stride)
                utils.plot_pcs_outliers(pc, outlier_runs_indices, p, phase1, phase2, stride, base_save_dir_condition)
                all_runs_to_remove[p] = runs_to_remove

            # combine the runs to remove from both phases for each mouse
            all_runs_to_remove_comb = {m: sorted(list(set(all_runs_to_remove[phase1][m] + all_runs_to_remove[phase2][m])))
                                       for m in condition_specific_settings[condition]['global_fs_mouse_ids']}

            utils.save_outliers_csv_updated(all_runs_to_remove_comb, base_save_dir_condition, condition, phase1, phase2,
                                            stride, stride_data)

            # Remove outliers from feature data
            for m, runs in all_runs_to_remove_comb.items():
                for run in runs:
                    # Construct the full multi-index key (stride, mouse, run)
                    key = (stride, m, run)
                    # Optionally, check if the key exists in your DataFrame index
                    if key in feature_data.index:
                        to_drop.add(key)

    feature_data = feature_data.drop(index=list(to_drop))
    print(f"\n--- Outliers removed for phases ---\nLength of feature data after outlier removal: {len(feature_data)}")
    return feature_data



