import matplotlib.pyplot as plt
import itertools
import datetime
import inspect
import os
import seaborn as sns
import random

from Analysis.ReduceFeatures.PCA import perform_pca, plot_pca, plot_scree, cross_validate_pca, plot_average_variance_explained_across_folds, compute_global_pca_for_phase
from Analysis.ReduceFeatures.LogisticRegression import compute_regression, compute_lasso_regression, run_regression, predict_runs, predict_compare_condition, compute_global_regression_model
from Analysis.ReduceFeatures.FeatureSelection import rfe_feature_selection, random_forest_feature_selection
from Analysis.ReduceFeatures.ClusterFeatures import cross_validate_k_clusters_folds_pca, cluster_features_run_space, get_global_feature_matrix, plot_feature_clustering, plot_feature_clusters_chart
from Analysis.ReduceFeatures.FindMousePools import compute_mouse_similarity, get_loading_matrix_from_nested, pool_mice_by_similarity
from Analysis.ReduceFeatures import utils_feature_reduction as utils
from Helpers.Config_23 import *
from Analysis.ReduceFeatures.config import base_save_dir_no_c, global_settings, condition_specific_settings, instance_settings

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

# ----------------------------
# Function Definitions
# ----------------------------
def log_settings(settings, log_dir):
    """
    Save the provided settings (a dict) to a timestamped log file.
    Also include the name of the running script and the current date.
    """
    os.makedirs(log_dir, exist_ok=True)
    # Get the name of the current script file
    script_name = os.path.basename(inspect.getfile(inspect.currentframe()))
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


# -----------------------------------------------------
# Main Processing Function for Each Phase Comparison
# -----------------------------------------------------
def process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, stride_data_compare, condition, exp, day,
                             base_save_dir_condition, selected_features=None, fs_df=None,
                             global_pca=None, compare_condition=None, cluster_mapping=None):
    """
    Process a single phase comparison for a given mouse. If selected_features is provided,
    that global feature set is used; otherwise local feature selection is performed.
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
        for fidx, feature in enumerate(selected_features):
            plt.figure()
            sns.histplot(selected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
            plt.axvline(0, color='red', label='True Accuracy')
            plt.title(feature)
            plt.xlabel('Shuffled Accuracy - True Accuracy')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(save_path, f'feature_significances\\feature{fidx}_feature_selection.png'))
            plt.close()

        # Plot non-significant features.
        nonselected_features = fs_df[~fs_df['significant']].index
        nonselected_features_accuracies = fs_df.loc[nonselected_features]
        if not os.path.exists(os.path.join(save_path, 'feature_nonsignificances')):
            os.makedirs(os.path.join(save_path, 'feature_nonsignificances'))
        for fidx, feature in enumerate(nonselected_features):
            plt.figure()
            sns.histplot(nonselected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
            plt.axvline(0, color='red', label='True Accuracy')
            plt.title(feature)
            plt.xlabel('Shuffled Accuracy - True Accuracy')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(save_path, f'feature_nonsignificances\\feature{fidx}_feature_selection.png'))
            plt.close()
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
        plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, save_path)
    else:
        # Local PCA (existing behavior).
        fold_variances = cross_validate_pca(reduced_feature_data_df, save_path, n_folds=10)
        plot_average_variance_explained_across_folds(fold_variances, reduced_feature_data_df)
        #n_components = 10 if allmice else 4
        n_components = 10 if len(selected_features) > 10 else len(selected_features)
        pca, pcs, loadings_df = perform_pca(reduced_feature_data_df, n_components=n_components)
        pcs_combined, labels, pcs_phase1, pcs_phase2 = utils.get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2)
        plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, save_path)


    # ---------------- Prediction - Regression-Based Feature Contributions ----------------
    smoothed_scaled_pred, feature_weights, w = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, save_path, condition)

    # ---------------- Map selected features to respective clusters ----------------
    if global_settings["allmice"] == False: # todo maybe extend this to allmice too
        #cluster_mapping = joblib.load('feature_clusters.pkl')

        # For the selected features (from local selection), map them to clusters.
        feature_cluster_assignments = {feat: cluster_mapping.get(feat, -1) for feat in selected_features}

        # Sum the regression loadings by cluster.
        cluster_loadings = {}
        for feat, weight in feature_weights.items():
            cluster = feature_cluster_assignments.get(feat)
            if cluster is not None and cluster != -1:
                cluster_loadings[cluster] = cluster_loadings.get(cluster, 0) + weight

        # Optionally, store this info for later aggregated plotting:
        # (e.g., add it to the tuple returned by process_phase_comparison)
        print(f"Mouse {mouse_id} - Cluster loadings: {cluster_loadings}")
    else:
        cluster_loadings = None

    return (mouse_id, list(reduced_feature_data_df.index),
                smoothed_scaled_pred), feature_weights, reduced_feature_data_df, cluster_loadings




# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------
def main(mouse_ids, stride_numbers, phases, condition='LowHigh', exp='Extended', day=None, compare_condition='None'):
    # construct path
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
    stride_data = utils.load_stride_data(stride_data_path)
    stride_data_compare = utils.load_stride_data(stride_data_path_compare)

    cval = condition_specific_settings[condition]['c']
    base_save_dir = base_save_dir_no_c + f'-c={cval}'
    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')

    for stride_number in stride_numbers:
        for phase1, phase2 in itertools.combinations(phases, 2):
            # get feature clusters
            # find k
            optimal_k, avg_sil_scores = cross_validate_k_clusters_folds_pca(
                condition_specific_settings[condition]['global_fs_mouse_ids'], stride_number, condition, exp, day,
                stride_data, phase1, phase2)
            print(f"Optimal k based on silhouette score is: {optimal_k}")

            save_file = os.path.join(base_save_dir_condition,
                                     f'feature_clusters_{phase1}_vs_{phase2}_stride{stride_number}.pkl')
            cluster_mapping = cluster_features_run_space(condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                         stride_number, condition, exp, day, stride_data, phase1,
                                                         phase2, n_clusters=optimal_k, save_file=save_file)
            feature_matrix = get_global_feature_matrix(condition_specific_settings[condition]['global_fs_mouse_ids'], stride_number, condition, exp, day,
                                                       stride_data, phase1, phase2)
            plot_feature_clustering(feature_matrix, cluster_mapping, save_file=f"feature_clustering_{phase1}_vs_{phase2}_stride{stride_number}.png")
            plot_feature_clusters_chart(cluster_mapping, save_file=f"feature_clusters_chart_{phase1}_vs_{phase2}_stride{stride_number}.png")

    if global_settings["allmice"]:
        # Directory for global feature selection.
        global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
        os.makedirs(global_fs_dir, exist_ok=True)

        global_fs_results = {}
        global_pca_results = {}
        for phase1, phase2 in itertools.combinations(phases, 2):
            for stride_number in stride_numbers:
                print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
                selected_features, fs_df = utils.global_feature_selection(
                    condition_specific_settings[condition]['global_fs_mouse_ids'],
                    stride_number,
                    phase1, phase2,
                    condition, exp, day,
                    stride_data,
                    save_dir=global_fs_dir,
                    c=condition_specific_settings[condition]['c'],
                    nFolds=global_settings["nFolds_selection"],
                    n_iterations=global_settings["n_iterations_selection"],
                    overwrite=global_settings["overwrite_FeatureSelection"],
                    method=global_settings["method"]
                )
                global_fs_results[(phase1, phase2, stride_number)] = (selected_features, fs_df)

                # Compute global PCA using the selected features.
                pca, loadings_df = compute_global_pca_for_phase(condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                                stride_number,
                                                                phase1, phase2,
                                                                condition, exp, day,
                                                                stride_data,
                                                                selected_features)
                global_pca_results[(phase1, phase2, stride_number)] = (pca, loadings_df)

    # Initialize an aggregation dictionary keyed by (phase1, phase2)
    aggregated_predictions = {}
    aggregated_feature_weights = {}
    aggregated_raw_features = {}
    aggregated_cluster_loadings = {}

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                try:
                    if global_settings["allmice"]:
                        tup = global_fs_results.get((phase1, phase2, stride_number), None)
                        if tup is not None:
                            selected_features, fs_df = tup
                            global_pca = global_pca_results.get((phase1, phase2, stride_number), None)
                        else:
                            selected_features, fs_df, global_pca = None, None, None
                    else:
                        selected_features, fs_df, global_pca = None, None, None

                    # Process phase comparison and collect aggregated info
                    agg_info, feature_weights, raw_features, cluster_loadings = process_phase_comparison(
                        mouse_id, stride_number, phase1, phase2,
                        stride_data, stride_data_compare, condition, exp, day,
                        base_save_dir_condition, allmice=global_settings["allmice"],
                        selected_features=selected_features, fs_df=fs_df,
                        method=global_settings["method"],
                        global_pca=global_pca,
                        compare_condition=compare_condition,
                        select_features=global_settings["select_features"],
                        cluster_mapping=cluster_mapping
                    )
                    # agg_info is a tuple: (mouse_id, x_vals, smoothed_scaled_pred)
                    aggregated_predictions.setdefault((phase1, phase2, stride_number), []).append(agg_info)
                    aggregated_feature_weights[(mouse_id, phase1, phase2, stride_number)] = feature_weights
                    aggregated_raw_features[(mouse_id, phase1, phase2, stride_number)] = raw_features
                    aggregated_cluster_loadings.setdefault((phase1, phase2, stride_number), {})[mouse_id] = cluster_loadings
                except ValueError as e:
                    print(f"Error processing {mouse_id}, stride {stride_number}, {phase1} vs {phase2}: {e}")

    # if allmice == False:
    #     sim_df = compute_cosine_similarity_local(aggregated_feature_weights, (phases[0], phases[1]), mouse_ids)
    #     plot_cosine_similarity(sim_df)

    # After processing all mice, create aggregated plots for each phase pair.
    # Optionally, use a subfolder (e.g., "Aggregated") within your base_save_dir_condition.
    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    os.makedirs(aggregated_save_dir, exist_ok=True)

    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    utils.plot_cluster_loadings(aggregated_cluster_loadings, aggregated_save_dir)

    mouse_groups = {}
    for (phase1, phase2, stride_number), agg_data in aggregated_predictions.items():
        utils.plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_aggregated_feature_weights(aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_aggregated_raw_features(aggregated_raw_features, aggregated_save_dir, phase1, phase2, stride_number)

        # -------------------------- Process global clustering --------------------------
        loading_df = get_loading_matrix_from_nested(aggregated_cluster_loadings, (phase1, phase2, stride_number))
        loading_dict = loading_df.to_dict(orient="index")
        sim_df, loading_matrix = compute_mouse_similarity(loading_dict)
        groups = pool_mice_by_similarity(sim_df, threshold=global_settings["mouse_pool_thresh"])
        mouse_groups[(phase1, phase2, stride_number)] = groups


    unique_phase_pairs = set((p1, p2, s) for (_, p1, p2, s) in aggregated_feature_weights.keys())

    for phase_pair in unique_phase_pairs:
        cluster_df, kmeans_model = utils.cluster_regression_weights_across_mice(
            aggregated_feature_weights, phase_pair, aggregated_save_dir, n_clusters=3
        )
        if cluster_df is not None:
            print(f"Global clustering for phase pair {phase_pair}:")
            print(cluster_df)

    # -------------------------- Process compare condition predictions in a separate loop --------------------------
    global_regression_params = {}
    for phase1, phase2 in itertools.combinations(phases, 2):
        for stride_number in stride_numbers:
            selected_features, fs_df = global_fs_results[(phase1, phase2, stride_number)]
            pca, loadings_df = global_pca_results[(phase1, phase2, stride_number)]

            regression_params = compute_global_regression_model(
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                stride_number,
                phase1, phase2,
                condition, exp, day,
                stride_data,
                selected_features, loadings_df
            )
            global_regression_params[(phase1, phase2, stride_number)] = regression_params

    aggregated_compare_predictions = {}
    if compare_condition != 'None':
        compare_mouse_ids = condition_specific_settings[compare_condition]['global_fs_mouse_ids']
        for phase1, phase2 in itertools.combinations(phases, 2):
            for stride_number in stride_numbers:
                # Retrieve regression parameters computed from the base condition.
                regression_params = global_regression_params.get((phase1, phase2, stride_number), None)
                if regression_params is None:
                    print(f"No regression model for phase pair {phase1} vs {phase2}, stride {stride_number}.")
                    continue
                w = regression_params['w']
                loadings_df = regression_params['loadings_df']
                selected_features = regression_params['selected_features']

                for mouse_id in compare_mouse_ids:
                    try:
                        compare_save_path = os.path.join(base_save_dir_condition, f"{compare_condition}_predictions",
                                                         mouse_id)
                        os.makedirs(compare_save_path, exist_ok=True)

                        smoothed_scaled_pred, runs = predict_compare_condition(
                            mouse_id, compare_condition, stride_number, exp, day,
                            stride_data_compare, phase1, phase2,
                            selected_features, loadings_df, w, compare_save_path
                        )

                        #x_vals = np.arange(len(smoothed_scaled_pred))
                        aggregated_compare_predictions.setdefault((phase1, phase2, stride_number), []).append(
                            (mouse_id, runs, smoothed_scaled_pred)
                        )

                    except Exception as e:
                        print(
                            f"Error processing compare condition for mouse {mouse_id}, phase pair {phase1} vs {phase2}: {e}")

        # Plot aggregated compare predictions.
        for (phase1, phase2, stride_number), agg_data in aggregated_compare_predictions.items():
            utils.plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=f"vs_{compare_condition}")


    # # After processing all mice, aggregate the contributions.
    # # You can choose a save directory for these summary plots:
    # summary_save_path = os.path.join(base_save_dir_condition, 'Summary_Cosine_Similarity')
    # os.makedirs(summary_save_path, exist_ok=True)
    #
    # pivot_contributions = utils.aggregate_feature_contributions(all_contributions)
    #
    # # Compute and plot the pairwise cosine similarity and the boxplots.
    # similarity_matrix = utils.plot_pairwise_cosine_similarity(pivot_contributions, summary_save_path)


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
    # Log the settings.
    log_settings(settings_to_log, base_save_dir_no_c)

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
        )
