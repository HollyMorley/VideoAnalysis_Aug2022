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
Similar to Analysis/Characterisation/FeatureReduction_SingleCon.py, but only with allmice conditions set to True.
Pipeline:

'''

# ----------------------------
# Library Imports
# ----------------------------
from Analysis.Tools.PredictPhase import (
    cluster_features_main, global_feature_selection_main, global_pca_main, process_mice_main, get_mouse_pcs_by_run,
    find_pcs_outliers_to_remove)
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

    # collect feature data from each mouse and stride
    feature_data = {}
    feature_data_compare = {}
    for stride in global_settings["stride_numbers"]:
        for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
            # Load and preprocess data for each mouse.
            scaled_data_df = utils.load_and_preprocess_data(mouse_id, stride, condition, exp, day)
            feature_data[(stride, mouse_id)] = scaled_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            scaled_data_comparison_df = utils.load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day)
            feature_data_compare[(stride, mouse_id)] = scaled_data_comparison_df

    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_df_compare = pd.concat(feature_data_compare, axis=0)

    return feature_data_df, feature_data_df_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition


# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------

def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):

    # Initialize experiment (data collection, directories, logging).
    feature_data, feature_data_df_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)
    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")

    if global_settings.get("overwrite_data_collection", True) or not os.path.exists(global_data_path):
        """
        # -------- Find Outliers --------
            - Run PCA on all data 
            - Get PCs for each mouse
            - Find outlier runs across mice based on PCA   
        """
        all_feats_pca = global_pca_main(feature_data, None, phases, stride_numbers, condition, stride_data, select_feats=False)
        pcs_p1 = {}
        pcs_p2 = {}
        for mouse in condition_specific_settings[condition]['global_fs_mouse_ids']:
            for stride in stride_numbers:
                for phase1, phase2 in itertools.combinations(phases, 2):
                    stride_pca = all_feats_pca[(phase1, phase2, stride)]
                    features = list(feature_data.loc(axis=0)[stride, mouse].columns)
                    pcs_phase1, pcs_phase2 = get_mouse_pcs_by_run(mouse, stride, phase1, phase2, stride_data, feature_data, base_save_dir_condition, stride_pca, features)
                    pcs_p1[(mouse, phase1, phase2, stride)] = pcs_phase1
                    pcs_p2[(mouse, phase1, phase2, stride)] = pcs_phase2
        phase1, phase2 = phases
        to_drop = []
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
            all_runs_to_remove_comb = {m: sorted(list(set(all_runs_to_remove[phase1][m] + all_runs_to_remove[phase2][m]))) for m in condition_specific_settings[condition]['global_fs_mouse_ids']}

            utils.save_outliers_csv_updated(all_runs_to_remove_comb, base_save_dir_condition, condition, phase1, phase2, stride, stride_data)

            # Remove outliers from feature data
            for m, runs in all_runs_to_remove_comb.items():
                for run in runs:
                    # Construct the full multi-index key (stride, mouse, run)
                    key = (stride, m, run)
                    # Optionally, check if the key exists in your DataFrame index
                    if key in feature_data.index:
                        to_drop.append(key)
        feature_data = feature_data.drop(index=to_drop)
        print(f"\n--- Outliers removed ---\nLength of feature data after outlier removal: {len(feature_data)}")


        """
            # -------- Feature Clustering --------
            For each stride number and phase pair, find K with cross-validation and then cluster features. Mapping is saved and 
            used to plot feature clustering and a chart describing the content of each cluster.
        """
        cluster_mappings = cluster_features_main(feature_data, phases, stride_numbers, condition, exp, day, stride_data, base_save_dir_condition)

        """
            # -------- Global Feature Selection + Global PCA --------
            Perform feature selection on data from all mice combined    
        """
        global_fs_results, global_pca_results, global_stride_fs_results = global_feature_selection_main(feature_data, phases, stride_numbers, condition, exp, day, stride_data, base_save_dir)

        if global_settings["combine_stride_features"]: # otherwise would have been computed in global_feature_selection_main with stride specific features
            global_pca_results = global_pca_main(feature_data, global_stride_fs_results, phases, stride_numbers, condition, stride_data)

        """
            # -------- Process Each Mouse --------
            For each mouse, process the phase comparison and collect aggregated info.
                - Use the global feature selection results to process each mouse.
                - Run regression and predict runs for each mouse.
        """
        (aggregated_predictions, aggregated_feature_weights, aggregated_raw_features, aggregated_raw_features_all,
         aggregated_cluster_loadings, multi_stride_data, even_ws, odd_ws,
         phase1_pc, phase2_pc) = process_mice_main(mouse_ids, phases, stride_numbers, condition, compare_condition, exp, day,
                                        stride_data, stride_data_compare, base_save_dir_condition,
                                      global_fs_results, global_pca_results, global_stride_fs_results, cluster_mappings)


        # Save the aggregated data to a pickle file.
        print("Saving computed global data...")
        global_data = {
            "cluster_mappings": cluster_mappings,
            "global_pca_results": global_pca_results,
            "global_fs_results": global_fs_results,
            "global_stride_fs_results": global_stride_fs_results,
            "aggregated_predictions": aggregated_predictions,
            "aggregated_feature_weights": aggregated_feature_weights,
            "aggregated_raw_features": aggregated_raw_features,
            "aggregated_raw_features_all": aggregated_raw_features_all,
            "aggregated_cluster_loadings": aggregated_cluster_loadings,
            "multi_stride_data": multi_stride_data,
            "even_ws": even_ws,
            "odd_ws": odd_ws,
            "phase1_pc": phase1_pc,
            "phase2_pc": phase2_pc,
            "stride_data": stride_data,
            "stride_data_compare": stride_data_compare,
        }
        with open(global_data_path, "wb") as f:
            pickle.dump(global_data, f)

    """
        # -------- Aggregated Plots and Further Processing --------
        After processing all mice, create aggregated plots for each phase pair.
    """
    # Load the aggregated data from the pickle file if not already calculated.
    if not global_settings.get("overwrite_data_collection", True) and os.path.exists(global_data_path):
        with open(global_data_path, "rb") as f:
            global_data = pickle.load(f)

    aggregated_save_dir = os.path.join(base_save_dir_condition, "Aggregated")
    os.makedirs(aggregated_save_dir, exist_ok=True)

    # Plot aggregated cluster loadings.
    utils.plot_cluster_loadings_lines(global_data["aggregated_cluster_loadings"], aggregated_save_dir)

    for (phase1, phase2, stride_number), agg_data in global_data["aggregated_predictions"].items():
        top_features = utils.get_top_features(global_data["aggregated_feature_weights"], global_data["global_fs_results"][(phase1,phase2,stride_number)], phase1, phase2, stride_number, n_features=15)
        top_feature_data = utils.get_top_feature_data(global_data["aggregated_raw_features"], phase1, phase2, stride_number, top_features)
        for cm in [True, False]:
            utils.plot_top_feature_phase_comparison(top_feature_data, base_save_dir, phase1, phase2, stride_number, condition_label=condition, connect_mice=cm)
        utils.plot_top_feature_phase_comparison_differences(top_feature_data, base_save_dir, phase1, phase2, stride_number, condition_label=condition)
        # Now plot all back values during stance as phase1 - phase2
        back_data = utils.get_back_data(global_data["aggregated_raw_features_all"], phase1, phase2, stride_number)
        utils.plot_back_phase_comparison(back_data, base_save_dir, phase1, phase2, stride_number, condition_label=condition)

    # Single stride data plots
    for (phase1, phase2, stride_number), agg_data in global_data["aggregated_predictions"].items():
        # create a selected feature order based on cluster mapping
        selected_features = global_data["global_stride_fs_results"][(phase1, phase2)]
        cluster_mapping = global_data["cluster_mappings"][(phase1, phase2, stride_number)]
        selected_features_sorted = sorted(selected_features, key=lambda f: cluster_mapping.get(f))
        feature_cluster_assignments = {feat: cluster_mapping.get(feat) for feat in selected_features_sorted}

        utils.plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_aggregated_feature_weights_byFeature(global_data["aggregated_feature_weights"], selected_features_sorted, feature_cluster_assignments, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        #utils.plot_aggregated_feature_weights_byRun(aggregated_feature_weights, aggregated_raw_features, aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_aggregated_raw_features(global_data["aggregated_raw_features"], aggregated_save_dir, phase1, phase2, stride_number)
        utils.plot_regression_loadings_PC_space_across_mice(global_data["global_pca_results"], global_data["aggregated_feature_weights"], aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.plot_even_odd_PCs_across_mice(global_data["even_ws"], global_data["odd_ws"], aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)
        utils.rank_within_vs_between_differences(global_data["even_ws"], global_data["odd_ws"], aggregated_save_dir, phase1, phase2, stride_number, condition_label=condition)

        for p in [phase1, phase2]:
            pc = global_data["phase1_pc"] if p == phase1 else global_data["phase2_pc"] # todo check this works with more phase combinations
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
    for (phase1, phase2), stride_dict in global_data["multi_stride_data"].items():
        for normalize in [True, False]:
            for smooth in [True, False]:
                utils.plot_multi_stride_predictions(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=smooth, normalize=normalize)
                utils.plot_multi_stride_predictions_difference(stride_dict, phase1, phase2, aggregated_save_dir, condition_label=condition, smooth=smooth, normalize=normalize)
        summary_curves_dict, summary_derivatives_dict, plateau_dict, learning_rate_dict,_ = utils.fit_exponential_to_prediction(stride_dict, aggregated_save_dir, phase1, phase2, condition, exp)


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
                feature_data,
                condition_specific_settings[condition]['global_fs_mouse_ids'],
                stride_number, 'APA1', 'APA2', stride_data, features)
            global_pca_derivative_results[('APA1', 'APA2', stride_number)] = (pca, loadings_df)

            # Retrieve the stored cluster mapping for this (phase1, phase2, stride_number)
            current_cluster_mapping = cluster_mappings.get((phase1, phase2, stride_number), {})

            X_all = pd.concat(allmice_data_selected.values(), axis=1).T # todo check axis
            Xdr_all = np.dot(loadings_df.T, X_all.T)
            Xdr_all, normalize_mean_all, normalize_std_all = utils.normalize_Xdr(Xdr_all)

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
