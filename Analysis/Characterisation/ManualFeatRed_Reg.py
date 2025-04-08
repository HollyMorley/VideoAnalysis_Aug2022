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
import math

'''
Independent feature selection and PCA for each condition.
Then run separate regression models for each condition.
If look similar, this is justification for PCA-BothLhHl+RegLhHlSep.py
'''

# ----------------------------
# Library Imports
# ----------------------------
from Analysis.Tools.PredictPhase import (
    cluster_features_main, global_feature_selection_main, global_pca_main, process_mice_main, find_outliers)
from Analysis.Tools.ClusterFeatures import plot_corr_matrix_sorted_manually, get_global_feature_matrix
from Analysis.Tools import utils_feature_reduction as utils
from Helpers.Config_23 import *
from Analysis.Tools.config import (
    global_settings, condition_specific_settings, instance_settings
)

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round24-9mice_ManualFeatRed_NoSelection_PCAallstr')

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
            filtered_data_df = utils.load_and_preprocess_data(mouse_id, stride, condition, exp, day, measures_list_manual_reduction)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = utils.load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day, measures_list_manual_reduction)
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
        outlier_data_path = os.path.join(base_save_dir, f"outlier_data_{condition}.pkl")
        if not os.path.exists(outlier_data_path):
            print("Finding outliers...")
            feature_data_notscaled = find_outliers(feature_data_notscaled, None, condition, compare_condition, exp, day, stride_data, None, phases, stride_numbers, base_save_dir_condition)
            feature_data_compare_notscaled = find_outliers(feature_data_compare_notscaled, None, compare_condition, condition, exp, day, stride_data_compare, None, phases, stride_numbers, base_save_dir_condition)
            # Save the outlier removed data for later use.
            with open(outlier_data_path, "wb") as f:
                pickle.dump((feature_data_notscaled, feature_data_compare_notscaled), f)
        else:
            print("Loading previously saved outlier data...")
            with open(outlier_data_path, "rb") as f:
                feature_data_notscaled, feature_data_compare_notscaled = pickle.load(f)

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
            (Main condition)
            For each stride number and phase pair, find K with cross-validation and then cluster features. Mapping is saved and 
            used to plot feature clustering and a chart describing the content of each cluster.
        """
        print("Clustering features...")
        cluster_mappings = {}
        for p1, p2 in itertools.combinations(phases,2):
            # features x runs
            feature_matrix = get_global_feature_matrix(feature_data,
                                                       condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                       'all', stride_data, p1, p2, smooth=False)
            plot_corr_matrix_sorted_manually(feature_matrix, base_save_dir_condition,
                                             f'CorrMatrix_manualclustering_{p1}-{p2}_all')
            for s in stride_numbers:
                cluster_mappings[(p1, p2, s)] = manual_clusters['cluster_mapping']


        """
            # -------- Global Feature Selection + Global PCA --------
            (Main condition)
            Perform feature selection on data from all mice combined    
        """
        print("Performing global feature selection...")
        global_fs_results, global_pca_results, global_stride_fs_results = global_feature_selection_main(feature_data, phases, stride_numbers, condition, exp, day, stride_data, base_save_dir)

        # # force global_fs_results and global_stride_fs_results to just be the full feature set
        # global_stride_fs_results = {key: feature_data.columns for key in itertools.combinations(phases, 2)}
        # global_fs_results = {}
        # for p1, p2 in itertools.combinations(phases, 2):
        #     for stride in stride_numbers:
        #         global_fs_results[(p1, p2, stride)] = feature_data.columns

        if global_settings["combine_stride_features"]: # otherwise would have been computed in global_feature_selection_main with stride specific features
            global_pca_results = global_pca_main(feature_data, feature_data_compare, global_stride_fs_results, phases,
                                                 stride_numbers, condition, compare_condition, stride_data, stride_data_compare,
                                                 base_save_dir_condition, combine_conditions=global_settings["pca_CombineAllConditions"],
                                                 combine_strides=global_settings["pca_CombineAllStrides"])

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
        (aggregated_predictions, aggregated_feature_weights, aggregated_contributions, aggregated_raw_features, aggregated_raw_features_all,
         aggregated_cluster_loadings, multi_stride_data, even_ws, odd_ws,
         phase1_pc, phase2_pc, normalize_mean_pc, normalize_std_pc) = results

        results_compare = process_mice_main(mouse_ids, phases, stride_numbers, compare_condition, exp, day,
                                            stride_data_compare, base_save_dir_condition, feature_data_compare,
                                            global_fs_results, global_pca_results, global_stride_fs_results,
                                            cluster_mappings)
        (aggregated_predictions_compare, aggregated_feature_weights_compare, aggregated_contributions_compare,
         aggregated_raw_features_compare, aggregated_raw_features_all_compare,
         aggregated_cluster_loadings_compare, multi_stride_data_compare, even_ws_compare, odd_ws_compare,
         phase1_pc_compare, phase2_pc_compare, normalize_mean_pc_compare, normalize_std_pc_compare) = results_compare

        # Save the aggregated data to a pickle file.
        print("Saving computed global data...")
        global_data = {
            "cluster_mappings": cluster_mappings,
            "global_pca_results": global_pca_results,
            "global_fs_results": global_fs_results,
            "global_stride_fs_results": global_stride_fs_results,
            "aggregated_predictions": aggregated_predictions,
            "aggregated_predictions_compare": aggregated_predictions_compare,
            "aggregated_feature_weights": aggregated_feature_weights,
            "aggregated_feature_weights_compare": aggregated_feature_weights_compare,
            "aggregated_contributions": aggregated_contributions,
            "aggregated_contributions_compare": aggregated_contributions_compare,
            "aggregated_raw_features": aggregated_raw_features,
            "aggregated_raw_features_compare": aggregated_raw_features_compare,
            "aggregated_raw_features_all": aggregated_raw_features_all,
            "aggregated_raw_features_all_compare": aggregated_raw_features_all_compare,
            "aggregated_cluster_loadings": aggregated_cluster_loadings,
            "aggregated_cluster_loadings_compare": aggregated_cluster_loadings_compare,
            "multi_stride_data": multi_stride_data,
            "multi_stride_data_compare": multi_stride_data_compare,
            "even_ws": even_ws,
            "even_ws_compare": even_ws_compare,
            "odd_ws": odd_ws,
            "odd_ws_compare": odd_ws_compare,
            "phase1_pc": phase1_pc,
            "phase1_pc_compare": phase1_pc_compare,
            "phase2_pc": phase2_pc,
            "phase2_pc_compare": phase2_pc_compare,
            "stride_data": stride_data,
            "stride_data_compare": stride_data_compare,
            "normalize": Normalize,
            "normalize_compare": Normalize_compare,
            "normalize_mean_pc": normalize_mean_pc,
            "normalize_std_pc": normalize_std_pc,
            "normalize_mean_pc_compare": normalize_mean_pc_compare,
            "normalize_std_pc_compare": normalize_std_pc_compare,
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

    def unnormalize_top_features(df_p1, df_p2, norm_dict, mouse_ids, stride):
        df1, df2 = df_p1.copy(), df_p2.copy()
        idx = pd.IndexSlice

        for mouse in mouse_ids:
            norm = norm_dict[(stride, mouse)]
            top_norm = norm.loc(axis=1)[df1.columns]

            for feat in top_norm.columns:
                sel = idx[mouse, :]
                mean = top_norm.loc(axis=1)[feat]['mean']
                std = top_norm.loc(axis=1)[feat]['std']
                df1.loc[sel, feat] = df1.loc[sel, feat] * std + mean
                df2.loc[sel, feat] = df2.loc[sel, feat] * std + mean

        return df1, df2

    def build_condition_dict_and_plot(preds, contributions, raw_feats, norm_dict, mouse_ids, label,top_feats_type, top_feats_preset=None):
        out = {}
        Top_Feats = {}
        All_Feats = {}
        for (phase1, phase2, stride), _ in preds.items():
            if top_feats_preset is None:
                top_feats_top10, top_feats_q, all_features = utils.get_top_features(contributions, global_data["global_fs_results"][(phase1, phase2, stride)],
                                                   phase1, phase2, stride, n_features=10, quartile=0.9)
                top_feats = top_feats_top10 if top_feats_type == 'top10' else top_feats_q
            else:
                top_feats = top_feats_preset[(phase1, phase2, stride)]
            p1, p2 = utils.get_top_feature_data(raw_feats, phase1, phase2, stride, top_feats)
            real_p1, real_p2 = unnormalize_top_features(p1, p2, norm_dict, mouse_ids, stride)
            out[(phase1, phase2, stride)] = [real_p1, real_p2]

            if top_feats_preset is None:
                # for cm in (True, False):
                #     utils.plot_top_feature_phase_comparison([real_p1, real_p2],
                #                                             base_save_dir, phase1, phase2, stride,
                #                                             condition_label=label, connect_mice=cm)
                utils.plot_top_feature_phase_comparison_connected_means([real_p1, real_p2],
                                                            base_save_dir, phase1, phase2, stride,
                                                            condition_label=label)
                utils.plot_top_feature_phase_comparison_differences([real_p1, real_p2],
                                                                    base_save_dir, phase1, phase2, stride,
                                                                    condition_label=label)
            Top_Feats[(phase1, phase2, stride)] = top_feats
            All_Feats[(phase1, phase2, stride)] = all_features
        return out, Top_Feats, All_Feats

    real_dict, top_feats, all_feats = build_condition_dict_and_plot(global_data["aggregated_predictions"],
                                     global_data["aggregated_contributions"],
                                     global_data["aggregated_raw_features"],
                                     global_data["normalize"],
                                     condition_specific_settings[condition]['global_fs_mouse_ids'],
                                     condition,
                                     top_feats_type='top10')

    # all_ordered_feats_names = all_feats['APA2', 'Wash2', -1].abs().sort_values(ascending=False).index
    # ordered_feats = {}
    # for key, val in all_feats.items():
    #     ordered_feats[key] = val.reindex(all_ordered_feats_names)
    # plt.figure()
    # plt.plot(ordered_feats['APA2', 'Wash2', -1], c='darkblue')
    # plt.plot(ordered_feats['APA2', 'Wash2', -2], c='blue')
    # plt.plot(ordered_feats['APA2', 'Wash2', -3], c='lightblue')
    # plt.xticks(ordered_feats['APA2', 'Wash2', -1].index, rotation=90, fontsize=8)
    # plt.tight_layout()


    utils.plot_common_across_strides_top_features(top_feats, real_dict, condition, base_save_dir)

    # compare_dict, _ = build_condition_dict_and_plot(global_data["aggregated_predictions_compare"],
    #                                     global_data["aggregated_feature_weights_compare"],
    #                                     global_data["aggregated_raw_features_compare"],
    #                                     global_data["aggregated_raw_features_all_compare"],
    #                                     global_data["normalize_compare"],
    #                                     condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
    #                                     compare_condition)
    #
    # compare_BaseCon_feats_dict, _ = build_condition_dict_and_plot(global_data["aggregated_predictions_compare"],
    #                                     global_data["aggregated_feature_weights_compare"],
    #                                     global_data["aggregated_raw_features_compare"],
    #                                     global_data["aggregated_raw_features_all_compare"],
    #                                     global_data["normalize_compare"],
    #                                     condition_specific_settings[compare_condition]['global_fs_mouse_ids'],
    #                                     compare_condition,
    #                                     top_feats_preset=top_feats)


    # for key, real_data in real_dict.items():
    #     comp_data = compare_dict[key]
    #     comp_BaseCon_Feat_data = compare_BaseCon_feats_dict[key]
    #     utils.plot_top_feature_phase_comparison_differences_BothConditions(real_data,
    #                                                                        comp_data,
    #                                                                        base_save_dir,
    #                                                                        *key,
    #                                                                        condition_label=condition,
    #                                                                        compare_condition_label=compare_condition,
    #                                                                        suffix='Sep Top Features')
    #     utils.plot_top_feature_phase_comparison_differences_BothConditions(real_data,
    #                                                                        comp_BaseCon_Feat_data,
    #                                                                        base_save_dir,
    #                                                                        *key,
    #                                                                        condition_label=condition,
    #                                                                        compare_condition_label=compare_condition,
    #                                                                        suffix='LowHigh Top Features')



    # Single stride data plots
    for (phase1, phase2, stride_number), agg_data in global_data["aggregated_predictions"].items():
        # create a selected feature order based on cluster mapping
        fs_results = global_data["global_stride_fs_results"][(phase1, phase2)] if global_settings["combine_stride_features"] else global_data["global_fs_results"][(phase1, phase2, stride_number)]
        selected_features = fs_results
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
        utils.plot_top_feature_pc_single_contributors(global_data["aggregated_contributions"],phase1, phase2, stride_number, condition, aggregated_save_dir)


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

    print("Done!")



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
