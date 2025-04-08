import os
import random
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import inspect
import itertools
import matplotlib.pyplot as plt
import math
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# ----------------------------
# Library Imports
# ----------------------------
from Helpers.Config_23 import *
from Analysis.Tools import utils_feature_reduction as utils
from Analysis.Tools.config import (
    global_settings, condition_specific_settings, instance_settings
)
from Analysis.Tools.SignificanceTesting import ShufflingTest_ComparePhases, ShufflingTest_CompareConditions

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round23-9mice_AllFeatDescriptivesAndStats')

# ---------------------------------
# Parallel processing helpers
# ---------------------------------

def compute_feature_significance(feat, p1_data, p2_data, p1_means, p2_means, p1_vars, p2_vars, phase1, phase2, measure):
    if measure == "mean":
        p_value, _, _ = ShufflingTest_ComparePhases(
            p1_data.loc(axis=1)[feat],
            p2_data.loc(axis=1)[feat],
            p1_means.loc(axis=1)[feat],
            p2_means.loc(axis=1)[feat],
            phase1, phase2,
            type="mean"
        )
    elif measure == "variance":
        p_value, _, _ = ShufflingTest_ComparePhases(
            p1_data.loc(axis=1)[feat],
            p2_data.loc(axis=1)[feat],
            p1_vars.loc(axis=1)[feat],
            p2_vars.loc(axis=1)[feat],
            phase1, phase2,
            type="var"
        )
    return feat, p_value

def compute_feature_significance_between_conditions(feat, p1_data_main, p2_data_main,
                                                    p1_data_compare, p2_data_compare,
                                                    mice_diff_main, mice_diff_compare,
                                                    phase1, phase2, measure):
    if measure == "mean":
        p_value, _, _ = ShufflingTest_CompareConditions(
            Obs_p1=p1_data_main[feat],
            Obs_p2=p2_data_main[feat],
            Obs_p1c=p1_data_compare[feat],
            Obs_p2c=p2_data_compare[feat],
            pdiff_Obs=mice_diff_main[feat],
            pdiff_c_Obs=mice_diff_compare[feat],
            phase1=phase1, phase2=phase2, type='mean', num_iter=1000
        )
    elif measure == "variance":
        p_value, _, _ = ShufflingTest_CompareConditions(
            Obs_p1=p1_data_main[feat],
            Obs_p2=p2_data_main[feat],
            Obs_p1c=p1_data_compare[feat],
            Obs_p2c=p2_data_compare[feat],
            pdiff_Obs=mice_diff_main[feat],
            pdiff_c_Obs=mice_diff_compare[feat],
            phase1=phase1, phase2=phase2, type='var', num_iter=1000
        )
    return feat, p_value


# -----------------------------------------------------
# Initialization Helper (unchanged)
# -----------------------------------------------------
def initialize_experiment(condition, exp, day, compare_condition, settings_to_log,
                          base_save_dir_no_c, condition_specific_settings) -> Tuple:
    stride_data, stride_data_compare = utils.collect_stride_data(condition, exp, day, compare_condition)
    base_save_dir, base_save_dir_condition = utils.set_up_save_dir(
        condition, exp, condition_specific_settings[condition]['c'], base_save_dir_no_c
    )
    script_name = os.path.basename(inspect.getfile(inspect.currentframe()))
    utils.log_settings(settings_to_log, base_save_dir, script_name)
    feature_data = {}
    feature_data_compare = {}
    for stride in global_settings["stride_numbers"]:
        for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
            filtered_data_df = utils.load_and_preprocess_data(mouse_id, stride, condition, exp, day, measures_list_feature_reduction)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = utils.load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day, measures_list_feature_reduction)
            feature_data_compare[(stride, mouse_id)] = filtered_data_comparison_df
    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_df_compare = pd.concat(feature_data_compare, axis=0)
    return feature_data_df, feature_data_df_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition

# -----------------------------------------------------
# Data & Significance Function (now with measure parameter)
# -----------------------------------------------------
def get_data_and_significance(feature_data, p1_runs, p2_runs, phase1, phase2, stride, mice, measure="mean"):
    p1_data_notscaled = feature_data.loc[(stride, mice, p1_runs)]
    p2_data_notscaled = feature_data.loc[(stride, mice, p2_runs)]
    combined_data = pd.concat([p1_data_notscaled, p2_data_notscaled])
    global_means = combined_data.mean()
    global_stds = combined_data.std()

    # Apply z-score scaling
    p1_data_scaled = (p1_data_notscaled - global_means) / global_stds
    p2_data_scaled = (p2_data_notscaled - global_means) / global_stds

    # Drop stride level
    p1_data = p1_data_scaled.droplevel('Stride')
    p2_data = p2_data_scaled.droplevel('Stride')

    p1_mice_feat_means = p1_data.groupby('MouseID').mean()
    p2_mice_feat_means = p2_data.groupby('MouseID').mean()
    p1_mice_feat_vars = p1_data.groupby('MouseID').var()
    p2_mice_feat_vars = p2_data.groupby('MouseID').var()

    mice_mean_diff = p2_mice_feat_means - p1_mice_feat_means
    mice_var_diff = p2_mice_feat_vars - p1_mice_feat_vars

    significance_dict = {}
    # for feat in p1_data.columns:
    #     if measure == "mean":
    #         p_value, _, _ = ShufflingTest_ComparePhases(
    #             p1_data.loc(axis=1)[feat], p2_data.loc(axis=1)[feat],
    #             p1_mice_feat_means.loc(axis=1)[feat], p2_mice_feat_means.loc(axis=1)[feat],
    #             phase1, phase2)
    #     elif measure == "variance":
    #         p_value, _, _ = ShufflingTest_ComparePhases_variance(
    #             p1_data.loc(axis=1)[feat], p2_data.loc(axis=1)[feat],
    #             p1_mice_feat_vars.loc(axis=1)[feat], p2_mice_feat_vars.loc(axis=1)[feat],
    #             phase1, phase2)
    #     significance_dict[feat] = p_value

    results = Parallel(n_jobs=-1)(
        delayed(compute_feature_significance)(
            feat, p1_data, p2_data, p1_mice_feat_means, p2_mice_feat_means,
            p1_mice_feat_vars, p2_mice_feat_vars, phase1, phase2, measure
        )
        for feat in p1_data.columns
    )
    # Collect the results in a dictionary:
    significance_dict = dict(results)

    # Apply correction
    # Convert p-values to an array (order matters, so you'll want to track the corresponding features)
    features = list(significance_dict.keys())
    raw_pvals = np.array([significance_dict[feat] for feat in features])

    # Adjust the p-values using the FDR method (Benjamini-Hochberg)
    alpha = 0.05  # desired family-wise error rate or FDR level
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')
    # Create a new dictionary with the adjusted p-values
    adjusted_significance_dict = dict(zip(features, pvals_corrected))

    all_features = list(mice_mean_diff.columns)
    sig_array = np.array(list(adjusted_significance_dict.values()))
    significant_feats = np.array(all_features)[sig_array < 0.05]
    non_significant_feats = np.array(all_features)[sig_array >= 0.05]

    # Return the appropriate diff and the scaled data for later use
    diff = mice_mean_diff if measure=="mean" else mice_var_diff
    return diff, adjusted_significance_dict, all_features, significant_feats, non_significant_feats, p1_data, p2_data

# -----------------------------------------------------
# Plotting Functions (now with measure parameter)
# -----------------------------------------------------
def plot_all_features(mice_diff, sig_dict, all_features, significant_feats,
                      phase1, phase2, condition, base_save_dir, stride, measure="mean"):
    chunk_size = 66
    n_chunks = math.ceil(len(all_features) / chunk_size)
    fig, axs = plt.subplots(n_chunks, 1, figsize=(50, 20 * n_chunks))
    if n_chunks == 1:
        axs = [axs]
    name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_feats = all_features[start:end]
        short_feats = []
        for feat in chunk_feats:
            name_bits = feat.split(', ')
            name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
            short_feats.append(', '.join(name_to_keep))
        chunk_vals = mice_diff[chunk_feats].mean(axis=0)
        chunk_errs = mice_diff[chunk_feats].sem(axis=0) * 1.645
        ax = axs[i]
        colors = ['green' if feat in significant_feats else 'C0' for feat in chunk_feats]
        ax.bar(short_feats, chunk_vals, yerr=chunk_errs, capsize=1, color=colors)
        for j, feat in enumerate(chunk_feats):
            if chunk_vals.iloc[j] > 0:
                ypos = chunk_vals.iloc[j] + chunk_errs.iloc[j] + 0.1
            else:
                ypos = chunk_vals.iloc[j] - chunk_errs.iloc[j] - 0.1
            if sig_dict[feat] < 0.001:
                ax.text(j, ypos, '***', ha='center', va='bottom', rotation=90)
            elif sig_dict[feat] < 0.01:
                ax.text(j, ypos, '**', ha='center', va='bottom', rotation=90)
            elif sig_dict[feat] < 0.05:
                ax.text(j, ypos, '*', ha='center', va='bottom', rotation=90)
        ax.tick_params(axis='x', rotation=90)
        ticklabels = ax.get_xticklabels()
        for j, label in enumerate(ticklabels):
            if chunk_feats[j] in significant_feats:
                label.set_color('green')
        ax.set_ylabel('Difference in ' + ('Mean' if measure=="mean" else 'Variance'))
        ax.set_xlabel('Feature')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"{'Mean' if measure=='mean' else 'Variance'} Difference ({phase2} - {phase1}) for {condition} (Features {start+1}-{min(end,len(all_features))})", pad=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.01)
    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
    os.makedirs(save_dir, exist_ok=True)
    fname = f"AllFeats_{'Mean' if measure=='mean' else 'Var'}Diff_{phase2}-{phase1}_{condition}_stride{stride}.pdf"
    fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_significant_features(mice_diff, sig_dict, significant_feats,
                              phase1, phase2, condition, base_save_dir, stride, measure="mean"):
    name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']
    sig_features_list = list(significant_feats)
    sig_short_feats = []
    for feat in sig_features_list:
        name_bits = feat.split(', ')
        name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
        sig_short_feats.append(', '.join(name_to_keep))
    sig_vals = mice_diff[sig_features_list].mean(axis=0)
    sig_errs = mice_diff[sig_features_list].sem(axis=0) * 1.645
    fig2, ax2 = plt.subplots(1, 1, figsize=(50, 20))
    ax2.bar(sig_short_feats, sig_vals, yerr=sig_errs, capsize=1)
    for i, feat in enumerate(sig_features_list):
        if sig_vals.iloc[i] > 0:
            ypos = sig_vals.iloc[i] + sig_errs.iloc[i] + 0.1
        else:
            ypos = sig_vals.iloc[i] - sig_errs.iloc[i] - 0.1
        if sig_dict[feat] < 0.001:
            ax2.text(i, ypos, '***', ha='center', va='bottom', rotation=90)
        elif sig_dict[feat] < 0.01:
            ax2.text(i, ypos, '**', ha='center', va='bottom', rotation=90)
        elif sig_dict[feat] < 0.05:
            ax2.text(i, ypos, '*', ha='center', va='bottom', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    ax2.set_ylabel(('Mean' if measure=="mean" else 'Variance') + " Difference")
    ax2.set_xlabel('Feature')
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(f"{'Mean' if measure=='mean' else 'Variance'} Difference ({phase2} - {phase1}) for {condition} (Significant Features Only)", pad=20)
    plt.tight_layout()
    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
    os.makedirs(save_dir, exist_ok=True)
    fname2 = f"AllSigFeats_{'Mean' if measure=='mean' else 'Var'}Diff_{phase2}-{phase1}_{condition}_stride{stride}.pdf"
    fig2.savefig(os.path.join(save_dir, fname2), dpi=300, bbox_inches='tight')
    plt.close()


def plot_between_conditions(mice_diff_main, mice_diff_compare,
                            features_to_plot, phase1, phase2, condition, compare_condition,
                            base_save_dir, stride, measure="mean", between_sig_dict=None):
    name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']
    short_feats = []
    for feat in features_to_plot:
        name_bits = feat.split(', ')
        name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
        short_feats.append(', '.join(name_to_keep))
    main_vals = mice_diff_main[features_to_plot].mean(axis=0)
    main_errs = mice_diff_main[features_to_plot].sem(axis=0) * 1.645
    compare_vals = mice_diff_compare[features_to_plot].mean(axis=0)
    compare_errs = mice_diff_compare[features_to_plot].sem(axis=0) * 1.645
    x = np.arange(len(features_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(features_to_plot) * 0.5), 10))

    rects1 = ax.bar(x - width / 2, main_vals, width, yerr=main_errs, capsize=5, label=condition)
    rects2 = ax.bar(x + width / 2, compare_vals, width, yerr=compare_errs, capsize=5, label=compare_condition)

    ax.set_ylabel(('Mean' if measure == "mean" else 'Variance') + " Difference")
    ax.set_xlabel('Feature')
    ax.set_title(f"Between-Condition {'Mean' if measure == 'mean' else 'Variance'} Differences ({phase2} - {phase1})")
    ax.set_xticks(x)
    ax.set_xticklabels(short_feats, rotation=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend()

    # Add significance stars if between_sig_dict is provided.
    if between_sig_dict is not None:
        for i, feat in enumerate(features_to_plot):
            p_val = between_sig_dict.get(feat, 1.0)
            stars = ""
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            if stars:
                # Place the star above the higher bar.
                bar_top = max(main_vals.iloc[i] + main_errs.iloc[i],
                              compare_vals.iloc[i] + compare_errs.iloc[i])
                ax.text(x[i], bar_top + 0.05, stars, ha='center', va='bottom', fontsize=16, color='red')

    plt.tight_layout()
    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
    os.makedirs(save_dir, exist_ok=True)
    fname = f"BetweenConditions_Features_{'Mean' if measure == 'mean' else 'Var'}Diff_{phase2}-{phase1}_{condition}_stride{stride}.pdf"
    fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_features_between_conditions(mice_diff_main, mice_diff_compare,
                                         all_features, phase1, phase2,
                                         condition, compare_condition,
                                         base_save_dir, stride, measure="mean",
                                         between_sig_dict=None):
    """
    Plots all features for between-condition differences.
    The bars are colored by condition (as in your original plots),
    but significant features (if provided via between_sig_dict)
    have their x-axis labels colored green and are annotated with stars.
    """
    # Prepare feature labels (remove unwanted bits)
    name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']
    short_feats = []
    for feat in all_features:
        name_bits = feat.split(', ')
        name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
        short_feats.append(', '.join(name_to_keep))

    # Compute means and errors for each condition for all features
    main_vals = mice_diff_main[all_features].mean(axis=0)
    main_errs = mice_diff_main[all_features].sem(axis=0) * 1.645
    compare_vals = mice_diff_compare[all_features].mean(axis=0)
    compare_errs = mice_diff_compare[all_features].sem(axis=0) * 1.645

    x = np.arange(len(all_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(all_features) * 0.5), 10))

    # Draw bars for each condition (using default colors)
    rects1 = ax.bar(x - width / 2, main_vals, width, yerr=main_errs, capsize=5, label=condition)
    rects2 = ax.bar(x + width / 2, compare_vals, width, yerr=compare_errs, capsize=5, label=compare_condition)

    ax.set_ylabel(('Mean' if measure == "mean" else "Variance") + " Difference")
    ax.set_xlabel('Feature')
    ax.set_title(f"Between-Condition {'Mean' if measure == 'mean' else 'Variance'} Differences ({phase2} - {phase1})")
    ax.set_xticks(x)
    ax.set_xticklabels(short_feats, rotation=90)

    # If a between_sig_dict is provided, check significance for each feature
    # and annotate the plot and color labels green.
    if between_sig_dict is not None:
        for i, feat in enumerate(all_features):
            p_val = between_sig_dict.get(feat, 1.0)
            stars = ""
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            if stars:
                # Place the star above the higher bar
                bar_top = max(main_vals.iloc[i] + main_errs.iloc[i],
                              compare_vals.iloc[i] + compare_errs.iloc[i])
                ax.text(x[i], bar_top + 0.05, stars, ha='center', va='bottom', fontsize=16, color='red')
            # Color x-tick label green if feature is significant.
            if p_val < 0.05:
                ax.get_xticklabels()[i].set_color('green')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend()

    plt.tight_layout()
    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
    os.makedirs(save_dir, exist_ok=True)
    fname = f"AllBetweenConditions_Features_{'Mean' if measure == 'mean' else 'Var'}Diff_{phase2}-{phase1}_{condition}_stride{stride}.pdf"
    fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------
# Main Function (now with a measure parameter)
# -----------------------------------------------------
def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None, measure: str = "mean"):
    feature_data_notscaled, feature_data_df_compare_notscaled, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(
        condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)
    feature_data_notscaled.index.names = ['Stride', 'MouseID', 'Run']
    feature_data_df_compare_notscaled.index.names = ['Stride', 'MouseID', 'Run']
    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")
    for stride in stride_numbers:
        print(f"Stride {stride}...")
        for phase1, phase2 in itertools.combinations(phases, 2):
            print(f"Comparing {phase1} and {phase2}...")
            p1_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
            p2_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
            common_mice = np.array(list(set(condition_specific_settings[condition]['global_fs_mouse_ids']).intersection(
                set(condition_specific_settings[compare_condition]['global_fs_mouse_ids']))))
            # Process MAIN data
            print("Processing main data...")
            results_main = get_data_and_significance(feature_data_notscaled, p1_runs, p2_runs, phase1, phase2, stride,
                                                     common_mice, measure)
            (mice_diff_main, sig_dict_main, all_features_main, significant_feats_main,
             non_significant_feats_main, p1_data_main, p2_data_main) = results_main
            # Process COMPARE data
            print("Processing compare data...")
            results_compare = get_data_and_significance(feature_data_df_compare_notscaled, p1_runs, p2_runs, phase1,
                                                        phase2, stride, common_mice, measure)
            (mice_diff_compare, sig_dict_compare, all_features_compare, significant_feats_compare,
             non_significant_feats_compare, p1_data_compare, p2_data_compare) = results_compare

            # Produce individual plots for each dataset
            for con, results in [(condition, results_main), (compare_condition, results_compare)]:
                mm_diff, mds, all_feats, sig_feats, _, _, _ = results
                print(f"Plotting for {con}...")
                plot_all_features(mm_diff, mds, all_feats, sig_feats,
                                  phase1, phase2, con, base_save_dir, stride, measure)
                plot_significant_features(mm_diff, mds, sig_feats,
                                          phase1, phase2, con, base_save_dir, stride, measure)
            # Now, select features that are significant in at least one dataset
            # and that are significantly different between conditions (using your shuffling test)
            # Initialize lists to collect features and raw p-values from the between conditions tests
            candidate_features = []
            raw_between_pvals = []

            # Loop through all features that are significant in at least one dataset
            # Run the between-condition significance tests in parallel
            results_between = Parallel(n_jobs=-1)(
                delayed(compute_feature_significance_between_conditions)(
                    feat, p1_data_main, p2_data_main, p1_data_compare, p2_data_compare,
                    mice_diff_main, mice_diff_compare, phase1, phase2, measure
                )
                for feat in all_features_main
                if (feat in significant_feats_main) or (feat in significant_feats_compare)
            )

            # Unpack results
            candidate_features, raw_between_pvals = zip(*results_between)

            # Apply the multiple comparisons correction
            reject, pvals_corrected, _, _ = multipletests(raw_between_pvals, alpha=0.05, method='fdr_bh')

            # Create a dictionary for features that pass the adjusted threshold
            between_sig_dict = {}
            features_to_plot = []
            for feat, p_corr, r in zip(candidate_features, pvals_corrected, reject):
                # Only include if the corrected p-value is significant (and/or r is True)
                if p_corr < 0.05:
                    features_to_plot.append(feat)
                    between_sig_dict[feat] = p_corr

            plot_all_features_between_conditions(mice_diff_main, mice_diff_compare,
                                                 all_features_main, phase1, phase2,
                                                 condition, compare_condition,
                                                 base_save_dir, stride, measure,
                                                 between_sig_dict)

            if len(features_to_plot) > 0:
                print(f"Plotting between-conditions differences for {len(features_to_plot)} features...")
                plot_between_conditions(mice_diff_main, mice_diff_compare,
                                        features_to_plot, phase1, phase2, condition, compare_condition, base_save_dir, stride, measure,
                                        between_sig_dict)

            else:
                print("No features met the criteria for between-conditions significance.")
    print("Done!")

if __name__ == "__main__":
    global_settings["LowHigh_c"] = condition_specific_settings['APAChar_LowHigh']['c']
    global_settings["HighLow_c"] = condition_specific_settings['APAChar_HighLow']['c']
    global_settings["LowHigh_mice"] = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
    global_settings["HighLow_mice"] = condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']
    settings_to_log = {
        "global_settings": global_settings,
        "instance_settings": instance_settings
    }
    # Run each instance for both mean and variance comparisons.
    for inst in instance_settings:
        # For mean differences:
        main(
            global_settings["mouse_ids"],
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log,
            measure="mean"
        )
        # For variance differences:
        main(
            global_settings["mouse_ids"],
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log,
            measure="variance"
        )
