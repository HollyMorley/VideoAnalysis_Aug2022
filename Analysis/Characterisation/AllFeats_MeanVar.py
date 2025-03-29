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

# ----------------------------
# Library Imports
# ----------------------------
from Helpers.Config_23 import *
from Analysis.Tools import utils_feature_reduction as utils
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
            filtered_data_df = utils.load_and_preprocess_data(mouse_id, stride, condition, exp, day)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = utils.load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day)
            feature_data_compare[(stride, mouse_id)] = filtered_data_comparison_df

    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_df_compare = pd.concat(feature_data_compare, axis=0)

    return feature_data_df, feature_data_df_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition



def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):

    # Initialize experiment (data collection, directories, logging). - NOT SCALED YET!!
    feature_data_notscaled, feature_data_df_compare_notscaled, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = initialize_experiment(condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)
    feature_data_notscaled.index.names = ['Stride', 'MouseID', 'Run']
    feature_data_df_compare_notscaled.index.names = ['Stride', 'MouseID', 'Run']

    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")

    # find difference between phases
    for stride in stride_numbers:
        for phase1, phase2 in itertools.combinations(phases, 2):
            p1_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
            p2_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]

            common_mice = np.array(list(set(condition_specific_settings[condition]['global_fs_mouse_ids']).intersection(set(condition_specific_settings[compare_condition]['global_fs_mouse_ids']))))

            p1_data_notscaled = feature_data_notscaled.loc[(stride, common_mice, p1_runs)]
            p2_data_notscaled = feature_data_notscaled.loc[(stride, common_mice, p2_runs)]
            combined_data = pd.concat([p1_data_notscaled, p2_data_notscaled])
            global_means = combined_data.mean()
            global_stds = combined_data.std()

            # Apply z-score scaling to both datasets
            p1_data_scaled = (p1_data_notscaled - global_means) / global_stds
            p2_data_scaled = (p2_data_notscaled - global_means) / global_stds

            # drop stride level
            p1_data = p1_data_scaled.droplevel('Stride')
            p2_data = p2_data_scaled.droplevel('Stride')

            p1_mice_feat_means = p1_data.groupby('MouseID').mean()
            p2_mice_feat_means = p2_data.groupby('MouseID').mean()

            p1_mice_feat_vars = p1_data.groupby('MouseID').var()
            p2_mice_feat_vars = p2_data.groupby('MouseID').var()

            mice_mean_diff = p2_mice_feat_means - p1_mice_feat_means
            mice_var_diff = p2_mice_feat_vars - p1_mice_feat_vars

            # Calculate significance
            mean_diff_sig = {}
            var_diff_sig = {}
            for feat in p1_data.columns:
                p_value_mean, _, _ = utils.ShufflingTest_ComparePhases(
                    p1_data.loc(axis=1)[feat], p2_data.loc(axis=1)[feat], p1_mice_feat_means.loc(axis=1)[feat], p2_mice_feat_means.loc(axis=1)[feat], phase1, phase2)
                mean_diff_sig[feat] = p_value_mean
                # todo adjust significance test for variance

            all_features = list(mice_mean_diff.columns)
            significances_array = np.array(list(mean_diff_sig.values()))
            significant_feats = np.array(all_features)[significances_array < 0.05]
            non_significant_feats = np.array(all_features)[significances_array >= 0.05]

            # Plotting
            chunk_size = 66
            n_chunks = math.ceil(len(all_features) / chunk_size)  # Expected to be 4 rows

            # Reduced figure size to avoid memory issues (width=50, height=20 per row)
            fig, axs = plt.subplots(n_chunks, 1, figsize=(50, 20 * n_chunks))
            if n_chunks == 1:
                axs = [axs]

            name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']

            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk_feats = all_features[start:end]

                # Process feature names for display
                short_feats = []
                for feat in chunk_feats:
                    name_bits = feat.split(', ')
                    name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
                    short_feats.append(', '.join(name_to_keep))

                # Calculate means and errors for the current chunk
                chunk_means = mice_mean_diff[chunk_feats].mean(axis=0)
                chunk_errs = mice_mean_diff[chunk_feats].sem(axis=0) * 1.645

                ax = axs[i]

                # Create a list of colors: green for significant features, default color for others.
                colors = ['green' if feat in significant_feats else 'C0' for feat in chunk_feats]

                # Plot the bar chart for this chunk with individual bar colors.
                ax.bar(short_feats, chunk_means, yerr=chunk_errs, capsize=1, color=colors)

                # Add significance stars
                for j, feat in enumerate(chunk_feats):
                    if chunk_means.iloc[j] > 0:
                        ypos = chunk_means.iloc[j] + chunk_errs.iloc[j] + 0.1
                    else:
                        ypos = chunk_means.iloc[j] - chunk_errs.iloc[j] - 0.1

                    if mean_diff_sig[feat] < 0.001:
                        ax.text(j, ypos, '***', ha='center', va='bottom', rotation=90)
                    elif mean_diff_sig[feat] < 0.01:
                        ax.text(j, ypos, '**', ha='center', va='bottom', rotation=90)
                    elif mean_diff_sig[feat] < 0.05:
                        ax.text(j, ypos, '*', ha='center', va='bottom', rotation=90)

                # Rotate x labels and change their color for significant features
                ax.tick_params(axis='x', rotation=90)
                ticklabels = ax.get_xticklabels()
                for j, label in enumerate(ticklabels):
                    if chunk_feats[j] in significant_feats:
                        label.set_color('green')
                ax.set_ylabel('Mean Difference')
                ax.set_xlabel('Feature')
                ax.set_ylim(-1, 1)
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title(
                    f"Mean Difference between {phase2} - {phase1} for {condition} condition (Features {start + 1}-{min(end, len(all_features))})",
                    pad=20)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.01)

            save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
            os.makedirs(save_dir, exist_ok=True)

            fname = f'AllFeats_MeanDiff_{phase2}-{phase1}_{condition}_stride{stride}.pdf'
            fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()

            #############################
            # 2. Plot only significant features on one subplot.
            #############################

            # Only plot the significant features
            sig_features_list = list(significant_feats)

            # Process feature names for display
            sig_short_feats = []
            for feat in sig_features_list:
                name_bits = feat.split(', ')
                name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
                sig_short_feats.append(', '.join(name_to_keep))

            # Calculate means and errors for significant features
            sig_means = mice_mean_diff[sig_features_list].mean(axis=0)
            sig_errs = mice_mean_diff[sig_features_list].sem(axis=0) * 1.645

            fig2, ax2 = plt.subplots(1, 1, figsize=(50, 20))

            # Plot the significant features (default color)
            ax2.bar(sig_short_feats, sig_means, yerr=sig_errs, capsize=1)

            # Add significance stars for significant features
            for i, feat in enumerate(sig_features_list):
                if sig_means.iloc[i] > 0:
                    ypos = sig_means.iloc[i] + sig_errs.iloc[i] + 0.1
                else:
                    ypos = sig_means.iloc[i] - sig_errs.iloc[i] - 0.1

                if mean_diff_sig[feat] < 0.001:
                    ax2.text(i, ypos, '***', ha='center', va='bottom', rotation=90)
                elif mean_diff_sig[feat] < 0.01:
                    ax2.text(i, ypos, '**', ha='center', va='bottom', rotation=90)
                elif mean_diff_sig[feat] < 0.05:
                    ax2.text(i, ypos, '*', ha='center', va='bottom', rotation=90)

            ax2.tick_params(axis='x', rotation=90)
            ax2.set_ylim(-1, 1)
            ax2.set_ylabel('Mean Difference')
            ax2.set_xlabel('Feature')
            ax2.grid(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_title(
                f"Mean Difference between {phase2} - {phase1} for {condition} condition (Significant Features Only)",
                pad=20)

            plt.tight_layout()

            fname2 = f'AllSigFeats_MeanDiff_{phase2}-{phase1}_{condition}_stride{stride}.png'
            fig2.savefig(os.path.join(save_dir, fname2), dpi=300, bbox_inches='tight')
            plt.close()
    print(f"Finished {condition} condition.")




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
