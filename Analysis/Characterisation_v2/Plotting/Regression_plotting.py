import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import medfilt

from Helpers.Config_23 import *
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)


def plot_weights_in_feature_space(feature_weights, save_path, mouse_id, phase1, phase2, stride_number, condition, x_label_offset=0.5):
    """
    Plot the feature-space weights as a bar plot with feature names on the x-axis.
    """
    # Create a DataFrame for plotting and sort for easier visualization
    df = pd.DataFrame({'Feature': feature_weights.index, 'weight': feature_weights.values})
    df['Display'] = df['Feature'].apply(lambda x: short_names.get(x, x))
    df['cluster'] = df['Feature'].map(manual_clusters['cluster_mapping'])
    df = df.dropna(subset=['cluster'])
    df['cluster'] = df['cluster'].astype(int)

    order_map = {feat: idx for idx, feat in enumerate(manual_clusters['cluster_mapping'].keys())}
    df['order'] = df['Feature'].map(order_map)
    df = df.sort_values(by='order').reset_index(drop=True)

    #sort df by weight
    #df = df.sort_values(by='weight', ascending=False)
    fig, ax = plt.subplots(figsize=(14, max(8, int(len(df) * 0.3))))
    sns.barplot(x='weight', y='Display', data=df, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('')
    plt.title(f'Feature Weights in Original Space for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    lower_x_lim = df['weight'].min()
    upper_x_lim = df['weight'].max()
    x_range = upper_x_lim - lower_x_lim

    if not df['Feature'].str.contains('PC').all():
        for i, cl in enumerate(sorted(df['cluster'].unique())):
            group_indices = df.index[df['cluster'] == cl].tolist()
            x_pos = lower_x_lim - x_range * x_label_offset
            y_positions = group_indices
            y0 = min(y_positions) - 0.05
            y1 = max(y_positions) + 0.05
            k_r = 0.1
            span = abs(y1 - y0)
            desired_depth = 0.1  # or any value that gives you the uniform look you want
            k_r_adjusted = desired_depth / span if span != 0 else k_r

            # Alternate the int_line_num value for every other cluster:
            base_line_num = 2
            int_line_num = base_line_num + 0.5 if i % 2 else base_line_num

            cluster_label = [k for k, v in manual_clusters['cluster_values'].items() if v == cl][0]

            pu.add_vertical_brace_curly(ax, y0, y1, x_pos, k_r=k_r_adjusted, int_line_num=int_line_num,
                                     xoffset=0.2, label=cluster_label, rot_label=90)
        plt.subplots_adjust(left=0.35)
        plt.xlim(lower_x_lim, upper_x_lim)

    plot_file = os.path.join(save_path, f'feature_space_weights_{mouse_id}_{phase1}_vs_{phase2}_stride{stride_number}_{condition}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical feature-space weights plot saved to: {plot_file}")

def plot_weights_in_pc_space(pc_weights, save_path, mouse_id, phase1, phase2, stride_number, condition):
    """
    Plot the PC-space weights as a bar plot with PC names on the x-axis.
    """
    # Create a DataFrame for plotting
    df = pd.DataFrame({'PC': pc_weights.index, 'weight': pc_weights.values})

    fig, ax = plt.subplots(figsize=(14, max(8, int(len(df) * 0.3))))
    sns.barplot(x='weight', y='PC', data=df, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('')
    plt.title(f'PC Weights in PCA Space for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    plot_file = os.path.join(save_path, f'pc_space_weights_{mouse_id}_{phase1}_vs_{phase2}_stride{stride_number}_{condition}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical PC-space weights plot saved to: {plot_file}")

def plot_run_prediction(data, run_pred, run_pred_smoothed, save_path, mouse_id, phase1, phase2, stride_number, scale_suffix, dataset_suffix):
    # plot run prediction
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, run_pred[0], color='lightblue', ls='--', label='Prediction')
    plt.plot(data.index, run_pred_smoothed, color='blue', ls='-', label='Smoothed Prediction')
    # Exp phases
    plt.vlines(x=9.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    plt.vlines(x=109.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    # Days
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=79.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=119.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)

    # plot a shaded box over x=60 to x=110 and x=135 to x=159, ymin to ymax
    plt.fill_between(x=range(60, 110), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)
    plt.fill_between(x=range(135, 160), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)

    plt.title(f'Run Prediction for Mouse {mouse_id} - {phase1} vs {phase2}')
    plt.xlabel('Run Number')
    plt.ylabel('Prediction')

    legend_elements = [Line2D([0], [0], color='red', linestyle='--', label='Experimental Phases'),
                          Line2D([0], [0], color='black', linestyle='--', label='Days'),
                          Patch(facecolor='gray', edgecolor='black', alpha=0.1, label='Training Portion'),
                          Line2D([0], [0], color='lightblue', label='Prediction', linestyle='--'),
                          Line2D([0], [0], color='blue', label='Smoothed Prediction', linestyle='-')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Run_Prediction_{phase1}_vs_{phase2}_stride{stride_number}_{scale_suffix}_{dataset_suffix}.png"), dpi=300)
    plt.close()

def plot_PCA_pred_heatmap(pca_all, pca_pred, feature_data, stride_data, phases, stride_numbers, condition, save_path):
    p1 = phases[0]
    p2 = phases[1]
    if len(pca_all) == 1 and pca_all[0].phase[0] == p1 and pca_all[0].phase[1] == p2:
        pca = pca_all[0].pca
        loadings = pca_all[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
        pca_weights = pca_pred[0].pc_weights

    for s in stride_numbers:
        blanked_preds_byPC = {col: {} for col in loadings.columns}

        for midx in condition_specific_settings[condition]['global_fs_mouse_ids']:
            # Get mouse run data
            featsxruns, featsxruns_phaseruns, run_ns, stepping_limbs, mask_p1, mask_p2 = gu.select_runs_data(
                midx, s, feature_data, stride_data, p1, p2)

            # Get the PCA data for the current mouse and stride
            pcs = pca.transform(featsxruns)
            # Trim by number of PCs to use
            pcs = pcs[:, :global_settings['pcs_to_use']]

            for pc_idx in range(min(global_settings['pcs_to_use'], pca_weights.shape[1])):
                blanked_pc_weights = np.zeros_like(pca_weights)[0]
                blanked_pc_weights[pc_idx] = pca_weights[0][pc_idx]
                y_pred = np.dot(pcs, blanked_pc_weights.T).squeeze()
                blanked_preds_byPC[loadings.columns[pc_idx]][midx] = y_pred

        pc_x_run_pred = {}
        for pc in blanked_preds_byPC.keys():
            mouse_pcs = blanked_preds_byPC[pc]

            mouse_pc_y_pred = {}
            for midx in mouse_pcs.keys():
                run_vals = feature_data.loc(axis=0)[s,midx].index.tolist()

                target_runs = np.arange(0,160)

                # interpolate the y_pred values to match the run numbers
                y_pred = mouse_pcs[midx]
                y_pred_interp = np.interp(target_runs, run_vals, y_pred)
                mouse_pc_y_pred[midx] = y_pred_interp
            mouse_pc_y_pred_df = pd.DataFrame(mouse_pc_y_pred, index=target_runs)
            median_pc_ypred = mouse_pc_y_pred_df.median(axis=1)
            pc_x_run_pred[pc] = median_pc_ypred
        pc_x_run_pred_df = pd.DataFrame(pc_x_run_pred).T

        heatmap_data_smooth = pc_x_run_pred_df.apply(lambda x: medfilt(x, kernel_size=3), axis=1)
        heatmap_df = pd.DataFrame(heatmap_data_smooth.tolist(), index=pc_x_run_pred_df.index, columns=pc_x_run_pred_df.columns)

        # plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        h = sns.heatmap(heatmap_df, cmap='coolwarm', cbar=True, yticklabels=loadings.columns.tolist(),
                        # vmin=-10, vmax=10, cbar_kws={'label': 'Prediction', 'orientation': 'vertical'})
                        cbar_kws={'label': 'Prediction', 'orientation': 'vertical'})

                        # Change x-axis tick labels
        ax.set_xticks(np.arange(0, 161, 10))
        ax.set_xticklabels(np.arange(0, 161, 10))
        ax.tick_params(axis='both', which='major', labelsize=7)

        # Change the colorbar tick labels font size
        cbar = h.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        # Optionally, change the colorbar label font size as well:
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=7)

        plt.axvline(x=10, color='black', linestyle='--')
        plt.axvline(x=110, color='black', linestyle='--')
        plt.ylabel('')
        plt.xlabel('Trial')

        plt.savefig(os.path.join(save_path, f"PCAXRuns_Heatmap__RunPrediction_{p1}_{p2}_{s}_{condition}.png"), dpi=300)
        plt.close()

def plot_aggregated_run_predictions(run_pred, save_dir, p1, p2, s, condition, normalization_method='maxabs'):
    plt.figure(figsize=(10, 8))

    # Collect common x-axis values.
    all_x_vals = []
    for data in run_pred:
        all_x_vals.extend(data.x_vals)
    global_min_x = min(all_x_vals)
    global_max_x = max(all_x_vals)
    common_npoints = max(len(data.x_vals) for data in run_pred)
    common_x = np.linspace(global_min_x, global_max_x, common_npoints)

    plt.axvspan(9.5, 109.5, color='lightblue', alpha=0.2)

    interpolated_curves = []
    for data in run_pred:
        mouse_id = data.mouse_id
        x_vals = data.x_vals
        smoothed_pred = data.y_pred_smoothed

        # Normalize the curve.
        if normalization_method == 'zscore':
            mean_val = np.mean(smoothed_pred)
            std_val = np.std(smoothed_pred)
            normalized_curve = (smoothed_pred - mean_val) / std_val if std_val != 0 else smoothed_pred
        elif normalization_method == 'maxabs':
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred
        else:
            normalized_curve = smoothed_pred

        interp_curve = np.interp(common_x, x_vals, normalized_curve)
        interpolated_curves.append(interp_curve)

        plt.plot(common_x, interp_curve, label=f'Mouse {mouse_id}', alpha=0.3, color='grey')

    all_curves_array = np.vstack(interpolated_curves)
    mean_curve = np.mean(all_curves_array, axis=0)
    plt.plot(common_x, mean_curve, color='black', linewidth=2, label='Mean Curve')

    # plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(
        f'Aggregated {normalization_method.upper()} Scaled Run Predictions for {p1} vs {p2}, stride {s}\n{condition}')
    plt.xlabel('Run Number')
    plt.ylabel('Normalized Prediction (Smoothed)')
    if normalization_method == 'maxabs':
        plt.ylim(-1, 1)
    # plt.legend(loc='upper right')
    plt.grid(False)
    # plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path_full = os.path.join(save_dir,
                                  f"Aggregated_{normalization_method.upper()}_Run_Predictions_{p1}_vs_{p2}_stride{s}_{condition}.png")
    plt.savefig(save_path_full, dpi=300)
    plt.close()

def plot_multi_stride_predictions(stride_dict, p1, p2, condition, save_dir, smooth: bool = False,
                                  smooth_window: int = 21,
                                  normalize: bool = False):
    plt.figure(figsize=(10, 8))
    mean_curves = {}  # Stores: stride_number -> (common_x, mean_curve)











