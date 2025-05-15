import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.signal
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib import ticker

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

    plot_file = os.path.join(save_path, f'feature_space_weights_{mouse_id}_{phase1}_vs_{phase2}_stride{stride_number}_{condition}')
    plt.savefig(plot_file + '.png')
    plt.savefig(plot_file + '.svg')
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

    plot_file = os.path.join(save_path, f'pc_space_weights_{mouse_id}_{phase1}_vs_{phase2}_stride{stride_number}_{condition}')
    plt.savefig(plot_file + '.png')
    plt.savefig(plot_file + '.svg')
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
    plt.savefig(os.path.join(save_path, f"Run_Prediction_{phase1}_vs_{phase2}_stride{stride_number}_{scale_suffix}_{dataset_suffix}.svg"), dpi=300)
    plt.close()

def plot_PCA_pred_heatmap(pca_all, pca_pred, feature_data, stride_data, phases, stride_numbers, condition, save_path, white_width=0.1, cbar_scaling=1):
    p1 = phases[0]
    p2 = phases[1]
    if len(pca_all) == 1 and pca_all[0].phase[0] == p1 and pca_all[0].phase[1] == p2:
        pca = pca_all[0].pca
        loadings = pca_all[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
        # pca_weights = pca_pred[0].pc_weights

    for s in stride_numbers:
        blanked_preds_byPC = {col: {} for col in loadings.columns}
        # stride_pca_preds = [pred for pred in pca_pred if pred.stride == s]

        for midx in condition_specific_settings[condition]['global_fs_mouse_ids']:
            pca_weights = [pred.pc_weights for pred in pca_pred if pred.stride == s and pred.mouse_id == midx][0]
            # Get mouse run data
            featsxruns, featsxruns_phaseruns, run_ns, stepping_limbs, mask_p1, mask_p2 = gu.select_runs_data(
                midx, s, feature_data, stride_data, p1, p2)

            # Get the PCA data for the current mouse and stride
            pcs = pca.transform(featsxruns)
            # Trim by number of PCs to use
            pcs = pcs[:, :global_settings['pcs_to_use']]

            for pc_idx in range(min(global_settings['pcs_to_use'], pca_weights.shape[1])):
                # blanked_pc_weights = np.zeros_like(pca_weights)[0]
                # blanked_pc_weights[pc_idx] = pca_weights[0][pc_idx]
                pc_weights = pca_weights[0][pc_idx]
                y_pred = np.dot(pcs, pc_weights.T).squeeze()
                blanked_preds_byPC[loadings.columns[pc_idx]][midx] = y_pred

        pc_x_run_pred = {}
        for pc_idx, pc in enumerate(blanked_preds_byPC.keys()):
            mouse_pcs = blanked_preds_byPC[pc]

            mouse_pc_y_pred = {}
            for midx in mouse_pcs.keys():
                run_vals = feature_data.loc(axis=0)[s,midx].index.tolist()

                target_runs = np.arange(0,160)

                # interpolate the y_pred values to match the run numbers
                y_pred = mouse_pcs[midx][:,pc_idx]
                y_pred_interp = np.interp(target_runs, run_vals, y_pred)
                mouse_pc_y_pred[midx] = y_pred_interp
            mouse_pc_y_pred_df = pd.DataFrame(mouse_pc_y_pred, index=target_runs)
            median_pc_ypred = mouse_pc_y_pred_df.mean(axis=1)
            pc_x_run_pred[pc] = median_pc_ypred
        pc_x_run_pred_df = pd.DataFrame(pc_x_run_pred).T

        heatmap_data_smooth = pc_x_run_pred_df.apply(lambda x: median_filter(x, size=7, mode='nearest'), axis=1)
        heatmap_df = pd.DataFrame(heatmap_data_smooth.tolist(), index=pc_x_run_pred_df.index, columns=pc_x_run_pred_df.columns)

        # --- Custom colormap creation ---
        # Get your phase-specific colors.
        p1_color = pu.get_color_phase(p1)
        p2_color = pu.get_color_phase(p2)
        p1_rgb = pu.hex_to_rgb(p1_color)
        p2_rgb = pu.hex_to_rgb(p2_color)
        white_rgb = (0.93, 0.93, 0.93)  # approximate for "#EEEEEE"
        # custom_cmap = pu.create_custom_colormap(p2_rgb, white_rgb, p1_rgb, cbar_scaling)

        cdict = {
            'red': [(0.0, p2_rgb[0], p2_rgb[0]),
                    (0.01, p2_rgb[0], p2_rgb[0]),  # hold low color longer
                    (0.5, white_rgb[0], white_rgb[0]),
                    (0.99, p1_rgb[0], p1_rgb[0]),
                    (1.0, p1_rgb[0], p1_rgb[0])],
            'green': [(0.0, p2_rgb[1], p2_rgb[1]),
                      (0.01, p2_rgb[1], p2_rgb[1]),
                      (0.5, white_rgb[1], white_rgb[1]),
                      (0.99, p1_rgb[1], p1_rgb[1]),
                      (1.0, p1_rgb[1], p1_rgb[1])],
            'blue': [(0.0, p2_rgb[2], p2_rgb[2]),
                     (0.01, p2_rgb[2], p2_rgb[2]),
                     (0.5, white_rgb[2], white_rgb[2]),
                     (0.99, p1_rgb[2], p1_rgb[2]),
                     (1.0, p1_rgb[2], p1_rgb[2])]
        }
        custom_cmap = mcolors.LinearSegmentedColormap("custom_div", segmentdata=cdict)

        # Use TwoSlopeNorm so that 0 maps to 0.5 (center of the colormap)
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        # -----------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 8))
        h = sns.heatmap(heatmap_df, cmap=custom_cmap, norm=norm, cbar=True, yticklabels=loadings.columns.tolist(),
                        vmin=-1, vmax=1, cbar_kws={'label': 'Prediction', 'orientation': 'vertical'})
                        #cbar_kws={'label': 'Prediction', 'orientation': 'vertical'})

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
        plt.savefig(os.path.join(save_path, f"PCAXRuns_Heatmap__RunPrediction_{p1}_{p2}_{s}_{condition}.svg"), dpi=300)
        plt.close()

def plot_aggregated_run_predictions(run_pred, save_dir, p1, p2, s, condition, normalization_method='maxabs', smooth_kernel=13, fs=7):
    stride_pred = [pred for pred in run_pred if pred.stride == s]
    common_x = np.arange(0, 160, 1)

    interpolated_curves = []
    fig,ax = plt.subplots(figsize=(5, 5))
    apa1_color = pu.get_color_phase('APA1')
    apa2_color = pu.get_color_phase('APA2')
    wash1_color = pu.get_color_phase('Wash1')
    wash2_color = pu.get_color_phase('Wash2')

    boxy = 1
    height = 0.02
    patch1 = plt.axvspan(xmin=9.5, xmax=59.5, ymin=boxy, ymax=boxy+height, color=apa1_color, lw=0)
    patch2 = plt.axvspan(xmin=59.5, xmax=109.5, ymin=boxy, ymax=boxy+height, color=apa2_color, lw=0)
    patch3 = plt.axvspan(xmin=109.5, xmax=134.5, ymin=boxy, ymax=boxy+height, color=wash1_color, lw=0)
    patch4 = plt.axvspan(xmin=134.5, xmax=159.5, ymin=boxy, ymax=boxy+height, color=wash2_color, lw=0)
    patch1.set_clip_on(False)
    patch2.set_clip_on(False)
    patch3.set_clip_on(False)
    patch4.set_clip_on(False)

    for data in stride_pred:
        mouse_id = data.mouse_id
        x_vals = data.x_vals
        pred = data.y_pred[0]

        smoothed_pred = median_filter(pred, size=smooth_kernel, mode='nearest') if smooth_kernel > 1 else pred

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
        # f = interp1d(x_vals, normalized_curve, kind='cubic')
        # interp_curve = f(common_x)
        interpolated_curves.append(interp_curve)

        ax.plot(common_x, interp_curve, linewidth=0.6, label=f'Mouse {mouse_id}', alpha=0.3, color='grey')

    all_curves_array = np.vstack(interpolated_curves)
    mean_curve = np.mean(all_curves_array, axis=0)
    ax.plot(common_x, mean_curve, color='black', linewidth=1, label='Mean Curve')

    # plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_title(s, fontsize=fs, pad=10)
    ax.set_xlabel('Trial', fontsize=fs)
    ax.set_ylabel('Normalized Prediction', fontsize=fs)
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.arange(-0.5, 0.6, 0.5))
    ax.set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    ax.set_xticks(np.arange(0, 161, 20))
    ax.set_xticklabels(np.arange(0, 161, 20), fontsize=fs)
    ax.set_xlim(0, 160)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.grid(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)

    save_path_full = os.path.join(save_dir,
                                  f"Aggregated_{normalization_method.upper()}_Run_Predictions_{p1}_vs_{p2}_stride{s}_{condition}")
    plt.savefig(f"{save_path_full}.png", dpi=300)
    plt.savefig(f"{save_path_full}.svg", dpi=300)
    plt.close()
    return mean_curve

def plot_multi_stride_predictions(Y_pred_bystride: dict, p1, p2, condition, save_dir,
                                  mean_smooth_window: int = 11, fs=7):
    fig,ax = plt.subplots(figsize=(5,5))
    mean_curves = {}  # Stores: stride_number -> (common_x, mean_curve)

    common_x = np.arange(0, 160, 1)

    apa1_color = pu.get_color_phase('APA1')
    apa2_color = pu.get_color_phase('APA2')
    wash1_color = pu.get_color_phase('Wash1')
    wash2_color = pu.get_color_phase('Wash2')
    boxy = 1
    height = 0.02
    patch1 = plt.axvspan(xmin=9.5, xmax=59.5, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
    patch2 = plt.axvspan(xmin=59.5, xmax=109.5, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
    patch3 = plt.axvspan(xmin=109.5, xmax=134.5, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
    patch4 = plt.axvspan(xmin=134.5, xmax=159.5, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
    patch1.set_clip_on(False)
    patch2.set_clip_on(False)
    patch3.set_clip_on(False)
    patch4.set_clip_on(False)

    for s, stride_pred_mean in Y_pred_bystride.items():
        # Further smooth the mean line
        stride_pred_mean = median_filter(stride_pred_mean, size=mean_smooth_window, mode='nearest') if mean_smooth_window > 1 else stride_pred_mean
        ls = pu.get_ls_stride(s)
        ax.plot(common_x, stride_pred_mean, linewidth=1, label=f"Stride {s}", color="k", linestyle=ls)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('Trial', fontsize=fs)
    ax.set_ylabel('Normalized Prediction', fontsize=fs)
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.arange(-0.5, 0.6, 0.5))
    ax.set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    ax.set_xticks(np.arange(0, 161, 20))
    ax.set_xticklabels(np.arange(0, 161, 20), fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper right')

    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(save_dir,f"Aggregated_MultiStride_Predictions_{p1}_vs_{p2}_{condition}_sw{mean_smooth_window}")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()

def plot_regression_loadings_PC_space_across_mice(pca_all, pca_pred, s, p1, p2, condition, save_dir, fs=7):
    loadings = [pca.pca_loadings for pca in pca_all if pca.stride == 'all'][0]
    w = np.array([pca.pc_weights[0] for pca in pca_pred if pca.stride == s and pca.phase == (p1, p2)])

    fig, ax = plt.subplots(figsize=(4,8))

    w_norm = np.zeros_like(w)
    for midx, w_mouse in enumerate(w):
        mouse_name = condition_specific_settings[condition]['global_fs_mouse_ids'][midx]
        # max abs normalisation
        max_abs = np.abs(w_mouse).max()
        w_mouse_norm = w_mouse / max_abs if max_abs != 0 else w_mouse
        w_norm[midx] = w_mouse_norm

        ms = pu.get_marker_style_mice(mouse_name)
        #ls = pu.get_line_style_mice(mouse_name)
        ls='-'

        ax.plot(w_mouse_norm[:global_settings["pcs_to_plot"]], np.arange(1, global_settings["pcs_to_plot"]+1),
                 marker=ms, markersize=4, linestyle=ls, linewidth=0.5, label=mouse_name, color='k')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.4)

    ax.set_xlabel("Normalized PC Weights", fontsize=fs)
    ax.set_ylabel("PC", fontsize=fs)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_xticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
    ax.set_ylim(0, global_settings["pcs_to_plot"]+1)
    ax.set_yticks(np.arange(1, global_settings["pcs_to_plot"]+1))
    ax.set_yticklabels(np.arange(1, global_settings["pcs_to_plot"]+1), fontsize=fs)
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=f"Mouse ID", title_fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.grid(False)

    ax.invert_yaxis()

    plt.subplots_adjust(right=0.7, top=0.95)

    fig.savefig(os.path.join(save_dir, f"Normalized_PC_regression_weights_{p1}_{p2}_stride{s}_{condition}.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, f"Normalized_PC_regression_weights_{p1}_{p2}_stride{s}_{condition}.svg"), dpi=300)
    plt.close()

def plot_top3_pcs_run_projections(feature_data, pca_all, stride, condition, save_dir, fs=7):
    data = feature_data.loc(axis=0)[stride]
    common_x = np.arange(0, 160)
    pca_obj = pca_all[0].pca

    pcs_all_mice = np.zeros((len(condition_specific_settings[condition]['global_fs_mouse_ids']), 160, 3))
    for midx, m in enumerate(condition_specific_settings[condition]['global_fs_mouse_ids']):
        mdata = data.loc(axis=0)[m]
        pcs = pca_obj.transform(mdata)
        pcs = pcs[:, [0, 2, 6]]
        pcs_interp = np.zeros((160, pcs.shape[1]))
        for pc in range(pcs.shape[1]):
            pcs_interp[:, pc] = np.interp(common_x, mdata.index, pcs[:, pc])
        pcs_all_mice[midx, :, :] = pcs_interp
    pcs_mean = np.mean(pcs_all_mice, axis=0)
    pcs_mean = pcs_mean[10:, :]
    # smooth across runs
    import scipy
    pcs_mean = scipy.signal.savgol_filter(pcs_mean, window_length=10, polyorder=2, axis=0)
    # pcs_mean_df = pd.DataFrame(index=common_x, data=pcs_mean, columns=['PC1', 'PC3', 'PC7'])
    # pcs_mean_df = pcs_mean_df.iloc(axis=0)[10:]

    # get colours for each run
    apa1_grad = pu.gradient_colors('#AAAAAA', pu.get_color_phase('APA2'), 50)
    apa2_grad = pu.gradient_colors(pu.get_color_phase('APA2'), pu.get_color_phase('APA2'), 50)
    wash1_grad = pu.gradient_colors('#AAAAAA', pu.get_color_phase('Wash2'), 25)
    wash2_grad = pu.gradient_colors(pu.get_color_phase('Wash2'), pu.get_color_phase('Wash2'), 25)
    clrs = np.concatenate((apa1_grad, apa2_grad, wash1_grad, wash2_grad), axis=0)

    # plot 3d plot of pcs
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    pcs_mean_exSm = scipy.signal.savgol_filter(pcs_mean, window_length=20, polyorder=2, axis=0)
    for run in range(0, 149):
        # ax.scatter(pcs_mean[run,0], pcs_mean[run, 1], pcs_mean[run, 2], c=clrs[run], marker='o', s=10, alpha=0.5)
        # ax.plot([pcs_mean[run, 0], pcs_mean[run+1, 0]], [pcs_mean[run, 1], pcs_mean[run+1, 1]], [pcs_mean[run, 2], pcs_mean[run+1, 2]], c=clrs[run], alpha=0.5)
        ax.plot([pcs_mean_exSm[run, 0], pcs_mean_exSm[run + 1, 0]], [pcs_mean_exSm[run, 1], pcs_mean_exSm[run + 1, 1]],
                [pcs_mean_exSm[run, 2], pcs_mean_exSm[run + 1, 2]], c=clrs[run], alpha=0.8)
    # ax.scatter(pcs_mean[:, 0], pcs_mean[:, 1], pcs_mean[:, 2], c='b', marker='o')

    ax.set_xlabel('PC1', fontsize=7)
    ax.set_ylabel('PC3', fontsize=7)
    ax.set_zlabel('PC7', fontsize=7)
    # set tick fontsize to 7
    ax.tick_params(axis='both', which='major', labelsize=7)

    ax.set_facecolor('none')  # no pane fill
    fig.patch.set_facecolor('white')  # or 'none' if you want transparency
    ax.grid(False)  # no grid lines
    # turn off the built‑in spines/ticks
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.view_init(elev=25, azim=32)

    plt.subplots_adjust(left=0.25, right=0.95, top=0.99, bottom=0.1)

    plt.savefig(os.path.join(save_dir, f"Top3PCs_3D_projection_{stride}_{condition}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"Top3PCs_3D_projection_{stride}_{condition}.svg"), dpi=300)
    plt.close()














