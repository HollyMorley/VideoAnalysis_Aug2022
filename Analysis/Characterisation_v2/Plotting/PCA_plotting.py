import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.ndimage import median_filter
import matplotlib.ticker as ticker

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu

def plot_scree(pca, p1, p2, stride, condition, save_path, fs=7):
    """
    Plot and save the scree plot.
    """
    from Analysis.Tools.config import (global_settings)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', markersize=2, linewidth=1, color='blue', label='Individual Explained Variance')
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', markersize=2, linewidth=1, color='red', label='Cumulative Explained Variance')
    ax.set_title(stride, fontsize=fs)
    ax.set_xlabel('Principal Component', fontsize=fs)
    ax.set_ylabel('Explained Variance Ratio', fontsize=fs)
    # xtick range with every 10th label
    ax.set_xlim(0, len(pca.explained_variance_ratio_) + 1)
    ax.set_xticks(np.arange(0, len(pca.explained_variance_ratio_) + 1, 10))
    ax.set_xticklabels(np.arange(0, len(pca.explained_variance_ratio_) + 1, 10), fontsize=fs)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=2, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=1, width=1)
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_yticklabels(np.arange(0, 1.1, 0.25), fontsize=fs)
    ax.legend(loc='best')
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.svg"), dpi=300)
    plt.close()


def plot_pca(pca, pcs, labels, p1, p2, stride, stepping_limbs, run_numbers, mouse_id, condition_label, save_path):
    """
    Create and save 2D and 3D PCA scatter plots.
    """
    n_pc = pcs.shape[1]
    df_plot = pd.DataFrame(pcs, columns=[f'PC{i + 1}' for i in range(n_pc)])
    df_plot['Condition'] = labels
    df_plot['SteppingLimb'] = stepping_limbs
    df_plot['Run'] = run_numbers

    markers_all = {'ForepawL': 'X', 'ForepawR': 'o'}
    unique_limbs = df_plot['SteppingLimb'].unique()
    current_markers = {}
    for limb in unique_limbs:
        if limb in markers_all:
            current_markers[limb] = markers_all[limb]
        else:
            raise ValueError(f"No marker defined for stepping limb: {limb}")

    explained_variance = pca.explained_variance_ratio_
    # print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    # print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    # if pca.n_components_ >= 3:
    #     print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

    # 2D Scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Condition',
        style='SteppingLimb',
        markers=current_markers,
        s=100,
        alpha=0.7
    )
    plt.title(f'PCA: PC1 vs PC2 for Mouse {mouse_id}')
    plt.xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
    plt.legend(title='Condition & Stepping Limb', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    for _, row in df_plot.iterrows():
        plt.text(row['PC1'] + 0.02, row['PC2'] + 0.02, str(row['Run']), fontsize=8, alpha=0.7)
    padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
    padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
    plt.xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
    plt.ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"PCA_2D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"PCA_2D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.svg"), dpi=300)
    plt.close()

    # 3D Scatter (if available)
    if pca.n_components_ >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        palette = sns.color_palette("bright", len(df_plot['Condition'].unique()))
        conditions_unique = df_plot['Condition'].unique()
        for idx, condition in enumerate(conditions_unique):
            subset = df_plot[df_plot['Condition'] == condition]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                       label=condition, color=palette[idx], alpha=0.7, s=50, marker='o')
            for _, row in subset.iterrows():
                ax.text(row['PC1'] + 0.02, row['PC2'] + 0.02, row['PC3'] + 0.02,
                        str(row['Run']), fontsize=8, alpha=0.7)
        ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
        ax.set_zlabel(f'PC3 ({explained_variance[2] * 100:.1f}%)')
        ax.set_title(f'3D PCA for Mouse {mouse_id}')
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2)
        padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
        padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
        padding_pc3 = (df_plot['PC3'].max() - df_plot['PC3'].min()) * 0.05
        ax.set_xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
        ax.set_ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
        ax.set_zlim(df_plot['PC3'].min() - padding_pc3, df_plot['PC3'].max() + padding_pc3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"PCA_3D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.png"), dpi=300)
        plt.savefig(os.path.join(save_path, f"PCA_3D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.svg"), dpi=300)
        plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def pca_plot_feature_loadings(pca_data, phases, save_path, fs=7):
    if len(pca_data) == 1 and pca_data[0].phase[0] == phases[0] and pca_data[0].phase[1] == phases[1]:
        pca_loadings = pca_data[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    else:
        raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")

    # build display names
    display_names = [short_names.get(f, f) for f in pca_loadings.index]

    # build heatmap DataFrame: rows=PCs, columns=features
    heatmap_df = pca_loadings.copy()
    heatmap_df.index = pca_loadings.index  # original feature keys
    heatmap_df.columns = [f"PC{idx + 1}" for idx in range(heatmap_df.shape[1])]
    heatmap_df.columns.name = "Principal Component"
    heatmap_df.index = display_names     # pretty feature labels
    heatmap_df = heatmap_df.T            # now rows=PCs, cols=features

    # --- Raw loadings plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        cmap='coolwarm',
        cbar_kws={'label': 'Loading'},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    ax.set_title(f'PCA Feature Loadings: {phases[0]} vs {phases[1]}', fontsize=fs)
    ax.set_xlabel('Features', fontsize=fs)
    ax.set_ylabel('Principal Component', fontsize=fs)
    ax.tick_params(axis='x', rotation=90, labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fn = f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}_raw.{ext}'
        fig.savefig(os.path.join(save_path, fn), dpi=300)
    plt.close(fig)

    # prepare a colormap corresponding to the positive half of the original coolwarm
    full_cmap = plt.cm.get_cmap('coolwarm', 256)
    half = np.linspace(0.5, 1.0, 128)
    pos_cmap = ListedColormap(full_cmap(half))

    # compute maximum absolute loading for scaling
    max_abs = heatmap_df.abs().values.max()

    # --- Absolute loadings plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df.abs(),
        cmap=pos_cmap,
        vmin=0,
        vmax=max_abs,
        cbar_kws={'label': 'Absolute Loading'},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    ax.set_title(f'PCA Absolute Feature Loadings: {phases[0]} vs {phases[1]}', fontsize=fs)
    ax.set_xlabel('Features', fontsize=fs)
    ax.set_ylabel('Principal Component', fontsize=fs)
    ax.tick_params(axis='x', rotation=90, labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fn = f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}_absolute.{ext}'
        fig.savefig(os.path.join(save_path, fn), dpi=300)
    plt.close(fig)



def plot_top_features_per_PC(pca_data, feature_data, feature_data_notscaled, phases, stride_numbers, condition, save_path, n_top_features=5, fs=7):
    """
    Find the top features which load onto each principal component.
    """
    if len(pca_data) == 1 and pca_data[0].phase[0] == phases[0] and pca_data[0].phase[1] == phases[1]:
        pca_loadings = pca_data[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    else:
        raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")

    top_features = {}
    for pc in pca_loadings.columns:
        top_features[pc] = pca_loadings.loc(axis=1)[pc].abs().nlargest(n_top_features).index.tolist()

    for s in stride_numbers:
        feats = feature_data.loc(axis=0)[s]
        feats_raw = feature_data_notscaled.loc(axis=0)[s]
        for pc in pca_loadings.columns:
            top_feats_pc = top_features[pc]
            top_feats_loadings = pca_loadings.loc(axis=1)[pc].loc(axis=0)[top_feats_pc]
            top_feats_data = feats.loc(axis=1)[top_feats_pc]
            top_feats_display_names = [short_names.get(f, f) for f in top_feats_pc]

            mask_p1, mask_p2 = gu.get_mask_p1_p2(top_feats_data, phases[0], phases[1])
            feats_p1 = top_feats_data.loc(axis=0)[mask_p1]
            feats_p2 = top_feats_data.loc(axis=0)[mask_p2]
            # feats_raw_p1 = feats_raw.loc(axis=0)[mask_p1]
            # feats_raw_p2 = feats_raw.loc(axis=0)[mask_p2]

            plot_top_feat_descriptives(feats_p1, feats_p2, top_feats_pc, top_feats_loadings, pc, phases, s,
                                       top_feats_display_names, save_path, fs=fs)

            # # Plot the raw features
            # common_x = np.arange(160)
            # fig, axs = plt.subplots(n_top_features, 1, figsize=(4, 7))
            # for i, feat in enumerate(top_feats_pc):
            #     mice_feats = np.zeros((len(condition_specific_settings[condition]['global_fs_mouse_ids']), len(common_x)))
            #     for midx, mouse_id in enumerate(condition_specific_settings[condition]['global_fs_mouse_ids']):
            #         interpolated_data = np.interp(common_x, feats.loc(axis=0)[mouse_id].loc(axis=1)[feat].index,
            #                                         feats.loc(axis=0)[mouse_id].loc(axis=1)[feat].values)
            #         smoothed_data = median_filter(interpolated_data, size=11)
            #
            #         ms = pu.get_marker_style_mice(mouse_id)
            #         axs[i].plot(common_x, smoothed_data, label=mouse_id, alpha=0.3, color='grey', markersize=3, zorder=10, marker=ms, linewidth=0.5)
            #
            #         mice_feats[midx] = smoothed_data
            #     # find the median across mice
            #     median_feats = np.median(mice_feats, axis=0)
            #     axs[i].plot(common_x, median_feats, alpha=0.7, color='black', zorder=10, linewidth=1)


def plot_top_feat_descriptives(feats_p1, feats_p2, top_feats_pc, top_feats_loadings, pc, phases, s, top_feats_display_names, save_path, fs=7):
    feats_permouse_medians_p1 = feats_p1.groupby(level=0).median()
    feats_permouse_medians_p2 = feats_p2.groupby(level=0).median()

    # Prepare data lists for the boxplots
    data_p1 = [feats_permouse_medians_p1[feat].values for feat in top_feats_pc]
    data_p2 = [feats_permouse_medians_p2[feat].values for feat in top_feats_pc]

    # Get the phase colours and darker versions for the median and whiskers
    p1_color = pu.get_color_phase(phases[0])
    p2_color = pu.get_color_phase(phases[1])
    dark_color_p1 = pu.darken_color(p1_color, 0.7)
    dark_color_p2 = pu.darken_color(p2_color, 0.7)

    # Boxplot properties for phases
    boxprops_p1 = dict(facecolor=p1_color, color=p1_color)
    boxprops_p2 = dict(facecolor=p2_color, color=p2_color)
    medianprops_p1 = dict(color=dark_color_p1, linewidth=2)
    whiskerprops_p1 = dict(color=dark_color_p1, linewidth=1.5, linestyle='-')
    medianprops_p2 = dict(color=dark_color_p2, linewidth=2)
    whiskerprops_p2 = dict(color=dark_color_p2, linewidth=1.5, linestyle='-')

    x = np.arange(len(top_feats_pc))
    width = 0.35
    bar_multiple = 0.6
    positions_p1 = x - width / 2
    positions_p2 = x + width / 2

    # Create a figure with 3 subplots (loadings, phase values, and phase difference)
    fig, axs = plt.subplots(4, 1, figsize=(5, 9))

    ### Subplot 0: Feature Loadings
    axs[0].bar(x, top_feats_loadings, width * bar_multiple, alpha=0.7, color='k')
    axs[0].set_xticks([])
    axs[0].set_ylabel('Feature Loadings', fontsize=fs)
    axs[0].set_ylim(-0.6, 0.6)
    axs[0].set_yticks(np.arange(-0.5, 0.6, 0.5))

    ### Subplot 1: Phase Z-scored Feature Values
    # Boxplots for p1 and p2
    axs[2].boxplot(data_p1, positions=positions_p1, widths=width * bar_multiple,
                   patch_artist=True, boxprops=boxprops_p1,
                   medianprops=medianprops_p1, whiskerprops=whiskerprops_p1, showcaps=False, showfliers=False)
    axs[2].boxplot(data_p2, positions=positions_p2, widths=width * bar_multiple,
                   patch_artist=True, boxprops=boxprops_p2,
                   medianprops=medianprops_p2, whiskerprops=whiskerprops_p2, showcaps=False, showfliers=False)

    # Plot scatter lines connecting each mouse's data between phases:
    for midx in feats_permouse_medians_p1.index:
        axs[2].plot([positions_p1, positions_p2],
                    [feats_permouse_medians_p1.loc[midx], feats_permouse_medians_p2.loc[midx]],
                    'o-', alpha=0.3, color='grey', markersize=3, zorder=10)
    p1_patch = mpatches.Patch(color=p1_color, label=f'{phases[0]}')
    p2_patch = mpatches.Patch(color=p2_color, label=f'{phases[1]}')
    axs[2].set_xticklabels('')
    axs[2].set_ylabel('Z-scored Feature', fontsize=fs)
    # legend labels
    axs[2].legend(handles=[p1_patch, p2_patch], fontsize=fs, loc='upper right', bbox_to_anchor=(1.2, 1),
                  title='Phase', title_fontsize=fs)

    ### Subplot 2: Projection of features on PCA
    weighted_features_p1 = [feature * loading for feature, loading in zip(data_p1, top_feats_loadings.values)]
    weighted_features_p2 = [feature * loading for feature, loading in zip(data_p2, top_feats_loadings.values)]
    # boxplots
    axs[1].boxplot(weighted_features_p1, positions=positions_p1, widths=width * bar_multiple,
                   patch_artist=True, boxprops=boxprops_p1,
                   medianprops=medianprops_p1, whiskerprops=whiskerprops_p1, showcaps=False, showfliers=False)
    axs[1].boxplot(weighted_features_p2, positions=positions_p2, widths=width * bar_multiple,
                   patch_artist=True, boxprops=boxprops_p2,
                   medianprops=medianprops_p2, whiskerprops=whiskerprops_p2, showcaps=False, showfliers=False)
    # Convert the lists of weighted feature arrays into DataFrames.
    # Each column corresponds to a feature and each row to a mouse.
    weighted_df_p1 = pd.DataFrame(np.column_stack(weighted_features_p1),
                                  index=feats_permouse_medians_p1.index,
                                  columns=top_feats_pc)
    weighted_df_p2 = pd.DataFrame(np.column_stack(weighted_features_p2),
                                  index=feats_permouse_medians_p1.index,
                                  columns=top_feats_pc)

    # Plot paired translucent line plots for each mouse (like in ax[1])
    for midx in weighted_df_p1.index:
        axs[1].plot([positions_p1, positions_p2],
                    [weighted_df_p1.loc[midx].values, weighted_df_p2.loc[midx].values],
                    'o-', alpha=0.3, color='grey', markersize=3, zorder=10)
    axs[1].set_ylabel('PC Projection', fontsize=fs)
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].set_yticks(np.arange(-0.5, 0.6, 0.5))
    axs[1].set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    axs[1].set_xticklabels('')

    ### Subplot 3: Phase Difference (p2 - p1)
    # Compute the per-mouse differences for each feature
    feats_diff = feats_permouse_medians_p2 - feats_permouse_medians_p1
    # Create a list of arrays (one per feature) for the differences
    data_diff = [feats_diff[feat].values for feat in top_feats_pc]

    # Choose a neutral color for the phase differences
    diff_color = "#888888"
    dark_diff_color = pu.darken_color(diff_color, 0.7)
    boxprops_diff = dict(facecolor=diff_color, color=diff_color)
    medianprops_diff = dict(color=dark_diff_color, linewidth=2)
    whiskerprops_diff = dict(color=dark_diff_color, linewidth=1.5, linestyle='-')

    # Plot the boxplot for differences (one box per feature at positions given by x)
    axs[3].boxplot(data_diff, positions=x, widths=width * bar_multiple,
                   patch_artist=True, boxprops=boxprops_diff,
                   medianprops=medianprops_diff, whiskerprops=whiskerprops_diff, showcaps=False, showfliers=False)

    # Plot scatter points for each mouse; add a slight random jitter for visibility
    for i, feat in enumerate(top_feats_pc):
        diff_vals = feats_diff[feat].values
        # x_vals = np.random.normal(loc=x[i], scale=0.04, size=len(diff_vals))  # jitter for clarity
        axs[3].scatter([x[i]] * len(diff_vals), diff_vals, color='k', alpha=0.5, s=3, zorder=10)
    axs[3].set_xticklabels(top_feats_display_names, fontsize=fs, rotation=90)
    axs[3].set_ylabel('Phase2 - Phase1', fontsize=fs)

    for ax in axs:
        #ax.set_xticks(x)
        if ax != axs[1] and ax != axs[0]:
            ax.set_ylim(-1.4, 1.4)
            ax.set_yticks(np.arange(-1, 1.1, 1))
            ax.set_yticklabels(np.arange(-1, 1.1, 1), fontsize=fs)
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        #ax.tick_params(axis='y', labelsize=fs)
        ax.set_xlim(-0.5, len(top_feats_pc) - 0.5)
        ax.tick_params(axis='y', which='both', left=True, labelsize=fs)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis='y', which='minor', length=4, width=1, color='k')

    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.21, hspace=0.1)

    # title
    plt.suptitle(pc, fontsize=fs)

    plt.savefig(os.path.join(save_path, f'PCA_top_features_{phases[0]}vs{phases[1]}_stride{s}_{pc}.png'), dpi=300)
    plt.savefig(os.path.join(save_path, f'PCA_top_features_{phases[0]}vs{phases[1]}_stride{s}_{pc}.svg'), dpi=300)
    plt.close()









