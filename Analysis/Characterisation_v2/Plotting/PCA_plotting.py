import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu

def plot_scree(pca, p1, p2, stride, condition, save_path):
    """
    Plot and save the scree plot.
    """
    from Analysis.Tools.config import (global_settings)
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label='Individual Explained Variance')
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', linewidth=2, color='red', label='Cumulative Explained Variance')
    plt.title(f'Scree Plot with Cumulative Explained Variance\n{p1} vs {p2} - {condition} - {stride}\n'
              f'Num chosen PCs: {global_settings["pcs_to_use"]}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(False)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.png"), dpi=300)
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
    print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    if pca.n_components_ >= 3:
        print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

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
        plt.close()


def pca_plot_feature_loadings(pca_data, phases, save_path, fs=7):
    if len(pca_data) == 1 and pca_data[0].phase[0] == phases[0] and pca_data[0].phase[1] == phases[1]:
        pca_loadings = pca_data[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    else:
        raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")

    display_names = []
    for f in pca_loadings.index:
        display_names.append(short_names.get(f,f))

    fig, ax = plt.subplots(figsize=(7, 10))
    for pc in pca_loadings.columns:
        single_pc_loadings = pca_loadings.loc(axis=1)[pc]
        ax.plot(single_pc_loadings.values, single_pc_loadings.index, label=pc, marker='o', markersize=2)
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=fs)
    ax.set_xlabel('Loadings')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=fs)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.invert_yaxis()

    plt.subplots_adjust(left=0.4, right=0.9, top=0.99, bottom=0.1)

    plt.savefig(os.path.join(save_path, f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}.png'), dpi=300)
    plt.savefig(os.path.join(save_path, f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}.svg'), dpi=300)
    plt.close()

def plot_top_features_per_PC(pca_data, feature_data, phases, stride_numbers, save_path, n_top_features=5, fs=7):
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
        for pc in pca_loadings.columns:
            top_feats_pc = top_features[pc]
            top_feats_loadings = pca_loadings.loc(axis=1)[pc].loc(axis=0)[top_feats_pc]
            top_feats_data = feats.loc(axis=1)[top_feats_pc]
            top_feats_display_names = [short_names.get(f, f) for f in top_feats_pc]

            mask_p1, mask_p2 = gu.get_mask_p1_p2(top_feats_data, phases[0], phases[1])
            feats_p1 = top_feats_data.loc(axis=0)[mask_p1]
            feats_p2 = top_feats_data.loc(axis=0)[mask_p2]

            feats_permouse_medians_p1 = feats_p1.groupby(level=0).median()
            feats_permouse_medians_p2 = feats_p2.groupby(level=0).median()

            # Prepare data lists for the boxplots
            data_p1 = [feats_permouse_medians_p1[feat].values for feat in top_feats_pc]
            data_p2 = [feats_permouse_medians_p2[feat].values for feat in top_feats_pc]

            # Get the phase colours and darker versions for the median and whiskers
            p1_color, p2_color = pu.get_colors(phases)
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
            fig, axs = plt.subplots(3, 1, figsize=(5, 7))

            ### Subplot 0: Feature Loadings
            axs[0].bar(x, top_feats_loadings, width * bar_multiple, alpha=0.7, color='k')
            axs[0].set_xticks([])
            axs[0].set_ylabel('Feature Loadings', fontsize=fs)

            ### Subplot 1: Phase Z-scored Feature Values
            # Boxplots for p1 and p2
            axs[1].boxplot(data_p1, positions=positions_p1, widths=width * bar_multiple,
                           patch_artist=True, boxprops=boxprops_p1,
                           medianprops=medianprops_p1, whiskerprops=whiskerprops_p1, showcaps=False)
            axs[1].boxplot(data_p2, positions=positions_p2, widths=width * bar_multiple,
                           patch_artist=True, boxprops=boxprops_p2,
                           medianprops=medianprops_p2, whiskerprops=whiskerprops_p2, showcaps=False)

            # Plot scatter lines connecting each mouse's data between phases:
            for midx in feats_permouse_medians_p1.index:
                axs[1].plot([positions_p1, positions_p2],
                            [feats_permouse_medians_p1.loc[midx], feats_permouse_medians_p2.loc[midx]],
                            'o-', alpha=0.3, color='grey', markersize=3, zorder=10)
            p1_patch = mpatches.Patch(color=p1_color, label=f'{phases[0]}')
            p2_patch = mpatches.Patch(color=p2_color, label=f'{phases[1]}')
            axs[1].set_xticklabels('')
            axs[1].set_ylabel('Z-scored feature Value', fontsize=fs)
            # legend labels
            axs[1].legend(handles=[p1_patch, p2_patch], fontsize=fs, loc='upper right', bbox_to_anchor=(1.2, 1),
                          title='Phase', title_fontsize=fs)

            ### Subplot 2: Phase Difference (p2 - p1)
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
            axs[2].boxplot(data_diff, positions=x, widths=width * bar_multiple,
                           patch_artist=True, boxprops=boxprops_diff,
                           medianprops=medianprops_diff, whiskerprops=whiskerprops_diff, showcaps=False)

            # Plot scatter points for each mouse; add a slight random jitter for visibility
            for i, feat in enumerate(top_feats_pc):
                diff_vals = feats_diff[feat].values
                #x_vals = np.random.normal(loc=x[i], scale=0.04, size=len(diff_vals))  # jitter for clarity
                axs[2].scatter([x[i]]*len(diff_vals), diff_vals, color='black', alpha=0.5, s=10, zorder=10)
            axs[2].set_xticklabels(top_feats_display_names, fontsize=fs, rotation=45)
            axs[2].set_ylabel('Phase Difference (p2 - p1)', fontsize=fs)


            for ax in axs:
                ax.set_xticks(x)
                ax.set_ylim(-1.1, 1.1)
                ax.set_yticks(np.arange(-1, 1.1, 0.5))
                ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.grid(False)
                ax.tick_params(axis='y', labelsize=fs)
                ax.set_xlim(-0.5, len(top_feats_pc) - 0.5)


            plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.2, hspace=0.05)

            plt.savefig(os.path.join(save_path, f'PCA_top_features_{phases[0]}vs{phases[1]}_stride{s}_{pc}.png'), dpi=300)
            plt.close()









