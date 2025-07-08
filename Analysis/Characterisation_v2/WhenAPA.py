import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from scipy.stats import pearsonr

from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu
from Helpers.Config_23 import *

# load LH pred data
LH_MultiFeatPath = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
LH_preprocessed_data_file_path = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_stride_0_preprocessed_data_file_path = r"H:\Characterisation_v2\LH_LHpcsonly_res_0_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_pred_path = f"{LH_MultiFeatPath}\\pca_predictions_APAChar_LowHigh.pkl"
LH_pca_path = f"{LH_MultiFeatPath}\\pca_APAChar_LowHigh.pkl"
with open(LH_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_preprocessed_data = data['feature_data'] # this is the normalised! :)
with open(LH_stride_0_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_stride0_preprocessed_data = data['feature_data']
with open(LH_pred_path, 'rb') as f:
    LH_pred_data = pickle.load(f)
with open(LH_pca_path, 'rb') as f:
    LH_pca_data = pickle.load(f)

class WhenAPA:
    def __init__(self, LH_feature_data, LH_feature_data_s0, LH_pred_data, LH_pca_data, base_dir):
        self.LH_feature_data = LH_feature_data
        self.LH_feature_data_s0 = LH_feature_data_s0
        self.LH_pred = LH_pred_data
        self.LH_pca = LH_pca_data
        self.base_dir = base_dir
        self.strides = [-1, -2, -3]

    def plot_accuracy_of_each_stride_model(self, fs=7):
        # APPROACH: If Δ accuracy > 0 significantly, decoding is better than chance given that mouse’s data structure.

        # Collect accuracy data
        all_stride_accs = {}
        for s in self.strides:
            stride_mice_names = [pred.mouse_id for pred in self.LH_pred if pred.stride == s]

            stride_accs = [pred.cv_acc for pred in self.LH_pred if pred.stride == s]
            accs_df = pd.DataFrame(stride_accs, index=stride_mice_names)

            stride_null_accs = [pred.null_acc_circ for pred in self.LH_pred if pred.stride == s]
            null_df = pd.DataFrame(stride_null_accs, index=stride_mice_names)

            delta_accs = accs_df.mean(axis=1) - null_df.mean(axis=1)

            all_stride_accs[s] = delta_accs

        # Combine into single DataFrame
        all_stride_accs_by_stride = pd.concat(all_stride_accs).reset_index()
        all_stride_accs_by_stride.columns = ['Stride', 'Mouse', 'Accuracy']
        all_stride_accs_by_stride['Stride_abs'] = all_stride_accs_by_stride['Stride'].abs()
        df = all_stride_accs_by_stride

        # Define stride order and palette
        stride_order = sorted(df['Stride_abs'].unique())
        palette = {s: pu.get_color_stride(-s) for s in stride_order}

        fig, ax = plt.subplots(figsize=(4, 3))

        # Plot violin plots
        sns.violinplot(data=df, x='Stride_abs', y='Accuracy', ax=ax, inner=None,
                       linewidth=0.5, palette=palette, order=stride_order)

        # Calculate significance and overlay scatter points
        for i, s in enumerate(stride_order):
            accs = df[df['Stride_abs'] == s]['Accuracy']

            accs_mean = accs.mean()

            # T-test
            t_stat, p_value = ttest_1samp(accs, 0)
            print(f"Stride {int(-s)}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

            # Scatter black points with jitter
            jitter = np.random.uniform(-0.1, 0.1, size=len(accs))
            x_jittered = i + jitter
            ax.scatter(x_jittered, accs, marker='o', color='black', s=7)

            # Significance stars
            star = ''
            if p_value < 0.001 and accs_mean > 0:
                star = '***'
            elif p_value < 0.01 and accs_mean > 0:
                star = '**'
            elif p_value < 0.05 and accs_mean > 0:
                star = '*'

            if star:
                y_max = accs.max()
                ax.text(i, y_max + 0.12, star, ha='center', va='bottom', fontsize=fs)

        # Chance line
        ax.axhline(y=0, color='gray', linestyle='--')

        # Formatting
        # set y limits as they are
        # --- Automatic y‐axis limits & 5 nice ticks ---
        y_vals = df['Accuracy']
        data_min, data_max = y_vals.min(), y_vals.max()
        span = data_max - data_min
        pad = span * 0.5  # 10% padding
        y_lo = data_min - pad
        y_hi = data_max + pad

        # round to nearest tenth
        y_lo = np.floor(y_lo * 10) / 10
        y_hi = np.ceil(y_hi * 10) / 10

        # generate 5 ticks
        yticks = np.linspace(y_lo, y_hi, 5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=fs)

        ax.set_xlim(-0.5, len(stride_order) - 0.5)
        ax.set_xticks(range(len(stride_order)))
        ax.set_xticklabels([-s for s in stride_order], fontsize=fs)
        ax.set_xlabel('Stride', fontsize=fs)
        ax.set_ylabel('CV Accuracy', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.invert_xaxis()
        fig.tight_layout()

        savepath = os.path.join(self.base_dir, 'WhenAPA_StrideModelAccuracy')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()


    def _compute_corr_matrix(self, df1, df2):
        """Pearson‐r matrix for two [runs × PCs] DataFrames."""
        n = global_settings['pcs_to_use']
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    M[i, j], _ = pearsonr(df1.iloc[:, i], df2.iloc[:, j])
                except:
                    M[i, j] = np.nan
        return M

    def _compute_mean_r(self, baseline_stride, compare_stride, run_type, pcs_byStride_interpolated, eps=1e-6):
        """Returns the mean Pearson‐r matrix (after Fisher transform) for one stride comparison and run_type."""
        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()
        zs = []
        for midx in mice:
            pc1 = pcs_base.loc[midx]
            pc2 = pcs_cmp.loc[midx]
            if run_type == 'APAlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]
            C = self._compute_corr_matrix(pc1, pc2)
            C_clipped = np.clip(C, -1 + eps, 1 - eps)
            zs.append(np.arctanh(C_clipped))
        return np.tanh(np.nanmean(zs, axis=0))

    def _compute_delta_stats(self, baseline_stride, compare_stride,
                             run_type, pcs_byStride_interpolated,
                             eps=1e-6):
        """
        Returns (mean_delta, stars) where
          mean_delta[i,j] = average over mice of (r_cmp[i,j] - r_base[i,j])
          stars[i,j]     = '', '*', '**', or '***' depending on p-value
        of a one-sample t-test that Δr ≠ 0.
        """
        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        deltas = []
        for midx in mice:
            pc1 = pcs_base.loc[midx]
            pc2 = pcs_cmp.loc[midx]
            if run_type == 'APAlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
                pc1 = pc1.loc[runs];
                pc2 = pc2.loc[runs]

            # compute and clip both matrices
            Cb = np.clip(self._compute_corr_matrix(pc1, pc1), -1 + eps, 1 - eps)
            Cc = np.clip(self._compute_corr_matrix(pc1, pc2), -1 + eps, 1 - eps)
            deltas.append(Cc - Cb)

        deltas = np.stack(deltas, axis=0)  # shape (n_mice, n_pcs, n_pcs)
        mean_delta = np.nanmean(deltas, axis=0)

        # now compute p-values & stars
        n = mean_delta.shape[0]
        stars = np.full((n, n), '', dtype=object)
        for i in range(n):
            for j in range(n):
                vals = deltas[:, i, j]
                # omit NaNs
                vals = vals[~np.isnan(vals)]
                if len(vals) > 1:
                    _, p = ttest_1samp(vals, 0.0)
                    if p < 0.001:
                        stars[i, j] = '***'
                    elif p < 0.01:
                        stars[i, j] = '**'
                    elif p < 0.05:
                        stars[i, j] = '*'
        return mean_delta, stars

    def _plot_heatmap(self, mat, label, run_type, fs=7, suffix=""):
        """Generic heatmap plotting + saving."""
        pcs = global_settings['pcs_to_use']
        xl = [f"PC{i + 1} ({label})" for i in range(pcs)]
        yl = [f"PC{i + 1} (-1)" for i in range(pcs)]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mat, vmin=-1, vmax=1, cmap='coolwarm',
                    xticklabels=xl, yticklabels=yl,
                    cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title(f"{'Δ ' if suffix else ''}Stride -1 vs {label}  ({run_type})", fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.figure.axes[-1].tick_params(labelsize=fs)
        plt.tight_layout()
        fname = f"CorrPCs_{run_type}_Stride{label}{suffix}"
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                        bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_scatter(self, baseline_stride, compare_stride, pcs_byStride_interpolated, fs=7):
        """Scatter PC means (APA2 vs Wash2) for one stride comparison."""
        apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
        wash_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        apa_color = pu.get_color_phase('APA2')
        wash_color = pu.get_color_phase('Wash2')

        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        for pc_idx in range(global_settings['pcs_to_use']):
            fig, ax = plt.subplots(figsize=(2, 2))
            apa_x, apa_y = [], []
            wash_x, wash_y = [], []

            for midx in mice:
                pc1 = pcs_base.loc[midx]
                pc2 = pcs_cmp.loc[midx]
                # means
                m1a = pc1.loc[apa_runs].iloc[:, pc_idx].mean()
                m2a = pc2.loc[apa_runs].iloc[:, pc_idx].mean()
                m1w = pc1.loc[wash_runs].iloc[:, pc_idx].mean()
                m2w = pc2.loc[wash_runs].iloc[:, pc_idx].mean()
                apa_x.append(m1a)
                apa_y.append(m2a)
                wash_x.append(m1w)
                wash_y.append(m2w)

            ax.scatter(apa_x, apa_y, marker='x', label='APAlate', alpha=0.7, color=apa_color)
            ax.scatter(wash_x, wash_y, marker='o', label='Washlate', alpha=0.7, color=wash_color)
            ax.set_xlabel(f'PC{pc_idx + 1} ({baseline_stride})', fontsize=fs)
            ax.set_ylabel(f'PC{pc_idx + 1} ({compare_stride})', fontsize=fs)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, fontsize=fs)
            ax.tick_params(labelsize=fs)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.85)

            fname = f"ScatterAPA_vs_Wash_PC{pc_idx + 1}_Stride{compare_stride}"
            for ext in ('png', 'svg'):
                fig.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                            bbox_inches='tight', dpi=300)
            plt.close()

    def _plot_delta_heatmap(self, mat, stars, label, run_type, fs=7, suffix=""):
        pcs = global_settings['pcs_to_use']
        xl = [f"PC{i + 1} ({label})" for i in range(pcs)]
        yl = [f"PC{i + 1} (-1)" for i in range(pcs)]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mat, vmin=-1, vmax=1, cmap='coolwarm',
                    xticklabels=xl, yticklabels=yl, cbar_kws={'label': 'Δ Pearson r'},
                    annot=False, ax=ax)

        # overlay stars
        for i in range(pcs):
            for j in range(pcs):
                star = stars[i, j]
                if star:
                    ax.text(j + 0.5, i + 0.5, star,
                            ha='center', va='center', fontsize=fs, color='black')

        ax.set_title(f"Δ Stride -1 vs {label}  ({run_type})", fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.figure.axes[-1].tick_params(labelsize=fs)
        plt.tight_layout()

        fname = f"CorrPCs_{run_type}_Stride{label}{suffix}_delta"
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                        bbox_inches='tight', dpi=300)
        plt.close()

    def plot_corr_pcs_heatmap(self, fs=7):
        pca = self.LH_pca[0].pca
        pcs_byStride = {}
        pcs_byStride_interpolated = {}
        for s in self.strides:
            stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
            mice_names = stride_feature_data.index.get_level_values('MouseID').unique()

            pcs_byMouse = {}
            pcs_byMouse_interpolated = {}
            for midx in mice_names:
                pc_df = pd.DataFrame(
                    index=np.arange(160),
                    columns=[f'PC{i + 1}' for i in range(global_settings['pcs_to_use'])]
                )
                pc_interp_df = pc_df.copy()

                mouse_data = stride_feature_data.loc[midx]
                pcs = pca.transform(mouse_data)[:, :global_settings['pcs_to_use']]
                runs = mouse_data.index.get_level_values('Run').unique()

                pcs_interp = np.array([
                    np.interp(np.arange(160), runs, pcs[:, i])
                    for i in range(global_settings['pcs_to_use'])
                ]).T

                pc_df.loc[runs, :] = pcs
                pc_interp_df.loc[:, :] = pcs_interp

                pcs_byMouse[midx] = pc_df
                pcs_byMouse_interpolated[midx] = pc_interp_df

            pcs_byStride[s] = pd.concat(pcs_byMouse)
            pcs_byStride_interpolated[s] = pd.concat(pcs_byMouse_interpolated)

        # compute and plot
        for rt in ['All runs', 'APAlate']:
            # 1) raw heatmaps + scatters
            for s in (-1, -2, -3):
                mean_r = self._compute_mean_r(-1, s, rt, pcs_byStride_interpolated)
                self._plot_heatmap(mean_r, s, rt)
                self._plot_scatter(-1, s, pcs_byStride_interpolated, fs=fs)

            # 2) delta heatmaps with stars
            mean2, stars2 = self._compute_delta_stats(-1, -2, rt, pcs_byStride_interpolated)
            mean3, stars3 = self._compute_delta_stats(-1, -3, rt, pcs_byStride_interpolated)

            self._plot_delta_heatmap(mean2, stars2, '-2 minus -1', rt, fs=fs, suffix='_Delta2')
            self._plot_delta_heatmap(mean3, stars3, '-3 minus -1', rt, fs=fs, suffix='_Delta3')

    def plot_line_pcs_apa_vs_wash(self, fs=7):
        apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
        wash_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        strides = [-3, -2, -1] # removed 0 from here

        pcs_byStride = {}
        pca = self.LH_pca[0].pca

        # --- Extract mean per mouse per stride ---
        for s in strides:
            if s != 0:
                stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
            else:
                stride_feature_data = self.LH_feature_data_s0.loc(axis=0)[s]
            mice_names = stride_feature_data.index.get_level_values('MouseID').unique()

            pcs_byMouse = {}
            for midx in mice_names:
                mouse_data = stride_feature_data.loc[midx]
                pcs = pca.transform(mouse_data)[:, :global_settings['pcs_to_use']]
                pc_df = pd.DataFrame(pcs, index=mouse_data.index.get_level_values('Run'),
                                     columns=[f'PC{i + 1}' for i in range(global_settings['pcs_to_use'])])
                pcs_byMouse[midx] = pc_df

            pcs_byStride[s] = pcs_byMouse

        # --- Plot per PC ---
        for pc_idx in range(global_settings['pcs_to_use']):
            fig, ax1 = plt.subplots(figsize=(4, 3))
            ax2 = ax1.twinx()

            apa_means = []
            wash_means = []
            delta_means = []

            apa_CIs = []
            wash_CIs = []
            delta_CIs = []

            for s in strides:
                pcs_mouse = pcs_byStride[s]
                apa_vals = []
                wash_vals = []
                diff_vals = []

                for midx, pc_df in pcs_mouse.items():
                    # APA
                    apa_mouse_vals = pc_df.loc[pc_df.index.isin(apa_runs), f'PC{pc_idx + 1}']
                    apa_mean = apa_mouse_vals.mean()
                    apa_vals.append(apa_mean)

                    # Wash
                    wash_mouse_vals = pc_df.loc[pc_df.index.isin(wash_runs), f'PC{pc_idx + 1}']
                    wash_mean = wash_mouse_vals.mean()
                    wash_vals.append(wash_mean)

                    # Difference
                    diffs = apa_mean - wash_mean
                    diff_vals.append(diffs)

                # Mean across mice
                apa_means.append(np.nanmean(apa_vals))
                wash_means.append(np.nanmean(wash_vals))
                delta_means.append(np.nanmean(diff_vals))

                # Confidence intervals
                apa_CI = np.nanstd(apa_vals) / np.sqrt(len(apa_vals)) * 1.96  # 95% CI
                wash_CI = np.nanstd(wash_vals) / np.sqrt(len(wash_vals)) * 1.96  # 95% CI
                delta_CI = np.nanstd(diff_vals) / np.sqrt(len(diff_vals)) * 1.96  # 95% CI
                apa_CIs.append(apa_CI)
                wash_CIs.append(wash_CI)
                delta_CIs.append(delta_CI)

            stride_labels = [-s for s in strides]  # display as positive stride numbers

            # --- Plot APA and Wash on left y-axis ---
            apa_color = pu.get_color_phase('APA2')
            wash_color = pu.get_color_phase('Wash2')
            ax1.plot(stride_labels, apa_means, color=apa_color, marker='o', label='APA')
            ax1.plot(stride_labels, wash_means, color=wash_color, marker='o', label='Wash')
            ax1.set_ylabel('PC Value (mean)', fontsize=fs, color='black')
            ax1.tick_params(axis='y', labelsize=fs)
            ax1.set_xlabel('Stride', fontsize=fs)
            ax1.set_xticks(stride_labels)
            ax1.set_xticklabels(stride_labels, fontsize=fs)
            ax1.tick_params(axis='x', labelsize=fs)
            ax1.invert_xaxis()

            # --- Plot delta on right y-axis ---
            ax2.plot(stride_labels, delta_means, color='teal', marker='o', linestyle='--', label='Delta (APA-Wash)')
            ax2.set_ylabel('Delta APA-Wash', fontsize=fs, color='teal')
            ax2.tick_params(axis='y', labelsize=fs, colors='teal')
            ax2.spines['top'].set_visible(False)

            # --- Legends ---
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=fs, loc='best')

            ax1.set_title(f'PC{pc_idx + 1} APA vs Wash vs Delta', fontsize=fs)
            ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)
            ax1.spines['top'].set_visible(False)

            fig.tight_layout()

            # --- Save ---
            savepath = os.path.join(self.base_dir, f'LinePlot_PC{pc_idx + 1}_APA_Wash_Delta')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close()

            # --- Collect delta values into DataFrame for heatmap ---
            delta_df = pd.DataFrame(index=[f'PC{i + 1}' for i in range(global_settings['pcs_to_use'])],
                                    columns=[s for s in strides])

            for pc_idx in range(global_settings['pcs_to_use']):
                delta_means = []
                for s in strides:
                    pcs_mouse = pcs_byStride[s]
                    apa_vals = []
                    wash_vals = []
                    diff_vals = []
                    for midx, pc_df in pcs_mouse.items():
                        # APA
                        apa_mouse_vals = pc_df.loc[pc_df.index.isin(apa_runs), f'PC{pc_idx + 1}']
                        apa_mean = apa_mouse_vals.mean()
                        apa_vals.append(apa_mean)

                        # Wash
                        wash_mouse_vals = pc_df.loc[pc_df.index.isin(wash_runs), f'PC{pc_idx + 1}']
                        wash_mean = wash_mouse_vals.mean()
                        wash_vals.append(wash_mean)

                        # Difference
                        diffs = apa_mean - wash_mean
                        diff_vals.append(diffs)

                    mean_delta = np.nanmean(diff_vals)
                    # Mean across mice
                    # delta = np.nanmean(apa_vals) - np.nanmean(wash_vals)
                    delta_df.loc[f'PC{pc_idx + 1}', s] = mean_delta  # -s to match stride labels

            # --- Convert to float ---
            delta_df = delta_df.astype(float)

            # --- Plot heatmap ---
            fig, ax = plt.subplots(figsize=(6, 8))
            sns.heatmap(delta_df, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                        cbar_kws={'label': 'Delta (APA - Wash)'}, vmin=-2, vmax=2)
            ax.set_xlabel('Stride', fontsize=fs)
            ax.set_ylabel('PC', fontsize=fs)
            ax.set_title('Delta (APA - Wash) Heatmap', fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)

            plt.tight_layout()

            # --- Save ---
            savepath = os.path.join(self.base_dir, f'DeltaHeatmap_APA_Wash')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close()




def main():
    save_dir = r"H:\Characterisation_v2\WhenAPA_1"
    os.path.exists(save_dir) or os.makedirs(save_dir)

    # Initialize the WhenAPA class with LH prediction data
    when_apa = WhenAPA(LH_preprocessed_data, LH_stride0_preprocessed_data, LH_pred_data, LH_pca_data, save_dir)

    # Plot the accuracy of each stride model
    when_apa.plot_accuracy_of_each_stride_model()
    # when_apa.plot_corr_pc_weights_heatmap()
    when_apa.plot_corr_pcs_heatmap()
    when_apa.plot_line_pcs_apa_vs_wash()



if __name__ == '__main__':
    main()




