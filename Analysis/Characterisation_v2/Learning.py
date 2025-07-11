import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from scipy.stats import ttest_rel
from scipy.stats import t


from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu
from Helpers.Config_23 import *
from Helpers.utils import Utils
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg

# load LH pred data
LH_MultiFeatPath = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
LH_preprocessed_data_file_path = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_stride_0_preprocessed_data_file_path = r"H:\Characterisation_v2\LH_LHpcsonly_res_0_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_pred_path = f"{LH_MultiFeatPath}\\pca_predictions_APAChar_LowHigh.pkl"
LH_pca_path = f"{LH_MultiFeatPath}\\pca_APAChar_LowHigh.pkl"
with open(LH_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_preprocessed_data = data['feature_data']
with open(LH_stride_0_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_stride0_preprocessed_data = data['feature_data']
with open(LH_pred_path, 'rb') as f:
    LH_pred_data = pickle.load(f)
with open(LH_pca_path, 'rb') as f:
    LH_pca_data = pickle.load(f)

class Learning:
    def __init__(self, feature_data, feature_data_s0, pred_data, pca_data, save_dir, disturb_pred_file: str):
        self.LH_feature_data = feature_data
        self.LH_feature_data_s0 = feature_data_s0
        self.LH_pred = LH_pred_data
        self.LH_pca = LH_pca_data
        self.base_dir = save_dir
        self.strides = [-1, -2, -3]

        self.all_learners_learning = {}
        self.all_learners_extinction = {}
        self.fast_learning = {}
        self.slow_learning = {}
        self.fast_extinction = {}
        self.slow_extinction = {}

        with open(disturb_pred_file, 'rb') as f:
            self.disturb_pred_data = pickle.load(f)

    def get_disturb_preds(self):
        """
        Returns: { mouse_id: (x_vals, y_pred_array) }
        for stride==0 entries in self.disturb_pred_data
        """
        out = {}
        for p in self.disturb_pred_data:
            if p.stride == 0:
                x = np.array(list(p.x_vals))
                y = p.y_pred[0]
                out[p.mouse_id] = (x, y)
        return out


    def get_pcs(self, s=-1):
        pca = self.LH_pca[0].pca
        stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
        mice_names = stride_feature_data.index.get_level_values('MouseID').unique()

        pcs_bymouse = {}
        for midx in mice_names:
            # Get pcs from feature data: # n runs x pcs
            mouse_data = stride_feature_data.loc[midx]
            pcs = pca.transform(mouse_data)
            pcs = pcs[:, :global_settings['pcs_to_use']]
            run_vals = mouse_data.index.get_level_values('Run').unique()
            pcs_bymouse[midx] = {'pcs': pcs, 'run_vals': run_vals}
        return pcs_bymouse

    def get_preds(self, pcwise, s=-1):
        pcs_bymouse = self.get_pcs(s=s)

        stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
        mice_names = stride_feature_data.index.get_level_values('MouseID').unique()
        goal_runs = np.arange(160)

        # get pcs from feature data and pca for each mouse
        if pcwise:
            preds_byPC_bymouse = {f'PC{i + 1}': {} for i in range(global_settings['pcs_to_use'])}
        else:
            preds_byPC_bymouse = {}

        for midx in mice_names:
            pcs = pcs_bymouse[midx]['pcs']
            run_vals = pcs_bymouse[midx]['run_vals']

            # Get regression weights: 1 x pcs
            pc_pred_weights = [pred.pc_weights for pred in self.LH_pred if pred.stride == s and pred.mouse_id == midx][
                0]

            if pcwise:
                for pc_idx in range(min(global_settings['pcs_to_use'], pc_pred_weights.shape[1])):
                    pc_weights = pc_pred_weights[0][pc_idx]
                    y_pred = np.dot(pcs, pc_weights.T).squeeze()
                    y_pred_pc = y_pred[:, pc_idx]

                    y_pred_interp = np.interp(goal_runs, run_vals, y_pred_pc)

                    # normalise with max abs
                    max_abs = max(abs(y_pred_interp.min()), abs(y_pred_interp.max()))
                    y_pred_interp_norm = y_pred_interp / max_abs

                    preds_byPC_bymouse[f'PC{pc_idx + 1}'][midx] = y_pred_interp_norm
            else:
                # Get overall prediction for the mouse
                y_pred = np.dot(pcs, pc_pred_weights.T).squeeze()
                y_pred_interp = np.interp(goal_runs, run_vals, y_pred)
                # normalise with max abs
                max_abs = max(abs(y_pred_interp.min()), abs(y_pred_interp.max()))
                y_pred_interp_norm = y_pred_interp / max_abs
                preds_byPC_bymouse[midx] = y_pred_interp_norm
        return preds_byPC_bymouse

    def fit_pcwise_regression_model(self, chosen_pcs, s=-1):
        pcs_bymouse = self.get_pcs(s=s)

        mice_names = [midx for midx in pcs_bymouse.keys()]

        goal_runs = np.arange(160)

        pc_reg_models = {f'PC{pc}': {} for pc in chosen_pcs}
        for pc in chosen_pcs:
            pc_index = pc - 1  # Convert to zero-based index
            for midx in mice_names:
                current_pcs = pcs_bymouse[midx]['pcs'][:, pc_index]
                current_run_vals = pcs_bymouse[midx]['run_vals']

                apa_mask = np.isin(current_run_vals, expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
                wash_mask = np.isin(current_run_vals, expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])

                pcs_apa = current_pcs[apa_mask]
                pcs_wash = current_pcs[wash_mask]
                Xdr = np.concatenate([pcs_apa, pcs_wash])
                Xdr = Xdr.reshape(1, -1)  # Reshape to 2D array for regression

                Xdr_long = current_pcs.reshape(1, -1)  # Reshape to 2D array for regression

                y_reg = np.concatenate([np.ones_like(pcs_apa), np.zeros_like(pcs_wash)])

                null_acc_circ = reg.compute_null_accuracy_circular(Xdr_long, y_reg, apa_mask, wash_mask)

                num_folds = 10
                w, bal_acc, cv_acc, w_folds = reg.compute_regression(Xdr, y_reg, folds=num_folds)

                pred = np.dot(Xdr_long.T, w).squeeze()

                # Store the regression model for this PC and mouse
                pc_reg_models[f'PC{pc}'][midx] = {
                    'x_vals': current_run_vals,
                    'y_pred': pred,
                    'weights': w,
                    'balanced_accuracy': bal_acc,
                    'cv_accuracy': cv_acc,
                    'w_folds': w_folds,
                    'null_accuracy_circ': null_acc_circ
                }

        return pc_reg_models

    def plot_total_predictions_x_trial(self, fast_slow=None, s=-1, fs=7, smooth_window=3):
        # Get the total (not PC-wise) predictions for each mouse
        preds_by_mouse = self.get_preds(pcwise=False, s=s)

        # Decide which mice to plot
        speed_ordered_mice = list(self.all_learners_learning.keys())
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = speed_ordered_mice

        goal_runs = np.arange(160)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()

        # Loop over mice
        for midx in mice_names:
            y = preds_by_mouse[midx]
            # interpolate if you need (should already be length 160)
            # smooth
            y_smooth = median_filter(y, size=smooth_window, mode='nearest')
            # normalize
            max_abs = np.max(np.abs(y_smooth))
            y_norm = y_smooth / max_abs

            # styling
            c = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
            ls = pu.get_line_style_mice(midx)

            ax.plot(goal_runs + 1, y_norm, color=c, linestyle=ls, linewidth=1, label=str(midx))

        # Formatting exactly like your other plots
        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Total Prediction', fontsize=fs)
        title = 'Total predictions'
        if fast_slow is not None:
            title += f' ({fast_slow})'
        ax.set_title(title, fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10','60','110','135','160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # Save
        fname = f'TotalPreds_Stride{s}'
        if fast_slow:
            fname = f'TotalPreds_{fast_slow}_Stride{s}'
        savepath = os.path.join(self.base_dir, fname)
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{savepath}.svg", dpi=300, bbox_inches='tight')
        plt.close()


    def plot_line_important_pcs_preds_x_trial(self, chosen_pcs, fast_slow=None, s=-1, fs=7, smooth_window=3):
        pc_reg_models = self.fit_pcwise_regression_model(chosen_pcs, s=s)
        goal_runs = np.arange(160)

        speed_ordered_mice = list(self.all_learners_learning.keys())

        # Select mouse IDs
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = speed_ordered_mice

        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()
        for pc in chosen_pcs:
            fig_pc, ax_pc = plt.subplots(figsize=(6, 4))
            pc_index = pc - 1  # Convert to zero-based index
            pc_data = pc_reg_models[f'PC{pc}']

            pc_preds_df = pd.DataFrame(index=goal_runs, columns=mice_names)
            for midx in mice_names:
                current_preds = pc_data[midx]['y_pred']
                current_run_vals = pc_data[midx]['x_vals']

                # Interpolate to match goal runs
                current_preds_interp = np.interp(goal_runs, current_run_vals, current_preds)
                # smooth
                current_preds_smooth = median_filter(current_preds_interp, size=smooth_window, mode='nearest')
                # normalise with max abs
                max_abs = max(abs(current_preds_smooth.min()), abs(current_preds_smooth.max()))
                current_preds_norm = current_preds_smooth / max_abs

                mouse_color = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
                mouse_ls = pu.get_line_style_mice(midx)

                ax_pc.plot(goal_runs + 1, current_preds_norm, color=mouse_color, linestyle=mouse_ls, linewidth=1, label=f'PC{pc} - {midx}')

                pc_preds_df[midx] = current_preds_norm

            # Format and save individual PC plot
            ax_pc.set_xlabel('Trial number', fontsize=fs)
            ax_pc.set_ylabel('Normalised Prediction', fontsize=fs)
            ax_pc.set_title(f'PC{pc} predictions', fontsize=fs)
            ax_pc.set_xlim(0, 160)
            ax_pc.set_xticks([10, 60, 110, 135, 160])
            ax_pc.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
            ax_pc.tick_params(axis='both', which='major', labelsize=fs)
            ax_pc.spines['top'].set_visible(False)
            ax_pc.spines['right'].set_visible(False)
            ax_pc.legend(fontsize=fs, frameon=False)
            plt.tight_layout()
            # --- Save individual PC plot ---
            if fast_slow is not None:
                savepath = os.path.join(self.base_dir, f'PC{pc}_preds_{fast_slow}_Stride{s}')
            else:
                savepath = os.path.join(self.base_dir, f'PC{pc}_preds_Stride{s}')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig_pc)

            # --- compute mean and 95% CI across mice ---
            n_mice = pc_preds_df.shape[1]
            mean_series = pc_preds_df.mean(axis=1)
            sem_series = pc_preds_df.std(axis=1, ddof=1) / np.sqrt(n_mice)
            ci_mult = t.ppf(0.975, df=n_mice - 1)  # two-tailed 95%
            ci_series = sem_series * ci_mult

            pc_color = pu.get_color_pc(pc_index)
            # plot shaded CI
            ax.fill_between(goal_runs + 1,
                            mean_series - ci_series,
                            mean_series + ci_series,
                            color=pc_color,
                            alpha=0.08,
                            linewidth=0)
            # plot the mean line
            ax.plot(goal_runs + 1,
                    mean_series,
                    color=pc_color,
                    linewidth=1,
                    label=f'PC{pc}')

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)
        plt.tight_layout()
        # --- Save ---
        if fast_slow is not None:
            savepath = os.path.join(self.base_dir, f'PC_preds_{chosen_pcs}_{fast_slow}_Stride{s}')
        else:
            savepath = os.path.join(self.base_dir, f'PC_preds_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()




    def plot_line_important_pcs_x_trial(self, chosen_pcs, fast_slow=None, s=-1, fs=7, smooth_window=15):
        pcs_bymouse = self.get_pcs(s=s)

        speed_ordered_mice = list(self.all_learners_learning.keys())

        # Select mouse IDs
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = [midx for midx in speed_ordered_mice]

        goal_runs = np.arange(160)

        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()

        for pc in chosen_pcs:
            fig_pc, ax_pc = plt.subplots(figsize=(6, 4))

            pc_index = pc - 1  # Convert to zero-based index
            pcs_df = pd.DataFrame(columns=goal_runs, index=mice_names)
            for midx in mice_names:
                current_pcs = pcs_bymouse[midx]['pcs'][:, pc_index]
                current_run_vals = pcs_bymouse[midx]['run_vals']

                # Interpolate to match goal runs
                current_pcs_interp = np.interp(goal_runs, current_run_vals, current_pcs)
                # smooth
                current_pcs_smooth = median_filter(current_pcs_interp, size=smooth_window, mode='nearest')
                # normalise with max abs
                max_abs = max(abs(current_pcs_smooth.min()), abs(current_pcs_smooth.max()))
                current_pcs_norm = current_pcs_smooth / max_abs
                pcs_df.loc(axis=0)[midx] = current_pcs_norm

                mouse_color = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
                mouse_ls = pu.get_line_style_mice(midx)
                ax_pc.plot(goal_runs + 1, current_pcs_norm, color=mouse_color, linestyle=mouse_ls, linewidth=1, label=f'PC{pc} - {midx}')

            # Format and save individual PC plot
            ax_pc.set_xlabel('Trial number', fontsize=fs)
            ax_pc.set_ylabel('Normalised PC', fontsize=fs)
            ax_pc.set_title(f'PC{pc} values', fontsize=fs)
            ax_pc.set_xlim(0, 160)
            ax_pc.set_xticks([10, 60, 110, 135, 160])
            ax_pc.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
            ax_pc.tick_params(axis='both', which='major', labelsize=fs)
            ax_pc.spines['top'].set_visible(False)
            ax_pc.spines['right'].set_visible(False)
            ax_pc.legend(fontsize=fs, frameon=False)
            plt.tight_layout()
            # --- Save individual PC plot ---
            if fast_slow is not None:
                savepath = os.path.join(self.base_dir, f'PC{pc}_vals_{fast_slow}_Stride{s}')
            else:
                savepath = os.path.join(self.base_dir, f'PC{pc}_vals_Stride{s}')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig_pc)

            # --- compute mean and 95% CI across mice ---
            n_mice = pcs_df.shape[0]
            mean_vals = pcs_df.mean(axis=0)
            sem_vals = pcs_df.std(axis=0, ddof=1) / np.sqrt(n_mice)
            ci_mult = t.ppf(0.975, df=n_mice - 1)
            ci_vals = sem_vals * ci_mult

            pc_color = pu.get_color_pc(pc_index)
            # shaded CI
            ax.fill_between(goal_runs + 1,
                            mean_vals - ci_vals,
                            mean_vals + ci_vals,
                            color=pc_color,
                            alpha=0.08,
                            linewidth=0)
            # mean line
            ax.plot(goal_runs + 1,
                    mean_vals,
                    color=pc_color,
                    linewidth=1,
                    label=f'PC{pc}')

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised PC', fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)
        plt.tight_layout()
        # --- Save ---
        if fast_slow is not None:
            savepath = os.path.join(self.base_dir, f'PC_vals_{chosen_pcs}_{fast_slow}_Stride{s}')
        else:
            savepath = os.path.join(self.base_dir, f'PC_vals_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()





    def plot_line_important_pc_predictions_x_trial(self, chosen_pcs, s=-1, fs=7, smooth_window=15):
        preds_byPC_bymouse = self.get_preds(pcwise=True, s=s)

        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()

        for idx, pc_idx in enumerate(chosen_pcs):
            pc_name = f'PC{pc_idx}'
            preds_all_mice = preds_byPC_bymouse.get(pc_name, {})

            # Convert to DataFrame for easy mean calculation
            preds_df = pd.DataFrame.from_dict(preds_all_mice, orient='index')  # rows=mice, cols=runs

            # Calculate mean across mice (ignore NaNs)
            preds_mean = preds_df.mean(axis=0)

            # Smooth
            #preds_mean_smooth = pd.Series(preds_mean).rolling(window=smooth_window, center=True, min_periods=1).mean()

            # Smooth median filter
            preds_mean_smooth = median_filter(preds_mean, size=smooth_window, mode='nearest')

            # --- Plot ---
            pc_color = pu.get_color_pc(pc_idx - 1, n_pcs=global_settings['pcs_to_use'])
            ax.plot(np.arange(160)[10:], preds_mean_smooth[10:], color=pc_color, linewidth=1, label=pc_name)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_title(f'Smooth window={smooth_window}', fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10,60,110,135,160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'Predictions_perPC_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def find_learned_trials(self, smoothing=None, phase='learning', fast_threshold=15, learned_threshold=5, fs=7):
        preds = self.get_preds(pcwise=False, s=-1)
        if phase == 'learning':
            phase_mask = np.isin(np.arange(160), expstuff['condition_exp_runs']['APAChar']['Extended']['APA'])
        elif phase == 'extinction':
            phase_mask = np.isin(np.arange(160), expstuff['condition_exp_runs']['APAChar']['Extended']['Washout'])
        phase_preds = {
            mouse_id: preds[mouse_id][phase_mask] for mouse_id in preds.keys()
        }

        fig, ax = plt.subplots(figsize=(15, 10))
        # Plot predictions for each mouse
        smooth_window = 3
        for mouse_id, pred in phase_preds.items():
            smooth_pred = median_filter(pred, size=smooth_window, mode='nearest')
            ax.plot(np.arange(len(pred))+1, smooth_pred, label=mouse_id, marker='o')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Smoothed Prediction', fontsize=fs)
        ax.set_title(f'Smooth window={smooth_window}', fontsize=fs)
        savepath = os.path.join(self.base_dir, f'SMOOTHED_Predictions_{phase}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        # Find trial where prediction is above 0 for threshold runs
        learned_trials_xMice = {}
        for mouse_id, pred in phase_preds.items():
            if smoothing:
                # Apply smoothing
                pred = median_filter(pred, size=smoothing, mode='nearest')
            if phase == 'learning':
                learned_trials = np.where(pred > 0)[0] + 1  # +1 to convert from index to trial number
            elif phase == 'extinction':
                learned_trials = np.where(pred < 0)[0] + 1
            learned_blocks = Utils().find_blocks(learned_trials, gap_threshold=1, block_min_size=learned_threshold)
            if len(learned_blocks) > 0:
                learned_trial = learned_blocks[0][0]  # Take the first block's first trial
                learned_trials_xMice[mouse_id] = learned_trial
            else:
                learned_trials_xMice[mouse_id] = None  # No plateau found for this mouse

        setattr(self, f'learned_trials_{phase}', learned_trials_xMice)

        # pick out fast and slow learners relative to 'fast_threshold'
        fast_learners = {mouse_id: trial for mouse_id, trial in learned_trials_xMice.items() if trial is not None and trial <= fast_threshold}
        slow_learners = {mouse_id: trial for mouse_id, trial in learned_trials_xMice.items() if trial is not None and trial > fast_threshold}

        # get top 3
        fast_learners_sorted = dict(sorted(fast_learners.items(), key=lambda item: item[1]))
        slow_learners_sorted = dict(sorted(slow_learners.items(), key=lambda item: item[1]))
        all_learners_sorted = dict(sorted(learned_trials_xMice.items(), key=lambda item: item[1]))

        fast_learners_top3 = {k: v for k, v in list(fast_learners_sorted.items())[-3:]}
        slow_learners_top3 = {k: v for k, v in list(slow_learners_sorted.items())[-3:]}

        setattr(self, f'fast_{phase}', fast_learners_top3)
        setattr(self, f'slow_{phase}', slow_learners_top3)
        setattr(self, f'all_learners_{phase}', all_learners_sorted)

        # --- Prepare ordered table data with group labels merged ---
        table_rows = []

        # Fast learners block
        fast_mouse_ids = list(fast_learners_sorted.keys())
        for idx, mouse_id in enumerate(fast_mouse_ids):
            table_rows.append([mouse_id, fast_learners_sorted[mouse_id]])

        # Slow learners block
        slow_mouse_ids = list(slow_learners_sorted.keys())
        for idx, mouse_id in enumerate(slow_mouse_ids):
            table_rows.append([mouse_id, slow_learners_sorted[mouse_id]])

        # --- Plot table ---
        fig, ax = plt.subplots(figsize=(4, 2 + len(table_rows) * 0.2))
        ax.axis('off')

        table = ax.table(cellText=table_rows,
                         colLabels=['Mouse ID', f'Trials to {phase.capitalize()}'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fs)
        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'LearnedTrialsTable_{phase}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_fast_vs_slow_learners_pcs(self, fast_slow, chosen_pcs, s=-1, fs=7, smooth_window=15):
        preds_byPC_bymouse = self.get_preds(pcwise=True, s=s)

        # Select mouse IDs
        if fast_slow == 'fast':
            selected_mice = self.fast_learning.keys()
        elif fast_slow == 'slow':
            selected_mice = self.slow_learning.keys()
        else:
            raise ValueError("fast_slow must be 'fast' or 'slow'")

        fig, ax = plt.subplots(figsize=(6, 4))

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

        goal_runs = np.arange(160)

        for idx, pc_idx in enumerate(chosen_pcs):
            pc_name = f'PC{pc_idx}'

            # Extract predictions only for selected mice
            preds_selected_mice = {mouse: preds for mouse, preds in preds_byPC_bymouse[pc_name].items() if
                                   mouse in selected_mice}

            # Convert to DataFrame for easy mean calculation
            preds_df = pd.DataFrame.from_dict(preds_selected_mice, orient='index')

            # Calculate mean across mice (ignore NaNs)
            preds_mean = preds_df.mean(axis=0)

            # Smooth with median filter
            preds_mean_smooth = median_filter(preds_mean, size=smooth_window, mode='nearest')

            # Plot
            pc_color = pu.get_color_pc(pc_idx - 1, n_pcs=global_settings['pcs_to_use'])
            ax.plot(goal_runs[10:], preds_mean_smooth[10:], color=pc_color, linewidth=1, label=pc_name)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_title(f'{fast_slow.capitalize()} learners (smooth window={smooth_window})',
                     fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'Predictions_{fast_slow}_learners_perPC_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_prediction_delta(self):
        preds_byPC_bymouse = self.get_preds(pcwise=False, s=-1)

        # Calculate deltas
        delta_preds = {}
        for mouse_id, preds in preds_byPC_bymouse.items():
            smooth_preds = median_filter(preds, size=10, mode='nearest')
            delta_preds[mouse_id] = np.diff(smooth_preds)

        delta_df = pd.DataFrame.from_dict(delta_preds, orient='index')

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for mouse_id, delta in delta_df.iterrows():
            ax.plot(np.arange(len(delta)) + 1, delta, label=mouse_id, marker='o')
        plt.close()

    def plot_prediction_per_day(self,fs=7):
        preds_byPC_bymouse = self.get_preds(pcwise=False, s=-1)

        # smooth preds
        smooth_window = 3
        # smooth_preds = {}
        # for mouse_id, preds in preds_byPC_bymouse.items():
        #     smooth_preds[mouse_id] = median_filter(preds, size=smooth_window, mode='nearest')

        # Convert to DataFrame for easy plotting
        preds_df = pd.DataFrame.from_dict(preds_byPC_bymouse, orient='index')
        preds_df.index.name = 'Mouse ID'

        # plot
        # fig, ax = plt.subplots(figsize=(8, 6))
        # for mouse_id, pred in preds_df.iterrows():
        #     ax.plot(np.arange(len(pred)) + 1, pred, label=mouse_id, marker='o')

        day_runs = [np.arange(5, 20), np.arange(35, 50), np.arange(75, 90), np.arange(105, 120), np.arange(115, 130)]
        day_dividers = [40, 80, 120]

        fig, ax = plt.subplots(figsize=(8, 4))

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

        for day_idx, day in enumerate(day_runs):
            data = preds_df.loc(axis=1)[day]
            #data_smoothed = data.apply(lambda x: median_filter(x, size=smooth_window, mode='nearest'))
            data_av = data.mean(axis=0)

            # Determine x positions for this chunk within full x-axis
            x_positions = day + 1  # if your trial numbers start at 1

            for mouse_id, pred in data.iterrows():
                ax.plot(x_positions, pred, label=mouse_id, alpha=0.5)
            ax.plot(x_positions, data_av, color='black', linewidth=2, label=f'Average Chunk {day_idx + 1}')

        # Add vertical lines for day dividers
        for divider in day_dividers:
            ax.axvline(x=divider, color='grey', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Prediction', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_xlim(1, 160)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Optionally adjust legend
        # ax.legend(fontsize=fs, frameon=False, ncol=2)

        save_path = os.path.join(self.base_dir, 'Predictions_AllChunks')
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        speed_ordered_mice = list(self.all_learners_learning.keys())

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        boxy = 1
        height = 0.02
        patch1 = plt.axvspan(xmin=10, xmax=60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
        patch2 = plt.axvspan(xmin=60, xmax=110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
        patch3 = plt.axvspan(xmin=110, xmax=135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
        patch4 = plt.axvspan(xmin=135, xmax=160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
        patch1.set_clip_on(False)
        patch2.set_clip_on(False)
        patch3.set_clip_on(False)
        patch4.set_clip_on(False)

        all_chunk_means = []

        post5_means_days = {}
        for day_idx, day in enumerate(day_runs):
            data = preds_df.loc(axis=1)[day]
            data_smoothed = data.apply(lambda x: median_filter(x, size=smooth_window, mode='nearest'))

            # Define windows: -5 (first 5), +5 (next 5), +6-10 (final 5)
            baseline_window = day[:5]
            post5_window = day[5:10]
            post10_window = day[10:15]

            baseline_means = data_smoothed.loc[:, baseline_window].mean(axis=1)
            post5_means = data_smoothed.loc[:, post5_window].mean(axis=1)
            post5_means_days[day_idx] = post5_means
            post10_means = data_smoothed.loc[:, post10_window].mean(axis=1)

            # Store for later if needed
            all_chunk_means.append((baseline_means, post5_means, post10_means))

            # Significance tests
            tstat_a, pval_a = ttest_rel(baseline_means, post5_means)
            tstat_b, pval_b = ttest_rel(baseline_means, post10_means)

            print(f"Chunk {day_idx + 1} comparison a (-5 vs +5): p={pval_a:.3f}")
            print(f"Chunk {day_idx + 1} comparison b (-5 vs +6-10): p={pval_b:.3f}")

            # Calculate window midpoints for x positions
            x_baseline = baseline_window.mean() + 1  # +1 if trials start at 1
            x_post5 = post5_window.mean() + 1
            x_post10 = post10_window.mean() + 1

            xpos = np.array([x_baseline, x_post5, x_post10])
            data_to_plot = [baseline_means, post5_means, post10_means]

            # Scatter per mouse
            xpos_counter = 0
            for x, d in zip(xpos, data_to_plot):
                for mouse in speed_ordered_mice:
                    marker = pu.get_marker_style_mice(mouse)
                    color = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
                    ax2.scatter([x], d[mouse], color=color, marker=marker, s=15, alpha=0.6, linewidth=0, label= mouse if day_idx == 0 and xpos_counter < len(speed_ordered_mice) else "")
                    xpos_counter += 1

            means = [d.mean() for d in data_to_plot]
            ax2.plot(xpos, means, color='k', linewidth=2)

            # Add brackets with stars for clarity
            y_max = max([d.max() for d in data_to_plot]) + 0.05
            line_height = 0.02  # height of the bracket line

            # Comparison a: baseline vs +5
            # if pval_a < 0.05:
            if pval_a < 0.001:
                stars = '*' * 3  # Use three stars for p < 0.001
            elif pval_a < 0.01:
                stars = '*' * 2  # Use two stars for p < 0.01
            elif pval_a < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            # Draw bracket
            ax2.plot([x_baseline, x_baseline, x_post5, x_post5],
                     [y_max, y_max + line_height, y_max + line_height, y_max],
                     lw=0.8, c='k')
            # Add star
            ax2.text((x_baseline + x_post5) / 2, y_max + line_height + 0.01, stars, ha='center', fontsize=fs)


            # Comparison b: baseline vs +6-10
            if pval_b < 0.001:
                stars = '*' * 3
            elif pval_b < 0.01:
                stars = '*' * 2
            elif pval_b < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            y_max_b = y_max + 0.07  # offset second bracket if needed
            ax2.plot([x_baseline, x_baseline, x_post10, x_post10],
                     [y_max_b, y_max_b + line_height, y_max_b + line_height, y_max_b],
                     lw=0.8, c='k')
            ax2.text((x_baseline + x_post10) / 2, y_max_b + line_height + 0.01, stars, ha='center', fontsize=fs)

        # compare post 5 across chunks 1,2,3
        post5_means_df = pd.DataFrame(post5_means_days)
        tstats = []
        pvals = []
        for pairs in [(0, 1), (0, 2), (1, 2)]:
            # compare post5 means between chunks
            tstat, pval = ttest_rel(post5_means_df.loc(axis=1)[pairs[0]], post5_means_df.loc(axis=1)[pairs[1]])
            tstats.append(tstat)
            pvals.append(pval)

        # plot where significant
        for idx, (pair, pval) in enumerate(zip([(0, 1), (0, 2), (1, 2)], pvals)):
            if pval < 0.001:
                stars = '*' * 3
            elif pval < 0.01:
                stars = '*' * 2
            elif pval < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            y_max = max(post5_means_df.max()) + 0.15 * (idx + 1)  # offset each bracket higher

            # Use actual x_post5 positions from each chunk for correct plotting
            x1 = day_runs[pair[0]][5:10].mean() + 1  # +1 if trials are 1-based
            x2 = day_runs[pair[1]][5:10].mean() + 1

            ax2.plot([x1, x1, x2, x2],
                     [y_max, y_max + 0.02, y_max + 0.02, y_max],
                     lw=0.8, c='r')
            ax2.text((x1 + x2) / 2, y_max + 0.03, stars, ha='center', fontsize=fs, color='r')


        ax2.vlines(x=day_dividers, ymin=-1, ymax=1, color='grey', linestyle='--', linewidth=0.5)

        # Finalise plot
        ax2.set_xlabel('Trial number', fontsize=fs)
        ax2.set_ylabel('Mean Prediction', fontsize=fs)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(1, 160)
        ax2.set_xticks([10, 40, 60, 80, 110, 120, 135, 160])
        ax2.set_ylim(-1, 1)
        ax2.tick_params(axis='both', which='major', labelsize=fs)
        ax2.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

        save_path2 = os.path.join(self.base_dir, 'Predictions_WindowMeans_Significance_ScatterTrueX')
        plt.savefig(f"{save_path2}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path2}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_learning_by_extinction_scatter(self, fs=7):
        learning_trials = getattr(self, 'learned_trials_learning', {})
        extinction_trials = getattr(self, 'learned_trials_extinction', {})

        speed_ordered_mice = list(self.all_learners_learning.keys())

        learn_x_extinct_df = pd.DataFrame({
            'Mouse ID': list(learning_trials.keys()),
            'Learning Trial': list(learning_trials.values()),
            'Extinction Trial': [extinction_trials.get(mouse, np.nan) for mouse in learning_trials.keys()]
        })

        # sort into speed_ordered_mice order
        learn_x_extinct_df = learn_x_extinct_df.set_index('Mouse ID').reindex(speed_ordered_mice).reset_index()
        # make MouseID the index
        learn_x_extinct_df.set_index('Mouse ID', inplace=True)

        learn_extinct_diff = learn_x_extinct_df['Extinction Trial'] - learn_x_extinct_df['Learning Trial']
        _, p = ttest_1samp(learn_extinct_diff, 0)

        fig, (ax, ax_diff) = plt.subplots(1, 2, figsize=(5, 4), gridspec_kw={'width_ratios': [3, 1]})

        for mouse_id in learn_x_extinct_df.index:
            grp = learn_x_extinct_df.loc[mouse_id]
            mkr = pu.get_marker_style_mice(mouse_id)
            col = pu.get_color_mice(mouse_id, speedordered=speed_ordered_mice)
            ax.scatter(
                grp['Learning Trial'],
                grp['Extinction Trial'],
                marker=mkr,
                s=50,
                color=col,
                label=str(mouse_id),
                linewidth=0,
            )

        # plot equality line to show where learning and extinction trials are equal given unequal scale
        max_val = min(learn_x_extinct_df['Learning Trial'].max(), learn_x_extinct_df['Extinction Trial'].max())
        ax.plot([0, max_val], [0, max_val], color='grey', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Learning Time (Trials)', fontsize=fs)
        ax.set_ylabel('Extinction Time (Trials)', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(0, 5 * round(xmax/5))
        ax.set_ylim(0, 5 * round(ymax/5))
        ax.set_xticks(np.arange(0, xmax + 1, 20))
        ax.set_yticks(np.arange(0, ymax + 1, 5))

        # Scatter plot difference (right)
        for mouse in learn_extinct_diff.index:
            diff = learn_extinct_diff[mouse]
            mkr = pu.get_marker_style_mice(mouse)
            col = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
            jitter = np.random.normal(0, 0.01, size=1)  # small jitter for visibility
            ax_diff.scatter(1 + jitter, diff, marker=mkr, s=50, edgecolor=col, facecolor='none', linewidth=1)
        # plot mean difference and 95% CI
        mean_diff = learn_extinct_diff.mean()
        ci = 1.96 * learn_extinct_diff.std() / np.sqrt(len(learn_extinct_diff))
        ax_diff.errorbar(
            [1], mean_diff, yerr=ci, fmt='o', color='black', markersize=5, capsize=3, label='Mean ± 95% CI', elinewidth=0.5
        )

        # Format difference plot
        ax_diff.set_xlim(0.95, 1.05)
        ax_diff.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax_diff.set_ylabel('Extinction - Learning Trials', fontsize=fs)
        ax_diff.tick_params(axis='x', labelsize=fs)
        ax_diff.tick_params(axis='y', labelsize=fs)
        ax_diff.spines['top'].set_visible(False)
        ax_diff.spines['right'].set_visible(False)

        # Set x-ticks as mouse ids for clarity, but you can tweak if too crowded
        ax_diff.set_xticks([1])
        ax_diff.set_xticklabels(['Diff'], fontsize=fs)

        # Add significance text
        if p < 0.05:
            # Convert p-value to stars
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            else:
                stars = '*'
            sig_text = stars
        else:
            sig_text = 'n.s.'

        # Place significance text above scatter plot
        ylim = ax_diff.get_ylim()
        ax_diff.text(
            0.5, 1, sig_text,
            ha='center', va='top', fontsize=fs, transform=ax_diff.transAxes
        )

        plt.tight_layout()

        savepath = os.path.join(self.base_dir, 'Learning_vs_Extinction_Scatter')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_disturbance_by_prediction_interpolated(self,
                                                    phase: str = 'APA2',
                                                    desired_diameter: int = 3):
        """
        One plot of Low vs High APA tertiles → mean disturbance,
        coloring each mouse by fast (blue), slow (red), or other (gray).
        """
        # 1) grab APA & disturbance preds
        apa_preds = self.get_preds(pcwise=False, s=-1)  # {mouse: 160-array}
        disturb_preds = self.get_disturb_preds()  # {mouse: (x_vals, y_vals)}

        # 2) which mice?
        mice = list(apa_preds.keys())
        fast = set(self.fast_learning.keys())
        slow = set(self.slow_learning.keys())

        # 3) which trial-indices belong to this phase?
        runs = np.array(expstuff['condition_exp_runs']
                        ['APAChar']['Extended'][phase])

        # 4) set up figure
        fig, ax = plt.subplots(figsize=(4, 4))
        low_vals, high_vals, colors, diffs = [], [], [], []

        for m in mice:
            # --- APA for this mouse ---
            y_apa = apa_preds[m]
            trials = np.arange(len(y_apa))
            phase_mask = np.isin(trials, runs)
            x_apa = trials[phase_mask]
            y_apa = y_apa[phase_mask]

            # --- interp disturbance to the APA trial points ---
            x_dist, y_dist = disturb_preds[m]
            y_dist_on_apa = np.interp(x_apa, x_dist, y_dist)

            # --- tertiles of APA strength ---
            order = np.argsort(y_apa)
            third = len(order) // 3
            bot_idx = order[:third]
            top_idx = order[-third:]

            bot_mean = y_dist_on_apa[bot_idx].mean()
            top_mean = y_dist_on_apa[top_idx].mean()

            low_vals.append(bot_mean)
            high_vals.append(top_mean)
            diff = bot_mean - top_mean
            diffs.append(diff)

            # choose color
            if m in fast:
                c = 'blue'
            elif m in slow:
                c = 'red'
            else:
                c = 'gray'
            colors.append(c)

            # draw the per‐mouse connector
            ax.plot([1, 2], [bot_mean, top_mean],
                    marker='o', markersize=desired_diameter,
                    color=c, alpha=0.5)

        # 5) scatter the Δ at x=3
        s = np.pi * (desired_diameter / 2) ** 2
        jit = np.random.normal(0, 0.02, size=len(diffs))
        ax.scatter(3 + jit, diffs, s=s,
                   c=colors, edgecolors='none', alpha=0.7)

        # 6) legend handles
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color='blue', label='Fast learners'),
            mpatches.Patch(color='red', label='Slow learners'),
            mpatches.Patch(color='gray', label='Others'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=8)

        # 7) styling & save
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low', 'High', 'Δ'], fontsize=9)
        ax.set_ylabel('Disturbance prediction', fontsize=10)
        ax.set_title(f"{phase} — all mice", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        fname = f"Disturb_vs_Pred_interp_{phase}_all"
        fig.savefig(os.path.join(self.base_dir, fname + '.png'), dpi=300)
        fig.savefig(os.path.join(self.base_dir, fname + '.svg'), dpi=300)
        plt.close(fig)


def main():
    save_dir = r"H:\Characterisation_v2\Learning_1"
    os.path.exists(save_dir) or os.makedirs(save_dir)

    chosen_pcs = [1, 3, 7]
    other_pcs = [5, 6, 8]
    chosen_pcs_extended = [1, 3, 5, 6, 7, 8]

    # Initialize the WhenAPA class with LH prediction data
    learning = Learning(LH_preprocessed_data, LH_stride0_preprocessed_data, LH_pred_data, LH_pca_data, save_dir,
                        disturb_pred_file=r"H:\Characterisation\LH_subtract_res_0_APA1APA2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl"
                        )

    learning.find_learned_trials(smoothing=None, phase='learning')
    learning.find_learned_trials(smoothing=None, phase='extinction')
    print("Fast learners in learning phase:", learning.fast_learning)
    print("Slow learners in learning phase:", learning.slow_learning)
    print("Fast learners in extinction phase:", learning.fast_extinction)
    print("Slow learners in extinction phase:", learning.slow_extinction)

    learning.plot_learning_by_extinction_scatter()

    learning.plot_prediction_delta()
    learning.plot_prediction_per_day(fs=7)

    learning.plot_total_predictions_x_trial(smooth_window=3)
    learning.plot_total_predictions_x_trial(fast_slow='fast', smooth_window=3)
    learning.plot_total_predictions_x_trial(fast_slow='slow', smooth_window=3)

    learning.plot_line_important_pcs_x_trial(chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='fast', chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='slow', chosen_pcs=chosen_pcs, smooth_window=10)

    learning.plot_line_important_pcs_preds_x_trial(chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='fast', chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='slow', chosen_pcs=chosen_pcs, smooth_window=10)

    # repeat with extended PCs
    learning.plot_line_important_pcs_x_trial(chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='fast', chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='slow', chosen_pcs=other_pcs, smooth_window=10)

    learning.plot_line_important_pcs_preds_x_trial(chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='fast', chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='slow', chosen_pcs=other_pcs, smooth_window=10)




    learning.fit_pcwise_regression_model(chosen_pcs=chosen_pcs)

    for phase in ['APA1','APA2','APA']:
        learning.plot_disturbance_by_prediction_interpolated(phase)


if __name__ == '__main__':
    main()
