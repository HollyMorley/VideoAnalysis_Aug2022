import pickle
import pandas as pd
from scipy.linalg import null_space
import os
import itertools
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
from itertools import combinations
from scipy.stats import wilcoxon
import scipy.stats as stats
np.random.seed(42)


from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg
from Analysis.Characterisation_v2.Plotting import Regression_plotting as rplot
from Analysis.Characterisation_v2 import DataClasses as dc
from Analysis.Characterisation_v2 import Plotting_utils as pu


class RegRunner:
    def __init__(self, conditions, base_dir, other_condition=None):
        self.conditions = conditions
        self.base_dir = base_dir
        self.other_condition = other_condition
        self.reg_apa_predictions = []
        # self.lda_wash_predictions = []
        os.makedirs(self.base_dir, exist_ok=True)

        # Load all data
        self._load_data()

    def _load_data(self):
        base_paths = {
            'APAChar_LowHigh': r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2",
            'APAChar_LowMid': r"H:\Characterisation_v2\LM_LHpcsonly_res_-3-2-1_APA2Wash2",
            'APAChar_HighLow': r"H:\Characterisation_v2\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2"
        }

        # Load the LH PCs as base
        pca_path = os.path.join(base_paths['APAChar_LowHigh'], r"APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_APAChar_LowHigh.pkl")
        with open(pca_path,'rb') as f:
            pca_LH = pickle.load(f)
        self.pca = pca_LH[0].pca
        self.pca_data = pca_LH[0]

        # Map condition to file path
        file_map = {
            'APAChar_LowHigh': os.path.join(base_paths['APAChar_LowHigh'], 'preprocessed_data_APAChar_LowHigh.pkl'),
            'APAChar_LowMid': os.path.join(base_paths['APAChar_LowMid'], 'preprocessed_data_APAChar_LowMid.pkl'),
            'APAChar_HighLow': os.path.join(base_paths['APAChar_HighLow'], 'preprocessed_data_APAChar_HighLow.pkl')
        }

        feature_names = list(short_names.keys())

        # Load only what's needed
        for cond in self.conditions:
            with open(file_map[cond], 'rb') as f:
                data = pickle.load(f)
                # reorder feature_data by feature_names
                f_data = data['feature_data_notscaled'].reindex(columns=feature_names)
                f_data_norm = data['feature_data'].reindex(columns=feature_names)
            setattr(self, f'feature_data_{cond.split("_")[-1]}', f_data)  # e.g. feature_data_LowHigh
            setattr(self, f'feature_data_norm_{cond.split("_")[-1]}', f_data_norm)

        if self.other_condition is not None and self.other_condition not in self.conditions:
            with open(file_map[self.other_condition], 'rb') as f:
                data = pickle.load(f)
                f_data = data['feature_data_notscaled'].reindex(columns=feature_names)
                f_data_norm = data['feature_data'].reindex(columns=feature_names)
            setattr(self, f'feature_data_{self.other_condition.split("_")[-1]}', f_data)
            setattr(self, f'feature_data_norm_{self.other_condition.split("_")[-1]}', f_data_norm)

        apawash_pred_files = {
            'LowHigh': os.path.join(base_paths['APAChar_LowHigh'], 'APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl'),
            'LowMid': os.path.join(base_paths['APAChar_LowMid'], 'APAChar_LowMid_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowMid.pkl'),
            'HighLow': os.path.join(base_paths['APAChar_HighLow'], 'APAChar_HighLow_Extended\MultiFeaturePredictions\pca_predictions_APAChar_HighLow.pkl')
        }
        for cond in ['LowHigh', 'LowMid', 'HighLow']:
            with open(apawash_pred_files[cond], 'rb') as f:
                pred_data = pickle.load(f)
            setattr(self, f'apawash_predictions_{cond}', pred_data)


    def filter_data(self, feature_data, s, midx, apa_wash='apa'):
        apa_mask, wash_mask = gu.get_mask_p1_p2(feature_data.loc(axis=0)[s, midx], 'APA2', 'Wash2')
        mask = apa_mask if apa_wash == 'apa' else wash_mask
        apa_run_vals = feature_data.loc(axis=0)[s, midx].index[mask]
        pcs = self.pca.transform(feature_data.loc(axis=0)[s, midx])
        pcs_apa = pcs[mask]
        pcs_trimmed = pcs_apa[:, :global_settings['pcs_to_use']]

        return pcs_trimmed, apa_run_vals

    def run(self):
        # find intersection of mice in all conditions
        mice_across_all_conditions = [condition_specific_settings[cond]['global_fs_mouse_ids'] for cond in self.conditions]
        mice_in_all = set(mice_across_all_conditions[0]).intersection(*mice_across_all_conditions[1:])

        label_map_2way = {
            'APAChar_LowHigh': 1.0,
            'APAChar_LowMid': 0.0,
            'APAChar_HighLow': 0.0  # gets overridden per pair
        }

        is_three_way = len(self.conditions) == 3
        all_pairs = list(itertools.combinations(self.conditions, 2)) if not is_three_way else []

        for s in global_settings['stride_numbers']:
            for midx in mice_in_all:
                if is_three_way:
                    try:
                        self.compare_all3(s, midx)
                    except Exception as e:
                        print(f"Error in 3-way for stride {s}, mouse {midx}: {e}")
                else:
                    for cond1, cond2 in all_pairs:
                        try:
                            label_map = label_map_2way.copy()
                            label_map[cond1] = 1.0
                            label_map[cond2] = 0.0
                            self.compare_pairwise(s, midx, cond1, cond2, label_map)
                        except Exception as e:
                            print(f"Error in 2-way ({cond1} vs {cond2}) for stride {s}, mouse {midx}: {e}")
        if not is_three_way:
            feature_data = {
                self.conditions[0]: getattr(self, f'feature_data_{self.conditions[0].split("_")[-1]}'),
                self.conditions[1]: getattr(self, f'feature_data_{self.conditions[1].split("_")[-1]}')
            }
            feature_data_norm = {
                self.conditions[0]: getattr(self, f'feature_data_norm_{self.conditions[0].split("_")[-1]}'),
                self.conditions[1]: getattr(self, f'feature_data_norm_{self.conditions[1].split("_")[-1]}')
            }
        else:
            feature_data = {
                'LowHigh': getattr(self, 'feature_data_LowHigh'),
                'LowMid': getattr(self, 'feature_data_LowMid'),
                'HighLow': getattr(self, 'feature_data_HighLow')
            }
            feature_data_norm = {
                'LowHigh': getattr(self, 'feature_data_norm_LowHigh'),
                'LowMid': getattr(self, 'feature_data_norm_LowMid'),
                'HighLow': getattr(self, 'feature_data_norm_HighLow')
            }

        performance_measure = 'mse' if is_three_way else 'acc'
        rplot.plot_reg_weights_condition_comparison(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        mean_preds, interp_preds = rplot.plot_prediction_per_trial(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        rplot.plot_prediction_histogram_ConditionComp(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        if len(self.conditions) == 2:
            rplot.plot_prediction_histogram_with_projection(reg_data=self.reg_apa_predictions,s=-1,trained_conditions=self.conditions,
                                                            other_condition=self.other_condition,exp='APA_Char',save_dir=self.base_dir)
        rplot.plot_condition_comparison_pc_features(feature_data_norm, self.pca_data, self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        rplot.plot_prediction_discrete_conditions(interp_preds, -1, self.conditions, 'APA_Char', self.base_dir)
        gu.get_and_save_pcs_of_interest(self.reg_apa_predictions, [-1], self.base_dir, conditions=self.conditions, reglda='reg', accmse=performance_measure)

        if is_three_way:
            self.compare_conditions_loaded_apavswash()


    def compare_all3(self, s, midx):
        LH_pcs, LH_runs = self.filter_data(self.feature_data_norm_LowHigh, s, midx)
        LM_pcs, LM_runs = self.filter_data(self.feature_data_norm_LowMid, s, midx)
        HL_pcs, HL_runs = self.filter_data(self.feature_data_norm_HighLow, s, midx)

        pcs_all = np.vstack([LH_pcs, LM_pcs, HL_pcs])
        mean_pcs = pcs_all.mean(axis=0)
        std_pcs = pcs_all.std(axis=0)
        # Avoid divide by zero
        std_pcs[std_pcs == 0] = 1
        # Z-score the PCs across all conditions
        LH_pcs = (LH_pcs - mean_pcs) / std_pcs
        LM_pcs = (LM_pcs - mean_pcs) / std_pcs
        HL_pcs = (HL_pcs - mean_pcs) / std_pcs

        # Combine the run values and reindex them to be chronological according to phase length
        LH_runs_zeroed = LH_runs - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
        LM_runs_zeroed = LM_runs - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 1
        HL_runs_zeroed = HL_runs - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 2

        runs_zeroed = np.concatenate([LH_runs_zeroed, LM_runs_zeroed, HL_runs_zeroed])

        pcs = np.vstack([LH_pcs, LM_pcs, HL_pcs])

        # assign lables relative to transition magnitude/direction --> LH:1, LM:0.5, HL:-1
        labels = np.concatenate([np.full(LH_pcs.shape[0], 1),
                                 np.full(LM_pcs.shape[0], 0.5),
                                 np.full(HL_pcs.shape[0], -1)])#.astype(int)

        w, bal_acc, cv_acc, w_folds = reg.compute_linear_regression(pcs.T, labels, folds=10)
        pc_acc, y_preds, null_acc = reg.compute_linear_regression_pcwise_prediction(pcs.T, labels, w)

        y_pred = np.dot(pcs, w)  # shape = (n_runs,)

        reg_data = dc.RegressionPredicitonData(
            conditions= ['APAChar_LowHigh', 'APAChar_LowMid', 'APAChar_HighLow'],
            phase='apa',
            stride=s,
            mouse_id=midx,
            x_vals=runs_zeroed,
            y_pred=y_pred,
            y_preds_pcs=y_preds,  # shape = (12, n_trials)
            pc_weights=w,  # 12 weights
            accuracy=bal_acc,
            cv_acc=cv_acc,
            w_folds=w_folds,
            pc_acc=pc_acc,  # now length 12
            null_acc=null_acc  # now (12 × shuffles)
        )
        self.reg_apa_predictions.append(reg_data)

    def compare_pairwise(self, s, midx, cond1, cond2, label_map):
        data_map = {
            cond1: getattr(self, f'feature_data_norm_{cond1.split("_")[-1]}'),
            cond2: getattr(self, f'feature_data_norm_{cond2.split("_")[-1]}')
        }

        pcs1, runs1 = self.filter_data(data_map[cond1], s, midx)
        pcs2, runs2 = self.filter_data(data_map[cond2], s, midx)

        # 🔥 NEW: concatenate and z-score across conditions
        pcs_all = np.vstack([pcs1, pcs2])
        mean_pcs = pcs_all.mean(axis=0)
        std_pcs = pcs_all.std(axis=0)

        # Avoid divide by zero
        std_pcs[std_pcs == 0] = 1

        pcs1_z = (pcs1 - mean_pcs) / std_pcs
        pcs2_z = (pcs2 - mean_pcs) / std_pcs
        pcs = np.vstack([pcs1_z, pcs2_z])

        # Zero x-axis
        runs1_zeroed = runs1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
        runs2_zeroed = runs2 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
            expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])

        runs_zeroed = np.concatenate([runs1_zeroed, runs2_zeroed])
        labels = np.concatenate([
            np.full(pcs1.shape[0], label_map[cond1]),
            np.full(pcs2.shape[0], label_map[cond2])
        ])

        num_folds = 10
        w, bal_acc, cv_acc, w_folds = reg.compute_regression(pcs.T, labels, folds=num_folds)
        pc_acc, y_preds, null_acc = reg.compute_regression_pcwise_prediction(pcs.T, labels, w)
        y_pred = np.dot(pcs, w.T)

        pc_lesions_cv_acc = np.zeros((global_settings['pcs_to_use'], num_folds))
        pc_lesions_w_folds = np.zeros((global_settings['pcs_to_use'], num_folds, global_settings['pcs_to_use']))
        for pc in range(global_settings['pcs_to_use']):
            cv_acc_lesion, w_folds_lesion = reg.compute_regression_lesion(pcs.T, labels, folds=num_folds, regressor_to_shuffle=pc)
            pc_lesions_cv_acc[pc, :] = cv_acc_lesion
            pc_lesions_w_folds[pc, :, :] = w_folds_lesion.squeeze()

        reg_data = dc.RegressionPredicitonData(
            conditions=[cond1, cond2],
            phase='apa',
            stride=s,
            mouse_id=midx,
            x_vals=runs_zeroed,
            y_pred=y_pred,
            y_preds_pcs=y_preds,
            pc_weights=w,
            accuracy=bal_acc,
            cv_acc=cv_acc,
            w_folds=w_folds,
            pc_acc=pc_acc,
            null_acc=null_acc,
            pc_lesions_cv_acc=pc_lesions_cv_acc,
            pc_lesions_w_folds=pc_lesions_w_folds
        )
        self.reg_apa_predictions.append(reg_data)

        if self.other_condition and midx in condition_specific_settings[self.other_condition]['global_fs_mouse_ids']:
            other_cond = self.other_condition
            other_data = getattr(self, f'feature_data_norm_{other_cond.split("_")[-1]}')
            pcs_other, runs_other = self.filter_data(other_data, s, midx)
            runs_other_zeroed = runs_other - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 2
            y_pred_other = np.dot(pcs_other, w.T)

            reg_data_proj = dc.RegressionPredicitonData(
                conditions=[other_cond],
                phase='apa',
                stride=s,
                mouse_id=midx,
                x_vals=runs_other_zeroed,
                y_pred=y_pred_other, # this is all we really need, plus x_vals
                y_preds_pcs=None,
                pc_weights=None,
                accuracy=None,
                cv_acc=None,
                w_folds=None,
                pc_acc=None,
                null_acc=None
            )
            self.reg_apa_predictions.append(reg_data_proj)

    def compare_conditions_loaded_apavswash(self, fs=7):
        fig, ax = plt.subplots(figsize=(3, 2))
        num_bins = 30
        bins = np.linspace(-1, 1, num_bins)
        num_sigma = 3

        for cond in ['LowHigh', 'LowMid', 'HighLow']:
            pred_data = getattr(self, f'apawash_predictions_{cond}')
            y_preds = [pred.y_pred for pred in pred_data if pred.phase == ('APA2', 'Wash2') and pred.stride == -1]
            x_vals = [pred.x_vals for pred in pred_data if pred.phase == ('APA2', 'Wash2') and pred.stride == -1]
            mice_names = [pred.mouse_id for pred in pred_data if pred.phase == ('APA2', 'Wash2') and pred.stride == -1]

            apa_runs = set(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            wash_runs = set(expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])

            apa_masks = [x.isin(apa_runs) for x in x_vals]
            wash_masks = [x.isin(wash_runs) for x in x_vals]

            y_preds_apa = [np.ravel(yp)[mask] for yp, mask in zip(y_preds, apa_masks)]
            y_preds_wash = [np.ravel(yp)[mask] for yp, mask in zip(y_preds, wash_masks)]

            x_vals_apa = [x[mask] for x, mask in zip(x_vals, apa_masks)]
            x_vals_wash = [x[mask] for x, mask in zip(x_vals, wash_masks)]

            common_x_apa = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
            common_x_wash = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']

            preds_df_apa = pd.DataFrame(index=common_x_apa, columns=mice_names, dtype=float)
            preds_df_wash = pd.DataFrame(index=common_x_wash, columns=mice_names, dtype=float)
            for p in [y_preds_apa, y_preds_wash]:
                for midx, mouse_preds in enumerate(p):
                    mouse_name = mice_names[midx]
                    if p is y_preds_apa:
                        preds_df_apa.loc[x_vals_apa[midx], mouse_name] = mouse_preds.ravel()
                    elif p is y_preds_wash:
                        preds_df_wash.loc[x_vals_wash[midx], mouse_name] = mouse_preds.ravel()

            # intrerpolate, smooth and z-score for each mouse
            y_preds_apa_interp = preds_df_apa.interpolate(limit_direction='both')
            y_preds_wash_interp = preds_df_wash.interpolate(limit_direction='both')
            y_preds_apa_smooth = median_filter(y_preds_apa_interp, size=3, mode='nearest')
            y_preds_wash_smooth = median_filter(y_preds_wash_interp, size=3, mode='nearest')
            max_abs_apa = max(abs(np.nanmin(y_preds_apa_smooth)), abs(np.nanmax(y_preds_apa_smooth)))
            max_abs_wash = max(abs(np.nanmin(y_preds_wash_smooth)), abs(np.nanmax(y_preds_wash_smooth)))
            norm_preds_apa = y_preds_apa_smooth / max_abs_apa
            norm_preds_wash = y_preds_wash_smooth / max_abs_wash

            apa_all = np.concatenate(norm_preds_apa)
            wash_all = np.concatenate(norm_preds_wash)

            for d in [apa_all, wash_all]:
                phase = 'APA2' if d is apa_all else 'Wash2'
                ls = '-' if phase == 'APA2' else '--'
                color = pu.get_color_speedpair(cond)
                hist_vals, _ = np.histogram(d, bins=bins, density=True)
                smoothed_hist = gaussian_filter1d(hist_vals, sigma=num_sigma)  # Tune sigma as needed
                ax.plot(bins[:-1], smoothed_hist, linewidth=1.5, linestyle=ls, color=color, label=f"{cond}-{phase}")
        ax.set_xlabel('Z-scored Prediction Score', fontsize=fs)
        ax.set_ylabel('Probability Density', fontsize=fs)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs - 1, frameon=False, ncol=1)
        legend_elements = [
            mlines.Line2D([], [], color=pu.get_color_speedpair('LowHigh'), linestyle='-', label='LowHigh'),
            mlines.Line2D([], [], color=pu.get_color_speedpair('LowMid'), linestyle='-', label='LowMid'),
            mlines.Line2D([], [], color=pu.get_color_speedpair('HighLow'), linestyle='-', label='HighLow'),
            mlines.Line2D([], [], color='black', linestyle='-', label='APAlate'),
            mlines.Line2D([], [], color='black', linestyle='--', label='Washlate'),
        ]

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs - 1, frameon=False)

        # X-axis: labels + minor ticks
        ax.set_xlim(-1.2, 1.2)
        ax.set_xticks(np.arange(-1, 1.1, 0.5))
        ax.set_xticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis='x', which='minor', bottom=True, length=2, width=1, color='k')
        ax.tick_params(axis='x', which='major', bottom=True, length=4, width=1)

        # Y-axis: font size + minor ticks
        ax.tick_params(axis='y', labelsize=fs)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='y', which='minor', length=2, width=1, color='k')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.grid(False)
        plt.subplots_adjust(left=0.2, right=0.75, top=0.95, bottom=0.2)

        save_path_full = os.path.join(self.base_dir, 'Comparison_ApaWash_Predictions_Histogram.png')
        plt.savefig(f"{save_path_full}.png", dpi=300)
        plt.savefig(f"{save_path_full}.svg", dpi=300)
        plt.close()

    def compare_conditions_wash_vs_apa(self, s, fs=7):
        common_mice = set(condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']).intersection(
            condition_specific_settings['APAChar_LowMid']['global_fs_mouse_ids'],
            condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids'])

        pc_data_per_mouse = {pc: {'mouse': [], 'condition': [], 'value': []}
                             for pc in range(1, global_settings['pcs_to_use'] + 1)}

        for midx in common_mice:
            # Retrieve data for this mouse
            LH_apa_pcs, _ = self.filter_data(self.feature_data_LowHigh, s, midx, apa_wash='apa')
            LM_apa_pcs, _ = self.filter_data(self.feature_data_LowMid, s, midx, apa_wash='apa')
            HL_apa_pcs, _ = self.filter_data(self.feature_data_HighLow, s, midx, apa_wash='apa')

            LH_wash_pcs, _ = self.filter_data(self.feature_data_LowHigh, s, midx, apa_wash='wash')
            LM_wash_pcs, _ = self.filter_data(self.feature_data_LowMid, s, midx, apa_wash='wash')
            HL_wash_pcs, _ = self.filter_data(self.feature_data_HighLow, s, midx, apa_wash='wash')

            for pc in range(1, global_settings['pcs_to_use']+1):
                pc_idx = pc - 1  # zero-indexing

                # wash conditions
                for cond_label, pcs_data in zip(
                        ['wash_LH', 'wash_LM', 'wash_HL'],
                        [LH_wash_pcs, LM_wash_pcs, HL_wash_pcs]
                ):
                    if pcs_data.shape[0] > 0:
                        val = pcs_data[:, pc_idx].mean()  # or keep all values per trial if desired
                        pc_data_per_mouse[pc]['mouse'].append(midx)
                        pc_data_per_mouse[pc]['condition'].append(cond_label)
                        pc_data_per_mouse[pc]['value'].append(val)

                # apa conditions
                for cond_label, pcs_data in zip(
                        ['apa_LH', 'apa_LM', 'apa_HL'],
                        [LH_apa_pcs, LM_apa_pcs, HL_apa_pcs]
                ):
                    if pcs_data.shape[0] > 0:
                        val = pcs_data[:, pc_idx].mean()
                        pc_data_per_mouse[pc]['mouse'].append(midx)
                        pc_data_per_mouse[pc]['condition'].append(cond_label)
                        pc_data_per_mouse[pc]['value'].append(val)

        for pc in range(1, global_settings['pcs_to_use'] + 1):
            data = pc_data_per_mouse[pc]
            x_labels = ['wash_LH', 'wash_LM', 'wash_HL', 'apa_LH', 'apa_LM', 'apa_HL']

            # Convert to DataFrame for seaborn
            df_plot = pd.DataFrame({
                'condition': data['condition'],
                'value': data['value'],
                'mouse': data['mouse']
            })
            fig, ax = plt.subplots(figsize=(5, 4))

            # Define x positions
            wash_labels = ['wash_LH', 'wash_LM', 'wash_HL']
            apa_labels = ['apa_LH', 'apa_LM', 'apa_HL']
            x_labels = wash_labels + apa_labels
            x_positions = [-0.2, 0.0, 0.2, 1, 2, 3]

            # Get colours and darker edges
            palette = []
            edge_colors = []
            for label in x_labels:
                speed = label.split('_')[-1]
                speed_full = {'LH': 'LowHigh', 'LM': 'LowMid', 'HL': 'HighLow'}[speed]
                face_color = pu.get_color_speedpair(speed_full)
                palette.append(face_color)
                edge_colors.append(pu.darken_color(face_color, factor=0.7))

            # Plot boxplots using ax.boxplot to control positions precisely
            for xpos, label, face_c, edge_c in zip(x_positions, x_labels, palette, edge_colors):
                subset = df_plot[df_plot['condition'] == label]
                if subset.empty:
                    continue

                bp = ax.boxplot(
                    subset['value'], positions=[xpos], widths=0.2,
                    patch_artist=True, showfliers=True
                )

                for patch in bp['boxes']:
                    patch.set_facecolor(face_c)
                    patch.set_edgecolor(edge_c)
                    patch.set_alpha(0.7)
                    patch.set_linewidth(1)

                for whisker in bp['whiskers']:
                    whisker.set_color(edge_c)
                    whisker.set_linewidth(1)

                for cap in bp['caps']:
                    cap.set_color(edge_c)
                    cap.set_linewidth(1)

                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(1.5)

                for flier in bp['fliers']:
                    flier.set_marker('o')
                    flier.set_markersize(3)
                    flier.set_markerfacecolor(face_c)
                    flier.set_markeredgecolor(edge_c)

            # Overlay scatter points
            x_pos_map = dict(zip(x_labels, x_positions))
            x_numeric = [x_pos_map[cond] for cond in df_plot['condition']]
            ax.scatter(x_numeric, df_plot['value'], color='k', s=4, zorder=3)

            # Adjust x-axis
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['wash', 'apa_LH', 'apa_LM', 'apa_HL'], rotation=45, ha='right')

            ax.set_ylabel(f'PC{pc} value')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            #significance
            # Define y position for significance bars
            y_max = df_plot['value'].max()
            y_min = df_plot['value'].min()
            y_range = y_max - y_min
            height_offset = y_range * 0.05  # vertical spacing between bars
            bar_height = y_max + height_offset
            bar_spacing = height_offset * 1.5

            # All pairwise combinations
            comparisons = list(combinations(x_labels, 2))

            for i, (cond1, cond2) in enumerate(comparisons):
                data1 = df_plot[df_plot['condition'] == cond1]['value']
                data2 = df_plot[df_plot['condition'] == cond2]['value']

                if len(data1) > 0 and len(data2) > 0:
                    # Perform Mann-Whitney U test
                    stat, pval = wilcoxon(data1, data2, alternative='two-sided')

                    # Determine significance stars
                    if pval < 0.001:
                        sig = '***'
                    elif pval < 0.01:
                        sig = '**'
                    elif pval < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'

                    if sig == 'ns':
                        continue
                    # Get x positions
                    x1 = x_pos_map[cond1]
                    x2 = x_pos_map[cond2]
                    x_center = (x1 + x2) / 2

                    # Plot line
                    ax.plot([x1, x1, x2, x2],
                            [bar_height, bar_height + height_offset, bar_height + height_offset, bar_height],
                            lw=1.0, c='k')

                    # Plot significance text
                    ax.text(x_center, bar_height + height_offset + (height_offset * 0.2), sig,
                            ha='center', va='bottom', fontsize=fs)

                    # Increment bar height for next comparison
                    bar_height += bar_spacing


            savepath = os.path.join(self.base_dir, f'PC{pc}_wash_vs_apa_box')
            plt.savefig(f"{savepath}.png", dpi=300)
            plt.savefig(f"{savepath}.svg", dpi=300)
            plt.close()

    def compare_conditions_APA_correlations(self, conditions, s, chosen_pcs=[1,3,7], fs=7):
        assert len(conditions) == 2, "This method is designed for pairwise comparisons only."
        common_mice = set(condition_specific_settings[f'APAChar_{conditions[0]}']['global_fs_mouse_ids']).intersection(
            condition_specific_settings[f'APAChar_{conditions[1]}']['global_fs_mouse_ids'])

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        # Define data collections for both plots per pc
        pc_data_dict = {pc: {'apa': {'x': [], 'y': []}, 'diff': {'x': [], 'y': []}} for pc in chosen_pcs}

        # First gather all data
        for midx in common_mice:
            cond1_feature_data = getattr(self, f'feature_data_{conditions[0]}')
            cond2_feature_data = getattr(self, f'feature_data_{conditions[1]}')

            cond1_apa_pcs, _ = self.filter_data(cond1_feature_data, s, midx, apa_wash='apa')
            cond2_apa_pcs, _ = self.filter_data(cond2_feature_data, s, midx, apa_wash='apa')
            cond1_wash_pcs, _ = self.filter_data(cond1_feature_data, s, midx, apa_wash='wash')
            cond2_wash_pcs, _ = self.filter_data(cond2_feature_data, s, midx, apa_wash='wash')

            cond1_apa_pc_avgs = np.mean(cond1_apa_pcs, axis=0)
            cond2_apa_pc_avgs = np.mean(cond2_apa_pcs, axis=0)
            cond1_wash_pc_avgs = np.mean(cond1_wash_pcs, axis=0)
            cond2_wash_pc_avgs = np.mean(cond2_wash_pcs, axis=0)

            for pc in chosen_pcs:
                pc_idx = pc - 1

                # APA plot data
                pc_data_dict[pc]['apa']['x'].append(cond1_apa_pc_avgs[pc_idx])
                pc_data_dict[pc]['apa']['y'].append(cond2_apa_pc_avgs[pc_idx])

                # Diff plot data
                cond1_pc_diff = cond1_apa_pc_avgs[pc_idx] - cond1_wash_pc_avgs[pc_idx]
                cond2_pc_diff = cond2_apa_pc_avgs[pc_idx] - cond2_wash_pc_avgs[pc_idx]
                pc_data_dict[pc]['diff']['x'].append(cond1_pc_diff)
                pc_data_dict[pc]['diff']['y'].append(cond2_pc_diff)

        # Now plot mean + CI for each pc on each plot
        for pc in chosen_pcs:
            pc_idx = pc - 1
            pc_color = pu.get_color_pc(pc_idx)

            for ax, key in zip([ax1, ax2], ['apa', 'diff']):
                x_vals = np.array(pc_data_dict[pc][key]['x'])
                y_vals = np.array(pc_data_dict[pc][key]['y'])

                mean_x = np.mean(x_vals)
                mean_y = np.mean(y_vals)
                ci_x = stats.sem(x_vals) * stats.t.ppf((1 + 0.95) / 2., len(x_vals) - 1)
                ci_y = stats.sem(y_vals) * stats.t.ppf((1 + 0.95) / 2., len(y_vals) - 1)

                ax.scatter(x_vals, y_vals, color=pc_color, s=8, label=f'PC{pc} data')
                ax.errorbar(mean_x, mean_y, xerr=ci_x, yerr=ci_y,
                            fmt='o', color=pc_color, markersize=5, capsize=3, label=f'PC{pc} mean')

        ax1.set_xlabel(f'{conditions[0]} APAlate PC', fontsize=fs)
        ax1.set_ylabel(f'{conditions[1]} APAlate PC', fontsize=fs)

        ax2.set_xlabel(f'{conditions[0]} APAlate - Washlate PC', fontsize=fs)
        ax2.set_ylabel(f'{conditions[1]} APAlate - Washlate PC', fontsize=fs)

        for fig,ax,key in zip([fig1, fig2], [ax1, ax2], ['apa', 'diff']):
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            # add legend with pc colors
            legend_elements = [mlines.Line2D([], [], color=pu.get_color_pc(pc_num-1), marker='o', linestyle='',
                                                label=f'PC{pc_num}') for pc_num in chosen_pcs]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs - 1, frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.base_dir, f'Comparison_{conditions[0]}_{conditions[1]}_PCs{chosen_pcs}_{key}.png'), dpi=300)

    def report_overall_model_accuracy(self,f):
        """
        Computes and prints overall model accuracy:
        - Mean across folds per mouse
        - Mean across mice
        """
        per_mouse_means = []

        for pred in self.reg_apa_predictions:
            if pred.cv_acc is not None:
                per_mouse_means.append(np.mean(pred.cv_acc))

        if per_mouse_means:
            overall_mean = np.mean(per_mouse_means)
            overall_sem = stats.sem(per_mouse_means)
            print(f"Overall model accuracy: {overall_mean:.3f} ± {overall_sem:.3f}")
            f.write(f"Overall model accuracy: {overall_mean:.3f} ± {overall_sem:.3f}\n")
        else:
            print("No cv_acc data found in predictions.")



def main():
    all_conditions = ['APAChar_LowHigh', 'APAChar_LowMid', 'APAChar_HighLow']

    # 3-way regression
    base_dir_3way = r"H:\Characterisation_v2\Compare_LH_LM_HL_regression"
    runner = RegRunner(all_conditions, base_dir_3way)

    runner.compare_conditions_APA_correlations(['LowHigh','HighLow'], -1)
    runner.compare_conditions_APA_correlations(['LowHigh','LowMid'], -1)
    runner.compare_conditions_APA_correlations(['LowHigh','HighLow'], -1, chosen_pcs=list(np.arange(global_settings['pcs_to_use'])+1))
    runner.compare_conditions_APA_correlations(['LowHigh','LowMid'], -1, chosen_pcs=list(np.arange(global_settings['pcs_to_use'])+1))

    runner.compare_conditions_wash_vs_apa(-1)

    runner.run()
    # write report to a text file
    with open(os.path.join(base_dir_3way, 'overall_model_accuracy.txt'), 'w') as f:
        f.write("Overall model accuracy report:\n")
        runner.report_overall_model_accuracy(f)

    # 2-way comparisons
    for cond1, cond2 in itertools.combinations(all_conditions, 2):
        other_cond = list(set(all_conditions) - {cond1, cond2})[0]
        base_dir = rf"H:\Characterisation_v2\Compare_{cond1.split('_')[-1]}_vs_{cond2.split('_')[-1]}_regression"
        print(f"Running comparison for {cond1} vs {cond2} with projection of {other_cond} in directory {base_dir}")
        runner = RegRunner([cond1, cond2], base_dir, other_condition=other_cond)
        runner.run()
        with open(os.path.join(base_dir, 'overall_model_accuracy.txt'), 'w') as f:
            f.write("Overall model accuracy report:\n")
            runner.report_overall_model_accuracy(f)
        print("Comparison completed.")




if __name__ == '__main__':
    main()
