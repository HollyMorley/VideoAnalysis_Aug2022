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
        # Load the LH PCs as base
        with open(
                r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\APAChar_LowHigh_Extended\\MultiFeaturePredictions\\pca_APAChar_LowHigh.pkl",
                'rb') as f:
            pca_LH = pickle.load(f)
        self.pca = pca_LH[0].pca
        self.pca_data = pca_LH[0]

        # Map condition to file path
        file_map = {
            'APAChar_LowHigh': r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\preprocessed_data_APAChar_LowHigh.pkl",
            'APAChar_LowMid': r"H:\Characterisation\LM_LHpcsonly_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\preprocessed_data_APAChar_LowMid.pkl",
            'APAChar_HighLow': r"H:\\Characterisation\\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_HighLow.pkl",
        }

        # Load only what's needed
        for cond in self.conditions:
            with open(file_map[cond], 'rb') as f:
                data = pickle.load(f)
            setattr(self, f'feature_data_{cond.split("_")[-1]}', data['feature_data'])  # e.g. feature_data_LowHigh

        if self.other_condition is not None and self.other_condition not in self.conditions:
            with open(file_map[self.other_condition], 'rb') as f:
                data = pickle.load(f)
            setattr(self, f'feature_data_{self.other_condition.split("_")[-1]}', data['feature_data'])

        apawash_pred_files = {
            'LowHigh': r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl",
            'LowMid': r"H:\Characterisation\LM_LHpcsonly_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowMid_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowMid.pkl",
            'HighLow': r"H:\Characterisation\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\APAChar_HighLow_Extended\MultiFeaturePredictions\pca_predictions_APAChar_HighLow.pkl"
        }
        for cond in ['LowHigh', 'LowMid', 'HighLow']:
            with open(apawash_pred_files[cond], 'rb') as f:
                pred_data = pickle.load(f)
            setattr(self, f'apawash_predictions_{cond}', pred_data)

        # load null space

    def filter_data(self, feature_data, s, midx):
        apa_mask, _ = gu.get_mask_p1_p2(feature_data.loc(axis=0)[s, midx], 'APA2', 'Wash2')
        apa_run_vals = feature_data.loc(axis=0)[s, midx].index[apa_mask]
        pcs = self.pca.transform(feature_data.loc(axis=0)[s, midx])
        pcs_apa = pcs[apa_mask]
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
        else:
            feature_data = {
                'LowHigh': getattr(self, 'feature_data_LowHigh'),
                'LowMid': getattr(self, 'feature_data_LowMid'),
                'HighLow': getattr(self, 'feature_data_HighLow')
            }

        performance_measure = 'mse' if is_three_way else 'acc'
        rplot.plot_reg_weights_condition_comparison(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        mean_preds, interp_preds = rplot.plot_prediciton_per_trial(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        rplot.plot_prediction_histogram_ConditionComp(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        if len(self.conditions) == 2:
            rplot.plot_prediction_histogram_with_projection(reg_data=self.reg_apa_predictions,s=-1,trained_conditions=self.conditions,
                                                            other_condition=self.other_condition,exp='APA_Char',save_dir=self.base_dir)
        rplot.plot_condition_comparison_pc_features(feature_data, self.pca_data, self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        rplot.plot_prediction_discrete_conditions(interp_preds, -1, self.conditions, 'APA_Char', self.base_dir)
        gu.get_and_save_pcs_of_interest(self.reg_apa_predictions, [-1], self.base_dir, conditions=self.conditions, reglda='reg', accmse=performance_measure)

        if is_three_way:
            self.compare_conditions_loaded_apavswash()


    def compare_all3(self, s, midx):
        LH_pcs, LH_runs = self.filter_data(self.feature_data_LowHigh, s, midx)
        LM_pcs, LM_runs = self.filter_data(self.feature_data_LowMid, s, midx)
        HL_pcs, HL_runs = self.filter_data(self.feature_data_HighLow, s, midx)

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
            cond1: getattr(self, f'feature_data_{cond1.split("_")[-1]}'),
            cond2: getattr(self, f'feature_data_{cond2.split("_")[-1]}')
        }
        pcs1, runs1 = self.filter_data(data_map[cond1], s, midx)
        pcs2, runs2 = self.filter_data(data_map[cond2], s, midx)

        # Zero x-axis
        runs1_zeroed = runs1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
        runs2_zeroed = runs2 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
            expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])

        runs_zeroed = np.concatenate([runs1_zeroed, runs2_zeroed])
        pcs = np.vstack([pcs1, pcs2])
        labels = np.concatenate([
            np.full(pcs1.shape[0], label_map[cond1]),
            np.full(pcs2.shape[0], label_map[cond2])
        ])

        w, bal_acc, cv_acc, w_folds = reg.compute_regression(pcs.T, labels, folds=10)
        pc_acc, y_preds, null_acc = reg.compute_regression_pcwise_prediction(pcs.T, labels, w)
        y_pred = np.dot(pcs, w.T)

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
            null_acc=null_acc
        )
        self.reg_apa_predictions.append(reg_data)

        if self.other_condition and midx in condition_specific_settings[self.other_condition]['global_fs_mouse_ids']:
            other_cond = self.other_condition
            other_data = getattr(self, f'feature_data_{other_cond.split("_")[-1]}')
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
                hist_vals, _ = np.histogram(d, bins=bins)
                smoothed_hist = gaussian_filter1d(hist_vals, sigma=num_sigma)  # Tune sigma as needed
                ax.plot(bins[:-1], smoothed_hist, linewidth=1.5, linestyle=ls, color=color, label=f"{cond}-{phase}")
        ax.set_xlabel('Z-scored Prediction Score', fontsize=fs)
        ax.set_ylabel('Count', fontsize=fs)
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

    def compare_conditions_pc_vals(self, s, midx, cond1, cond2, label_map):
        data_map = {
            cond1: getattr(self, f'feature_data_{cond1.split("_")[-1]}'),
            cond2: getattr(self, f'feature_data_{cond2.split("_")[-1]}')
        }

def main():
    all_conditions = ['APAChar_LowHigh', 'APAChar_LowMid', 'APAChar_HighLow']

    # 3-way regression
    base_dir_3way = r"H:\Characterisation\Compare_LH_LM_HL_regression"
    runner = RegRunner(all_conditions, base_dir_3way)
    runner.run()

    # 2-way comparisons
    for cond1, cond2 in itertools.combinations(all_conditions, 2):
        other_cond = list(set(all_conditions) - {cond1, cond2})[0]
        base_dir = rf"H:\Characterisation\Compare_{cond1.split('_')[-1]}_vs_{cond2.split('_')[-1]}_regression"
        print(f"Running comparison for {cond1} vs {cond2} with projection of {other_cond} in directory {base_dir}")
        runner = RegRunner([cond1, cond2], base_dir, other_condition=other_cond)
        runner.run()
        print("Comparison completed.")


if __name__ == '__main__':
    main()
