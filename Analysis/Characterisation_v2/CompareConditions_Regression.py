import pickle
import pandas as pd
from scipy.linalg import null_space
import os
import itertools

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg
from Analysis.Characterisation_v2.Plotting import Regression_plotting as rplot
from Analysis.Characterisation_v2 import DataClasses as dc


class LDARunner:
    def __init__(self, conditions, base_dir):
        self.conditions = conditions
        self.base_dir = base_dir
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

        # Map condition to file path
        file_map = {
            'APAChar_LowHigh': r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\preprocessed_data_APAChar_LowHigh.pkl",
            'APAChar_LowMid': r"H:\Characterisation\LM_LHpcsonly_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\preprocessed_data_APAChar_LowMid.pkl",
            'APAChar_HighLow': r"H:\\Characterisation\\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_HighLow.pkl"
        }

        # Load only what's needed
        for cond in self.conditions:
            with open(file_map[cond], 'rb') as f:
                data = pickle.load(f)
            setattr(self, f'feature_data_{cond.split("_")[-1]}', data['feature_data'])  # e.g. feature_data_LowHigh

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

        performance_measure = 'mse' if is_three_way else 'acc'
        rplot.plot_reg_weights_condition_comparison(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        mean_preds, interp_preds = rplot.plot_prediciton_per_trial(self.reg_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        rplot.plot_prediction_discrete_conditions(interp_preds, -1, self.conditions, 'APA_Char', self.base_dir)
        gu.get_and_save_pcs_of_interest(self.reg_apa_predictions, [-1], self.base_dir, reglda='reg', accmse=performance_measure)


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






def main():
    all_conditions = ['APAChar_LowHigh', 'APAChar_LowMid', 'APAChar_HighLow']

    # # 3-way regression
    # base_dir_3way = r"H:\Characterisation\Compare_LH_LM_HL_regression"
    # runner = LDARunner(all_conditions, base_dir_3way)
    # runner.run()

    # 2-way comparisons
    for cond1, cond2 in itertools.combinations(all_conditions, 2):
        base_dir = rf"H:\Characterisation\Compare_{cond1.split('_')[-1]}_vs_{cond2.split('_')[-1]}_regression"
        print(f"Running comparison for {cond1} vs {cond2} in directory {base_dir}")
        runner = LDARunner([cond1, cond2], base_dir)
        runner.run()
        print("Comparison completed.")


if __name__ == '__main__':
    main()
