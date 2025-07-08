import pickle
import pandas as pd
from scipy.linalg import null_space
import os
import itertools

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.AnalysisTools import LDA
from Analysis.Characterisation_v2.Plotting import LDA_plotting
from Analysis.Characterisation_v2.Plotting import Regression_plotting as rplot
from Analysis.Characterisation_v2 import DataClasses as dc


class LDARunner:
    def __init__(self, method, conditions, base_dir, other_condition=None):
        self.method = method
        self.conditions = conditions
        self.base_dir = base_dir
        self.other_condition = other_condition
        self.lda_apa_predictions = []
        self.lda_wash_predictions = []
        os.makedirs(self.base_dir, exist_ok=True)

        self._load_data()

    def _load_data(self):
        base_paths = {
            'APAChar_LowHigh': r"H:\Characterisation\LH_allpca_LhWnrm_res_-3-2-1_APA2Wash2",
            'APAChar_LowMid': r"H:\Characterisation\LM_allpca_LHpcsonly_res_-3-2-1_APA2Wash2",
            'APAChar_HighLow': r"H:\Characterisation\HL_allpca_LHpcsonly_res_-3-2-1_APA2Wash2" ## todo THIS DOESNT EXIST YET!!
        }

        file_map = {
            'APAChar_LowHigh': os.path.join(base_paths['APAChar_LowHigh'], 'preprocessed_data_APAChar_LowHigh.pkl'),
            'APAChar_LowMid': os.path.join(base_paths['APAChar_LowMid'], 'preprocessed_data_APAChar_LowMid.pkl'),
            'APAChar_HighLow': os.path.join(base_paths['APAChar_HighLow'], 'preprocessed_data_APAChar_HighLow.pkl')
        }

        file_map_wash = {
            'APAChar_HighLow': r"H:\\Characterisation\\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_HighLow.pkl"
        }

        with open(r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\APAChar_LowHigh_Extended\\MultiFeaturePredictions\\pca_APAChar_LowHigh.pkl", 'rb') as f:
            pca_LH = pickle.load(f)
            self.pca = pca_LH[0].pca
            self.pca_data = pca_LH[0]

        self.feature_data = {}
        for cond in self.conditions:
            path = file_map_wash[cond] if self.method == 'wash_normalised' and cond in file_map_wash else file_map[cond]
            with open(path, 'rb') as f:
                self.feature_data[cond] = pickle.load(f)['feature_data']

        if self.other_condition and self.other_condition not in self.conditions:
            path = file_map_wash[
                self.other_condition] if self.method == 'wash_normalised' and self.other_condition in file_map_wash else \
            file_map[self.other_condition]
            with open(path, 'rb') as f:
                self.feature_data[self.other_condition] = pickle.load(f)['feature_data']

    def filter_data(self, data, s, midx, phase):
        # mask1, mask2 = gu.get_mask_p1_p2(data.loc(axis=0)[s, midx], 'APA2', 'Wash2')
        # pcs = self.pca.transform(data.loc(axis=0)[s, midx])
        # return pcs[:, :global_settings['pcs_to_use']], mask1, mask2, data.loc(axis=0)[s, midx]
        apa_mask, wash_mask = gu.get_mask_p1_p2(data.loc(axis=0)[s, midx], 'APA2', 'Wash2')
        phase_mask = apa_mask if phase == 'APA2' else wash_mask
        phase_run_vals = data.loc(axis=0)[s, midx].index[phase_mask]
        pcs = self.pca.transform(data.loc(axis=0)[s, midx])
        pcs_phase = pcs[phase_mask]
        pcs_trimmed = pcs_phase[:, :global_settings['pcs_to_use']]
        return pcs_trimmed, phase_run_vals


    def run(self):
        is_three_way = len(self.conditions) == 3
        all_pairs = list(itertools.combinations(self.conditions, 2)) if not is_three_way else [(self.conditions[0], self.conditions[1])]

        for cond1, cond2 in all_pairs:
            mice = list(set(condition_specific_settings[cond1]['global_fs_mouse_ids']) & set(condition_specific_settings[cond2]['global_fs_mouse_ids']))
            for s in global_settings['stride_numbers']:
                for midx in mice:
                    try:
                        self._run_lda_for_pair(s, midx, cond1, cond2)
                    except Exception as e:
                        print(f"Error in LDA ({cond1} vs {cond2}) for stride {s}, mouse {midx}: {e}")

        feature_data = {cond.split('_')[-1]: self.feature_data[cond] for cond in self.conditions}
        if self.other_condition and self.other_condition not in self.conditions:
            feature_data[self.other_condition.split('_')[-1]] = self.feature_data[self.other_condition]

        lda_dir = os.path.join(self.base_dir, f'LDA_{self.method}')
        os.makedirs(lda_dir, exist_ok=True)
        LDA_plotting.plot_lda_weights(self.lda_apa_predictions, -1, self.conditions[0].split('_')[-1], self.conditions[1].split('_')[-1], 'APA_Char', lda_dir)
        _, interp_preds = LDA_plotting.plot_prediction_per_trial(self.lda_apa_predictions, -1, self.conditions[0].split('_')[-1], self.conditions[1].split('_')[-1], 'APA_Char', lda_dir)
        rplot.plot_prediction_histogram_ConditionComp(self.lda_apa_predictions, -1, self.conditions, 'APA_Char', lda_dir, model_type='lda')
        rplot.plot_prediction_histogram_with_projection(reg_data=self.lda_apa_predictions, s=-1,
                                                        trained_conditions=self.conditions,
                                                        other_condition=self.other_condition, exp='APA_Char',
                                                        save_dir=lda_dir,
                                                        model_type='lda')
        rplot.plot_condition_comparison_pc_features(feature_data, self.pca_data, self.lda_apa_predictions, -1, self.conditions, 'APA_Char', self.base_dir)
        LDA_plotting.plot_prediction_discrete_conditions(interp_preds, -1, self.conditions[0].split('_')[-1], self.conditions[1].split('_')[-1], 'APA_Char', lda_dir)
        gu.get_and_save_pcs_of_interest(self.lda_apa_predictions, [-1], lda_dir, conditions=self.conditions, reglda='lda', accmse='acc')

    def _run_lda_for_pair(self, s, midx, cond1, cond2):
        # pcs1, mask1_1, mask1_2, df1 = self.filter_data(self.feature_data[cond1], s, midx)
        # pcs2, mask2_1, mask2_2, df2 = self.filter_data(self.feature_data[cond2], s, midx)
        #
        # runs1_p1 = df1.index[mask1_1]
        # runs2_p1 = df2.index[mask2_1]
        # runs1_p2 = df1.index[mask1_2]
        # runs2_p2 = df2.index[mask2_2]
        pcs_apa_1, runs_apa_1 = self.filter_data(self.feature_data[cond1], s, midx, 'APA2')
        pcs_apa_2, runs_apa_2 = self.filter_data(self.feature_data[cond2], s, midx, 'APA2')

        if self.method == 'wash_normalised':
            runs1_zeroed = runs_apa_1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            runs2_zeroed = runs_apa_2 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            runs_zeroed = np.concatenate([runs1_zeroed, runs2_zeroed])

            pcs = np.vstack([pcs_apa_1, pcs_apa_2])

            labels = np.array([0] * pcs_apa_1.shape[0] + [1] * pcs_apa_2.shape[0])
            results = LDA.compute_lda(pcs, labels, folds=5)
            y_pred, lda_weights, accuracy, w_folds, cv_acc, intercept = results
            pc_acc, null_acc, y_preds_pcs = LDA.compute_lda_pcwise(pcs, labels, lda_weights, intercept, shuffles=1000)

            lda_data = dc.LDAPredictionData(
                conditions= [cond1, cond2],
                phase='apa', stride=s, mouse_id=midx, x_vals=runs_zeroed,
                y_pred=y_pred, y_preds_pcs=y_preds_pcs, weights=lda_weights,
                accuracy=accuracy, cv_acc=cv_acc, w_folds=w_folds,
                pc_acc=pc_acc, null_acc=null_acc
            )
            self.lda_apa_predictions.append(lda_data)
            self.lda_wash_predictions.append(None)

            if self.other_condition and midx in condition_specific_settings[self.other_condition][
                'global_fs_mouse_ids']:
                pcs_other, runs_other = self.filter_data(self.feature_data[self.other_condition], s, midx, 'APA2')
                runs_other_zeroed = runs_other - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                    expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 2
                y_pred_other = np.dot(pcs_other[:, :global_settings['pcs_to_use']], lda_weights.T) + intercept

                # Store as projection-only
                lda_proj = dc.LDAPredictionData(
                    conditions=[self.other_condition],
                    phase='apa',
                    stride=s,
                    mouse_id=midx,
                    x_vals=runs_other_zeroed,
                    y_pred=y_pred_other,
                    y_preds_pcs=None,
                    weights=None,
                    accuracy=None,
                    cv_acc=None,
                    w_folds=None,
                    pc_acc=None,
                    null_acc=None
                )
                self.lda_apa_predictions.append(lda_proj)

        elif self.method == 'raw':
            pcs_wash_1, runs_wash_1 = self.filter_data(self.feature_data[cond1], s, midx, 'Wash2')
            pcs_wash_2, runs_wash_2 = self.filter_data(self.feature_data[cond2], s, midx, 'Wash2')

            runs_apa_zeroed_1 = runs_apa_1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            runs_apa_zeroed_2 = runs_apa_2 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            runs_wash_zeroed_1 = runs_wash_1 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0]
            runs_wash_zeroed_2 = runs_wash_2 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])

            pcs_apa = np.vstack([pcs_apa_1, pcs_apa_2])
            pcs_wash = np.vstack([pcs_wash_1, pcs_wash_2])

            labels_apa = np.array([0] * pcs_apa_1.shape[0] + [1] * pcs_apa_2.shape[0])
            labels_wash = np.array([0] * pcs_wash_1.shape[0] + [1] * pcs_wash_2.shape[0])

            y_pred2, w, acc, w_folds, cv_acc, intercept = LDA.compute_lda(pcs_wash, labels_wash, folds=5)
            pc_acc, null_acc, y_preds_pcs = LDA.compute_lda_pcwise(pcs_wash, labels_wash, w, intercept, shuffles=1000)

            lda_wash = dc.LDAPredictionData(
                conditions=[cond1, cond2],
                phase='wash', stride=s, mouse_id=midx, x_vals=np.concatenate([runs_wash_zeroed_1, runs_wash_zeroed_2]),
                y_pred=y_pred2, y_preds_pcs=y_preds_pcs, weights=w, accuracy=acc,
                cv_acc=cv_acc, w_folds=w_folds, pc_acc=pc_acc, null_acc=null_acc
            )
            self.lda_wash_predictions.append(lda_wash)

            null_basis = null_space(w.reshape(1, -1))
            pcs_proj_null = pcs_apa @ null_basis
            y_pred3, w2, acc2, w_folds2, cv_acc2, intercept2 = LDA.compute_lda(pcs_proj_null, labels_apa, folds=5)
            w2_in_full = null_basis @ w2
            pc_acc2, null_acc2, y_preds_pcs2 = LDA.compute_lda_pcwise(pcs_apa, labels_apa, w2_in_full, intercept2, shuffles=1000)

            lda_apa = dc.LDAPredictionData(
                conditions=[cond1, cond2],
                phase='apa', stride=s, mouse_id=midx, x_vals=np.concatenate([runs_apa_zeroed_1, runs_apa_zeroed_2]),
                y_pred=y_pred3, y_preds_pcs=y_preds_pcs2, weights=w2_in_full, accuracy=acc2,
                cv_acc=cv_acc2, w_folds=w_folds2, pc_acc=pc_acc2, null_acc=null_acc2
            )
            self.lda_apa_predictions.append(lda_apa)

            if self.other_condition and midx in condition_specific_settings[self.other_condition]['global_fs_mouse_ids']:
                pcs_other, runs_other = self.filter_data(self.feature_data[self.other_condition], s, midx, 'APA2')
                runs_other_zeroed_p1 = (runs_other - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] +
                                        len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 2)
                y_pred_other = np.dot(pcs_other[:, :global_settings['pcs_to_use']], w2_in_full.T) + intercept2

                lda_proj = dc.LDAPredictionData(
                    conditions=[self.other_condition],
                    phase='apa',
                    stride=s,
                    mouse_id=midx,
                    x_vals=runs_other_zeroed_p1,
                    y_pred=y_pred_other,
                    y_preds_pcs=None,
                    weights=None,
                    accuracy=None,
                    cv_acc=None,
                    w_folds=None,
                    pc_acc=None,
                    null_acc=None
                )
                self.lda_apa_predictions.append(lda_proj)

        else:
            raise ValueError("Method must be either 'wash_normalised' or 'raw'.")

def main():
    all_conditions = ['APAChar_LowHigh', 'APAChar_LowMid', 'APAChar_HighLow']
    all_pairs = list(itertools.combinations(all_conditions, 2))

    for cond1, cond2 in all_pairs:
        for method in ['wash_normalised', 'raw']:
            other_cond = list(set(all_conditions) - {cond1, cond2})[0]

            outdir = fr"H:\\Characterisation\\Compare_{cond1.split('_')[-1]}_vs_{cond2.split('_')[-1]}_LDA"
            print(f"Running comparison for {cond1} vs {cond2} using method {method} in directory {outdir}")
            runner = LDARunner(method, [cond1, cond2], outdir, other_condition=other_cond)
            runner.run()
            print("Comparison completed.")

if __name__ == '__main__':
    main()
