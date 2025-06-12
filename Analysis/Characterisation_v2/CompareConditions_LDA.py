import pickle
import pandas as pd
from scipy.linalg import null_space
import os

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.AnalysisTools import LDA
from Analysis.Characterisation_v2.Plotting import LDA_plotting
from Analysis.Characterisation_v2 import DataClasses as dc


class LDARunner:
    def __init__(self, method, base_dir):
        self.method = method
        self.base_dir = base_dir
        self.lda_apa_predictions = []
        self.lda_wash_predictions = []
        os.makedirs(self.base_dir, exist_ok=True)

        # Load all data
        self._load_data()

    def _load_data(self):
        with open(r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\preprocessed_data_APAChar_LowHigh.pkl", 'rb') as f:
            self.data_LH = pickle.load(f)
        with open(r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\APAChar_LowHigh_Extended\\MultiFeaturePredictions\\pca_APAChar_LowHigh.pkl", 'rb') as f:
            self.pca_LH = pickle.load(f)
        with open(r"H:\\Characterisation\\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_HighLow.pkl", 'rb') as f:
            self.data_HL_norm = pickle.load(f)
        with open(r"H:\\Characterisation\\HL_LHpcsonly_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\preprocessed_data_APAChar_HighLow.pkl", 'rb') as f:
            self.data_HL = pickle.load(f)

        self.pca = self.pca_LH[0].pca
        self.LH_feature_data = self.data_LH['feature_data']
        self.HL_feature_data = self.data_HL_norm['feature_data'] if self.method == 'wash_normalised' else self.data_HL['feature_data']

    def run(self):
        for s in global_settings['stride_numbers']:
            for midx in condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']:
                self._run_lda_for_pair(s, midx)

        lda_dir = os.path.join(self.base_dir, f'LDA_{self.method}')
        os.makedirs(lda_dir, exist_ok=True)

        LDA_plotting.plot_lda_weights(self.lda_apa_predictions, -1, 'LowHigh', 'HighLow', 'APA_Char', lda_dir)
        mean_preds, interp_preds = LDA_plotting.plot_prediction_per_trial(self.lda_apa_predictions, -1, 'LowHigh', 'HighLow', 'APA_Char', lda_dir)
        LDA_plotting.plot_prediction_discrete_conditions(interp_preds, -1, 'LowHigh', 'HighLow', 'APA_Char', lda_dir)
        gu.get_and_save_pcs_of_interest(self.lda_apa_predictions, [-1], lda_dir, reglda='lda', accmse='acc')

    def _run_lda_for_pair(self, s, midx):
        LH_pcs = self.pca.transform(self.LH_feature_data.loc(axis=0)[s, midx])
        HL_pcs = self.pca.transform(self.HL_feature_data.loc(axis=0)[s, midx])

        LH_mask1, LH_mask2 = gu.get_mask_p1_p2(self.LH_feature_data.loc(axis=0)[s, midx], global_settings['phases'][0],
                                               global_settings['phases'][1])
        HL_mask1, HL_mask2 = gu.get_mask_p1_p2(self.HL_feature_data.loc(axis=0)[s, midx], global_settings['phases'][0],
                                               global_settings['phases'][1])

        #########################################################
        ##### LDA separating wash normalised LH and HL data #####
        #########################################################
        if self.method == 'wash_normalised':
            LH_runs_p1 = self.LH_feature_data.loc(axis=0)[s, midx].index[LH_mask1]
            HL_runs_p1 = self.HL_feature_data.loc(axis=0)[s, midx].index[HL_mask1]

            LH_runs_zeroed = LH_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            HL_runs_zeroed = HL_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            runs_zeroed = np.concatenate([LH_runs_zeroed, HL_runs_zeroed])

            LH_pcs_p1 = LH_pcs[LH_mask1]
            HL_pcs_p1 = HL_pcs[HL_mask1]

            # Trim to pcs_to_use
            LH_pcs_p1_trim = LH_pcs_p1[:, :global_settings['pcs_to_use']]
            HL_pcs_p1_trim = HL_pcs_p1[:, :global_settings['pcs_to_use']]

            pcs = np.vstack([LH_pcs_p1_trim, HL_pcs_p1_trim])

            # get labels for lH and HL
            labels = np.array([0] * LH_pcs_p1_trim.shape[0] + [1] * HL_pcs_p1_trim.shape[0])

            results = LDA.compute_lda(pcs, labels, folds=5)
            y_pred, lda_weights, accuracy, w_folds, cv_acc, intercept = results

            results_pcwise = LDA.compute_lda_pcwise(pcs, labels, lda_weights, intercept, shuffles=1000)
            pc_acc, null_acc, y_preds_pcs = results_pcwise

            # Store results
            lda_data = dc.LDAPredictionData(
                phase='apa',
                stride=s,
                mouse_id=midx,
                x_vals=runs_zeroed,
                y_pred=y_pred,
                y_preds_pcs=y_preds_pcs,
                weights=lda_weights,
                accuracy=accuracy,
                cv_acc=cv_acc,
                w_folds=w_folds,
                pc_acc=pc_acc,
                null_acc=null_acc
            )
            self.lda_apa_predictions.append(lda_data)
            self.lda_wash_predictions.append(None)

        ######################################################
        ##### LDA separating wash in raw HL and HL data ######
        ######################################################
        elif self.method == 'raw':
            LH_runs_p1 = self.LH_feature_data.loc(axis=0)[s, midx].index[LH_mask1]
            HL_runs_p1 = self.HL_feature_data.loc(axis=0)[s, midx].index[HL_mask1]
            LH_runs_p2 = self.LH_feature_data.loc(axis=0)[s, midx].index[LH_mask2]
            HL_runs_p2 = self.HL_feature_data.loc(axis=0)[s, midx].index[HL_mask2]

            LH_runs_zeroed_p1 = LH_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            HL_runs_zeroed_p1 = HL_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            runs_zeroed_p1 = np.concatenate([LH_runs_zeroed_p1, HL_runs_zeroed_p1])
            LH_runs_zeroed_p2 = LH_runs_p2 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0]
            HL_runs_zeroed_p2 = HL_runs_p2 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0] + len(
                expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])
            runs_zeroed_p2 = np.concatenate([LH_runs_zeroed_p2, HL_runs_zeroed_p2])

            # Get pcs for p1
            LH_pcs_p1 = LH_pcs[LH_mask1]
            HL_pcs_p1 = HL_pcs[HL_mask1]
            # Get pcs for p2
            LH_pcs_p2 = LH_pcs[LH_mask2]
            HL_pcs_p2 = HL_pcs[HL_mask2]

            # Trim to pcs_to_use
            LH_pcs_p1_trim = LH_pcs_p1[:, :global_settings['pcs_to_use']]
            HL_pcs_p1_trim = HL_pcs_p1[:, :global_settings['pcs_to_use']]
            LH_pcs_p2_trim = LH_pcs_p2[:, :global_settings['pcs_to_use']]
            HL_pcs_p2_trim = HL_pcs_p2[:, :global_settings['pcs_to_use']]

            pcs1 = np.vstack([LH_pcs_p1_trim, HL_pcs_p1_trim])
            pcs2 = np.vstack([LH_pcs_p2_trim, HL_pcs_p2_trim])

            # get labels for lH and HL
            labels1 = np.array([0] * LH_pcs_p1_trim.shape[0] + [1] * HL_pcs_p1_trim.shape[0])
            labels2 = np.array([0] * LH_pcs_p2_trim.shape[0] + [1] * HL_pcs_p2_trim.shape[0])

            results = LDA.compute_lda(pcs2, labels2, folds=5)
            y_pred, lda_weights, accuracy, w_folds, cv_acc, intercept = results
            results_pcwise = LDA.compute_lda_pcwise(pcs2, labels2, lda_weights, intercept, shuffles=1000)
            pc_acc, null_acc, y_preds_pcs = results_pcwise

            lda_wash_data = dc.LDAPredictionData(
                phase='wash',
                stride=s,
                mouse_id=midx,
                x_vals=runs_zeroed_p2,
                y_pred=y_pred,
                y_preds_pcs=y_preds_pcs,
                weights=lda_weights,
                accuracy=accuracy,
                cv_acc=cv_acc,
                w_folds=w_folds,
                pc_acc=pc_acc,
                null_acc=null_acc
            )
            self.lda_wash_predictions.append(lda_wash_data)

            null_basis = null_space(lda_weights.reshape(1, -1))  # lda_weights is shape (n_features,)
            pcs_proj_null = pcs1 @ null_basis  # shape: (n_trials, n_pcs_used - 1)

            results2 = LDA.compute_lda(pcs_proj_null, labels1, folds=5)
            y_pred2, lda2_weights, accuracy2, w_folds2, cv_acc2, intercept2 = results2
            lda2_w_in_pca_space = null_basis @ lda2_weights
            # Now compute per‐PC accuracy/null‐accuracy back on the ORIGINAL 12 PCs (pcs1):
            results_pcwise2_orig = LDA.compute_lda_pcwise(
                pcs1,  # the 12‐D data for phase 1
                labels1,
                lda2_w_in_pca_space,  # mapped‐back 12‐D weight vector
                intercept2,
                shuffles=1000
            )
            pc_acc2_orig, null_acc2_orig, y_preds_pcs2_orig = results_pcwise2_orig

            lda_apa_data = dc.LDAPredictionData(
                phase='apa',
                stride=s,
                mouse_id=midx,
                x_vals=runs_zeroed_p1,
                y_pred=y_pred2,
                y_preds_pcs=y_preds_pcs2_orig,  # shape = (12, n_trials)
                weights=lda2_w_in_pca_space,  # 12 weights
                accuracy=accuracy2,
                cv_acc=cv_acc2,
                w_folds=w_folds2,
                pc_acc=pc_acc2_orig,  # now length 12
                null_acc=null_acc2_orig  # now (12 × shuffles)
            )
            self.lda_apa_predictions.append(lda_apa_data)

        else:
            raise ValueError("Method must be either 'wash_normalised' or 'raw'.")


def main():
    base_save_dir = r"H:\\Characterisation\\Compare_LH_HL_LDA"
    for method in ['wash_normalised', 'raw']:
        runner = LDARunner(method, base_save_dir)
        runner.run()


if __name__ == '__main__':
    main()


