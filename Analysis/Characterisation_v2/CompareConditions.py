import pickle
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from scipy.linalg import null_space
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.AnalysisTools import LDA
from Analysis.Characterisation_v2 import DataClasses as dc


"""
Run LDA on LH vs HL pca.
Remember using LH pcs for HL too.
"""
# 'wash_normalised' or 'raw'
method = 'wash_normalised'

# import LowHigh data
with open(r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\preprocessed_data_APAChar_LowHigh.pkl", 'rb') as f:
    data_LH = pickle.load(f)
with open(r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_APAChar_LowHigh.pkl", 'rb') as f:
    pca_LH = pickle.load(f)

# import HighLow data
with open(r"H:\Characterisation\HL_LHpcsonly_LhWnrm_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_HighLow.pkl", 'rb') as f:
    data_HL_norm = pickle.load(f)
with open(r"H:\Characterisation\HL_LHpcsonly_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\preprocessed_data_APAChar_HighLow.pkl", 'rb') as f:
    data_HL = pickle.load(f)

pca = pca_LH[0].pca

LH_feature_data = data_LH['feature_data']

HL_feature_data = data_HL_norm['feature_data'] if method == 'wash_normalised' else data_HL['feature_data']

index = pd.MultiIndex.from_product([global_settings['stride_numbers'], condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']],
                                      names=['stride', 'mouse_id'])
accuracies = pd.DataFrame(index=index, columns=['accuracy', 'cv_accuracy'])
weights = pd.DataFrame(index=index, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])
runs_to_get = len(expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']) * 2 if method == 'raw' else len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']) * 2
run_scores = pd.DataFrame(index=index, columns=np.arange(runs_to_get))
lda_predictions = []
for s in global_settings['stride_numbers']:
    for midx in condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']:
        LH_pcs = pca.transform(LH_feature_data.loc(axis=0)[s, midx])
        HL_pcs = pca.transform(HL_feature_data.loc(axis=0)[s, midx])

        LH_mask1, LH_mask2 = gu.get_mask_p1_p2(LH_feature_data.loc(axis=0)[s, midx], global_settings['phases'][0], global_settings['phases'][1])
        HL_mask1, HL_mask2 = gu.get_mask_p1_p2(HL_feature_data.loc(axis=0)[s, midx], global_settings['phases'][0], global_settings['phases'][1])

        #########################################################
        ##### LDA separating wash normalised LH and HL data #####
        #########################################################
        if method == 'wash_normalised':
            LH_runs_p1 = LH_feature_data.loc(axis=0)[s, midx].index[LH_mask1]
            HL_runs_p1 = HL_feature_data.loc(axis=0)[s, midx].index[HL_mask1]

            LH_runs_zeroed = LH_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            HL_runs_zeroed = HL_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
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
            lda_weights, accuracy, w_folds, cv_acc = results

            results_pcwise = LDA.compute_lda_pcwise(pcs, labels, lda_weights, shuffles=1000)
            pc_acc, null_acc, y_preds = results_pcwise

            # Store results
            lda_data = dc.LDAPredictionData(
                phase='apa',
                stride=s,
                mouse_id=midx,
                x_vals=runs_zeroed,
                y_pred=y_preds,  # Linear combination of features and weights
                weights=lda_weights,
                accuracy=accuracy,
                cv_acc=cv_acc,
                w_folds=w_folds,
                pc_acc= pc_acc,  # Not computed in this function
                null_acc= null_acc  # Not computed in this function
            )

            # Run LDA to
            lda = LDA()
            lda.fit(pcs, labels)
            lda_predictions = lda.predict(pcs)
            accuracy = lda.score(pcs, labels)

            cv_scores = cross_val_score(lda, pcs, labels, cv=5)
            cv_accuracy = cv_scores.mean()

            accuracies.loc[(s, midx), 'accuracy'] = accuracy
            accuracies.loc[(s, midx), 'cv_accuracy'] = cv_accuracy

            lda_run_scores = pcs @ lda.coef_[0] + lda.intercept_[0]

            lda_weights = lda.coef_[0]

            weights.loc[(s, midx)] = lda_weights

            run_scores.loc[(s, midx), runs_zeroed] = lda_run_scores

        ######################################################
        ##### LDA separating wash in raw HL and HL data ######
        ######################################################
        elif method == 'raw':
            LH_runs_p1 = LH_feature_data.loc(axis=0)[s, midx].index[LH_mask1]
            HL_runs_p1 = HL_feature_data.loc(axis=0)[s, midx].index[HL_mask1]
            LH_runs_p2 = LH_feature_data.loc(axis=0)[s, midx].index[LH_mask2]
            HL_runs_p2 = HL_feature_data.loc(axis=0)[s, midx].index[HL_mask2]

            LH_runs_zeroed_p1 = LH_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0]
            HL_runs_zeroed_p1 = HL_runs_p1 - expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
            runs_zeroed_p1 = np.concatenate([LH_runs_zeroed_p1, HL_runs_zeroed_p1])
            LH_runs_zeroed_p2 = LH_runs_p2 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0]
            HL_runs_zeroed_p2 = HL_runs_p2 - expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'][0] + len(expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])
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

            # Run LDA to
            lda = LDA()
            lda.fit(pcs2, labels2)
            lda_predictions = lda.predict(pcs2)
            accuracy = lda.score(pcs2, labels2)

            cv_scores = cross_val_score(lda, pcs2, labels2, cv=5)
            cv_accuracy = cv_scores.mean()

            # accuracies.loc[(s, midx), 'accuracy'] = accuracy
            # accuracies.loc[(s, midx), 'cv_accuracy'] = cv_accuracy

            lda_run_scores = pcs2 @ lda.coef_[0] + lda.intercept_[0]

            lda_weights = lda.coef_[0]

            # weights.loc[(s, midx)] = lda_weights

            # run_scores.loc[(s, midx), runs_zeroed] = lda_run_scores

            # then find null space of lda_weights
            null_basis = null_space(lda.coef_) # lda.coef_ is shape (1, n_features)

            pcs_proj_null = pcs1 @ null_basis  # shape: (n_trials, n_pcs_used - 1)

            lda2 = LDA()
            lda2.fit(pcs_proj_null, labels1)
            lda2_weights = lda2.coef_[0]

            lda2_in_pca_space = null_basis @ lda2_weights

            # Accuracy and CV accuracy of the second LDA (on null-projected data)
            accuracy2 = lda2.score(pcs_proj_null, labels1)
            cv_scores2 = cross_val_score(lda2, pcs_proj_null, labels1, cv=5)
            cv_accuracy2 = cv_scores2.mean()

            accuracies.loc[(s, midx), 'accuracy'] = accuracy2
            accuracies.loc[(s, midx), 'cv_accuracy'] = cv_accuracy2

            # Run scores along the new LDA2 axis (in original PC space)
            lda2_run_scores = pcs1 @ lda2_in_pca_space + lda2.intercept_[0]

            run_scores.loc[(s, midx), runs_zeroed_p1] = lda2_run_scores
            weights.loc[(s, midx)] = lda2_in_pca_space


        else:
            raise ValueError("Method must be either 'wash_normalised' or 'raw'.")

run_scores = run_scores.reindex(sorted(run_scores.columns, key=int), axis=1)

# plot stride -1 weights across PCs for all mice
import matplotlib.pyplot as plt

plt.figure()
for midx in weights.index.levels[1]:
    plt.plot(weights.loc[-1, midx], label=midx)


run_score_means = run_scores.loc(axis=0)[-1].mean(axis=0)
plt.figure()
plt.plot(run_score_means, label='Mean Run Scores')
plt.axvline(49.5, linestyle='--', color='red')
plt.axhline(0, linestyle='--', color='red')

run_score_mouse_means_p1 = run_scores.loc(axis=0)[-1].loc(axis=1)[:len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])-1].mean(axis=1)
run_score_mouse_means_p2 = run_scores.loc(axis=0)[-1].loc(axis=1)[len(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']):].mean(axis=1)
fig, ax = plt.subplots(figsize=(1, 4))
for midx in run_score_mouse_means_p1.index:
    ax.plot([run_score_mouse_means_p1[midx], run_score_mouse_means_p2[midx]], label=midx, linestyle='--', marker='o', markersize=3, linewidth=1, alpha=1)
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['LowHigh', 'HighLow'])
ax.set_xlim(-0.5, 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# change all font sizes to 7
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel('Condition', fontsize=7)
plt.ylabel('Prediction Score', fontsize=7)


###### run lda with raw HL data on wash's to find the belt1 speed element and then find orthogonal dimension (null space) to it for finding difference between APAs









