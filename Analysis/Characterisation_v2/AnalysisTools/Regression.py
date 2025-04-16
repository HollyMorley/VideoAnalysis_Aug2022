import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, ttest_1samp
import matplotlib.pyplot as plt
import os

from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.Plotting import Regression_plotting as rp
from Analysis.Tools.config import condition_specific_settings, global_settings

def compute_regression(X, y, folds=5):
    model = LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear', C=0.5)

    # cross-validate
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_acc = []
    w_folds = []
    for train_idx, test_idx in kf.split(np.arange(n_samples), y):
        ### start loop through pcs
        # Create a new model instance for each fold
        model_fold = LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear', C=0.5)
        # Train using the training columns
        model_fold.fit(X[:, train_idx].T, y[train_idx])


        w_fold = model_fold.coef_
        y_pred = np.dot(w_fold, X[:, test_idx])
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0

        acc_fold = balanced_accuracy(y[test_idx], y_pred.ravel())
        cv_acc.append(acc_fold)
        w_folds.append(w_fold)


    cv_acc = np.array(cv_acc)
    w_folds = np.array(w_folds)

    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)

    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    # y_pred3 = np.dot(w[0][2], X[2, :])
    # plt.figure()
    # plt.plot(y_pred3)
    # plt.show()

    return w, bal_acc, cv_acc, w_folds

def compute_regression_pcwise_prediction(X, y, w, folds=10, shuffles=100):
    w = w[0]
    n_samples = X.shape[1]
   # kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    num_pcs = X.shape[0]

    pc_acc = np.zeros((num_pcs,)) # pcs x folds
    null_acc = np.zeros((num_pcs, shuffles)) # pcs x folds
    y_preds = np.zeros((num_pcs, X.shape[1])) # pcs x runs
    for pc in range(num_pcs):
        wpc = w[pc]
        y_pred = np.dot(wpc, X[pc, :])  # wpc is a row vector, X is a column vector
        y_preds[pc, :] = y_pred
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0
        pc_acc[pc] = balanced_accuracy(y, y_pred.ravel())  # not sure why ravel

        for idx in range(shuffles):
            x_shuffle = np.random.permutation(X[pc, :].T).T
            y_pred_shuffle = np.dot(wpc, x_shuffle)
            y_pred_shuffle[y_pred_shuffle > 0] = 1
            y_pred_shuffle[y_pred_shuffle < 0] = 0
            bal_acc = balanced_accuracy(y, y_pred_shuffle)  # not sure why ravel
            null_acc[pc, idx] = bal_acc

        # for idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples), y)):
        #     y_pred = np.dot(wpc, X[:, test_idx])
        #
        #     y_pred[y_pred > 0] = 1
        #     y_pred[y_pred < 0] = 0
        #
        #     bal_acc = balanced_accuracy(y[test_idx], y_pred.ravel()) # not sure why ravel
        #     cv_acc[pc, idx] = bal_acc
        #
        #     shuffle_accs = []
        #     for it in np.arange(shuffles):
        #         shuffle_x_test = np.random.permutation(X[:, test_idx].T).T
        #         y_pred_shuffle = np.dot(wpc, shuffle_x_test)
        #
        #         y_pred_shuffle[y_pred_shuffle > 0] = 1
        #         y_pred_shuffle[y_pred_shuffle < 0] = 0
        #
        #         bal_acc_shuffle = balanced_accuracy(y[test_idx], y_pred_shuffle.ravel())
        #         shuffle_accs.append(bal_acc_shuffle)
        #     # average_shuffle_acc = np.mean(shuffle_accs)
        #     # shuffle_cv_acc[pc, idx] = average_shuffle_acc
        #     shuffle_cv_acc[pc, idx, :] = np.array(shuffle_accs)



    # delta_cv_acc = cv_acc - shuffle_cv_acc

    return pc_acc, y_preds, null_acc #, delta_cv_acc

def run_regression_on_PCA_and_predict(loadings: pd.DataFrame,
                       pcs: np.ndarray,
                       feature_data: pd.DataFrame,
                       selected_feature_data: pd.DataFrame,
                       mask_p1: np.ndarray,
                       mask_p2: np.ndarray,
                       mouse_id: str,
                       p1: str, p2: str, s: int,
                       condition: str,
                       save_path: str):

    # Fit regression model on PCA data
    results = fit_regression_model(loadings, selected_feature_data, mask_p1, mask_p2)
    (w, normalize_mean, normalize_std, y_reg, full_accuracy, cv_acc, w_folds, pc_acc, y_preds, null_acc) = results

    # Trim the weights and normalization parameters to the number of PCs to use
    w = np.array(w[0][:global_settings['pcs_to_use']]).reshape(1, -1)
    normalize_mean = normalize_mean[:global_settings['pcs_to_use']]
    normalize_std = normalize_std[:global_settings['pcs_to_use']]
    loadings = loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()

    # Transform regression weights in PCA space to feature space
    feature_weights = loadings.dot(w.T).squeeze()

    pc_weights = pd.Series(w[0], index=loadings.columns)

    # Plot the weights in the original feature space
    rp.plot_weights_in_feature_space(feature_weights, save_path, mouse_id, p1, p2, s, condition)
    rp.plot_weights_in_pc_space(pc_weights, save_path, mouse_id, p1, p2, s, condition)

    # Predict runs using the full model
    smoothed_y_pred, y_pred = predict_runs(loadings, feature_data, normalize_mean, normalize_std, w, #todo check dtypes are correct
                                           save_path, mouse_id, p1, p2, s, condition)

    return y_pred, smoothed_y_pred, feature_weights, w, normalize_mean, normalize_std, full_accuracy, cv_acc, w_folds, pc_acc, y_preds, null_acc


def fit_regression_model(loadings: pd.DataFrame, selected_feature_data: pd.DataFrame,
                         mask_p1: np.ndarray, mask_p2: np.ndarray):
    # trim loadings
    loadings = loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings.T, selected_feature_data)

    # Normalize X
    _, normalize_mean, normalize_std = gu.normalize_Xdr(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_p1)), np.zeros(np.sum(mask_p2))])

    # Run logistic regression on the full model
    w, bal_acc, cv_acc, w_folds = compute_regression(Xdr, y_reg)
    pc_acc, y_preds, null_acc = compute_regression_pcwise_prediction(Xdr, y_reg, w)

    # mean_cv_acc_PCwise = np.mean(cv_acc_PCwise, axis=1)
    # mean_cv_acc_shuffle_PCwise = np.mean(cv_acc_shuffle_PCwise, axis=1)

    # w, full_accuracy, cv_acc = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {bal_acc:.3f}")

    return w, normalize_mean, normalize_std, y_reg, bal_acc, cv_acc, w_folds, pc_acc, y_preds, null_acc

def predict_runs(loadings: pd.DataFrame, feature_data: pd.DataFrame, normalize_mean: float, normalize_std: float,
                 w: np.ndarray, save_path: str, mouse_id: str, p1: str, p2:str, s: int, condition_name: str):
    # Apply the full model to all runs (scaled and unscaled)
    all_trials_dr = np.dot(loadings.T, feature_data.T)
    all_trials_dr = ((all_trials_dr.T - normalize_mean) / normalize_std).T # pc wise normalization
    run_pred = np.dot(w, np.dot(loadings.T, feature_data.T))
    run_pred_scaled = np.dot(w, all_trials_dr)

    # Compute smoothed scaled predictions for aggregation.
    kernel_size = 5
    padded_run_pred = np.pad(run_pred[0], pad_width=kernel_size, mode='reflect')
    padded_run_pred_scaled = np.pad(run_pred_scaled[0], pad_width=kernel_size, mode='reflect')
    smoothed_pred = medfilt(padded_run_pred, kernel_size=kernel_size)
    smoothed_scaled_pred = medfilt(padded_run_pred_scaled, kernel_size=kernel_size)
    smoothed_pred = smoothed_pred[kernel_size:-kernel_size]
    smoothed_scaled_pred = smoothed_scaled_pred[kernel_size:-kernel_size]

    # Plot run prediction
    rp.plot_run_prediction(feature_data, run_pred_scaled, smoothed_scaled_pred, save_path, mouse_id, p1, p2, s,
                              scale_suffix="scaled", dataset_suffix=condition_name)
    return smoothed_scaled_pred, run_pred_scaled

def calculate_PC_prediction_significances(pca_pred, stride, mice_thresh):
    mouse_stride_preds = [pred for pred in pca_pred if pred.stride == stride ]

    accuracies_x_pcs = []
    accuracies_pcs_x_shuffle = []
    pc_weights = []
    for mouse_pred in mouse_stride_preds:
        accuracies_x_pcs.append(mouse_pred.pc_acc)
        accuracies_pcs_x_shuffle.append(mouse_pred.null_acc)
        pc_weights.append(mouse_pred.pc_weights[0])
    accuracies_x_pcs = np.array(accuracies_x_pcs) # mice x pcs
    accuracies_shuffle_x_pcs = np.array(accuracies_pcs_x_shuffle) # mice x pcs x shuffle
    mean_accs = accuracies_x_pcs.mean(axis=0)

    pc_weights = np.array(pc_weights)
    pos_counts = (pc_weights > 0).sum(axis=0)
    neg_counts = (pc_weights < 0).sum(axis=0)
    max_counts = np.maximum(pos_counts, neg_counts)
    total_mice_num = len(mouse_stride_preds)
    ideal_mice_num  = total_mice_num - mice_thresh
    counts_more_than_thresh = max_counts >= ideal_mice_num


    delta_acc_by_mouse = accuracies_x_pcs - accuracies_shuffle_x_pcs.mean(axis=2)
    pc_significances = np.zeros((delta_acc_by_mouse.shape[1]))
    for pc in np.arange(delta_acc_by_mouse.shape[1]):
        pc_acc = delta_acc_by_mouse[:, pc]
        stat = ttest_1samp(pc_acc, 0)
        pc_significances[pc] = stat.pvalue


    return pc_significances, mean_accs, counts_more_than_thresh

def find_residuals(feature_data, stride_numbers, phases, savedir):
    all_residuals = []
    for s in stride_numbers:
        print(f"Stride {s} - finding residuals for {phases[0]} and {phases[1]}")
        stride_features = feature_data.loc(axis=0)[s]
        mask_p1, mask_p2 = gu.get_mask_p1_p2(stride_features, phases[0], phases[1])
        phase_stride_features = pd.concat(
            [stride_features[mask_p1], stride_features[mask_p2]])
        residuals = phase_stride_features.groupby(level='MouseID').apply(gu.compute_residuals, s, savedir)
        all_residuals.append(residuals)
    all_residuals_df =  pd.concat(all_residuals, keys=stride_numbers, names=['Stride'])

    # save
    all_residuals_df.to_hdf(os.path.join(savedir, "ResidualData.h5"), key="residuals", mode="w")
    return all_residuals_df

















