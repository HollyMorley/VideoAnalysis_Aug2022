import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2.Plotting import Regression_plotting as rp
from Analysis.Tools.config import condition_specific_settings, global_settings

def compute_regression(X, y, folds=10):
    model = LogisticRegression(penalty='none', fit_intercept=False)

    # cross-validate
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_acc = []
    w_folds = []
    for train_idx, test_idx in kf.split(np.arange(n_samples), y):
        ### start loop through pcs
        # Create a new model instance for each fold
        model_fold = LogisticRegression(penalty='none', fit_intercept=False)
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

    return w, bal_acc, cv_acc, w_folds

def compute_regression_pcwise_prediction(X, y, w, folds=10):
    w = w[0]
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    num_pcs = X.shape[0]

    cv_acc = np.zeros((num_pcs, folds)) # pcs x folds
    y_preds = np.zeros((num_pcs, X.shape[1])) # pcs x runs
    for pc in range(num_pcs):
        wpc = np.zeros(len(w))
        wpc[pc] = w[pc]

        for idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples), y)):
            y_pred = np.dot(wpc, X[:, test_idx])

            y_pred[y_pred > 0] = 1
            y_pred[y_pred < 0] = 0

            bal_acc = balanced_accuracy(y[test_idx], y_pred.ravel()) # not sure why ravel
            cv_acc[pc, idx] = bal_acc

        y_pred = np.dot(wpc, X)
        y_preds[pc, :] = y_pred

    return cv_acc, y_preds

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
    w, normalize_mean, normalize_std, y_reg, full_accuracy, cv_acc, w_folds, cv_acc_PCwise, y_preds_PCwise = fit_regression_model(loadings, selected_feature_data,
                                                                                  mask_p1, mask_p2)

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

    return y_pred, smoothed_y_pred, feature_weights, w, normalize_mean, normalize_std, full_accuracy, cv_acc, w_folds, cv_acc_PCwise, y_preds_PCwise


def fit_regression_model(loadings: pd.DataFrame, selected_feature_data: pd.DataFrame,
                         mask_p1: np.ndarray, mask_p2: np.ndarray):
    # trim loadings
    loadings = loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings.T, selected_feature_data)

    # Normalize X
    Xdr, normalize_mean, normalize_std = gu.normalize_Xdr(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_p1)), np.zeros(np.sum(mask_p2))])

    # Run logistic regression on the full model
    w, bal_acc, cv_acc, w_folds = compute_regression(Xdr, y_reg)
    cv_acc_PCwise, y_preds_PCwise = compute_regression_pcwise_prediction(Xdr, y_reg, w)
    # w, full_accuracy, cv_acc = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {bal_acc:.3f}")

    return w, normalize_mean, normalize_std, y_reg, bal_acc, cv_acc, w_folds, cv_acc_PCwise, y_preds_PCwise

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

def calculate_PC_prediction_significance(cv_acc_PCwise, stride, chance_level=0.5):
    accuracies = [accs for accs in cv_acc_PCwise if accs.stride == stride ]
    t_stat, p_value = wilcoxon(accuracies, chance_level)
    return t_stat, p_value

