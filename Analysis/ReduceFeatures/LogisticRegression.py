import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt

from Analysis.ReduceFeatures import utils_feature_reduction as utils

def compute_regression(X, y):
    model = LogisticRegression(penalty='none', fit_intercept=False)
    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)
    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    return w, bal_acc

def compute_lasso_regression(X, y):
    # Use L1 penalty for Lasso-style logistic regression.
    model = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1.0)
    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)
    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    return w, bal_acc


def find_unique_and_single_contributions(selected_scaled_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy):
    single_all = []
    unique_all = []

    # Shuffle X and run logistic regression
    for fidx, feature in enumerate(selected_scaled_data_df.index):
        #print(f"Shuffling feature {fidx + 1}/{len(selected_scaled_data_df.index)}")
        single_shuffle = utils.shuffle_single(feature, selected_scaled_data_df)
        unique_shuffle = utils.shuffle_unique(feature, selected_scaled_data_df)
        Xdr_shuffled_single = np.dot(loadings_df.T, single_shuffle)
        Xdr_shuffled_single = ((Xdr_shuffled_single.T - normalize_mean) / normalize_std).T
        Xdr_shuffled_unique = np.dot(loadings_df.T, unique_shuffle)
        Xdr_shuffled_unique = ((Xdr_shuffled_unique.T - normalize_mean) / normalize_std).T

        _, single_accuracy = compute_regression(Xdr_shuffled_single, y_reg)
        _, unique_accuracy = compute_regression(Xdr_shuffled_unique, y_reg)
        unique_contribution = full_accuracy - unique_accuracy
        single_all.append(single_accuracy)
        unique_all.append(unique_contribution)
    single_all = np.array(single_all)
    unique_all = np.array(unique_all)
    single_all_dict = dict(zip(selected_scaled_data_df.index, single_all.flatten()))
    unique_all_dict = dict(zip(selected_scaled_data_df.index, unique_all.flatten()))

    return single_all_dict, unique_all_dict

    # shuffle_all = utils.shuffle_single(feature, selected_scaled_data_df)
    # shuffle_all = utils.shuffle_unique(feature, shuffle_all)
    # shuffle_all = np.dot(loadings_df.T, shuffle_all)
    # shuffle_all = ((shuffle_all.T - normalize_mean) / normalize_std).T
    # _, full_accuracy_shuffled = compute_regression(shuffle_all, y_reg, selected_scaled_data_df, cv=5)
    #
    #
def find_full_shuffle_accuracy(selected_scaled_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy):
    feature = selected_scaled_data_df.index[0]

    shuffle_all = utils.shuffle_single(feature, selected_scaled_data_df)
    shuffle_all = utils.shuffle_unique(feature, shuffle_all)
    shuffle_all = np.dot(loadings_df.T, shuffle_all)
    shuffle_all = ((shuffle_all.T - normalize_mean) / normalize_std).T
    _, full_accuracy_shuffled = compute_regression(shuffle_all, y_reg)
    return full_accuracy_shuffled

def fit_regression_model(loadings_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2):
    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings_df.T, reduced_feature_selected_data_df)

    # Normalize X
    Xdr, normalize_mean, normalize_std = utils.normalize(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])

    # Run logistic regression on the full model
    w, full_accuracy = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {full_accuracy:.3f}")
    return w, normalize_mean, normalize_std, y_reg, full_accuracy


def predict_runs(loadings_df, reduced_feature_data_df, normalize_mean, normalize_std, w, save_path, mouse_id, phase1, phase2, condition_name):
    # Apply the full model to all runs (scaled and unscaled)
    all_trials_dr = np.dot(loadings_df.T, reduced_feature_data_df.T)
    all_trials_dr = ((all_trials_dr.T - normalize_mean) / normalize_std).T
    run_pred = np.dot(w, np.dot(loadings_df.T, reduced_feature_data_df.T))
    run_pred_scaled = np.dot(w, all_trials_dr)

    # Compute smoothed scaled predictions for aggregation.
    smoothed_pred = medfilt(run_pred[0], kernel_size=5)
    smoothed_scaled_pred = medfilt(run_pred_scaled[0], kernel_size=5)

    # Plot run prediction
    utils.plot_run_prediction(reduced_feature_data_df, run_pred, smoothed_pred, save_path, mouse_id, phase1, phase2,
                              scale_suffix="", dataset_suffix=condition_name)
    utils.plot_run_prediction(reduced_feature_data_df, run_pred_scaled, smoothed_scaled_pred, save_path, mouse_id, phase1, phase2,
                              scale_suffix="scaled", dataset_suffix=condition_name)
    return smoothed_scaled_pred

def regression_feature_contributions(loadings_df, reduced_feature_selected_data_df, mouse_id, phase1, phase2, save_path, normalize_mean, normalize_std, y_reg, full_accuracy):
    # Shuffle features and run logistic regression to find unique contributions and single feature contributions
    single_all_dict, unique_all_dict = find_unique_and_single_contributions(reduced_feature_selected_data_df,
                                                                            loadings_df, normalize_mean,
                                                                            normalize_std, y_reg, full_accuracy)

    # Find full shuffled accuracy (one shuffle may not be enough)
    full_shuffled_accuracy = find_full_shuffle_accuracy(reduced_feature_selected_data_df, loadings_df,
                                                        normalize_mean, normalize_std, y_reg, full_accuracy)
    print(f"Full model shuffled accuracy: {full_shuffled_accuracy:.3f}")

    # Plot unique and single feature contributions
    utils.plot_unique_delta_accuracy(unique_all_dict, mouse_id, save_path, title_suffix=f"{phase1}_vs_{phase2}")
    utils.plot_feature_accuracy(single_all_dict, mouse_id, save_path, title_suffix=f"{phase1}_vs_{phase2}")

def run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, save_path, condition):
    w, normalize_mean, normalize_std, y_reg, full_accuracy = fit_regression_model(loadings_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2)

    # Compute feature contributions
    regression_feature_contributions(loadings_df, reduced_feature_selected_data_df, mouse_id, phase1, phase2, save_path, normalize_mean, normalize_std, y_reg, full_accuracy)

    # Compute feature-space weights for this mouse
    feature_weights = loadings_df.dot(w.T).squeeze()
    # Plot the weights in the original feature space
    utils.plot_weights_in_feature_space(feature_weights, save_path, mouse_id, phase1, phase2)

    # Predict runs using the full model
    smoothed_scaled_pred = predict_runs(loadings_df, reduced_feature_data_df, normalize_mean, normalize_std, w, save_path, mouse_id, phase1, phase2, condition)

    return smoothed_scaled_pred, feature_weights, w



