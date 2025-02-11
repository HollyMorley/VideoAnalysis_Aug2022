import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy

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

