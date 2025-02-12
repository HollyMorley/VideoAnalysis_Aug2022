import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import random
from tqdm import tqdm
import ast
from joblib import Parallel, delayed
from collections import Counter

from Analysis.ReduceFeatures.PCA import perform_pca, plot_pca, plot_scree, cross_validate_pca, plot_average_variance_explained_across_folds
from Analysis.ReduceFeatures.LDA import perform_lda, compute_feature_contributions, plot_feature_contributions, plot_lda_transformed_data, plot_LDA_loadings
from Analysis.ReduceFeatures.LogisticRegression import compute_regression, compute_lasso_regression, find_unique_and_single_contributions, find_full_shuffle_accuracy
from Analysis.ReduceFeatures.KNN import ImbalancedKNN
from Analysis.ReduceFeatures import utils_feature_reduction as utils
from Helpers.Config_23 import *

# ----------------------------
# Configuration Section
# ----------------------------

# Set your parameters here
global_fs_mouse_ids = ['1035243', '1035250']
mouse_ids = ['1035243', '1035250']
# mouse_ids = [
#     '1035243', '1035244', '1035245', '1035246',
#     '1035249', '1035250', '1035297', '1035298',
#     '1035299', '1035301', '1035302'
# ]  # List of mouse IDs to analyze
stride_numbers = [-1]  # List of stride numbers to filter data
phases = ['APA2', 'Wash2']  # List of phases to compare
base_save_dir = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round9-20250212-global-lasso')
overwrite_FeatureSelection = False

n_iterations_selection = 100
nFolds_selection = 5

# ----------------------------
# Function Definitions
# ----------------------------
sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

def load_and_preprocess_data(mouse_id, stride_number, condition, exp, day):
    """
    Load data for the specified mouse and preprocess it by selecting desired features,
    imputing missing values, and standardizing.
    """
    if exp == 'Extended':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_single_kinematics_runXstride.h5")
    elif exp == 'Repeats':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\Wash\\Exp\\{day}\\MEASURES_single_kinematics_runXstride.h5")
    else:
        raise ValueError(f"Unknown experiment type: {exp}")
    data_allmice = pd.read_hdf(filepath, key='single_kinematics')

    try:
        data = data_allmice.loc[mouse_id]
    except KeyError:
        raise ValueError(f"Mouse ID {mouse_id} not found in the dataset.")

    # Build desired columns using the simplified build_desired_columns function
    measures = measures_list_feature_reduction

    col_names = []
    for feature in measures.keys():
        if any(measures[feature]):
            if feature != 'signed_angle':
                for param in itertools.product(*measures[feature].values()):
                    param_names = list(measures[feature].keys())
                    formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                    col_names.append((feature, formatted_params))
            else:
                for combo in measures['signed_angle'].keys():
                    col_names.append((feature, combo))
        else:
            col_names.append((feature, 'no_param'))

    col_names_trimmed = []
    for c in col_names:
        if np.logical_and('full_stride:True' in c[1], 'step_phase:None' not in c[1]):
            pass
        elif np.logical_and('full_stride:False' in c[1], 'step_phase:None' in c[1]):
            pass
        else:
            col_names_trimmed.append(c)

    selected_columns = col_names_trimmed


    filtered_data = data.loc[:, selected_columns]

    separator = '|'
    # Collapse MultiIndex columns to single-level strings including group info.
    filtered_data.columns = [
        f"{measure}{separator}{params}" if params != 'no_param' else f"{measure}"
        for measure, params in filtered_data.columns
    ]

    try:
        filtered_data = filtered_data.xs(stride_number, level='Stride', axis=0)
    except KeyError:
        raise ValueError(f"Stride number {stride_number} not found in the data.")

    filtered_data_imputed = filtered_data.fillna(filtered_data.mean())

    if filtered_data_imputed.isnull().sum().sum() > 0:
        print("Warning: There are still missing values after imputation.")
    else:
        print("All missing values have been handled.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data_imputed)
    scaled_data_df = pd.DataFrame(scaled_data, index=filtered_data_imputed.index,
                                  columns=filtered_data_imputed.columns)
    return scaled_data_df

def get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2):
    mask_phase1 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase1])
    mask_phase2 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase2])

    if not mask_phase1.any():
        raise ValueError(f"No runs found for phase '{phase1}'.")
    if not mask_phase2.any():
        raise ValueError(f"No runs found for phase '{phase2}'.")

    run_numbers_phase1 = scaled_data_df.index[mask_phase1]
    run_numbers_phase2 = scaled_data_df.index[mask_phase2]
    run_numbers = list(run_numbers_phase1) + list(run_numbers_phase2)

    # Determine stepping limbs.
    stepping_limbs = [utils.determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                      for run in run_numbers]

    return run_numbers, stepping_limbs, mask_phase1, mask_phase2

def get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2):
    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]

    labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
    labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
    labels = np.concatenate([labels_phase1, labels_phase2])
    pcs_combined = np.vstack([pcs_phase1, pcs_phase2])

    return pcs_combined, labels, pcs_phase1, pcs_phase2

def rfe_feature_selection(selected_scaled_data_df, y, cv=5):
    """
    Performs feature selection using RFECV with L1-regularized logistic regression.
    Assumes selected_scaled_data_df has features as rows and samples as columns.
    """
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    # Transpose so that rows are samples and columns are features.
    X = selected_scaled_data_df.T
    estimator = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1.0)
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='balanced_accuracy')
    rfecv.fit(X, y)
    selected_features = selected_scaled_data_df.index[rfecv.support_]
    print(f"RFECV selected {rfecv.n_features_} features.")
    return selected_features, rfecv

def random_forest_feature_selection(selected_scaled_data_df, y):
    """
    Performs feature selection using a Random Forest to rank features and selects those
    with importance above the median.
    """
    from sklearn.ensemble import RandomForestClassifier
    X = selected_scaled_data_df.T  # rows: samples, columns: features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    threshold = np.median(importances)
    selected_features = selected_scaled_data_df.index[importances > threshold]
    print(f"Random Forest selected {len(selected_features)} features (threshold: {threshold:.4f}).")
    return selected_features, rf


def unified_feature_selection(feature_data_df, y, method='rfecv', cv=nFolds_selection, n_iterations=n_iterations_selection, fold_assignment=None):
    """
    Unified feature selection function that can be used both in local (single-mouse)
    and global (aggregated) cases.

    Parameters:
      - feature_data_df: DataFrame with features as rows and samples as columns.
      - y: target vector (e.g. binary labels)
      - method: 'rfecv', 'rf', or 'regression'
      - cv: number of folds for cross-validation (if applicable)
      - n_iterations: number of shuffles for regression-based selection
      - fold_assignment: optional pre-computed fold assignment dictionary for regression-based selection.

    Returns:
      - selected_features: the index (or list) of features that were selected.
    """
    if method == 'rfecv':
        # Use RFECV for the entire data matrix.
        selected_features, rfecv_model = rfe_feature_selection(feature_data_df, y, cv=cv)
        print(f"RFECV selected {rfecv_model.n_features_} features.")
        return selected_features
    elif method == 'rf':
        # Use a Random Forest to rank features.
        selected_features, rf_model = random_forest_feature_selection(feature_data_df, y)
        print(f"Random Forest selected {len(selected_features)} features.")
        return selected_features
    elif method == 'regression':
        # Fall back to your current per-feature method.
        N = feature_data_df.shape[1]
        if fold_assignment is None:
            indices = list(range(N))
            random.shuffle(indices)
            fold_assignment = {i: (j % cv + 1) for j, i in enumerate(indices)}
        features = list(feature_data_df.index)
        results = Parallel(n_jobs=-1)(
            delayed(process_single_feature)(
                feature,
                feature_data_df.loc[feature].values,
                fold_assignment,
                y,
                list(range(N)),
                cv,
                n_iterations
            )
            for feature in tqdm(features, desc="Unified regression-based feature selection")
        )
        all_feature_accuracies = dict(results)
        all_feature_accuracies_df = pd.DataFrame.from_dict(all_feature_accuracies, orient='index')
        # Select features where the true accuracy is significantly better than shuffled.
        selected_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
        return selected_features
    else:
        raise ValueError("Unknown method specified for feature selection.")


def process_single_feature(feature, X, fold_assignment, y_reg, run_numbers, nFolds, n_iterations):
    fold_true_accuracies = []
    iteration_diffs_all = {i: [] for i in range(n_iterations)}
    #knn = KNeighborsClassifier(n_neighbors=5)
    # find custom weights based on class imbalance
    class_counts = Counter(y_reg)
    n_samples = len(y_reg)
    custom_weights = {
        cls: n_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    #knn = ImbalancedKNN(n_neighbors=5, weights=custom_weights)

    # Loop over folds
    for fold in range(1, nFolds + 1):
        test_mask = np.array([fold_assignment[run] == fold for run in run_numbers])
        train_mask = ~test_mask

        # Training on the training set
        X_fold_train = X[train_mask].reshape(1, -1) # Get feature values across training runs in current fold
        y_reg_fold_train = y_reg[train_mask]  #  Create y (regression target) - 1 for phase1, 0 for phase2 - for this fold

        w, _ = compute_lasso_regression(X_fold_train, y_reg_fold_train) # Run logistic regression on single feature to get weights
        #w, _ = compute_regression(X_fold_train, y_reg_fold_train) # Run logistic regression on single feature to get weights
        #knn.fit(X_fold_train.T, y_reg_fold_train)

        # Testing on the test set
        X_fold_test = X[test_mask].reshape(1, -1) # Get feature values across test runs in current fold
        y_reg_fold_test = y_reg[test_mask] # Create y (regression target) - 1 for phase1, 0 for phase2 - for this fold
        y_pred = np.dot(w, X_fold_test) # Get accuracy from test set
        y_pred[y_pred > 0] = 1 # change y_pred +ves to 1 and -ves to 0
        y_pred[y_pred < 0] = 0 # change y_pred +ves to 1 and -ves to 0
        #y_pred = knn.predict(X_fold_test.T) ## add in weights for
        feature_accuracy_test = utils.balanced_accuracy(y_reg_fold_test.T, y_pred.T) # Get balanced accuracy from test set
        fold_true_accuracies.append(feature_accuracy_test)


        # For each iteration: shuffle and compute difference in accuracy.
        for i in range(n_iterations):
            X_shuffled = X.copy()
            random.shuffle(X_shuffled)

            X_shuffled_fold_train = X_shuffled[train_mask].reshape(1, -1)
            # Run logistic regression on shuffled data
            w, _ = compute_lasso_regression(X_shuffled_fold_train, y_reg_fold_train)
            w, _ = compute_regression(X_shuffled_fold_train, y_reg_fold_train)
            # knn.fit(X_shuffled_fold_train.T, y_reg_fold_train)

            X_shuffled_fold_test = X_shuffled[test_mask].reshape(1, -1)
            y_pred_shuffle = np.dot(w, X_shuffled_fold_test)
            y_pred_shuffle[y_pred_shuffle > 0] = 1
            y_pred_shuffle[y_pred_shuffle < 0] = 0
            # y_pred_shuffle = knn.predict(X_shuffled_fold_test.T)
            shuffled_feature_accuracy_test = utils.balanced_accuracy(y_reg_fold_test.T, y_pred_shuffle.T)

            # Difference between true and shuffled accuracy.
            feature_diff = shuffled_feature_accuracy_test - feature_accuracy_test #feature_accuracy_test - shuffled_feature_accuracy_test
            iteration_diffs_all[i].append(feature_diff)

    # Average differences across folds
    avg_feature_diffs = {i: np.mean(iteration_diffs_all[i]) for i in range(n_iterations)}
    avg_true_accuracy = np.mean(fold_true_accuracies)

    return feature, {"true_accuracy": avg_true_accuracy, "iteration_diffs": avg_feature_diffs}


def global_feature_selection(mice_ids, stride_number, phase1, phase2, condition, exp, day, stride_data, save_dir,
                             nFolds=nFolds_selection, n_iterations=n_iterations_selection):
    results_file = os.path.join(save_dir, 'global_feature_selection_results.csv')
    if os.path.exists(results_file) and not overwrite_FeatureSelection:
        all_feature_accuracies_df = pd.read_csv(results_file, index_col=0)
        all_feature_accuracies_df['iteration_diffs'] = all_feature_accuracies_df['iteration_diffs'].apply(ast.literal_eval)
        print("Global feature selection results loaded from file.")
        significant_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
        return significant_features, all_feature_accuracies_df

    aggregated_data_list = []
    aggregated_y_list = []
    total_run_numbers = []

    for mouse_id in mice_ids:
        # Load and preprocess data for each mouse.
        scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
        # Get runs and phase masks.
        run_numbers, _, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id, stride_number,
                                                            phase1, phase2)
        selected_mask = mask_phase1 | mask_phase2
        # Transpose so that rows are features and columns are runs.
        selected_data = scaled_data_df.loc[selected_mask].T
        aggregated_data_list.append(selected_data)
        # Create the regression target.
        y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])
        aggregated_y_list.append(y_reg)
        # (Store run indices as simple integers for each mouse.)
        total_run_numbers.extend(list(range(selected_data.shape[1])))

    # Combine data across mice.
    aggregated_data_df = pd.concat(aggregated_data_list, axis=1)
    aggregated_y = np.concatenate(aggregated_y_list)
    N = aggregated_data_df.shape[1]

    # Create a fold assignment mapping for the aggregated runs.
    indices = list(range(N))
    random.shuffle(indices)
    fold_assignment = {i: (j % nFolds + 1) for j, i in enumerate(indices)}
    features = list(aggregated_data_df.index)

    # Run feature selection in parallel (using your existing process_single_feature).
    results = Parallel(n_jobs=-1)(
        delayed(process_single_feature)(
            feature,
            aggregated_data_df.loc[feature].values,
            fold_assignment,
            aggregated_y,
            list(range(N)),
            nFolds,
            n_iterations
        )
        for feature in tqdm(features, desc="Global feature selection")
    )

    all_feature_accuracies = dict(results)
    all_feature_accuracies_df = pd.DataFrame.from_dict(all_feature_accuracies, orient='index')
    # Determine significance (as in your original code).
    all_feature_accuracies_df['significant'] = 0 > all_feature_accuracies_df['iteration_diffs'].apply(
        lambda d: np.percentile(list(d.values()), 99)
    )
    significant_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
    # Save the results.
    all_feature_accuracies_df.to_csv(results_file)
    return significant_features, all_feature_accuracies_df


# -----------------------------------------------------
# Main Processing Function for Each Phase Comparison
# -----------------------------------------------------
def process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, condition, exp, day, base_save_dir_condition, selected_features=None, fs_df=None):
    """
    Process a single phase comparison for a given mouse. If selected_features is provided,
    that global feature set is used; otherwise local feature selection is performed.
    """
    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    # Load and preprocess data.
    scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day) # Load data for the specified mouse and preprocess it by selecting desired features, imputing missing values, and standardizing.
    print('Data loaded and preprocessed.')

    # Get runs and stepping limbs for each phase.
    run_numbers, stepping_limbs, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2)

    # Select only runs from the two phases in feature data
    selected_mask = mask_phase1 | mask_phase2
    selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

    if selected_features is None:
        # Assign each run randomly a number from 1 to 10 (for 10 folds) for cross-validation
        nFolds = nFolds_selection
        n_iterations = n_iterations_selection
        # Shuffle the list randomly
        run_numbers_shuffled = run_numbers.copy()
        random.shuffle(run_numbers_shuffled)
        # Assign fold numbers 1-10 in a round-robin fashion
        fold_assignment = {run: (i % nFolds + 1) for i, run in enumerate(run_numbers_shuffled)}
        y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])

        # ---------------- Feature Selection ----------------

        # Check if the feature selection results have already been computed
        if os.path.exists(os.path.join(save_path, 'feature_selection_results.csv')) and overwrite_FeatureSelection == False:
            all_feature_accuracies_df = pd.read_csv(os.path.join(save_path, 'feature_selection_results.csv'), index_col=0)
            all_feature_accuracies_df['iteration_diffs'] = all_feature_accuracies_df['iteration_diffs'].apply(ast.literal_eval)
            print("Feature selection results loaded from file.")
        else:
            print("Running feature selection...")
            features = list(selected_scaled_data_df.index)
            # Process features in parallel (using all available cores; adjust n_jobs as needed)
            results = Parallel(n_jobs=-1)(
                delayed(process_single_feature)(
                    feature,
                    selected_scaled_data_df.loc[feature].values,
                    fold_assignment,
                    y_reg,
                    run_numbers,
                    nFolds,
                    n_iterations
                )
                for feature in tqdm(features, desc="Processing features single mouse")
            )
            # Combine results into a dictionary
            all_feature_accuracies = dict(results)

            all_feature_accuracies_df = pd.DataFrame.from_dict(all_feature_accuracies, orient='index')

            # find if true accuracy is significant compared to shuffled accuracies ie > than 95% of shuffled accuracies
            all_feature_accuracies_df['significant'] = 0 > \
                                                       all_feature_accuracies_df['iteration_diffs'].apply(
                                                           lambda d: np.percentile(list(d.values()), 99))
            # Save the feature selection results
            all_feature_accuracies_df.to_csv(os.path.join(save_path, 'feature_selection_results.csv'))
        # find features that are significant
        significant_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
    else:
        print("Using globally selected features for feature reduction.")
        significant_features = selected_features
        all_feature_accuracies_df = fs_df


    # store significant features in a reduced df
    significant_features_accuracies = all_feature_accuracies_df.loc[significant_features]
    print(f"Length of significant features: {len(significant_features)}")

    # plot significant features
    if not os.path.exists(os.path.join(save_path, 'feature_significances')):
        os.makedirs(os.path.join(save_path, 'feature_significances'))
    for fidx, feature in enumerate(significant_features):
        plt.figure()
        sns.histplot(significant_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
        plt.axvline(0, color='red', label='True Accuracy')
        plt.title(feature)
        plt.xlabel('Shuffled Accuracy - True Accuracy')
        plt.ylabel('Frequency')
        plt.legend()

        plt.savefig(os.path.join(save_path, f'feature_significances\\feature{fidx}_feature_selection.png'))
        plt.close()

    nonsignificant_features = all_feature_accuracies_df[~all_feature_accuracies_df['significant']].index
    nonsignificant_features_accuracies = all_feature_accuracies_df.loc[nonsignificant_features]
    # plot non significant features
    if not os.path.exists(os.path.join(save_path, 'feature_nonsignificances')):
        os.makedirs(os.path.join(save_path, 'feature_nonsignificances'))
    for fidx, feature in enumerate(nonsignificant_features):
        plt.figure()
        sns.histplot(nonsignificant_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
        plt.axvline(0, color='red', label='True Accuracy')
        plt.title(feature)
        plt.xlabel('Shuffled Accuracy - True Accuracy')
        plt.ylabel('Frequency')
        plt.legend()

        plt.savefig(os.path.join(save_path, f'feature_nonsignificances\\feature{fidx}_feature_selection.png'))
        plt.close()

    # Now reduce the data to only the significant features
    reduced_feature_data_df = scaled_data_df.loc(axis=1)[significant_features]
    reduced_feature_selected_data_df = selected_scaled_data_df.loc[significant_features]
    # todo might want to also remove instances where have full stride AND both swing and stance for a single measure

    # ---------------- PCA Analysis ----------------

    # Cross-validate PCA and determine PCn
    fold_variances = cross_validate_pca(reduced_feature_data_df, save_path, n_folds=10)
    plot_average_variance_explained_across_folds(fold_variances, reduced_feature_data_df)

    # Perform PCA.
    pca, pcs, loadings_df = perform_pca(reduced_feature_data_df, n_components=10)

    # Get PCs and labels for each phase.
    pcs_combined, labels, pcs_phase1, pcs_phase2 = get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2)

    # Plot PCA and Scree.
    plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
    plot_scree(pca, save_path)

    # ---------------- LDA Analysis ----------------
    #
    # # Perform LDA on the PCA-transformed data.
    # y_labels_all = np.concatenate([np.ones(pcs_phase1.shape[0]), np.zeros(pcs_phase2.shape[0])])
    # lda_all, Y_lda_all, lda_loadings_all = perform_lda(pcs_combined, y_labels=y_labels_all,
    #                                                    phase1=phase1, phase2=phase2, n_components=1)
    # feature_contributions_df_all = compute_feature_contributions(loadings_df, lda_loadings_all)
    # plot_feature_contributions(feature_contributions_df_all, save_path, title_suffix="All_PCs")
    # plot_lda_transformed_data(Y_lda_all, phase1, phase2, save_path, title_suffix="All_PCs")
    # plot_LDA_loadings(lda_loadings_all, save_path, title_suffix="All_PCs")


    # ---------------- Prediction - Regression-Based Feature Contributions ----------------

    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings_df.T, reduced_feature_selected_data_df)

    # Normalize X
    Xdr, normalize_mean, normalize_std = utils.normalize(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])

    # Run logistic regression on the full model
    w, full_accuracy = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {full_accuracy:.3f}")

    # Shuffle features and run logistic regression to find unique contributions and single feature contributions
    single_all_dict, unique_all_dict = find_unique_and_single_contributions(reduced_feature_selected_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy)

    # Find full shuffled accuracy (one shuffle may not be enough)
    full_shuffled_accuracy = find_full_shuffle_accuracy(reduced_feature_selected_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy)
    print(f"Full model shuffled accuracy: {full_shuffled_accuracy:.3f}")

    # Plot unique and single feature contributions
    utils.plot_unique_delta_accuracy(unique_all_dict, save_path, title_suffix=f"{phase1}_vs_{phase2}")
    utils.plot_feature_accuracy(single_all_dict, save_path, title_suffix=f"{phase1}_vs_{phase2}")

    # Apply the full model to all runs (scaled and unscaled)
    all_trials_dr = np.dot(loadings_df.T, reduced_feature_data_df.T)
    all_trials_dr = ((all_trials_dr.T - normalize_mean) / normalize_std).T
    run_pred = np.dot(w, np.dot(loadings_df.T, reduced_feature_data_df.T))
    run_pred_scaled = np.dot(w, all_trials_dr)

    # Plot run prediction
    utils.plot_run_prediction(reduced_feature_data_df, run_pred, save_path, mouse_id, phase1, phase2, "")
    utils.plot_run_prediction(reduced_feature_data_df, run_pred_scaled, save_path, mouse_id, phase1, phase2, "scaled")





# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------
def main(mouse_ids, stride_numbers, phases, condition='LowHigh', exp='Extended', day=None, allmice=False):
    # construct path
    if exp == 'Extended':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_StrideInfo.h5")
    elif exp == 'Repeats':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
    stride_data = utils.load_stride_data(stride_data_path)

    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')

    if allmice:
        # Create a directory to store global feature selection results.
        global_fs_dir = os.path.join(base_save_dir, f'GlobalFeatureSelection_{condition}_{exp}')
        os.makedirs(global_fs_dir, exist_ok=True)

        global_fs_results = {}
        for phase1, phase2 in itertools.combinations(phases, 2):
            print(f"Performing global feature selection for {phase1} vs {phase2}")
            significant_features, fs_df = global_feature_selection(
                global_fs_mouse_ids,
                stride_numbers[0],
                phase1, phase2,
                condition, exp, day,
                stride_data,
                global_fs_dir
            )
            global_fs_results[(phase1, phase2)] = (significant_features, fs_df)


    all_contributions = []  # list to store each mouse's contributions
    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                try:
                    if allmice:
                        tup = global_fs_results.get((phase1, phase2), None)
                        if tup is not None:
                            selected_features, fs_df = tup
                        else:
                            selected_features, fs_df = None, None
                    else:
                        selected_features, fs_df = None, None

                    process_phase_comparison(
                        mouse_id, stride_number, phase1, phase2,
                        stride_data, condition, exp, day,
                        base_save_dir_condition,
                        selected_features=selected_features, fs_df=fs_df
                    )
                except ValueError as e:
                    print(f"Error processing {mouse_id}, stride {stride_number}, {phase1} vs {phase2}: {e}")

    # # After processing all mice, aggregate the contributions.
    # # You can choose a save directory for these summary plots:
    # summary_save_path = os.path.join(base_save_dir_condition, 'Summary_Cosine_Similarity')
    # os.makedirs(summary_save_path, exist_ok=True)
    #
    # pivot_contributions = utils.aggregate_feature_contributions(all_contributions)
    #
    # # Compute and plot the pairwise cosine similarity and the boxplots.
    # similarity_matrix = utils.plot_pairwise_cosine_similarity(pivot_contributions, summary_save_path)


# ----------------------------
# Execute Main Function
# ----------------------------

if __name__ == "__main__":
    main(mouse_ids, stride_numbers, phases, condition='APAChar_LowHigh', exp='Extended',day=None,allmice=True)
    # main(mouse_ids, stride_numbers, phases, condition='APAChar_LowMid', exp='Extended', day=None)
    # main(mouse_ids, stride_numbers, phases, condition='APAChar_HighLow', exp='Extended', day=None)
