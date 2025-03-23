import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random
from tqdm import tqdm
import ast
from joblib import Parallel, delayed
from collections import Counter

from Analysis.Tools.PCA import perform_pca, plot_pca, plot_scree, cross_validate_pca, plot_average_variance_explained_across_folds
from Analysis.Tools.LogisticRegression import compute_regression, compute_lasso_regression, run_regression, predict_runs
from Analysis.Tools.FeatureSelection import rfe_feature_selection, random_forest_feature_selection
from Analysis.Tools import utils_feature_reduction as utils
from Helpers.Config_23 import *

# ----------------------------
# Configuration Section
# ----------------------------

condition_configurations = {
    'APAChar_LowHigh': {
        'c': 1,
        'global_fs_mouse_ids': ['1035243', '1035244', '1035245', '1035246', '1035250','1035299','1035301'],
    },
    'APAChar_HighLow': {
        'c': 0.5,
        'global_fs_mouse_ids': ['1035243', '1035244', '1035245', '1035246','1035250','1035301', '1035302'],
    },
}

mouse_ids = [
    '1035243', '1035244', '1035245', '1035246',
    '1035249', '1035250', '1035297', '1035298',
    '1035299', '1035301', '1035302'
]  # List of mouse IDs to analyze
stride_numbers = [-1]  # List of stride numbers to filter data
phases = ['APA2', 'Wash2']  # List of phases to compare
exp = 'Extended'
day = None
method = 'rfecv'
# The two conditions want to process:
conditions = ['APAChar_LowHigh', 'APAChar_HighLow']

base_save_dir = os.path.join(paths['plotting_destfolder'], 'FeatureReduction', 'Round11-20250217-global-rfecv-multicon')

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

def select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2):
    # Load and preprocess data.
    scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp,
                                              day)  # Load data for the specified mouse and preprocess it by selecting desired features, imputing missing values, and standardizing.
    print('Data loaded and preprocessed.')

    # Get runs and stepping limbs for each phase.
    run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id,
                                                                     stride_number, phase1, phase2)

    # Select only runs from the two phases in feature data
    selected_mask = mask_phase1 | mask_phase2
    selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

    return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2


def unified_feature_selection(feature_data_df, y, c, method='regression', cv=5, n_iterations=100, fold_assignment=None, save_file=None, overwrite_FeatureSelection=False):
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
      - save_file: if provided, a file path to load/save results.

    Returns:
      - selected_features: the list (or index) of selected features.
      - results: For 'regression' method, a DataFrame of per-feature results; otherwise, None.
    """
    # Check if a results file exists and we are not overwriting.
    if save_file is not None and os.path.exists(save_file) and not overwrite_FeatureSelection:
        if method == 'regression':
            all_feature_accuracies_df = pd.read_csv(save_file, index_col=0)
            # Convert the string representation back into dictionaries.
            all_feature_accuracies_df['iteration_diffs'] = all_feature_accuracies_df['iteration_diffs'].apply(
                ast.literal_eval)
            print("Global feature selection results loaded from file.")
            selected_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
            return selected_features, all_feature_accuracies_df
        else:
            df = pd.read_csv(save_file)
            # Assuming selected features were saved in a column 'selected_features'
            selected_features = df['selected_features'].tolist()
            print("Global feature selection results loaded from file.")
            return selected_features, None

    # Compute feature selection using the chosen method.
    if method == 'rfecv':
        print("Running RFECV for feature selection.")
        selected_features, rfecv_model = rfe_feature_selection(feature_data_df, y, cv=cv, min_features_to_select=5, C=c)
        print(f"RFECV selected {rfecv_model.n_features_} features.")
        results = None
    elif method == 'rf':
        print("Running Random Forest for feature selection.")
        selected_features, rf_model = random_forest_feature_selection(feature_data_df, y)
        print(f"Random Forest selected {len(selected_features)} features.")
        results = None
    elif method == 'regression':
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
        # Mark features as significant if the 99th percentile of shuffled differences is below zero.
        all_feature_accuracies_df['significant'] = 0 > all_feature_accuracies_df['iteration_diffs'].apply(
            lambda d: np.percentile(list(d.values()), 99)
        )
        selected_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
        results = all_feature_accuracies_df
    else:
        raise ValueError("Unknown method specified for feature selection.")

    # Save results if a file path is provided.
    if save_file is not None:
        if method == 'regression':
            results.to_csv(save_file)
        else:
            # Save the list of selected features in a simple CSV.
            pd.DataFrame({'selected_features': selected_features}).to_csv(save_file, index=False)

    if method == 'regression':
        return selected_features, results
    else:
        return selected_features, None

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
            #w, _ = compute_regression(X_shuffled_fold_train, y_reg_fold_train)
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
                             nFolds=5, n_iterations=100, method='regression'):
    results_file = os.path.join(save_dir, 'global_feature_selection_results.csv')

    aggregated_data_list = []
    aggregated_y_list = []
    total_run_numbers = []

    for mouse_id in mice_ids:
        # Load and preprocess data for each mouse.
        scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
        # Get runs and phase masks.
        run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id, stride_number,
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

    # Call unified_feature_selection (choose method as desired: 'rfecv', 'rf', or 'regression')
    selected_features, fs_results_df = unified_feature_selection(
                                        feature_data_df=aggregated_data_df,
                                        y=aggregated_y,
                                        c=condition_configurations[condition]['c'],
                                        method=method,  # change as desired
                                        cv=nFolds,
                                        n_iterations=n_iterations,
                                        save_file=results_file,
                                        overwrite_FeatureSelection=overwrite_FeatureSelection
                                    )
    print("Global selected features:", selected_features)
    return selected_features, fs_results_df


def compute_global_pca_for_phase(global_mouse_ids, stride_number, phase1, phase2,
                                 condition, exp, day, stride_data, selected_features,
                                 n_components=10):
    """
    Aggregates data from all mice in global_mouse_ids (using only runs for phase1 and phase2),
    restricts to the globally selected features, and computes PCA.
    """
    aggregated_data = []
    for mouse_id in global_mouse_ids:
        scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
        # Get run masks for the two phases.
        run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2)
        # Select only runs corresponding to the phases.
        selected_mask = mask_phase1 | mask_phase2
        selected_data = scaled_data_df.loc[selected_mask]
        # Restrict to the globally selected features.
        reduced_data = selected_data[selected_features]
        aggregated_data.append(reduced_data)
    # Concatenate all runs (rows) across mice.
    global_data = pd.concat(aggregated_data)
    # Compute PCA on the aggregated data.
    pca, pcs, loadings_df = perform_pca(global_data, n_components=n_components)
    return pca, loadings_df

def predict_compare_condition(mouse_id, compare_condition, stride_number, exp, day, stride_data_compare, phase1, phase2, selected_features, loadings_df, w, save_path):
    # Retrieve reduced feature data for the comparison condition
    # _, comparison_selected_scaled_data, _, _, _, _ = select_runs_data(mouse_id, stride_number, compare_condition, exp, day, stride_data_compare, phase1, phase2)
    comparison_scaled_data = load_and_preprocess_data(mouse_id, stride_number, compare_condition, exp, day)
    comparison_reduced_feature_data_df = comparison_scaled_data.loc(axis=1)[selected_features]
    runs = list(comparison_reduced_feature_data_df.index)

    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings_df.T, comparison_reduced_feature_data_df.T)
    # Normalize X
    Xdr, normalize_mean, normalize_std = utils.normalize_Xdr(Xdr)

    save_path_compare = os.path.join(save_path, f"vs_{compare_condition}")
    # prefix path wth \\?\ to avoid Windows path length limit
    save_path_compare = "\\\\?\\" + save_path_compare
    os.makedirs(save_path_compare, exist_ok=True)
    smoothed_scaled_pred = predict_runs(loadings_df, comparison_reduced_feature_data_df, normalize_mean, normalize_std, w, save_path_compare, mouse_id, phase1, phase2, compare_condition)

    return smoothed_scaled_pred, runs


def compute_global_regression_model(global_mouse_ids, stride_number, phase1, phase2, condition, exp, day, stride_data,
                                    selected_features, loadings_df):
    aggregated_data_list = []
    y_list = []
    for mouse_id in global_mouse_ids:
        scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
        # Get phase masks and runs.
        run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id, stride_number,
                                                            phase1, phase2)
        selected_mask = mask_phase1 | mask_phase2
        selected_data = scaled_data_df.loc[selected_mask][selected_features]
        aggregated_data_list.append(selected_data)
        # Create labels: 1 for phase1, 0 for phase2.
        y_list.append(np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))]))

    # Combine all data across mice.
    global_data_df = pd.concat(aggregated_data_list)
    y_global = np.concatenate(y_list)

    # Project aggregated data into PCA space using the global loadings.
    Xdr = np.dot(loadings_df.T, global_data_df.T)
    Xdr, norm_mean, norm_std = utils.normalize_Xdr(Xdr)

    # Compute regression weights (using your chosen regression function).
    w, full_accuracy = compute_regression(Xdr, y_global)
    print(f"Global regression model accuracy for {phase1} vs {phase2}: {full_accuracy:.3f}")

    return {'w': w, 'norm_mean': norm_mean, 'norm_std': norm_std, 'selected_features': selected_features,
            'loadings_df': loadings_df}



# -----------------------------------------------------
# Main Processing Function for Each Phase Comparison
# -----------------------------------------------------
def process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, stride_data_compare, condition, exp, day,
                             base_save_dir_condition, selected_features=None, fs_df=None, method='regression',
                             global_pca=None, compare_condition=None):
    """
    Process a single phase comparison for a given mouse. If selected_features is provided,
    that global feature set is used; otherwise local feature selection is performed.
    """
    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    save_path = "\\\\?\\" + save_path
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2 = select_runs_data(mouse_id, stride_number, condition, exp, day, stride_data, phase1, phase2)

    if selected_features is None:
        y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])
        # For local selection you might choose not to save to file (or provide a unique save_file path).
        selected_features, _ = unified_feature_selection(
            feature_data_df=selected_scaled_data_df,
            y=y_reg,
            c=condition_configurations[condition]['c'],
            method=method,  # or 'rf' or 'regression'
            cv=nFolds_selection,
            n_iterations=n_iterations_selection,
            save_file=None,
            overwrite_FeatureSelection=overwrite_FeatureSelection
        )
        print("Selected features (local):", selected_features)
    else:
        print("Using globally selected features for feature reduction.")
        fs_df = fs_df

    if fs_df is not None:
        # Detailed regression results are available; proceed with plotting.
        selected_features_accuracies = fs_df.loc[selected_features]
        print(f"Length of significant features: {len(selected_features)}")

        # Plot significant features.
        if not os.path.exists(os.path.join(save_path, 'feature_significances')):
            os.makedirs(os.path.join(save_path, 'feature_significances'))
        for fidx, feature in enumerate(selected_features):
            plt.figure()
            sns.histplot(selected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
            plt.axvline(0, color='red', label='True Accuracy')
            plt.title(feature)
            plt.xlabel('Shuffled Accuracy - True Accuracy')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(save_path, f'feature_significances\\feature{fidx}_feature_selection.png'))
            plt.close()

        # Plot non-significant features.
        nonselected_features = fs_df[~fs_df['significant']].index
        nonselected_features_accuracies = fs_df.loc[nonselected_features]
        if not os.path.exists(os.path.join(save_path, 'feature_nonsignificances')):
            os.makedirs(os.path.join(save_path, 'feature_nonsignificances'))
        for fidx, feature in enumerate(nonselected_features):
            plt.figure()
            sns.histplot(nonselected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
            plt.axvline(0, color='red', label='True Accuracy')
            plt.title(feature)
            plt.xlabel('Shuffled Accuracy - True Accuracy')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(save_path, f'feature_nonsignificances\\feature{fidx}_feature_selection.png'))
            plt.close()
    else:
        # Detailed per-feature results are not available (i.e. when using RFECV or RF).
        print("No detailed per-feature selection results available for plotting; skipping per-feature plots.")

    # Now reduce the data to only the significant features
    reduced_feature_data_df = scaled_data_df.loc(axis=1)[selected_features]
    reduced_feature_selected_data_df = selected_scaled_data_df.loc[selected_features]

    # ---------------- PCA Analysis ----------------

    if global_pca is not None:
        # Use the global PCA transformation.
        pca, loadings_df = global_pca
        # Project this mouse's data using the global PCA.
        pcs = pca.transform(reduced_feature_data_df)
        pcs_phase1 = pcs[mask_phase1]
        pcs_phase2 = pcs[mask_phase2]
        labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
        labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
        labels = np.concatenate([labels_phase1, labels_phase2])
        pcs_combined = np.vstack([pcs_phase1, pcs_phase2])
        # Plot using the global PCA.
        plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, save_path)
    else:
        # Local PCA (existing behavior).
        fold_variances = cross_validate_pca(reduced_feature_data_df, save_path, n_folds=10)
        plot_average_variance_explained_across_folds(fold_variances, reduced_feature_data_df)
        pca, pcs, loadings_df = perform_pca(reduced_feature_data_df, n_components=10)
        pcs_combined, labels, pcs_phase1, pcs_phase2 = utils.get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2)
        plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
        plot_scree(pca, save_path)


    # ---------------- Prediction - Regression-Based Feature Contributions ----------------
    smoothed_scaled_pred, feature_weights, w = run_regression(loadings_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, save_path, condition)

    return (mouse_id, list(reduced_feature_data_df.index), smoothed_scaled_pred), feature_weights, reduced_feature_data_df

# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------
def main_feature_reduction_aggregate():
    # Dictionary to hold selected feature lists for each condition
    feature_lists = {}

    # Loop over each condition and perform global feature selection
    for cond in conditions:
        print(f"\n=== Processing feature selection for condition: {cond} ===")
        # Build a save directory for feature selection results
        fs_save_dir = os.path.join(base_save_dir, f'{cond}_{exp}', 'GlobalFeatureSelection')
        os.makedirs(fs_save_dir, exist_ok=True)

        # Load stride data for this condition
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{cond}\\{exp}\\MEASURES_StrideInfo.h5")
        stride_data = utils.load_stride_data(stride_data_path)

        # For feature selection we use the global mouse ids defined in your condition_configurations.
        global_mouse_ids = condition_configurations[cond]['global_fs_mouse_ids']
        # For simplicity, here we take the first pair of phases.
        phase1, phase2 = phases[0], phases[1]

        selected_features, fs_df = global_feature_selection(
            global_mouse_ids,
            stride_numbers[0],
            phase1, phase2,
            cond, exp, day,
            stride_data,
            save_dir=fs_save_dir,
            method=method
        )
        feature_lists[cond] = list(selected_features)
        # Save the selected features for this condition as CSV.
        pd.DataFrame({'feature': list(selected_features)}).to_csv(
            os.path.join(fs_save_dir, f'features_{cond}.csv'), index=False
        )
        print(f"Condition {cond}: {len(selected_features)} features selected and saved.")

    # Aggregate the features from both conditions (union)
    aggregated_features = list(set(feature_lists[conditions[0]]) | set(feature_lists[conditions[1]]))
    # print the features which appear in both conditions
    print(f"Features common to both conditions: {len(set(feature_lists[conditions[0]]) & set(feature_lists[conditions[1]]))}")
    agg_fs_save_dir = os.path.join(base_save_dir, 'Aggregated_Features')
    os.makedirs(agg_fs_save_dir, exist_ok=True)
    pd.DataFrame({'feature': aggregated_features}).to_csv(
        os.path.join(agg_fs_save_dir, 'aggregated_features.csv'), index=False
    )
    print(f"\nAggregated feature list (total {len(aggregated_features)} features) saved at {agg_fs_save_dir}")

    # Now, for each condition, run PCA/regression analyses using the aggregated features.
    agg_feature_w = []
    for cond in conditions:
        print(f"\n=== Running PCA and Regression for condition: {cond} ===")
        # Use the same stride data for analysis in this condition.
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{cond}\\{exp}\\MEASURES_StrideInfo.h5")
        stride_data = utils.load_stride_data(stride_data_path)

        # Define a save directory for analysis results.
        analysis_save_dir = os.path.join(base_save_dir, f'{cond}_{exp}', 'Analysis')
        os.makedirs(analysis_save_dir, exist_ok=True)

        # Compute a global PCA on the global mouse set using the aggregated features.
        global_mouse_ids = condition_configurations[cond]['global_fs_mouse_ids']
        phase1, phase2 = phases[0], phases[1]
        pca, loadings_df = compute_global_pca_for_phase(
            global_mouse_ids,
            stride_numbers[0],
            phase1, phase2,
            cond, exp, day,
            stride_data,
            aggregated_features
        )

        # Process each individual mouse in your list.
        aggregated_predictions = {}
        aggregated_feature_weights = {}
        for mouse_id in mouse_ids:
            try:
                info, feat_weights, _ = process_phase_comparison(
                    mouse_id, stride_numbers[0],
                    phase1, phase2,
                    stride_data,  # use same stride data for both arguments if no comparison condition
                    stride_data,
                    cond, exp, day,
                    base_save_dir_condition=analysis_save_dir,
                    selected_features=aggregated_features,
                    fs_df=None,  # detailed per-feature results not needed here
                    method=method,
                    global_pca=(pca, loadings_df),
                    compare_condition='None'
                )
                # info is (mouse_id, x_vals, smoothed_scaled_pred)
                aggregated_predictions.setdefault((phase1, phase2), []).append(info)
                aggregated_feature_weights[(mouse_id, phase1, phase2)] = feat_weights
                print(f"Processed mouse {mouse_id} for condition {cond}.")
            except ValueError as e:
                print(f"Error processing mouse {mouse_id} for condition {cond}: {e}")

        # After processing individual mice, plot aggregated results.
        utils.plot_aggregated_run_predictions(
            aggregated_predictions.get((phase1, phase2), []),
            analysis_save_dir, phase1, phase2, condition_label=cond
        )
        utils.plot_aggregated_feature_weights_byFeature(
            aggregated_feature_weights, analysis_save_dir, phase1, phase2, condition_label=cond
        )
        agg_feature_w.append(aggregated_feature_weights)
        print(f"Aggregated plots saved for condition {cond}.")

    # Compare the feature weights between the two conditions
    aggregated_feature_weights_cond1 = agg_feature_w[0]
    aggregated_feature_weights_cond2 = agg_feature_w[1]
    utils.plot_aggregated_feature_weights_comparison(aggregated_feature_weights_cond1,
                                               aggregated_feature_weights_cond2,
                                               save_path=base_save_dir,
                                               phase1='APA2', phase2='Wash2',
                                               cond1_label=conditions[0],
                                               cond2_label=conditions[1])
    # and other way around
    utils.plot_aggregated_feature_weights_comparison(aggregated_feature_weights_cond2,
                                               aggregated_feature_weights_cond1,
                                               save_path=base_save_dir,
                                               phase1='APA2', phase2='Wash2',
                                               cond1_label=conditions[1],
                                               cond2_label=conditions[0])


if __name__ == "__main__":
    main_feature_reduction_aggregate()

