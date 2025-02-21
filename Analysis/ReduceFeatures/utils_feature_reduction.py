from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import os
import re
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ast
from joblib import Parallel, delayed
import random
from collections import Counter
from tqdm import tqdm


from Helpers.Config_23 import *
from Analysis.ReduceFeatures.LogisticRegression import compute_regression, compute_lasso_regression, run_regression, predict_runs
from Analysis.ReduceFeatures.FeatureSelection import rfe_feature_selection, random_forest_feature_selection


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
    run_numbers, stepping_limbs, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id,
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
        feature_accuracy_test = balanced_accuracy(y_reg_fold_test.T, y_pred.T) # Get balanced accuracy from test set
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
            shuffled_feature_accuracy_test = balanced_accuracy(y_reg_fold_test.T, y_pred_shuffle.T)

            # Difference between true and shuffled accuracy.
            feature_diff = shuffled_feature_accuracy_test - feature_accuracy_test #feature_accuracy_test - shuffled_feature_accuracy_test
            iteration_diffs_all[i].append(feature_diff)

    # Average differences across folds
    avg_feature_diffs = {i: np.mean(iteration_diffs_all[i]) for i in range(n_iterations)}
    avg_true_accuracy = np.mean(fold_true_accuracies)

    return feature, {"true_accuracy": avg_true_accuracy, "iteration_diffs": avg_feature_diffs}


def global_feature_selection(mice_ids, stride_number, phase1, phase2, condition, exp, day, stride_data, save_dir,
                             c, nFolds, n_iterations, overwrite, method='regression'):
    results_file = os.path.join(save_dir, f'global_feature_selection_results_{phase1}_{phase2}_stride{stride_number}.csv')

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

    # Call unified_feature_selection (choose method as desired: 'rfecv', 'rf', or 'regression')
    selected_features, fs_results_df = unified_feature_selection(
                                        feature_data_df=aggregated_data_df,
                                        y=aggregated_y,
                                        c=c,
                                        method=method,
                                        cv=nFolds,
                                        n_iterations=n_iterations,
                                        save_file=results_file,
                                        overwrite_FeatureSelection=overwrite
                                    )
    print(f"Global selected features: {selected_features}, \nLength: {len(selected_features)}")
    return selected_features, fs_results_df

def normalize(Xdr):
    normalize_mean = []
    normalize_std = []
    for row in range(Xdr.shape[0]):
        mean = np.mean(Xdr[row, :])
        std = np.std(Xdr[row, :])
        Xdr[row, :] = (Xdr[row, :] - mean) / std
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return Xdr, normalize_mean, normalize_std

def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy: average of sensitivity and specificity.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        float: balanced accuracy score
    """
    # Calculate true positives and true negatives
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))

    # Calculate false positives and false negatives
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    # Calculate sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced accuracy is the average of sensitivity and specificity
    return (sensitivity + specificity)/2

def shuffle_single(feature, raw_features):
    shuffled = raw_features.copy()
    for col in raw_features.columns:
        if col != feature:
            shuffled[col] = np.random.permutation(shuffled[col].values)
    return shuffled

def shuffle_unique(feature, raw_features):
    shuffled = raw_features.copy()
    shuffled.loc(axis=0)[feature] = np.random.permutation(shuffled.loc(axis=0)[feature].values)
    return shuffled

def plot_feature_accuracy(single_cvaccuracy, mouseID, save_path, title_suffix="Single_Feature_cvaccuracy"):
    """
    Plots the single-feature model accuracy values.

    Parameters:
        single_cvaccuracy (dict): Mapping of feature names to accuracy values.
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """
    df = pd.DataFrame(list(single_cvaccuracy.items()), columns=['Feature', 'cvaccuracy'])
    # Replace the separator so that group headers appear as "Group: FeatureName"
    df['Display'] = df['Feature'].apply(lambda x: x.replace('|', ': '))
    #df = df.sort_values(by='cvaccuracy', ascending=False)

    plt.figure(figsize=(14, max(8, len(df) * 0.3)))
    sns.barplot(data=df, x='cvaccuracy', y='Display', palette='viridis')
    plt.title(f'{mouseID}\nSingle Feature Model accuracy ' + title_suffix)
    plt.xlabel('accuracy')
    plt.ylabel('Feature (Group: FeatureName)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Single_Feature_cvaccuracy_{title_suffix}.png"), dpi=300)
    plt.close()

def plot_unique_delta_accuracy(unique_delta_accuracy, mouseID, save_path, title_suffix="Unique_Δaccuracy"):
    """
    Plots the unique contribution (Δaccuracy) for each feature.

    Parameters:
        unique_delta_accuracy (dict): Mapping of feature names to unique Δaccuracy.
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """

    df = pd.DataFrame(list(unique_delta_accuracy.items()), columns=['Feature', 'Unique_Δaccuracy'])
    df['Display'] = df['Feature'].apply(lambda x: x.replace('|', ': '))
    #df = df.sort_values(by='Unique_Δaccuracy', ascending=False)

    plt.figure(figsize=(14, max(8, len(df) * 0.3)))
    sns.barplot(data=df, x='Unique_Δaccuracy', y='Display', palette='magma')
    plt.title(f'{mouseID}\nUnique Feature Contributions (Δaccuracy) ' + title_suffix)
    plt.xlabel('Δaccuracy')
    plt.ylabel('Feature (Group: FeatureName)')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Unique_delta_accuracy_{title_suffix}.png"), dpi=300)
    plt.close()

def plot_run_prediction(scaled_data_df, run_pred, run_pred_smoothed, save_path, mouse_id, phase1, phase2, scale_suffix, dataset_suffix):
    # median filter smoothing on run_pred
    #run_pred_smoothed = medfilt(run_pred[0], kernel_size=5)

    # plot run prediction
    plt.figure(figsize=(8, 6))
    plt.plot(scaled_data_df.index, run_pred[0], color='lightblue', ls='--', label='Prediction')
    plt.plot(scaled_data_df.index, run_pred_smoothed, color='blue', ls='-', label='Smoothed Prediction')
    # Exp phases
    plt.vlines(x=9.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    plt.vlines(x=109.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    # Days
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=79.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=119.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)

    # plot a shaded box over x=60 to x=110 and x=135 to x=159, ymin to ymax
    plt.fill_between(x=range(60, 110), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)
    plt.fill_between(x=range(135, 160), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)

    plt.title(f'Run Prediction for Mouse {mouse_id} - {phase1} vs {phase2}')
    plt.xlabel('Run Number')
    plt.ylabel('Prediction')

    legend_elements = [Line2D([0], [0], color='red', linestyle='--', label='Experimental Phases'),
                          Line2D([0], [0], color='black', linestyle='--', label='Days'),
                          Patch(facecolor='gray', edgecolor='black', alpha=0.1, label='Training Portion'),
                          Line2D([0], [0], color='lightblue', label='Prediction', linestyle='--'),
                          Line2D([0], [0], color='blue', label='Smoothed Prediction', linestyle='-')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(False)
    # horizontal grid lines only
    plt.gca().yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Run_Prediction_{phase1}_vs_{phase2}_{scale_suffix}_{dataset_suffix}.png"), dpi=300)
    plt.close()


def plot_aggregated_run_predictions(aggregated_data, save_dir, phase1, phase2, stride_number, condition_label, normalization_method='maxabs'):
    """
    aggregated_data: list of tuples (mouse_id, x_values, smoothed_scaled_pred)
    normalization_method: 'maxabs' or 'zscore'
      - 'maxabs': each curve is divided by its maximum absolute value (range becomes [-1, 1])
      - 'zscore': each curve is standardized (mean=0, unit variance)
    This function interpolates each curve onto a common x-axis, plots individual curves (with lower alpha),
    and plots the mean curve.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # Determine a common x-axis.
    all_x_vals = []
    for _, x_vals, _ in aggregated_data:
        all_x_vals.extend(x_vals)
    global_min_x = min(all_x_vals)
    global_max_x = max(all_x_vals)
    # Use the maximum number of points among curves as the common number.
    common_npoints = max(len(x_vals) for _, x_vals, _ in aggregated_data)
    common_x = np.linspace(global_min_x, global_max_x, common_npoints)

    interpolated_curves = []  # List to store each curve interpolated to common_x

    for mouse_id, x_vals, smoothed_pred in aggregated_data:
        # Normalize the curve based on the chosen method.
        if normalization_method == 'zscore':
            mean_val = np.mean(smoothed_pred)
            std_val = np.std(smoothed_pred)
            normalized_curve = (smoothed_pred - mean_val) / std_val if std_val != 0 else smoothed_pred
        elif normalization_method == 'maxabs':
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred
        else:
            normalized_curve = smoothed_pred

        # Interpolate the normalized curve onto the common x-axis.
        interp_curve = np.interp(common_x, x_vals, normalized_curve)
        interpolated_curves.append(interp_curve)

        # Plot individual curve with reduced alpha.
        plt.plot(common_x, interp_curve, label=f'Mouse {mouse_id}', alpha=0.3)

    # Compute the mean curve across all mice.
    all_curves_array = np.vstack(interpolated_curves)  # shape: (n_mice, n_points)
    mean_curve = np.mean(all_curves_array, axis=0)

    # Plot the mean curve as a bold black line.
    plt.plot(common_x, mean_curve, color='black', linewidth=2, label='Mean Curve')

    # Optionally add vertical lines (e.g., at x=9.5 and 109.5).
    plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')

    plt.title(f'Aggregated {normalization_method.upper()} Scaled Run Predictions for {phase1} vs {phase2}, stride {stride_number}\n{condition_label}')
    plt.xlabel('Run Number')
    plt.ylabel('Normalized Prediction (Smoothed)')
    if normalization_method == 'maxabs':
        plt.ylim(-1, 1)
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir,
                             f"Aggregated_{normalization_method.upper()}_Run_Predictions_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def cluster_weights(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=2):
    """
    Cluster the regression weights (w, from PCA space) using KMeans and plot the original clustering,
    including the cluster centroids.

    This function:
      1. Clusters the regression weight vector (w) in PCA space.
      2. Plots a scatter of PCA component indices vs. regression weights, colored by cluster,
         and overlays the centroids (each centroid is labeled with its corresponding cluster number).
      3. Transforms the regression weights back to feature space (via loadings_df) to assign features
         to clusters and saves this mapping to CSV for later use.

    Parameters:
      - w: Regression weight vector from PCA space (numpy array of shape (n_components,)).
      - loadings_df: DataFrame of PCA loadings (rows: features, columns: components).
      - save_path: Directory to save the clustering results and visualization.
      - mouse_id, phase1, phase2: Identifiers used in filenames.
      - n_clusters: Number of clusters to form (default: 2).

    Returns:
      - cluster_df: DataFrame mapping each feature to its weight and assigned cluster label.
      - kmeans: The fitted KMeans model (from clustering w).
    """
    # --- Step 1: Cluster the regression weights (w) in PCA space ---
    w_2d = w.reshape(-1, 1)  # KMeans expects 2D data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(w_2d)
    cluster_labels_pca = kmeans.labels_  # one label per PCA component

    # --- Step 2: Visualize the original clustering of regression weights ---
    plt.figure(figsize=(8, 6))
    component_indices = np.arange(len(w))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(n_clusters):
        color = colors[k % len(colors)]
        mask = (cluster_labels_pca == k)
        plt.scatter(component_indices[mask], w[mask], color=color, label=f'Cluster {k}', s=100, alpha=0.8)
        # Compute centroid for this cluster:
        centroid_x = np.mean(component_indices[mask])
        centroid_y = np.mean(w[mask])
        plt.scatter(centroid_x, centroid_y, color=color, marker='X', s=200, edgecolor='black',
                    label=f'Centroid {k}')
    plt.xlabel('PCA Component Index')
    plt.ylabel('Regression Weight')
    plt.title(f'Regression Weights Clustering for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.legend()
    vis_path = os.path.join(save_path, f'cluster_regression_weights_{mouse_id}_{phase1}_vs_{phase2}.png')
    plt.savefig(vis_path, dpi=300)
    plt.close()

    # --- Step 3: Map the clustering back into feature space ---
    # Compute feature-space weights as the weighted combination of PCA loadings.
    feature_weights = loadings_df.dot(w).squeeze()
    if isinstance(feature_weights, pd.DataFrame):
        feature_weights = feature_weights.iloc[:, 0]

    n_features = loadings_df.shape[0]
    cluster_scores = np.zeros((n_features, n_clusters))
    for j in range(len(w)):
        cluster_idx = cluster_labels_pca[j]
        cluster_scores[:, cluster_idx] += loadings_df.iloc[:, j].values * w[j]

    # Assign each feature to the cluster whose (absolute) score is largest.
    feature_cluster = np.argmax(np.abs(cluster_scores), axis=1)

    # --- Step 4: Create and save a DataFrame with the clustering results ---
    cluster_df = pd.DataFrame({
        'feature': loadings_df.index,
        'weight': feature_weights,
        'cluster': feature_cluster
    })

    cluster_file = os.path.join(save_path, f'cluster_weights_{mouse_id}_{phase1}_vs_{phase2}.csv')
    cluster_df.to_csv(cluster_file, index=False)
    print(f"Cluster weights saved to: {cluster_file}")

    return cluster_df, kmeans


def plot_weights_in_feature_space(feature_weights, save_path, mouse_id, phase1, phase2):
    """
    Plot the feature-space weights as a bar plot with feature names on the x-axis.
    """
    # Create a DataFrame for plotting and sort for easier visualization
    df = pd.DataFrame({'feature': feature_weights.index, 'weight': feature_weights.values})

    #sort df by weight
    #df = df.sort_values(by='weight', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='weight', y='feature', data=df, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Weights in Original Space for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    plot_file = os.path.join(save_path, f'feature_space_weights_{mouse_id}_{phase1}_vs_{phase2}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical feature-space weights plot saved to: {plot_file}")


def plot_aggregated_feature_weights(weights_dict, save_path, phase1, phase2, stride_number, condition_label):
    """
    Plot aggregated feature-space weights across mice for a specific phase pair,
    summarizing the mean (with error bars) for each feature while overlaying individual mouse lines.

    Parameters:
      - weights_dict: dict where keys are tuples (mouse_id, phase1, phase2)
                      and values are pandas Series of feature weights.
      - save_path: directory to save the resulting plot.
      - phase1, phase2: phase names to include in the plot title.
    """
    # Filter weights for the current phase pair.
    filtered_weights = {
        mouse_id: weights
        for (mouse_id, p1, p2, s), weights in weights_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }

    if not filtered_weights:
        print(f"No weights found for {phase1} vs {phase2}.")
        return

    # Combine into a DataFrame (rows = features, columns = mouse_ids)
    weights_df = pd.DataFrame(filtered_weights).sort_index()
    # Scale the weights so they are comparable (optional)
    weights_df = weights_df / weights_df.abs().max()

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot a faint line for each mouse
    for mouse in weights_df.columns:
        ax.plot(weights_df[mouse].values, weights_df.index, alpha=0.3,
                marker='o', markersize=3, linestyle='-', label=mouse)

    # Compute summary statistics: mean and standard error (SEM) for each feature
    mean_weights = weights_df.mean(axis=1)
    std_weights = weights_df.std(axis=1)
    sem = std_weights / np.sqrt(len(weights_df.columns))

    # Plot the mean with error bars
    ax.errorbar(mean_weights, weights_df.index, xerr=sem, fmt='o-', color='black',
                label='Mean ± SEM', linewidth=2, capsize=4)

    # Add a vertical reference line at 0
    ax.axvline(x=0, color='red', linestyle='--')

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Feature')
    ax.set_title(f'Aggregated Feature Space Weights Across Mice ({phase1} vs {phase2}), stride {stride_number}\n{condition_label}')
    plt.tight_layout()
    plt.legend(title='Mouse ID / Summary', loc='upper right')

    output_file = os.path.join(save_path, f'aggregated_feature_weights_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Aggregated feature weights plot saved to: {output_file}")


def plot_aggregated_feature_weights_comparison(weights_dict1, weights_dict2, save_path, phase1, phase2, cond1_label,
                                               cond2_label):
    """
    Plot the average (with SEM error bars) aggregated feature weights for two conditions on the same plot.
    Features are ordered by the absolute mean weight (descending) of condition 1.

    Parameters:
      - weights_dict1: dict for condition 1 (keys: (mouse_id, phase1, phase2), values: pandas Series of feature weights)
      - weights_dict2: dict for condition 2 (same format as weights_dict1)
      - save_path: directory where the resulting plot is saved.
      - phase1, phase2: phase names (for the plot title).
      - cond1_label, cond2_label: labels for condition 1 and condition 2.
    """
    # Extract aggregated weights for each condition.
    def aggregate_weights(weights_dict):
        # Filter weights for the given phase pair.
        filtered = {
            mouse_id: weights
            for (mouse_id, p1, p2), weights in weights_dict.items()
            if p1 == phase1 and p2 == phase2
        }
        if not filtered:
            raise ValueError("No weights found for the specified phase pair.")
        # Build a DataFrame (rows = features, columns = mouse IDs)
        df = pd.DataFrame(filtered).sort_index()
        # Scale weights (optional; here we keep as-is; remove or adjust scaling as needed)
        df = df / df.abs().max()
        return df

    df1 = aggregate_weights(weights_dict1)
    df2 = aggregate_weights(weights_dict2)

    # Compute mean and SEM for each feature
    mean1 = df1.mean(axis=1)
    sem1 = df1.std(axis=1) / np.sqrt(df1.shape[1])
    mean2 = df2.mean(axis=1)
    sem2 = df2.std(axis=1) / np.sqrt(df2.shape[1])

    # Order features by descending absolute mean from condition 1
    ordered_features = mean1.abs().sort_values(ascending=False).index.tolist()

    # Reorder statistics accordingly.
    mean1 = mean1.loc[ordered_features]
    sem1 = sem1.loc[ordered_features]
    mean2 = mean2.loc[ordered_features]
    sem2 = sem2.loc[ordered_features]

    # Create the plot.
    fig, ax = plt.subplots(figsize=(10, len(ordered_features) * 0.3 + 3))

    # Plot condition 1: horizontal errorbar (mean ± SEM)
    ax.errorbar(mean1, ordered_features, xerr=sem1, fmt='o-', color='blue',
                label=f'{cond1_label} Mean ± SEM', capsize=3, linewidth=2)

    # Plot condition 2: horizontal errorbar (mean ± SEM)
    ax.errorbar(mean2, ordered_features, xerr=sem2, fmt='s-', color='green',
                label=f'{cond2_label} Mean ± SEM', capsize=3, linewidth=2)

    # Vertical reference line at 0
    ax.axvline(x=0, color='red', linestyle='--')

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Feature')
    ax.set_title(f'Comparison of Aggregated Feature Weights ({phase1} vs {phase2})')
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_file = os.path.join(save_path,
                               f'aggregated_feature_weights_comparison_{phase1}_vs_{phase2}_{cond1_label}_vs_{cond2_label}.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to: {output_file}")


def plot_aggregated_raw_features(raw_features_dict, save_path, phase1, phase2, stride_number):
    """
    Plot aggregated raw features across mice for a specific phase pair,
    summarizing the mean (with error bars) for each feature while overlaying individual mouse lines.

    Parameters:
      - raw_features_dict: dict where keys are tuples (mouse_id, phase1, phase2)
                           and values are pandas DataFrame of raw features.
      - save_path: directory to save the resulting plot.
      - phase1, phase2: phase names to include in the plot title.
    """
    # Filter raw features for the current phase pair.
    filtered_features = {
        mouse_id: features
        for (mouse_id, p1, p2, s), features in raw_features_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }

    if not filtered_features:
        print(f"No raw features found for {phase1} vs {phase2}.")
        return

    # Combine into a DataFrame (rows = features, columns = mouse_ids)
    features_df = pd.concat(filtered_features, axis=0).sort_index()

    for feature in features_df.columns:
        feature_df = features_df[feature]
        # make mousid the column
        feature_df = feature_df.unstack(level=0)
        #smooth the data with median filter
        #feature_df = feature_df.apply(lambda x: medfilt(x, kernel_size=5), axis=0)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot a faint line for each mouse
        for mouse in feature_df.columns:
            ax.plot(feature_df.index, feature_df.loc(axis=1)[mouse], alpha=0.3,
                    marker='o', markersize=3, linestyle='-', label=mouse)

        # Compute summary statistics: mean and standard error (SEM) for each feature
        mean_features = feature_df.mean(axis=1)
        std_features = feature_df.std(axis=1)
        sem = std_features / np.sqrt(len(feature_df.columns))

        # Plot the mean with error bars
        ax.errorbar(feature_df.index, mean_features, xerr=sem, fmt='o-', color='black',
                    label='Mean ± SEM', linewidth=2, capsize=4)

        # Compute the global values for this feature (flattening across all mice)
        all_values = feature_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]  # remove any NaNs

        # Compute the first and third quartiles and the IQR
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        IQR = Q3 - Q1

        # Define lower and upper bounds (1.5 times the IQR below Q1 and above Q3)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Set the y-axis limits using these bounds
        ax.set_ylim(lower_bound, upper_bound)

        # Now draw vertical lines using the filtered bounds:
        ax.vlines(x=[9.5, 109.5], ymin=lower_bound, ymax=upper_bound, color='black', linestyle='--')

        ax.set_xlabel('Run')
        ax.set_ylabel(f'{feature}')
        ax.set_title(f'Aggregated {feature} Across Mice ({phase1} vs {phase2}), stride {stride_number}')
        plt.tight_layout()
        plt.legend(title='Mouse ID / Summary', loc='upper right')
        plt.grid(False)
        plt.gca().yaxis.grid(True)

        filename = f"{feature}_{phase1}_vs_{phase2}_stride{stride_number}"
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_filename = re.sub(r'\s+', '_', safe_filename)

        subdir = os.path.join(save_path, 'aggregated_raw_features')
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        output_file = os.path.join(subdir, f'{safe_filename}.png')
        output_file = r'\\?\{}'.format(output_file)
        plt.savefig(output_file)
        plt.close()

def cluster_regression_weights_across_mice(aggregated_feature_weights, phase_pair, save_dir, n_clusters=2):
    """
    Clusters regression weight vectors across mice for a given phase pair
    and labels the points with their mouse IDs.

    Parameters:
      - aggregated_feature_weights: dict with keys (mouse_id, phase1, phase2) and value = regression weight vector.
      - phase_pair: tuple (phase1, phase2) specifying the phase comparison.
      - save_dir: directory to save the clustering plot.
      - n_clusters: number of clusters to form.

    Returns:
      - cluster_df: DataFrame mapping mouse_id to its assigned cluster.
      - kmeans: The fitted KMeans model.
    """
    phase1, phase2, stride_number = phase_pair
    weights_list = []
    mouse_ids = []

    # Collect weights for the given phase pair.
    for (mouse_id, p1, p2, s), weights in aggregated_feature_weights.items():
        if p1 == phase1 and p2 == phase2 and s == stride_number:
            weights_list.append(weights)
            mouse_ids.append(mouse_id)

    if not weights_list:
        print(f"No regression weights found for phase pair {phase_pair}.")
        return None, None

    # Stack into a matrix: each row corresponds to one mouse.
    weights_matrix = np.vstack(weights_list)

    # Optionally standardize the weights.
    scaler = StandardScaler()
    weights_matrix_scaled = scaler.fit_transform(weights_matrix)

    # Cluster using KMeans.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(weights_matrix_scaled)

    # Create a DataFrame with the clustering results.
    cluster_df = pd.DataFrame({
        'mouse_id': mouse_ids,
        'cluster': clusters
    })

    # Project to 2D for visualization using PCA.
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(weights_matrix_scaled)

    # Plot the results with annotations for each point.
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=clusters, palette='viridis', s=50, ax=ax)

    # Annotate each point with the corresponding mouse id.
    for i, mouse in enumerate(mouse_ids):
        ax.text(pcs[i, 0] + 0.02, pcs[i, 1] + 0.02, str(mouse),
                fontsize=9, color='black', weight='bold')

    ax.set_title(f"Clustering of Regression Weights across Mice: {phase1} vs {phase2}, stride {stride_number}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.legend(title="Cluster")

    plot_path = os.path.join(save_dir, f"regression_weights_clustering_{phase1}_vs_{phase2}_stride{stride_number}.png")
    plt.savefig(plot_path)
    plt.close()

    return cluster_df, kmeans


def find_cluster_features(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=2):
    """
    Identify and save the features contributing to each cluster.

    Returns:
      - cluster_dict: Dictionary mapping cluster labels to lists of features.
    """
    # Cluster the weights (this call also saves the clustering CSV)
    cluster_df, _ = cluster_weights(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=n_clusters)

    # Create a dictionary grouping features by their cluster label
    cluster_dict = {}
    for cluster in range(n_clusters):
        cluster_features = cluster_df[cluster_df['cluster'] == cluster]['feature'].tolist()
        cluster_dict[f'Cluster {cluster}'] = cluster_features
        print(f"Features in Cluster {cluster} for Mouse {mouse_id} ({phase1} vs {phase2}):")
        print(cluster_features)

    # Save the cluster details to a text file
    output_file = os.path.join(save_path, f'cluster_features_{mouse_id}_{phase1}_vs_{phase2}.txt')
    with open(output_file, 'w') as f:
        for cluster, features in cluster_dict.items():
            f.write(f"{cluster}:\n")
            for feat in features:
                f.write(f"  {feat}\n")
    print(f"Cluster features details saved to: {output_file}")

    return cluster_dict


def plot_clustered_weights(cluster_df, save_path, mouse_id, phase1, phase2):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Sort the DataFrame for better visualization
    cluster_df_sorted = cluster_df.sort_values(['cluster', 'weight'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='weight', y='feature', hue='cluster', data=cluster_df_sorted, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('Feature')
    plt.title(f'Clustered Feature Weights for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    plot_file = os.path.join(save_path, f'clustered_feature_weights_{mouse_id}_{phase1}_vs_{phase2}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical clustered feature weights plot saved to: {plot_file}")


def plot_cluster_loadings_lines(aggregated_cluster_loadings, save_dir):
    """
    For each (phase1, phase2, stride) combination, create a line plot where each mouse
    is represented by a single line that shows its regression loading across clusters.
    """
    os.makedirs(save_dir, exist_ok=True)

    def plot_cluster_loadings_lines(aggregated_cluster_loadings, save_dir):
        """
        For each (phase1, phase2, stride) combination, create a line plot where each mouse
        is represented by a single line that shows its regression loading across clusters.
        If a mouse does not have a value for a given cluster, it is plotted as zero.
        """
        os.makedirs(save_dir, exist_ok=True)

        for key, mouse_cluster_dict in aggregated_cluster_loadings.items():
            phase1, phase2, stride_number = key

            # Compute the union of all cluster IDs for this key.
            all_clusters = set()
            for cl_loadings in mouse_cluster_dict.values():
                all_clusters.update(cl_loadings.keys())
            sorted_clusters = sorted(all_clusters)

            plt.figure(figsize=(10, 6))

            for mouse, cl_loadings in mouse_cluster_dict.items():
                # For each cluster in the union, get the loading, or 0 if missing.
                loadings = [cl_loadings.get(cluster, 0) for cluster in sorted_clusters]
                # scale loadings
                loadings = np.array(loadings) / np.max(np.abs(loadings))
                plt.plot(sorted_clusters, loadings, marker='o', label=mouse)

            plt.title(f"Regression Loadings by Cluster: {phase1} vs {phase2} | Stride {stride_number}")
            plt.xlabel("Cluster")
            plt.ylabel("Regression Loading")
            plt.legend(title="Mouse ID")
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'cluster_loadings_lines_{phase1}_{phase2}_stride{stride_number}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved cluster loading line plot to {save_path}")


def create_save_directory(base_dir, mouse_id, stride_number, phase1, phase2):
    """
    Create a directory path based on the settings to save plots.

    Parameters:
        base_dir (str): Base directory where plots will be saved.
        mouse_id (str): Identifier for the mouse.
        stride_number (int): Stride number used in analysis.
        phase1 (str): First experimental phase.
        phase2 (str): Second experimental phase.

    Returns:
        str: Path to the directory where plots will be saved.
    """
    # Construct the directory name
    dir_name = f"Mouse_{mouse_id}_Stride_{stride_number}_Compare_{phase1}_vs_{phase2}"
    save_path = os.path.join(base_dir, dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    return save_path


def load_stride_data(stride_data_path):
    """
    Load stride data from the specified HDF5 file.

    Parameters:
        stride_data_path (str): Path to the stride data HDF5 file.

    Returns:
        pd.DataFrame: Loaded stride data.
    """
    stride_data = pd.read_hdf(stride_data_path, key='stride_info')
    return stride_data


def determine_stepping_limbs(stride_data, mouse_id, run, stride_number):
    """
    Determine the stepping limb (ForepawL or ForepawR) for a given MouseID, Run, and Stride.

    Parameters:
        stride_data (pd.DataFrame): Stride data DataFrame.
        mouse_id (str): Identifier for the mouse.
        run (str/int): Run identifier.
        stride_number (int): Stride number.

    Returns:
        str: 'ForepawL' or 'ForepawR' indicating the stepping limb.
    """
    paws = stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any().index[
        stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any()]
    if len(paws) > 1:
        raise ValueError(f"Multiple paws found for Mouse {mouse_id}, Run {run}.")
    else:
        return paws[0]


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
    stepping_limbs = [determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
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





