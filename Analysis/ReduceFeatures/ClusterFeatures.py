import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

from Analysis.ReduceFeatures import utils_feature_reduction as utils
from Analysis.ReduceFeatures.config import base_save_dir_no_c

def get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1, phase2):
    """
    Build the global runs-by-features matrix from the provided mouse IDs.
    Returns the transposed feature matrix (rows = features, columns = runs).
    """
    all_runs_data = []
    for mouse in global_fs_mouse_ids:
        data = utils.load_and_preprocess_data(mouse, stride_number, condition, exp, day)
        # Get runs for the two phases; here we use phase1 and phase2 for example.
        run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(data, stride_data, mouse, stride_number, phase1,
                                                                  phase2)
        selected_mask = mask_phase1 | mask_phase2
        run_data = data.loc[selected_mask]
        all_runs_data.append(run_data)

    if not all_runs_data:
        raise ValueError("No run data found for global clustering.")

    global_data = pd.concat(all_runs_data, axis=0)
    # Transpose so that rows are features, columns are runs.
    feature_matrix = global_data.T
    return feature_matrix


def cross_validate_db_score_folds(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1,
                                  phase2, k_range=range(2, 11), n_splits=10, n_init=10):
    """
    Build the global feature matrix and perform k-fold cross-validation to select the optimal number
    of clusters using the Davies–Bouldin score.

    For each fold, KMeans is fitted on the training subset, and the DB score is computed on the test subset.

    Returns:
      - avg_db_scores: dict mapping each k to its average Davies–Bouldin score across folds.
    """
    # Build the global feature matrix (features as rows, runs as columns)
    feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
                                               phase1, phase2)

    # Prepare KFold cross-validation over features (rows of feature_matrix)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    db_scores = {k: [] for k in k_range}

    for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
        train_data = feature_matrix.iloc[train_idx]
        test_data = feature_matrix.iloc[test_idx]

        for k in k_range:
            # Use built-in n_init for efficiency.
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(train_data)
            # Predict test labels.
            test_labels = kmeans.predict(test_data)
            # Compute the Davies–Bouldin score on the test data.
            db_score = davies_bouldin_score(test_data, test_labels)
            db_scores[k].append(db_score)

    # Average DB scores across folds for each candidate k.
    avg_db_scores = {k: np.mean(scores) for k, scores in db_scores.items()}
    for k, score in avg_db_scores.items():
        print(f"k={k}, Average Davies–Bouldin Score (CV): {score:.3f}")
    return avg_db_scores

def cross_validate_inertia_folds(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1,
                                 phase2, k_range=range(2, 11), n_splits=10, n_init=10):
    """
    Build the global feature matrix and perform k-fold cross-validation to evaluate the average inertia
    (within-cluster sum-of-squares) for each candidate number of clusters k.
    Returns:
      - avg_inertia: dict mapping each k to its average inertia across folds.
    """
    # Build the global feature matrix (features as rows, runs as columns)
    feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
                                               phase1, phase2)

    # Prepare KFold cross-validation over features (rows of feature_matrix)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    inertia_scores = {k: [] for k in k_range}

    for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
        train_data = feature_matrix.iloc[train_idx]
        test_data = feature_matrix.iloc[test_idx]

        for k in k_range:
            # Use built-in n_init for efficiency.
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(train_data)
            # Compute inertia on the test set by assigning test_data to the nearest cluster center.
            # (This is a simple approximation; you could also compute it on the training set.)
            test_labels = kmeans.predict(test_data)
            inertia = kmeans.inertia_  # inertia from the training set.
            inertia_scores[k].append(inertia)

    # Average inertia scores across folds for each candidate k.
    avg_inertia = {k: np.mean(scores) for k, scores in inertia_scores.items()}
    for k, inertia in avg_inertia.items():
        print(f"k={k}, Average Inertia (CV): {inertia:.3f}")
    return avg_inertia

def cross_validate_k_clusters_folds(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1,
                                    phase2,
                                    k_range=range(2, 11), n_splits=10, n_init=10):
    """
    Build the global feature matrix and perform k-fold cross-validation to select the optimal number
    of clusters. In each fold, the clustering model is fit on the training subset of features and then
    applied to the test subset to compute the silhouette score.

    Parameters:
      - global_fs_mouse_ids: list of mouse IDs for global feature selection.
      - stride_number, condition, exp, day, stride_data, phase1, phase2: parameters used to load data.
      - k_range: range of candidate k values (number of clusters).
      - n_splits: number of folds in the KFold cross-validation.
      - n_init: number of random initializations per fold for stability.

    Returns:
      - optimal_k: the k value with the highest average silhouette score across folds.
      - avg_sil_scores: dict mapping each k to its average silhouette score.
    """
    # Build the global feature matrix (features as rows, runs as columns)
    feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
                                               phase1, phase2)

    # Prepare KFold cross-validation over features (rows of feature_matrix)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Dictionary to store silhouette scores per candidate k for each fold.
    fold_scores = {k: [] for k in k_range}

    # Wrap the outer loop with tqdm to show fold progress.
    for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
        train_data = feature_matrix.iloc[train_idx]
        test_data = feature_matrix.iloc[test_idx]

        for k in k_range:
            # Use built-in n_init instead of manual loop.
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(train_data)
            test_labels = kmeans.predict(test_data)
            score = silhouette_score(test_data, test_labels)
            fold_scores[k].append(score)

    # Average silhouette scores across folds for each candidate k.
    avg_sil_scores = {k: np.mean(scores_list) for k, scores_list in fold_scores.items()}
    for k, score in avg_sil_scores.items():
        print(f"k={k}, Average Silhouette Score (CV): {score:.3f}")

    # Select the k with the highest average silhouette score.
    optimal_k = max(avg_sil_scores, key=avg_sil_scores.get)
    print(f"Optimal k determined to be: {optimal_k}")
    return optimal_k, avg_sil_scores

def cross_validate_k_clusters_folds_pca(global_fs_mouse_ids, stride_number, condition, exp, day,
                                        stride_data, phase1, phase2,
                                        n_components=10,
                                        k_range=range(2, 11),
                                        n_splits=10, n_init=10):
    """
    Build the global feature matrix, apply PCA to reduce its dimensionality, and then perform
    k-fold cross-validation to select the optimal number of clusters using the silhouette score.

    Parameters:
      - global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1, phase2:
          parameters used to load data.
      - n_components: number of PCA components to retain.
      - k_range: range of candidate k values.
      - n_splits: number of folds in the KFold cross-validation.
      - n_init: number of initializations for KMeans.

    Returns:
      - optimal_k: the k value with the highest average silhouette score across folds.
      - avg_sil_scores: dict mapping each k to its average silhouette score.
    """
    # Build the global feature matrix (rows = features, columns = runs)
    feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day,
                                               stride_data, phase1, phase2)
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(feature_matrix.values)
    # Reconstruct a DataFrame with the same feature index.
    X_reduced_df = pd.DataFrame(X_reduced, index=feature_matrix.index)

    # Prepare KFold cross-validation over the features (rows of X_reduced_df)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = {k: [] for k in k_range}

    # Wrap the outer loop with tqdm to show progress.
    for train_idx, test_idx in tqdm(list(kf.split(X_reduced_df)), total=n_splits, desc="PCA Folds"):
        train_data = X_reduced_df.iloc[train_idx]
        test_data = X_reduced_df.iloc[test_idx]

        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(train_data)
            test_labels = kmeans.predict(test_data)
            score = silhouette_score(test_data, test_labels)
            fold_scores[k].append(score)

    # Average the silhouette scores over folds for each candidate k.
    avg_sil_scores = {k: np.mean(scores) for k, scores in fold_scores.items()}
    for k, score in avg_sil_scores.items():
        print(f"k={k}, Average Silhouette Score (CV, PCA): {score:.3f}")

    # Select the k with the highest average silhouette score.
    optimal_k = max(avg_sil_scores, key=avg_sil_scores.get)
    print(f"Optimal k (after PCA) determined to be: {optimal_k}")
    return optimal_k, avg_sil_scores


def cluster_features_run_space(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1, phase2,
                               n_clusters, save_file='feature_clusters.pkl'):
    """
    Build the global runs-by-features matrix, transpose it so that rows are features,
    cluster the features using k-means with n_clusters, and save the mapping.
    Returns:
      cluster_mapping: dict mapping feature names to cluster labels.
    """
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
                                               phase1, phase2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_matrix)

    # Mapping: feature -> cluster label
    cluster_mapping = dict(zip(feature_matrix.index, kmeans.labels_))
    joblib.dump(cluster_mapping, save_file)
    print(f"Feature clustering done and saved to {save_file} using k={n_clusters}.")
    return cluster_mapping


def plot_feature_clustering(feature_matrix, cluster_mapping, save_file="feature_clustering.png"):
    """
    Projects the features into 2D using PCA and plots them colored by cluster.
    Each point represents a feature (from the rows of feature_matrix).
    """
    # Perform PCA on the feature matrix (rows are features)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(feature_matrix.values)

    # Get cluster label for each feature
    features = feature_matrix.index
    clusters = [cluster_mapping.get(feat, -1) for feat in features]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="tab10", s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Feature Clustering (PCA Projection)")
    plt.colorbar(scatter, label="Cluster")

    # Optionally annotate features (can be crowded if many features)
    # for i, feat in enumerate(features):
    #     plt.annotate(feat, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=6, alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(base_save_dir_no_c, save_file)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature clustering plot to {save_path}")


def plot_feature_clusters_chart(cluster_mapping, save_file="feature_clusters_chart.png"):
    """
    Creates and saves a chart that arranges feature names by their assigned cluster.
    Each column represents one cluster.
    """
    import matplotlib.pyplot as plt

    # Invert the cluster_mapping: cluster -> list of features
    clusters = {}
    for feat, cl in cluster_mapping.items():
        clusters.setdefault(cl, []).append(feat)

    # Sort clusters by cluster id and sort features within each cluster (optional)
    sorted_clusters = {cl: sorted(feats) for cl, feats in sorted(clusters.items())}
    k = len(sorted_clusters)

    # Find the maximum number of features in any cluster
    max_features = max(len(feats) for feats in sorted_clusters.values())

    # Build table data: each column corresponds to a cluster.
    table_data = []
    for i in range(max_features):
        row = []
        for cl in sorted(sorted_clusters.keys()):
            feats = sorted_clusters[cl]
            if i < len(feats):
                row.append(feats[i])
            else:
                row.append("")  # empty if no feature for this row in the cluster
        table_data.append(row)

    # Create the figure and table
    fig, ax = plt.subplots(figsize=(k * 8, max_features * 0.5 + 1))
    ax.axis("tight")
    ax.axis("off")
    col_labels = [f"Cluster {cl}" for cl in sorted(sorted_clusters.keys())]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    save_path = os.path.join(base_save_dir_no_c, save_file)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature clusters chart to {save_path}")
