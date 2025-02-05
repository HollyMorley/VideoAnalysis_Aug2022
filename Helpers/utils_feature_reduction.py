from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import os
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pairwise_cosine_similarity(pivot_df, save_path):
    """
    Computes the pairwise cosine similarity between mice based on their unique contributions,
    and plots:
      - A histogram of all pairwise cosine similarity values.
      - Boxplots of the unique contribution distributions per feature.
      - (Optionally) a summary boxplot of average cosine similarity per mouse.
    """
    # Fill missing values (if any) with 0 (or another appropriate value)
    data_matrix = pivot_df.fillna(0).values

    # Compute cosine similarity matrix (rows: mice)
    similarity_matrix = cosine_similarity(data_matrix)

    # Get the upper triangle (excluding the diagonal) as a 1D array
    # (Alternatively, you can use the lower triangle)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarity_values = similarity_matrix[triu_indices]

    # Histogram of pairwise cosine similarities
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Pairwise Cosine Similarities Across Mice')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Cosine_Similarity_Histogram.png'), dpi=300)
    plt.close()

    # Boxplot: unique contributions per feature across mice
    plt.figure(figsize=(14, 8))
    pivot_df.boxplot(rot=45)
    plt.title('Unique Contribution Distribution per Feature Across Mice')
    plt.xlabel('Feature')
    plt.ylabel('Unique Contribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Unique_Contribution_Boxplot.png'), dpi=300)
    plt.close()

    # (Optional) Single variable plot: average cosine similarity per mouse.
    # Compute the average similarity (excluding self similarity)
    avg_similarity = similarity_matrix.sum(axis=1) / (similarity_matrix.shape[1] - 1)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=avg_similarity)
    plt.title('Distribution of Average Cosine Similarity per Mouse')
    plt.xlabel('Average Cosine Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Average_Cosine_Similarity_Boxplot.png'), dpi=300)
    plt.close()

    return similarity_matrix


def aggregate_feature_contributions(contrib_list):
    """
    Given a list of DataFrames (each with columns 'Mouse', 'Feature', 'Contribution'),
    combine them into one pivot table with rows = Mouse and columns = Feature.
    """
    df_all = pd.concat(contrib_list, ignore_index=True)

    # If a mouse appears multiple times, you might average the contributions per feature:
    df_avg = df_all.groupby(['Mouse', 'Feature'], as_index=False)['Contribution'].mean()

    # Pivot so that each row is a mouse and each column is a feature
    pivot_df = df_avg.pivot(index='Mouse', columns='Feature', values='Contribution')
    return pivot_df


def compute_feature_importances_regression(X, y, cv=5):
    """
    Computes the full model cvR², the cvR² for each single-feature model,
    and the unique contribution (ΔR²) for each feature, where
      ΔR² = full_model_cvR² - cvR²(without the feature).

    Parameters:
        X (pd.DataFrame): Standardized data (PCs by runs).
        y (np.array): Binary labels (e.g. 1 for phase1, 0 for phase2).
        cv (int): Number of cross-validation folds.

    Returns:
        full_R2 (float): The full model cross-validated R².
        single_cvR2 (dict): Dictionary mapping feature name -> cvR² using only that feature.
        unique_delta_R2 (dict): Dictionary mapping feature name -> unique ΔR².
    """
    model = LogisticRegression()
    full_R2 = np.mean(cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')) #dont want to cross validate the full model

    model.fit(X, y)
    ws = model.coef_
    w = np.mean(ws, axis=1)[0]

    single_cvR2 = {}
    unique_delta_R2 = {}

    for feature in X.columns:
        # Single-feature model
        X_feature = X[[feature]]
        cvR2_feature = np.mean(cross_val_score(model, X_feature, y, cv=cv, scoring='balanced_accuracy'))
        single_cvR2[feature] = cvR2_feature

        # Reduced model (all features except this one)
        X_reduced = X.drop(columns=[feature])
        cvR2_reduced = np.mean(cross_val_score(model, X_reduced, y, cv=cv, scoring='balanced_accuracy'))
        unique_delta_R2[feature] = full_R2 - cvR2_reduced

    return full_R2, single_cvR2, unique_delta_R2, w


def plot_feature_accuracy(single_cvR2, save_path, title_suffix="Single_Feature_cvR2"):
    """
    Plots the single-feature model cvR² values.

    Parameters:
        single_cvR2 (dict): Mapping of feature names to cvR² values.
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """
    df = pd.DataFrame(list(single_cvR2.items()), columns=['Feature', 'cvR2'])
    # Replace the separator so that group headers appear as "Group: FeatureName"
    df['Display'] = df['Feature'].apply(lambda x: x.replace('|', ': '))
    #df = df.sort_values(by='cvR2', ascending=False)

    plt.figure(figsize=(14, max(8, len(df) * 0.3)))
    sns.barplot(data=df, x='cvR2', y='Display', palette='viridis')
    plt.title('Single Feature Model accuracy ' + title_suffix)
    plt.xlabel('accuracy')
    plt.ylabel('Feature (Group: FeatureName)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Single_Feature_cvR2_{title_suffix}.png"), dpi=300)
    plt.close()


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

def plot_unique_delta_accuracy(unique_delta_R2, save_path, title_suffix="Unique_ΔR2"):
    """
    Plots the unique contribution (ΔR²) for each feature.

    Parameters:
        unique_delta_R2 (dict): Mapping of feature names to unique ΔR².
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(list(unique_delta_R2.items()), columns=['Feature', 'Unique_ΔR2'])
    df['Display'] = df['Feature'].apply(lambda x: x.replace('|', ': '))
    #df = df.sort_values(by='Unique_ΔR2', ascending=False)

    plt.figure(figsize=(14, max(8, len(df) * 0.3)))
    sns.barplot(data=df, x='Unique_Δaccuracy', y='Display', palette='magma')
    plt.title('Unique Feature Contributions (Δaccuracy) ' + title_suffix)
    plt.xlabel('Δaccuracy')
    plt.ylabel('Feature (Group: FeatureName)')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Unique_delta_R2_{title_suffix}.png"), dpi=300)
    plt.close()


def build_desired_columns(measures_list_feature_reduction, data, separator=', '):
    """
    Optimized version of build_desired_columns.

    It first groups available columns by their measure (the first element of the tuple)
    and then, for each measure in measures_list_feature_reduction, it only generates
    allowed parameter combinations if that measure exists in the DataFrame.

    Parameters:
        measures_list_feature_reduction (dict): Dictionary where each key is a measure name and
            each value is either an empty list (for no parameters) or a dictionary of parameter lists.
        data (pd.DataFrame): DataFrame whose MultiIndex columns (tuples) will be checked.
        separator (str): Separator used to join parameter keys and values.

    Returns:
        selected_columns (list of tuple): List of tuples (measure, params) that exist in data.columns.
    """
    # Group available column parameter parts by measure for fast lookup.
    columns_by_measure = {}
    for col in data.columns:
        measure, param = col
        columns_by_measure.setdefault(measure, set()).add(param)

    selected_columns = []
    for measure, allowed in measures_list_feature_reduction.items():
        # Skip measures that are not present at all.
        if measure not in columns_by_measure:
            continue
        available_params = columns_by_measure[measure]
        if not allowed:  # No parameters expected; the DataFrame should have 'no_param'
            if 'no_param' in available_params:
                selected_columns.append((measure, 'no_param'))
        elif isinstance(allowed, dict):
            param_keys = list(allowed.keys())
            param_values = [allowed[key] for key in param_keys]
            # Generate allowed parameter combinations only for this measure.
            for combination in itertools.product(*param_values):
                params_str = separator.join(f"{key}:{value}" for key, value in zip(param_keys, combination))
                if params_str in available_params:
                    selected_columns.append((measure, params_str))
        else:
            # If the parameter format is unexpected, you can add additional handling here.
            pass

    if not selected_columns:
        raise ValueError("No desired columns found in the dataset based on measures_list_feature_reduction.")
    return selected_columns

