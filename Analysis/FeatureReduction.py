import pandas as pd
import itertools
import os  # For directory operations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from Helpers import utils_feature_reduction as utils
from Helpers.Config_23 import *

# ----------------------------
# Configuration Section
# ----------------------------

# Set your parameters here
mouse_ids = [
    '1035243', '1035244', '1035245', '1035246',
    '1035249', '1035250', '1035297', '1035298',
    '1035299', '1035301', '1035302'
]  # List of mouse IDs to analyze
stride_numbers = [-1]  # List of stride numbers to filter data
phases = ['APA2', 'Wash2']  # List of phases to compare
base_save_dir = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round2-20250204')

# ----------------------------
# Function Definitions
# ----------------------------
sns.set(style="whitegrid")


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


def perform_pca(scaled_data_df, n_components):
    """
    Perform PCA on the standardized data.
    """
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data_df)
    pcs = pca.transform(scaled_data_df)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=scaled_data_df.columns,
                               columns=[f'PC{i + 1}' for i in range(n_components)])
    return pca, pcs, loadings_df


def plot_pca(pca, pcs, labels, stepping_limbs, run_numbers, mouse_id, save_path):
    """
    Create and save 2D and 3D PCA scatter plots.
    """
    df_plot = pd.DataFrame(pcs, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])
    df_plot['Condition'] = labels
    df_plot['SteppingLimb'] = stepping_limbs
    df_plot['Run'] = run_numbers

    markers_all = {'ForepawL': 'X', 'ForepawR': 'o'}
    unique_limbs = df_plot['SteppingLimb'].unique()
    current_markers = {}
    for limb in unique_limbs:
        if limb in markers_all:
            current_markers[limb] = markers_all[limb]
        else:
            raise ValueError(f"No marker defined for stepping limb: {limb}")

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    if pca.n_components_ >= 3:
        print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

    # 2D Scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Condition',
        style='SteppingLimb',
        markers=current_markers,
        s=100,
        alpha=0.7
    )
    plt.title(f'PCA: PC1 vs PC2 for Mouse {mouse_id}')
    plt.xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
    plt.legend(title='Condition & Stepping Limb', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    for _, row in df_plot.iterrows():
        plt.text(row['PC1'] + 0.02, row['PC2'] + 0.02, str(row['Run']), fontsize=8, alpha=0.7)
    padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
    padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
    plt.xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
    plt.ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"PCA_Mouse_{mouse_id}_PC1_vs_PC2.png"), dpi=300)
    plt.close()

    # 3D Scatter (if available)
    if pca.n_components_ >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        palette = sns.color_palette("bright", len(df_plot['Condition'].unique()))
        conditions_unique = df_plot['Condition'].unique()
        for idx, condition in enumerate(conditions_unique):
            subset = df_plot[df_plot['Condition'] == condition]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                       label=condition, color=palette[idx], alpha=0.7, s=50, marker='o')
            for _, row in subset.iterrows():
                ax.text(row['PC1'] + 0.02, row['PC2'] + 0.02, row['PC3'] + 0.02,
                        str(row['Run']), fontsize=8, alpha=0.7)
        ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
        ax.set_zlabel(f'PC3 ({explained_variance[2] * 100:.1f}%)')
        ax.set_title(f'3D PCA for Mouse {mouse_id}')
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2)
        padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
        padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
        padding_pc3 = (df_plot['PC3'].max() - df_plot['PC3'].min()) * 0.05
        ax.set_xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
        ax.set_ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
        ax.set_zlim(df_plot['PC3'].min() - padding_pc3, df_plot['PC3'].max() + padding_pc3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"PCA_Mouse_{mouse_id}_PC1_vs_PC2_vs_PC3_3D.png"), dpi=300)
        plt.close()


def plot_scree(pca, save_path):
    """
    Plot and save the scree plot.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label='Individual Explained Variance')
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', linewidth=2, color='red', label='Cumulative Explained Variance')
    plt.title('Scree Plot with Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Scree_Plot.png"), dpi=300)
    plt.close()


def perform_lda(pcs, y_labels, phase1, phase2, n_components):
    """
    Perform LDA on the PCA-transformed data.
    """
    lda = LDA(n_components=n_components)
    lda.fit(pcs, y_labels)
    Y_lda = lda.transform(pcs)
    lda_loadings = lda.coef_[0]
    return lda, Y_lda, lda_loadings


def compute_feature_contributions(loadings_df, lda_loadings):
    """
    Compute original feature contributions to LDA.
    """
    original_feature_contributions = loadings_df.dot(lda_loadings)
    feature_contributions_df = pd.DataFrame({
        'Feature': original_feature_contributions.index,
        'Contribution': original_feature_contributions.values
    })
    return feature_contributions_df


def plot_feature_contributions(feature_contributions_df, save_path, title_suffix=""):
    """
    Plot the LDA feature contributions.
    """
    plt.figure(figsize=(14, 50))
    sns.barplot(data=feature_contributions_df, x='Contribution', y='Feature', palette='viridis')
    plt.title(f'Original Feature Contributions to LDA Component {title_suffix}')
    plt.xlabel('Contribution to LDA')
    plt.ylabel('Original Features')
    plt.axvline(0, color='red', linewidth=1)
    plt.tight_layout()
    plot_filename = "Feature_Contributions_to_LDA"
    if title_suffix:
        plot_filename += f"_{title_suffix.replace(' ', '_')}"
    plot_filename += ".png"
    plt.savefig(os.path.join(save_path, plot_filename), dpi=300)
    plt.close()


def plot_lda_transformed_data(Y_lda, phase1, phase2, save_path, title_suffix=""):
    """
    Plot the LDA transformed data (scatter and box plots).
    """
    num_phase1 = int(len(Y_lda) / 2)
    num_phase2 = len(Y_lda) - num_phase1
    df_lda = pd.DataFrame(Y_lda, columns=['LDA_Component'])
    df_lda['Condition'] = [phase1] * num_phase1 + [phase2] * num_phase2

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_lda, x='LDA_Component', y=[0] * df_lda.shape[0],
                    hue='Condition', style='Condition', s=100, alpha=0.7)
    plt.title(f'LDA Transformed Data for {phase1} vs {phase2} ({title_suffix})')
    plt.xlabel('LDA Component')
    plt.yticks([])
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    plt.tight_layout()
    scatter_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Scatter.png"
    plt.savefig(os.path.join(save_path, scatter_plot_filename), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Condition', y='LDA_Component', data=df_lda, palette='Set2')
    plt.title(f'LDA Component Distribution by Condition ({phase1} vs {phase2}) ({title_suffix})')
    plt.xlabel('Condition')
    plt.ylabel('LDA Component')
    plt.grid(True)
    plt.tight_layout()
    box_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Box.png"
    plt.savefig(os.path.join(save_path, box_plot_filename), dpi=300)
    plt.close()

def plot_LDA_loadings(lda_loadings_all, save_path, title_suffix=""):
    # Optional LDA loadings bar plot.
    plt.figure(figsize=(12, 6))
    pc_indices_all = [f'PC{i + 1}' for i in range(len(lda_loadings_all))]
    plt.bar(pc_indices_all, lda_loadings_all, color='skyblue')
    plt.title(f'LDA Loadings on PCA Components {title_suffix}')
    plt.xlabel('Principal Components')
    plt.ylabel('LDA Coefficients')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"LDA_Loadings_on_PCA_Components_All_{title_suffix}.png"), dpi=300)
    plt.close()


def cross_validate_pca(scaled_data_df, save_path, n_folds=10):
    """
    Perform PCA on folds of the data. For each fold, perform PCA with n_components equal to
    the number of features (or the number of training samples if lower) and record the explained variance ratio.
    This version also plots the cumulative explained variance for each fold and adds a horizontal line at 80%.

    Parameters:
        scaled_data_df (pd.DataFrame): The standardized data.
        save_path (str): Directory where the plot will be saved.
        n_folds (int): Number of cross-validation folds.

    Returns:
        fold_explained_variances (list of np.ndarray): A list containing the explained variance
                                                       ratios for each fold.
    """
    num_features = scaled_data_df.shape[1]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_explained_variances = []

    plt.figure(figsize=(40, 15))

    # For tracking the maximum number of components across folds.
    max_n_components = 0

    # Loop through folds, applying PCA on the training split of each fold.
    for fold_idx, (train_index, test_index) in enumerate(kf.split(scaled_data_df)):
        X_train = scaled_data_df.iloc[train_index]
        current_n_components = min(num_features, X_train.shape[0])
        max_n_components = max(max_n_components, current_n_components)
        print(f"Fold {fold_idx + 1}: {current_n_components} components")

        pca = PCA(n_components=current_n_components)
        pca.fit(X_train)
        explained = pca.explained_variance_ratio_
        fold_explained_variances.append(explained)

        # Plot individual explained variance for this fold.
        plt.plot(range(1, current_n_components + 1), explained, marker='o', label=f'Fold {fold_idx + 1} EV')

        # Compute and plot cumulative explained variance for this fold.
        cumulative_explained = np.cumsum(explained)
        plt.plot(range(1, current_n_components + 1), cumulative_explained, marker='s', linestyle='--',
                 label=f'Fold {fold_idx + 1} Cumulative')

    # Add a horizontal line at 80% explained variance.
    plt.axhline(y=0.8, color='k', linestyle=':', linewidth=2, label='80% Threshold')

    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Principal Components across Folds')

    # Set x-axis ticks to show every other number.
    plt.xticks(range(1, max_n_components + 1, 2))
    plt.yticks(np.linspace(0,1,21))

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "PCA_CV_Scree_Plot.png"), dpi=300)
    plt.close()

    return fold_explained_variances


def plot_average_variance_explained_across_folds(fold_variances, scaled_data_df):
    # Determine the minimum number of components across folds.
    min_components = min(len(arr) for arr in fold_variances)

    # Trim each fold's explained variance array to min_components.
    trimmed_variances = [arr[:min_components] for arr in fold_variances]

    # Compute the average explained variance ratio across folds.
    avg_variance = np.mean(np.vstack(trimmed_variances), axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, min_components + 1), avg_variance, marker='s', color='black')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Average Explained Variance Ratio')
    plt.title('Average PCA Scree Plot across Folds')
    plt.xticks(range(1, min_components + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Average_PCA_CV_Scree_Plot.png", dpi=300)
    plt.close()


# -----------------------------------------------------
# Main Processing Function for Each Phase Comparison
# -----------------------------------------------------
def process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, condition, exp, day, base_save_dir_condition):
    # Create directory for saving plots.
    save_path = utils.create_save_directory(base_save_dir_condition, mouse_id, stride_number, phase1, phase2)
    print(f"Processing Mouse {mouse_id}, Stride {stride_number}: {phase1} vs {phase2} (saving to {save_path})")

    # Load and preprocess data.
    scaled_data_df = load_and_preprocess_data(mouse_id, stride_number, condition, exp, day) # Load data for the specified mouse and preprocess it by selecting desired features, imputing missing values, and standardizing.
    print('Data loaded and preprocessed.')

    fold_variances = cross_validate_pca(scaled_data_df, save_path, n_folds=10)
    plot_average_variance_explained_across_folds(fold_variances, scaled_data_df)

    # Perform PCA.
    pca, pcs, loadings_df = perform_pca(scaled_data_df, n_components=15)

    # Create masks based on Run numbers.
    mask_phase1 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase1])
    mask_phase2 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase2])

    if not mask_phase1.any():
        raise ValueError(f"No runs found for phase '{phase1}'.")
    if not mask_phase2.any():
        raise ValueError(f"No runs found for phase '{phase2}'.")

    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]
    run_numbers_phase1 = scaled_data_df.index[mask_phase1]
    run_numbers_phase2 = scaled_data_df.index[mask_phase2]
    run_numbers = list(run_numbers_phase1) + list(run_numbers_phase2)

    # Determine stepping limbs.
    stepping_limbs = [utils.determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                      for run in run_numbers]

    labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
    labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
    labels = np.concatenate([labels_phase1, labels_phase2])
    pcs_combined = np.vstack([pcs_phase1, pcs_phase2])

    # Plot PCA and Scree.
    plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)
    plot_scree(pca, save_path)

    # normalize the PCs
    pcs_combined_norm = StandardScaler().fit_transform(pcs_combined) # todo new

    # ---------------- LDA Analysis ----------------
    y_labels_all = np.concatenate([np.ones(pcs_phase1.shape[0]), np.zeros(pcs_phase2.shape[0])])
    lda_all, Y_lda_all, lda_loadings_all = perform_lda(pcs_combined_norm, y_labels=y_labels_all,
                                                       phase1=phase1, phase2=phase2, n_components=1)
    feature_contributions_df_all = compute_feature_contributions(loadings_df, lda_loadings_all)
    plot_feature_contributions(feature_contributions_df_all, save_path, title_suffix="All_PCs")
    plot_lda_transformed_data(Y_lda_all, phase1, phase2, save_path, title_suffix="All_PCs")
    plot_LDA_loadings(lda_loadings_all, save_path, title_suffix="All_PCs")

    # ---------------- Regression-Based Feature Contributions ----------------
    # mask_all = mask_phase1 | mask_phase2
    # X_reg = scaled_data_df.loc[mask_all]
    pcs_combined_norm_df = pd.DataFrame(pcs_combined_norm, index=run_numbers, columns=[f'PC{i + 1}' for i in range(pcs_combined_norm.shape[1])])
    X_reg = pcs_combined_norm_df
    y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])
    full_R2, single_cvR2, unique_delta_R2, w = utils.compute_feature_importances_regression(X_reg, y_reg, cv=5) #todo new/adjusted
    print(f"Full model cvR²: {full_R2:.3f}")
    utils.plot_feature_cvR2(single_cvR2, save_path, title_suffix=f"{phase1}_vs_{phase2}")
    utils.plot_unique_delta_R2(unique_delta_R2, save_path, title_suffix=f"{phase1}_vs_{phase2}")

    # Convert unique_delta_R2 (a dict) to a DataFrame and add a Mouse identifier.
    contrib_df = pd.DataFrame(list(unique_delta_R2.items()), columns=['Feature', 'Contribution'])
    contrib_df['Mouse'] = mouse_id  # assuming mouse_id is defined in your function scope

    # Return the contributions_df so that you can aggregate them later.
    return contrib_df


# -----------------------------------------------------
# Main Execution Function
# -----------------------------------------------------
def main(mouse_ids, stride_numbers, phases, condition='LowHigh', exp='Extended', day=None):
    # construct path
    if exp == 'Extended':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_StrideInfo.h5")
    elif exp == 'Repeats':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
    stride_data = utils.load_stride_data(stride_data_path)

    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')

    all_contributions = []  # list to store each mouse's contributions
    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                contrib_df = process_phase_comparison(mouse_id, stride_number, phase1, phase2, stride_data, condition,
                                                      exp, day, base_save_dir_condition)
                all_contributions.append(contrib_df)

    # After processing all mice, aggregate the contributions.
    # You can choose a save directory for these summary plots:
    summary_save_path = os.path.join(base_save_dir_condition, 'Summary_Cosine_Similarity')
    os.makedirs(summary_save_path, exist_ok=True)

    pivot_contributions = utils.aggregate_feature_contributions(all_contributions)

    # Compute and plot the pairwise cosine similarity and the boxplots.
    similarity_matrix = utils.plot_pairwise_cosine_similarity(pivot_contributions, summary_save_path)


# ----------------------------
# Execute Main Function
# ----------------------------

if __name__ == "__main__":
    main(mouse_ids, stride_numbers, phases, condition='APAChar_LowHigh', exp='Extended',day=None)
