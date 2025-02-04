import pandas as pd
import numpy as np
import itertools
import os  # For directory operations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder


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
stride_numbers = [-1, -2]  # List of stride numbers to filter data
phases = ['APA1', 'APA2', 'Wash2']  # List of phases to compare

# Define the base directory where plots will be saved
base_save_dir = os.path.join(paths['plotting_destfolder'], 'FeatureReduction')

# ----------------------------
# Function Definitions
# ----------------------------
sns.set(style="whitegrid")

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

def load_and_preprocess_data(mouse_id, stride_number):
    """
    Load data for the specified mouse and preprocess it by selecting desired features,
    handling missing values, and standardizing the data.

    Parameters:
        mouse_id (str): Identifier for the mouse.
        stride_number (int): Stride number to filter the data.

    Returns:
        pd.DataFrame: Preprocessed and standardized data.
    """
    # Load data
    data_allmice = pd.read_hdf(
        r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\MEASURES_single_kinematics_runXstride.h5",
        key='single_kinematics'
    )

    # Select data for the specified mouse
    try:
        data = data_allmice.loc[mouse_id]
    except KeyError:
        raise ValueError(f"Mouse ID {mouse_id} not found in the dataset.")

    # Filter desired columns based on measures_list_feature_reduction
    desired_columns = []
    for measure, params in measures_list_feature_reduction.items():
        if not params:
            # Measures with no parameters
            desired_columns.append((measure, 'no_param'))
        elif isinstance(params, dict):
            # Extract parameter keys and their possible values
            param_keys = list(params.keys())
            param_values = [params[key] for key in param_keys]

            # Generate all possible combinations of parameters
            for combination in itertools.product(*param_values):
                # Create a Params string in the format "key1:value1, key2:value2, ..."
                params_str = ', '.join(f"{key}:{value}" for key, value in zip(param_keys, combination))
                desired_columns.append((measure, params_str))
        else:
            # Handle unexpected parameter formats if necessary
            pass

    # Convert DataFrame columns to a set for faster lookup
    data_columns_set = set(data.columns)

    # Filter out only those columns that exist in the DataFrame
    selected_columns = [col for col in desired_columns if col in data_columns_set]

    if not selected_columns:
        raise ValueError("No desired columns found in the dataset based on measures_list_feature_reduction.")

    # Extract the filtered DataFrame
    filtered_data = data.loc[:, selected_columns]

    # Collapse MultiIndex to single-level index
    separator = '|'  # Choose a separator
    filtered_data.columns = [
        f"{measure}{separator}{params}" if params != 'no_param' else f"{measure}"
        for measure, params in filtered_data.columns
    ]

    # Filter by the specified stride number
    try:
        filtered_data = filtered_data.xs(stride_number, level='Stride', axis=0)
    except KeyError:
        raise ValueError(f"Stride number {stride_number} not found in the data.")

    # Handle missing data by imputing with the mean of each feature
    filtered_data_imputed = filtered_data.fillna(filtered_data.mean())

    # Verify if there are still missing values
    if filtered_data_imputed.isnull().sum().sum() > 0:
        print("Warning: There are still missing values after imputation.")
    else:
        print("All missing values have been handled.")

    # Z-Scoring (Standardization)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data_imputed)
    scaled_data_df = pd.DataFrame(scaled_data, index=filtered_data_imputed.index, columns=filtered_data_imputed.columns)

    return scaled_data_df


def perform_pca(scaled_data_df, n_components=10):
    """
    Perform Principal Component Analysis (PCA) on the standardized data.

    Parameters:
        scaled_data_df (pd.DataFrame): Standardized data.
        n_components (int): Number of principal components to compute.

    Returns:
        PCA: Fitted PCA object.
        np.ndarray: PCA-transformed data.
        pd.DataFrame: PCA loadings.
    """
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data_df)
    pcs = pca.transform(scaled_data_df)

    # Compute PCA loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=scaled_data_df.columns,
                               columns=[f'PC{i + 1}' for i in range(n_components)])

    return pca, pcs, loadings_df


def plot_pca(pca, pcs, labels, stepping_limbs, run_numbers, mouse_id, save_path):
    """
    Create 2D and 3D scatter plots of the PCA-transformed data with run number labels and save them.

    Parameters:
        pca (PCA): Fitted PCA object.
        pcs (np.ndarray): PCA-transformed data.
        labels (np.ndarray): Labels for each data point.
        stepping_limbs (list): Stepping limb identifiers ('ForepawL' or 'ForepawR') for each data point.
        run_numbers (list): Run numbers corresponding to each data point.
        mouse_id (str): Identifier for the mouse (used in plot titles).
        save_path (str): Directory path where plots will be saved.
    """
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(pcs, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])
    df_plot['Condition'] = labels
    df_plot['SteppingLimb'] = stepping_limbs
    df_plot['Run'] = run_numbers  # Add run numbers for labeling

    # Define all possible markers
    markers_all = {'ForepawL': 'X', 'ForepawR': 'o'}

    # Get unique stepping limbs
    unique_limbs = df_plot['SteppingLimb'].unique()
    print(f"Unique Stepping Limbs: {unique_limbs}")  # For debugging

    current_markers = {}
    for limb in unique_limbs:
        if limb in markers_all:
            current_markers[limb] = markers_all[limb]
        else:
            raise ValueError(f"No marker defined for stepping limb: {limb}")

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    if pca.n_components_ >= 3:
        print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

    # 2D Scatter Plot using Seaborn
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Condition',
        style='SteppingLimb',
        markers=current_markers,  # Use the dynamically created markers dictionary
        s=100,
        alpha=0.7
    )
    plt.title(f'PCA: PC1 vs PC2 for Mouse {mouse_id}')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.1f}%)')
    plt.legend(title='Condition & Stepping Limb', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)

    # Annotate points with run numbers
    for _, row in df_plot.iterrows():
        plt.text(row['PC1'] + 0.02, row['PC2'] + 0.02, str(row['Run']),
                 fontsize=8, alpha=0.7)

    # Add padding to axes
    padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
    padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
    plt.xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
    plt.ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)

    plt.tight_layout()
    # Save the 2D PCA scatter plot
    plt.savefig(os.path.join(save_path, f"PCA_Mouse_{mouse_id}_PC1_vs_PC2.png"), dpi=300)
    plt.close()

    # 3D Scatter Plot using Matplotlib (if at least 3 components)
    if pca.n_components_ >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        palette = sns.color_palette("bright", len(df_plot['Condition'].unique()))
        conditions_unique = df_plot['Condition'].unique()

        for idx, condition in enumerate(conditions_unique):
            subset = df_plot[df_plot['Condition'] == condition]
            ax.scatter(
                subset['PC1'], subset['PC2'], subset['PC3'],
                label=condition,
                color=palette[idx],
                alpha=0.7,
                s=50,
                marker='o'  # Default marker
            )
            # Annotate points with run numbers and stepping limbs
            for _, row in subset.iterrows():
                ax.text(
                    row['PC1'] + 0.02,
                    row['PC2'] + 0.02,
                    row['PC3'] + 0.02,
                    str(row['Run']),
                    fontsize=8,
                    alpha=0.7
                )

        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.1f}%)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.1f}%)')
        ax.set_zlabel(f'Principal Component 3 ({explained_variance[2] * 100:.1f}%)')
        ax.set_title(f'3D PCA for Mouse {mouse_id}')
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # Add padding to axes
        padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
        padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
        padding_pc3 = (df_plot['PC3'].max() - df_plot['PC3'].min()) * 0.05
        ax.set_xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
        ax.set_ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
        ax.set_zlim(df_plot['PC3'].min() - padding_pc3, df_plot['PC3'].max() + padding_pc3)

        plt.tight_layout()
        # Save the 3D PCA scatter plot
        plt.savefig(os.path.join(save_path, f"PCA_Mouse_{mouse_id}_PC1_vs_PC2_vs_PC3_3D.png"), dpi=300)
        plt.close()

def plot_relationship_pc1_pc2(df_plot, save_path, mouse_id):
    """
    Plot the relationship between PC1 and PC2.

    Parameters:
        df_plot (pd.DataFrame): DataFrame with PCA-transformed data.
        save_path (str): Directory path where plots will be saved.
        mouse_id (str): Identifier for the mouse (used in plot titles).
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Condition',
        style='SteppingLimb',
        markers={'ForepawL': 'X', 'ForepawR': 'o'},
        palette='Set1',
        s=100,
        alpha=0.7
    )
    plt.title(f'Relationship between PC1 and PC2 for Mouse {mouse_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Condition & Stepping Limb', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"PC1_vs_PC2_Relationship_Mouse_{mouse_id}.png"), dpi=300)
    plt.close()




def plot_scree(pca, save_path):
    """
    Plot the Scree plot to visualize explained variance by each principal component,
    including cumulative explained variance, and save it.

    Parameters:
        pca (PCA): Fitted PCA object.
        save_path (str): Directory path where plots will be saved.
    """
    plt.figure(figsize=(12, 8))

    # Individual explained variance
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label='Individual Explained Variance')

    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', linewidth=2, color='red', label='Cumulative Explained Variance')

    plt.title('Scree Plot with Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)  # Extend y-axis to accommodate cumulative variance up to 100%
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the Scree plot
    plt.savefig(os.path.join(save_path, "Scree_Plot.png"), dpi=300)
    plt.close()


def perform_lda(pcs, y_labels, phase1, phase2, n_components=1):
    """
    Perform Linear Discriminant Analysis (LDA) on PCA-transformed data.

    Parameters:
        pcs (np.ndarray): PCA-transformed data.
        y_labels (np.ndarray): Binary labels for each data point.
        phase1 (str): Name of the first phase.
        phase2 (str): Name of the second phase.
        n_components (int): Number of LDA components to compute.

    Returns:
        LDA: Fitted LDA object.
        np.ndarray: LDA-transformed data.
        np.ndarray: LDA coefficients (loadings).
    """
    lda = LDA(n_components=n_components)
    lda.fit(pcs, y_labels)
    Y_lda = lda.transform(pcs)

    lda_loadings = lda.coef_[0]  # Shape: (n_features_in_LDA, )

    return lda, Y_lda, lda_loadings


def compute_feature_contributions(loadings_df, lda_loadings):
    """
    Compute the contribution of each original feature to the LDA component.

    Parameters:
        loadings_df (pd.DataFrame): DataFrame containing PCA loadings.
        lda_loadings (np.ndarray): LDA coefficients (loadings).

    Returns:
        pd.DataFrame: DataFrame containing features and their contributions to LDA.
    """
    # Compute original feature contributions to LDA
    original_feature_contributions = loadings_df.dot(lda_loadings)

    # Create a DataFrame for plotting
    feature_contributions_df = pd.DataFrame({
        'Feature': original_feature_contributions.index,
        'Contribution': original_feature_contributions.values
    })

    # # Sort features by absolute contribution for better visualization
    # feature_contributions_df['Abs_Contribution'] = feature_contributions_df['Contribution'].abs()
    # feature_contributions_df.sort_values(by='Abs_Contribution', ascending=False, inplace=True)

    return feature_contributions_df


def plot_feature_contributions(feature_contributions_df, save_path, title_suffix=""):
    """
    Plot the contribution of each original feature to the LDA component and save it.

    Parameters:
        feature_contributions_df (pd.DataFrame): DataFrame with feature contributions.
        save_path (str): Directory path where plots will be saved.
        title_suffix (str): Suffix to add to the plot titles indicating the LDA analysis type.
    """
    plt.figure(figsize=(14, 20))  # Adjusted height for more features
    sns.barplot(
        data=feature_contributions_df,
        x='Contribution',
        y='Feature',
        palette='viridis'
    )
    plt.title(f'Original Feature Contributions to LDA Component {title_suffix}')
    plt.xlabel('Contribution to LDA')
    plt.ylabel('Original Features')
    plt.axvline(0, color='red', linewidth=1)
    # Reduce text size and handle overlap if necessary
    plt.tight_layout()
    # Save the Feature Contributions plot
    plot_filename = "Feature_Contributions_to_LDA"
    if title_suffix:
        plot_filename += f"_{title_suffix.replace(' ', '_')}"
    plot_filename += ".png"
    plt.savefig(os.path.join(save_path, plot_filename), dpi=300)
    plt.close()


def plot_lda_transformed_data(Y_lda, phase1, phase2, save_path, title_suffix=""):
    """
    Visualize the LDA-transformed data using scatter and box plots and save them.

    Parameters:
        Y_lda (np.ndarray): LDA-transformed data.
        phase1 (str): Name of the first phase.
        phase2 (str): Name of the second phase.
        save_path (str): Directory path where plots will be saved.
        title_suffix (str): Suffix to add to the plot titles indicating the LDA analysis type.
    """
    # Calculate number of samples in each phase
    num_phase1 = int(len(Y_lda) / 2)
    num_phase2 = len(Y_lda) - num_phase1  # Handle odd numbers

    # Create a DataFrame for LDA-transformed data
    df_lda = pd.DataFrame(Y_lda, columns=['LDA_Component'])
    df_lda['Condition'] = [phase1] * num_phase1 + [phase2] * num_phase2

    # Scatter Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_lda,
        x='LDA_Component',
        y=[0] * df_lda.shape[0],
        hue='Condition',
        style='Condition',
        s=100,
        alpha=0.7
    )
    plt.title(f'LDA Transformed Data for {phase1} vs {phase2} ({title_suffix})')
    plt.xlabel('LDA Component')
    plt.yticks([])  # Hide y-axis as it's not informative in this context
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)

    plt.tight_layout()
    # Save the LDA Scatter plot
    scatter_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Scatter.png"
    plt.savefig(os.path.join(save_path, scatter_plot_filename), dpi=300)
    plt.close()

    # Box Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Condition', y='LDA_Component', data=df_lda, palette='Set2')
    plt.title(f'LDA Component Distribution by Condition ({phase1} vs {phase2}) ({title_suffix})')
    plt.xlabel('Condition')
    plt.ylabel('LDA Component')
    plt.grid(True)
    plt.tight_layout()
    # Save the LDA Box plot
    box_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Box.png"
    plt.savefig(os.path.join(save_path, box_plot_filename), dpi=300)
    plt.close()

# ----------------------------
# Check relationship with handedness
# ----------------------------
def handedness_correlation(df_plot, save_path):
    """
    Check the correlation between PC2 and handedness (SteppingLimb) and visualize the relationship.

    Parameters:
        df_plot (pd.DataFrame): DataFrame with PCA-transformed data and SteppingLimb.
        save_path (str): Directory path where plots will be saved.
    """
    # Encode 'SteppingLimb' for correlation
    le = LabelEncoder()
    df_plot['SteppingLimb_Encoded'] = le.fit_transform(df_plot['SteppingLimb'])

    # Compute correlation
    correlation = df_plot['PC2'].corr(df_plot['SteppingLimb_Encoded'])
    print(f"Correlation between PC2 and SteppingLimb: {correlation:.2f}")

    # Visualize the relationship
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='SteppingLimb', y='PC2', data=df_plot, palette='Set3')
    plt.title('PC2 Scores by Stepping Limb (Handedness)')
    plt.xlabel('Stepping Limb')
    plt.ylabel('PC2 Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "PC2_vs_SteppingLimb_Boxplot.png"), dpi=300)
    plt.close()

def handedness_stats(df_plot, save_path):
    """
    Perform statistical tests to assess differences in PC2 scores between handedness groups.

    Parameters:
        df_plot (pd.DataFrame): DataFrame with PCA-transformed data and SteppingLimb.
        save_path (str): Directory path where plots will be saved.
    """
    # Assuming two groups: 'ForepawL' and 'ForepawR'
    group1 = df_plot[df_plot['SteppingLimb'] == 'ForepawL']['PC2']
    group2 = df_plot[df_plot['SteppingLimb'] == 'ForepawR']['PC2']

    # Perform t-test
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    print(f"T-test between ForepawL and ForepawR on PC2: t-stat={t_stat:.2f}, p-value={p_val:.4f}")

    # Save the t-test results to a text file
    with open(os.path.join(save_path, "PC2_Handedness_TTest.txt"), 'w') as f:
        f.write(f"T-test between ForepawL and ForepawR on PC2:\n")
        f.write(f"t-statistic = {t_stat:.2f}\n")
        f.write(f"p-value = {p_val:.4f}\n")



# ----------------------------
# Main Execution
# ----------------------------

def main(mouse_ids, stride_numbers, phases, base_save_dir):
    # Load stride_data once before the loops
    stride_data_path = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\MEASURES_StrideInfo.h5"
    stride_data = load_stride_data(stride_data_path)

    for mouse_id in mouse_ids:
        for stride_number in stride_numbers:
            for phase1, phase2 in itertools.combinations(phases, 2):
                # Create save directory based on settings
                save_path = create_save_directory(base_save_dir, mouse_id, stride_number, phase1, phase2)
                print(
                    f"Plots for Mouse ID: {mouse_id}, Stride: {stride_number}, Comparing {phase1} vs {phase2} will be saved to: {save_path}")

                # Load and preprocess data
                scaled_data_df = load_and_preprocess_data(mouse_id, stride_number)

                # Perform PCA
                pca, pcs, loadings_df = perform_pca(scaled_data_df, n_components=10)

                # Prepare masks for the specified phases
                mask_phase1 = scaled_data_df.index.get_level_values('Run').isin(
                    expstuff['condition_exp_runs']['APAChar']['Extended'][phase1])
                mask_phase2 = scaled_data_df.index.get_level_values('Run').isin(
                    expstuff['condition_exp_runs']['APAChar']['Extended'][phase2])

                # Validate that there are data points for both phases
                if not mask_phase1.any():
                    raise ValueError(f"No runs found for phase '{phase1}'. Please check the phase name and data.")
                if not mask_phase2.any():
                    raise ValueError(f"No runs found for phase '{phase2}'. Please check the phase name and data.")

                # Extract PCA-transformed data for each phase
                pcs_phase1 = pcs[mask_phase1]
                pcs_phase2 = pcs[mask_phase2]

                # Extract run numbers for each phase
                run_numbers_phase1 = scaled_data_df.index[mask_phase1]
                run_numbers_phase2 = scaled_data_df.index[mask_phase2]
                run_numbers = run_numbers_phase1.tolist() + run_numbers_phase2.tolist()

                # Determine stepping limb for each run
                stepping_limbs = []
                for run in run_numbers:
                    stepping_limb = determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                    stepping_limbs.append(stepping_limb)

                # Create labels for plotting
                labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
                labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
                labels = np.concatenate([labels_phase1, labels_phase2])

                # Combine PCA-transformed data for plotting
                pcs_combined = np.vstack([pcs_phase1, pcs_phase2])

                # Plot PCA results and save, with run number labels and stepping limb markers
                plot_pca(pca, pcs_combined, labels, stepping_limbs, run_numbers, mouse_id, save_path)

                # Plot Scree Plot and save
                plot_scree(pca, save_path)

                # ----------------------------
                # Perform LDA on All PCs
                # ----------------------------
                # Prepare labels for LDA (1 for phase1, 0 for phase2)
                y_labels_all = np.concatenate([np.ones(pcs_phase1.shape[0]), np.zeros(pcs_phase2.shape[0])])

                lda_all, Y_lda_all, lda_loadings_all = perform_lda(
                    pcs_combined,
                    y_labels=y_labels_all,
                    phase1=phase1,
                    phase2=phase2,
                    n_components=1
                )

                # Compute Feature Contributions to LDA for All PCs
                feature_contributions_df_all = compute_feature_contributions(loadings_df, lda_loadings_all)

                # Plot Feature Contributions to LDA and save for All PCs
                plot_feature_contributions(feature_contributions_df_all, save_path, title_suffix="All_PCs")

                # Visualize LDA-transformed Data and save for All PCs
                plot_lda_transformed_data(Y_lda_all, phase1, phase2, save_path, title_suffix="All_PCs")

                # Optionally, plot LDA loadings as a bar chart and save for All PCs
                plt.figure(figsize=(12, 6))

                # *** Dynamic Generation of PC Labels ***
                # Replace fixed pc_indices with a dynamically generated list based on lda_loadings_all
                pc_indices_all = [f'PC{i + 1}' for i in range(len(lda_loadings_all))]

                plt.bar(pc_indices_all, lda_loadings_all, color='skyblue')
                plt.title('LDA Loadings on PCA Components (All PCs)')
                plt.xlabel('Principal Components')
                plt.ylabel('LDA Coefficients')
                plt.grid(axis='y')
                plt.tight_layout()
                # Save the LDA Loadings plot for All PCs
                plt.savefig(os.path.join(save_path, f"LDA_Loadings_on_PCA_Components_All_PCs.png"), dpi=300)
                plt.close()

                # ----------------------------
                # Perform LDA on PC2 Only
                # ----------------------------
                # Extract only PC2 from the PCA-transformed data
                pc2_phase1 = pcs_phase1[:, 1].reshape(-1, 1)  # PC2 is index 1
                pc2_phase2 = pcs_phase2[:, 1].reshape(-1, 1)
                pcs_pc2 = np.vstack([pc2_phase1, pc2_phase2])

                # Perform LDA on PC2
                lda_pc2, Y_lda_pc2, lda_loadings_pc2 = perform_lda(
                    pcs_pc2,
                    y_labels=np.concatenate([np.ones(pcs_phase1.shape[0]), np.zeros(pcs_phase2.shape[0])]),
                    phase1=phase1,
                    phase2=phase2,
                    n_components=1
                )

                # Compute Feature Contributions to LDA for PC2
                # Since we're using only PC2, the contribution is directly from PC2
                loadings_df_pc2 = loadings_df[['PC2']]
                feature_contributions_df_pc2 = compute_feature_contributions(loadings_df_pc2, lda_loadings_pc2)

                # Plot Feature Contributions to LDA and save for PC2
                plot_feature_contributions(feature_contributions_df_pc2, save_path, title_suffix="PC2")

                # Visualize LDA-transformed Data and save for PC2
                plot_lda_transformed_data(Y_lda_pc2, phase1, phase2, save_path, title_suffix="PC2 Only")

                # Optionally, plot LDA loadings as a bar chart and save for PC2
                # **User Request:** Do not create this plot
                # Therefore, we comment it out
                # plt.figure(figsize=(6, 4))
                # pc_indices_pc2 = ['PC2']  # Only PC2
                # plt.bar(pc_indices_pc2, lda_loadings_pc2, color='skyblue')
                # plt.title('LDA Loadings on PC2 Only')
                # plt.xlabel('Principal Components')
                # plt.ylabel('LDA Coefficients')
                # plt.grid(axis='y')
                # plt.tight_layout()
                # plt.savefig(os.path.join(save_path, f"LDA_Loadings_on_PC2_Only.png"), dpi=300)
                # plt.close()

                # ----------------------------
                # Handedness Correlation and Stats
                # ----------------------------
                # Create a combined DataFrame for correlation and stats
                df_plot = pd.DataFrame(pcs_combined, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])
                df_plot['Condition'] = labels
                df_plot['SteppingLimb'] = stepping_limbs
                df_plot['Run'] = run_numbers

                # Perform correlation analysis between PC2 and Handedness
                handedness_correlation(df_plot, save_path)

                # Perform statistical tests on PC2 between handedness groups
                handedness_stats(df_plot, save_path)

                # ----------------------------
                # Include Handedness as a Covariate in LDA
                # ----------------------------
                # Encode 'SteppingLimb' as a numerical feature
                le = LabelEncoder()
                handedness_encoded = le.fit_transform(df_plot['SteppingLimb'])  # 0 and 1

                # Combine PCs with handedness as covariate
                pcs_with_covariate = np.hstack((pcs_combined, handedness_encoded.reshape(-1, 1)))

                # Perform LDA with Handedness as a covariate
                lda_cov, Y_lda_cov, lda_loadings_cov = perform_lda(
                    pcs_with_covariate,
                    y_labels_all,
                    phase1=phase1,
                    phase2=phase2,
                    n_components=1
                )

                # Compute Feature Contributions to LDA with Covariate
                # Note: The last column corresponds to Handedness
                loadings_df_cov = pd.DataFrame(
                    loadings_df.copy(),
                    index=loadings_df.index,
                    columns=list(loadings_df.columns) + ['Handedness']
                )
                # Assign a loading of 1 for the Handedness covariate
                loadings_df_cov['Handedness'] = 1  # Since it's directly included

                feature_contributions_df_cov = compute_feature_contributions(loadings_df_cov, lda_loadings_cov)

                # Plot Feature Contributions to LDA with Covariate and save
                plot_feature_contributions(feature_contributions_df_cov, save_path,
                                           title_suffix="All_PCs_With_Handedness")

                # Visualize LDA-transformed Data with Covariate and save
                plot_lda_transformed_data(Y_lda_cov, phase1, phase2, save_path, title_suffix="All_PCs_With_Handedness")

                # ----------------------------
                # Relationship between PC1 and PC2
                # ----------------------------
                plot_relationship_pc1_pc2(df_plot, save_path, mouse_id)


# ----------------------------
# Execute Main Function
# ----------------------------

if __name__ == "__main__":
    main(mouse_ids, stride_numbers, phases, base_save_dir)
