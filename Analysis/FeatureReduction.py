import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import seaborn as sns

from Helpers.Config_23 import *

"""
first attempt:
- start with 1 mouse - 1035243
- start with 1 condition - LowHigh Extended
- compare APA2 by Washout2 (second halves) (ultimately, but start with all runs)

Steps:
1) num features x n runs
2) PCA gives num features x num features ** wrong, its the other way around
3) reduce dimensions num PCs x n runs
4) LDA gives 1 x num PCs
"""

# ----------------------------
# Preprocessing
# ----------------------------

# Load data
data_allmice = pd.read_hdf(
    r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\MEASURES_single_kinematics_runXstride.h5",
    key='single_kinematics'
)

# Select one mouse
mouse_id = '1035243'
data = data_allmice.loc[mouse_id]

# Filter data features
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

# Extract the filtered DataFrame
filtered_data = data.loc[:, selected_columns]

# Collapse MultiIndex to single-level index
separator = '|'  # You can choose any separator you prefer
filtered_data.columns = [
    f"{measure}{separator}{params}" if params != 'no_param' else f"{measure}"
    for measure, params in filtered_data.columns
]

# Filter by stride -1
filtered_data = filtered_data.xs(-1, level='Stride', axis=0)

# ----------------------------
# Handling Missing Data
# ----------------------------

# Fill missing values with the mean of each feature
filtered_data_imputed = filtered_data.fillna(filtered_data.mean())

# ----------------------------
# Z-Scoring (Standardization)
# ----------------------------

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform
scaled_data = scaler.fit_transform(filtered_data_imputed)

# Convert the scaled data back to a DataFrame for convenience
scaled_data_df = pd.DataFrame(scaled_data, index=filtered_data_imputed.index, columns=filtered_data_imputed.columns)

# ----------------------------
# PCA
# ----------------------------

# Initialize PCA with desired number of components
pca = PCA(n_components=3)

# Fit PCA on the standardized data
pca.fit(scaled_data_df)

# Transform data for each condition
baseline_mask = scaled_data_df.index.get_level_values('Run').isin(expstuff['condition_exp_runs']['APAChar']['Extended']['Baseline'])
apa2_mask = scaled_data_df.index.get_level_values('Run').isin(expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
wash2_mask = scaled_data_df.index.get_level_values('Run').isin(expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])

# Extract data subsets
baseline_data = scaled_data_df[baseline_mask]
apa2_data = scaled_data_df[apa2_mask]
wash2_data = scaled_data_df[wash2_mask]

# Transform the data using the fitted PCA
Yd = dict()
Yd["Baseline"] = pca.transform(baseline_data)
Yd["APA_2ndHalf"] = pca.transform(apa2_data)
Yd["Wash_2ndHalf"] = pca.transform(wash2_data)

# ----------------------------
# Preparing Data for Plotting
# ----------------------------

# Create labels for each condition
labels_Baseline = np.array(['Baseline'] * Yd["Baseline"].shape[0])
labels_APA = np.array(['APA_2ndHalf'] * Yd["APA_2ndHalf"].shape[0])
labels_Wash = np.array(['Wash_2ndHalf'] * Yd["Wash_2ndHalf"].shape[0])

# Combine the transformed data
PCs = np.vstack([Yd["Baseline"], Yd["APA_2ndHalf"], Yd["Wash_2ndHalf"]])
labels = np.concatenate([labels_Baseline, labels_APA, labels_Wash])

# Create a DataFrame for easier plotting
df_plot = pd.DataFrame(PCs, columns=['PC1', 'PC2', 'PC3'])
df_plot['Condition'] = labels

# ----------------------------
# Plotting
# ----------------------------

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained_variance[0]*100:.2f}%")
print(f"Explained variance by PC2: {explained_variance[1]*100:.2f}%")
print(f"Explained variance by PC3: {explained_variance[2]*100:.2f}%")

# 2D Scatter Plot using Seaborn
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Condition', style='Condition', s=100, alpha=0.7)
plt.title(f'PCA: PC1 vs PC2 for Mouse {mouse_id}')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)')
plt.legend(title='Condition')
plt.grid(True)
plt.show()

# 3D Scatter Plot using Matplotlib
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Define color palette
palette = sns.color_palette("bright", len(df_plot['Condition'].unique()))
conditions = df_plot['Condition'].unique()

for idx, condition in enumerate(conditions):
    subset = df_plot[df_plot['Condition'] == condition]
    ax.scatter(
        subset['PC1'], subset['PC2'], subset['PC3'],
        label=condition, color=palette[idx], alpha=0.7, s=50
    )

ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)')
ax.set_zlabel(f'Principal Component 3 ({explained_variance[2]*100:.1f}%)')
ax.set_title(f'3D PCA for Mouse {mouse_id}')
ax.legend(title='Condition')
plt.show()

# ----------------------------
# Additional Visualizations
# ----------------------------

## **a. Scree Plot**
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()

## **b. PCA Loadings Plot**
# Access PCA loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, index=filtered_data_imputed.columns, columns=['PC1', 'PC2', 'PC3'])

# # Plot the loadings
# plt.figure(figsize=(12, 8))
# loadings_df.plot(kind='bar', figsize=(14, 8))
# plt.title('PCA Loadings')
# plt.xlabel('Features')
# plt.ylabel('Loading Scores')
# plt.legend(title='Principal Components')
# # add more space under the plot for the x-labels
# plt.show()

# ----------------------------
# LDA
# ----------------------------
# Prepare data for LDA
X_lda = np.vstack([Yd["APA_2ndHalf"], Yd["Wash_2ndHalf"]])
y_lda = np.concatenate([np.ones(Yd["APA_2ndHalf"].shape[0]), np.zeros(Yd["Wash_2ndHalf"].shape[0])])

# Initialize LDA with desired number of components
lda = LDA(n_components=1)

# Fit LDA on the PCA-transformed data
lda.fit(X_lda, y_lda)

# Transform the data using LDA
Y_lda = lda.transform(X_lda)

# Extract LDA coefficients (loadings)
# lda.coef_ has shape (n_classes - 1, n_features), here (1, 3)
lda_loadings = lda.coef_[0]

# ----------------------------
# Visualizing LDA Loadings
# ----------------------------

# Compute original feature contributions to LDA
original_feature_contributions = loadings_df.dot(lda_loadings)

# Create a DataFrame for plotting
feature_contributions_df = pd.DataFrame({
    'Feature': original_feature_contributions.index,
    'Contribution': original_feature_contributions.values
})

# Sort features by absolute contribution for better visualization
feature_contributions_df['Abs_Contribution'] = feature_contributions_df['Contribution'].abs()
feature_contributions_df.sort_values(by='Abs_Contribution', ascending=False, inplace=True)

# Plot Original Feature Contributions to LDA
plt.figure(figsize=(14, 10))
sns.barplot(
    data=feature_contributions_df,
    x='Contribution',
    y='Feature',
    palette='viridis'
)
plt.title('Original Feature Contributions to LDA Component')
plt.xlabel('Contribution to LDA')
plt.ylabel('Original Features')
# make y labels text smaller
plt.yticks(fontsize=6)
plt.axvline(0, color='red', linewidth=1)
plt.tight_layout()
plt.show()

# ----------------------------
# Visualizing LDA-transformed Data
# ----------------------------

# Create a DataFrame for LDA-transformed data
df_lda = pd.DataFrame(Y_lda, columns=['LDA_Component'])
df_lda['Condition'] = np.concatenate([['APA_2ndHalf'] * Yd["APA_2ndHalf"].shape[0],
                                      ['Wash_2ndHalf'] * Yd["Wash_2ndHalf"].shape[0]])

# Plot LDA-transformed data
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_lda, x='LDA_Component', y=[0]*df_lda.shape[0], hue='Condition', style='Condition', s=100, alpha=0.7)
plt.title('LDA Transformed Data for APA_2ndHalf vs Wash_2ndHalf')
plt.xlabel('LDA Component')
plt.yticks([])  # Hide y-axis as it's not informative in this context
plt.legend(title='Condition')
plt.grid(True)
plt.show()

# Optionally, add density plots or box plots to better visualize the separation
plt.figure(figsize=(10, 6))
sns.boxplot(x='Condition', y='LDA_Component', data=df_lda, palette='Set2')
plt.title('LDA Component Distribution by Condition')
plt.xlabel('Condition')
plt.ylabel('LDA Component')
plt.grid(True)
plt.show()









