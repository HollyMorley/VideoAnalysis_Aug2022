import pandas as pd
import os
import glob
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    hamming_loss,
    confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib


def load_and_preprocess_data(data_path, label_columns):
    data = pd.read_csv(data_path)

    # Prepare feature columns (exclude label columns and non-feature columns)
    non_feature_columns = ['Filename', 'Frame'] + label_columns
    feature_columns = [col for col in data.columns if col not in non_feature_columns]

    # Keep 'Filename' and 'Frame' in X for later use
    X = data[feature_columns + ['Filename', 'Frame']]
    y = data[label_columns]

    # Remove rows with missing labels
    missing_labels = y.isnull().any(axis=1)
    if missing_labels.any():
        print(f"Removing {missing_labels.sum()} rows with missing labels.")
        X = X[~missing_labels]
        y = y[~missing_labels]

    # Encode labels if necessary
    label_encoders = {}
    for col in label_columns:
        if y[col].dtype == 'object' or y[col].dtype.name == 'category':
            label_enc = LabelEncoder()
            y[col] = label_enc.fit_transform(y[col].astype(str))
            label_encoders[col] = label_enc
            class_names = label_enc.classes_
            print(f"Classes for {col}:", class_names)

    # Reset indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return X, y, feature_columns, label_encoders


def train_model(X_train_features, y_train, output_dir, model_type='lightgbm'):
    if model_type == 'xgboost':
        # Define the base estimator for XGBoost
        base_estimator = XGBClassifier(
            objective='binary:logistic',
            booster='gbtree',
            learning_rate=0.05,
            n_estimators=100,
            max_depth=6,
            verbosity=1,
            use_label_encoder=False
        )
    else:
        # Default to LightGBM
        base_estimator = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            verbose=-1,
            is_unbalance=True  # Adjust if classes are imbalanced
        )

    # Create the multi-output classifier
    multi_target_model = MultiOutputClassifier(base_estimator, n_jobs=-1)

    # Train the model
    multi_target_model.fit(X_train_features, y_train)

    # Save the model to disk
    model_filename = f'limb_classification_model.pkl'
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(multi_target_model, model_path)
    print(f"Model saved as {model_filename}")

    return multi_target_model


def plot_misclassified_sample(base_directory, filename, frame_num, true_labels, predicted_labels, output_dir):
    # Construct the image path
    subdir = filename.replace('_mapped3D.h5', '')
    image_filename_png = f'img{frame_num}.png'
    image_path_pattern = os.path.join(base_directory, subdir + '*')
    matching_dirs = glob.glob(image_path_pattern)

    if matching_dirs:
        image_dir = matching_dirs[0]
        image_path = os.path.join(image_dir, image_filename_png)
    else:
        print(f"Image directory not found for filename {filename}")
        return

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load the image
    image = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    plt.axis('off')

    # Prepare the label text (convert numeric labels to 'Stance' or 'Swing')
    label_mapping = {1: 'Stance', 0: 'Swing'}
    true_label_text = '\n'.join([f'{limb}: {label_mapping.get(label, label)}' for limb, label in true_labels.items()])
    predicted_label_text = '\n'.join([f'{limb}: {label_mapping.get(label, label)}' for limb, label in predicted_labels.items()])

    # Add text annotations outside the image
    plt.gcf().text(0.05, 0.95, f"True Labels:\n{true_label_text}", fontsize=10, color='green', verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))
    plt.gcf().text(0.05, 0.75, f"Predicted Labels:\n{predicted_label_text}", fontsize=10, color='red', verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Misclassified Sample\nFilename: {filename}, Frame: {frame_num}")

    # Save the plot to a file
    os.makedirs(output_dir + "\Misclassified_Plots", exist_ok=True)
    plot_filename = f"{filename}_frame{frame_num}_misclassified.png"
    plot_path = os.path.join(output_dir + "\Misclassified_Plots", plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

def analyze_misclassifications(X_test, y_test, y_pred, label_columns, base_directory, output_dir):
    for idx, col in enumerate(label_columns):
        print(f"\nAnalyzing misclassifications for {col}:")

        # Compare y_test values with predictions directly
        misclassified = y_test[col].values != y_pred[:, idx]
        num_misclassified = misclassified.sum()
        print(f'Number of misclassified samples: {num_misclassified}')

        if num_misclassified > 0:
            # Extract misclassified samples
            X_misclassified = X_test.iloc[misclassified].reset_index(drop=True)
            y_true_misclassified = y_test.iloc[misclassified].reset_index(drop=True)
            y_pred_misclassified = pd.Series(y_pred[misclassified, idx], index=y_true_misclassified.index)

            # For each misclassified sample, plot and save the image
            for i in range(len(X_misclassified)):
                filename = X_misclassified.loc[i, 'Filename']
                frame_num = X_misclassified.loc[i, 'Frame']
                true_label = y_true_misclassified[col].iloc[i]
                predicted_label = y_pred_misclassified.iloc[i]

                # Prepare labels as dictionaries
                true_labels = {col: true_label}
                predicted_labels = {col: predicted_label}

                # Plot and save the misclassified sample
                plot_misclassified_sample(base_directory, filename, frame_num, true_labels, predicted_labels,
                                          output_dir)


def plot_feature_importance(model, feature_columns, output_dir):
    # Get feature importance from the trained model
    feature_importance = model.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(20, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))  # Plot top 20 features
    plt.title('Top 20 Feature Importance')
    #reduce y tick text size
    plt.yticks(fontsize=8)

    # save the plot
    plot_filename = 'feature_importance.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


def main():
    # Define data path and label columns
    data_path = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff\extracted_features.csv"
    output_dir = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff"
    label_columns = ['HindpawL', 'ForepawL', 'HindpawR', 'ForepawR']

    # Load and preprocess data
    X, y, feature_columns, label_encoders = load_and_preprocess_data(data_path, label_columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reset indices
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Prepare feature matrices for training and testing
    X_train_features = X_train[feature_columns]
    X_test_features = X_test[feature_columns]

    # Train the model
    multi_target_model = train_model(X_train_features, y_train, output_dir, model_type='xgboost')

    # Predict on test data
    y_pred = multi_target_model.predict(X_test_features)

    # Evaluate the model
    print("Model Evaluation:")
    for idx, col in enumerate(label_columns):
        print(f"\nEvaluation for {col}:")
        accuracy = accuracy_score(y_test[col], y_pred[:, idx])
        print(f'Accuracy: {accuracy:.4f}')
        print('Classification Report:')
        print(classification_report(y_test[col], y_pred[:, idx]))
        print('-' * 50)

    # Calculate overall hamming loss
    hloss = hamming_loss(y_test, y_pred)
    print(f'Overall Hamming Loss: {hloss:.4f}')

    # Optionally, calculate subset accuracy (exact match ratio)
    subset_accuracy = (y_pred == y_test.values).all(axis=1).mean()
    print(f'Subset Accuracy (Exact Match Ratio): {subset_accuracy:.4f}')

    # Analyze misclassifications and save images
    base_directory = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side"
    analyze_misclassifications(X_test, y_test, y_pred, label_columns, base_directory, output_dir)
    plot_feature_importance(multi_target_model.estimators_[0], feature_columns, output_dir)

    plt.show()


if __name__ == "__main__":
    main()
