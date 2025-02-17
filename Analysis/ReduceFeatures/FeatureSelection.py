import numpy as np
import pandas as pd
import os
import random
import ast
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def rfe_feature_selection(selected_scaled_data_df, y, cv=5, min_features_to_select=5, C=1.0):
    """
    Performs feature selection using RFECV with L1-regularized logistic regression.
    Parameters:
      - selected_scaled_data_df: DataFrame with features as rows and samples as columns.
      - y: target vector.
      - cv: number of folds for cross-validation.
      - min_features_to_select: minimum number of features RFECV is allowed to select.
      - C: Inverse regularization strength (higher values reduce regularization).
    """
    # Transpose so that rows are samples and columns are features.
    X = selected_scaled_data_df.T
    estimator = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=C)
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='balanced_accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X, y)
    selected_features = selected_scaled_data_df.index[rfecv.support_]
    print(f"RFECV selected {rfecv.n_features_} features.")
    return selected_features, rfecv

def random_forest_feature_selection(selected_scaled_data_df, y):
    """
    Performs feature selection using a Random Forest to rank features and selects those
    with importance above the median.
    """
    X = selected_scaled_data_df.T  # rows: samples, columns: features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    threshold = np.median(importances)
    selected_features = selected_scaled_data_df.index[importances > threshold]
    print(f"Random Forest selected {len(selected_features)} features (threshold: {threshold:.4f}).")
    return selected_features, rf
