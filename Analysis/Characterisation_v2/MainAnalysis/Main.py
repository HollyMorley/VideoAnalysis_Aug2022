import os
import itertools
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List
from collections import defaultdict

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import SingleFeaturePred_utils as sfpu
from Analysis.Characterisation_v2 import MultiFeaturePred_utils as mfpu
from Analysis.Characterisation_v2.AnalysisTools import ClusterFeatures as cf
from Analysis.Characterisation_v2.AnalysisTools import PCA as pca
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg
from Analysis.Characterisation_v2.Plotting import ClusterFeatures_plotting as cfp
from Analysis.Characterisation_v2.Plotting import SingleFeaturePred_plotting as sfpp
from Analysis.Characterisation_v2.Plotting import PCA_plotting as pcap
from Analysis.Characterisation_v2.Plotting import Regression_plotting as regp

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'Characterisation\\LowHigh')

def main(mouse_ids: List[str], stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None):

    """
    Initialize experiment (data collection, directories, interpolation, logging). - NOT SCALED YET!!
    """
    print(f"Running {condition} analysis...")
    feature_data_notscaled, feature_data_compare_notscaled, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = gu.initialize_experiment(
        condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)

    # Paths
    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")
    preprocessed_data_file_path = os.path.join(base_save_dir, f"preprocessed_data_{condition}.pkl")
    SingleFeatPath = os.path.join(base_save_dir_condition, 'SingleFeaturePredictions')
    MultiFeatPath = os.path.join(base_save_dir_condition, 'MultiFeaturePredictions')
    os.makedirs(SingleFeatPath, exist_ok=True)
    os.makedirs(MultiFeatPath, exist_ok=True)

    print(f"Base save directory: {base_save_dir_condition}")
    # Skipping outlier removal
    if not os.path.exists(preprocessed_data_file_path):
        """
            # -------- Scale Data --------
            (Both conditions)
            - Z-score scale data for each mouse and stride
        """
        print("Scaling data...")
        feature_data = feature_data_notscaled.copy()
        feature_data.index.names = ['Stride', 'MouseID', 'Run']
        feature_data_compare = feature_data_compare_notscaled.copy()
        feature_data_compare.index.names = ['Stride', 'MouseID', 'Run']
        Normalize = {}
        idx = pd.IndexSlice

        feature_names = list(short_names.keys())
        # reorder feature_data by feature_names
        feature_data = feature_data.reindex(columns=feature_names)
        feature_data_compare = feature_data_compare.reindex(columns=feature_names)

        for (stride, mouse_id), data in feature_data.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = gu.normalize_df(data)
            feature_data.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize[(stride, mouse_id)] = norm_df
        Normalize_compare = {}
        for (stride, mouse_id), data in feature_data_compare.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = gu.normalize_df(data)
            feature_data_compare.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize_compare[(stride, mouse_id)] = norm_df

        # Get average feature values for each feature in each stride (across mice)
        feature_data_average = feature_data.groupby(level=['Stride', 'Run']).median()
        feature_data_compare_average = feature_data_compare.groupby(level=['Stride', 'Run']).median()

        """
            # -------- Feature Clusters --------
            (Main condition)
            For all strides together, plot correlations between features, organised by manual cluster. 
            Save manual clusters for later access.
        """
        print("Clustering features...")
        cluster_mappings = {}
        for p1, p2 in itertools.combinations(phases, 2):
            # features x runs
            feature_matrix = cf.get_global_feature_matrix(feature_data,
                                                       condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                       'all', stride_data, p1, p2, smooth=False)
            cfp.plot_corr_matrix_sorted_manually(feature_matrix, base_save_dir_condition,
                                             f'CorrMatrix_manualclustering_{p1}-{p2}_all')
            for s in stride_numbers:
                cluster_mappings[(p1, p2, s)] = manual_clusters['cluster_mapping']

        # save everything so far to file
        with open(preprocessed_data_file_path, 'wb') as f:
            pickle.dump({
                'feature_names': feature_names,
                'feature_data': feature_data,
                'feature_data_compare': feature_data_compare,
                'feature_data_average': feature_data_average,
                'feature_data_compare_average': feature_data_compare_average,
                'stride_data': stride_data,
                'stride_data_compare': stride_data_compare,
                'Normalize': Normalize,
                'Normalize_compare': Normalize_compare,
                'cluster_mappings': cluster_mappings
            }, f)
    else:
        with open(preprocessed_data_file_path, 'rb') as f:
            data = pickle.load(f)
            feature_names = data['feature_names']
            feature_data = data['feature_data']
            feature_data_compare = data['feature_data_compare']
            feature_data_average = data['feature_data_average']
            feature_data_compare_average = data['feature_data_compare_average']
            stride_data = data['stride_data']
            stride_data_compare = data['stride_data_compare']
            Normalize = data['Normalize']
            Normalize_compare = data['Normalize_compare']
            cluster_mappings = data['cluster_mappings']

    """
    ------------------ Single feature predictions ------------------
    """
    # Get predictions for each phase combo, mouse, stride and feature
    filename = f'single_feature_predictions_{condition}.pkl'
    if os.path.exists(os.path.join(SingleFeatPath, filename)):
        print("Loading single feature predictions from file...")
        with open(os.path.join(SingleFeatPath, filename), 'rb') as f:
            single_feature_predictions = pickle.load(f)
    else:
        print("Running single feature predictions...")
        single_feature_predictions = sfpu.run_single_feature_regressions(phases, stride_numbers, condition, feature_names,
                                                                   feature_data, stride_data, SingleFeatPath,
                                                                   filename)

    # Get summary/average predictions for each phase combo, stride and feature, and find the top x features for each phase combo and stride
    filename_summary = f'single_feature_predictions_summary_{condition}.pkl'
    filename_top_feats = f'top_features_{condition}.pkl'

    if os.path.exists(os.path.join(SingleFeatPath, filename_summary)):
        print("Loading single feature predictions summary from file...")
        with open(os.path.join(SingleFeatPath, filename_summary), 'rb') as f:
            single_feature_predictions_summary = pickle.load(f)
        with open(os.path.join(SingleFeatPath, filename_top_feats), 'rb') as f:
            top_feats = pickle.load(f)
    else:
        print("Running single feature predictions summary...")
        single_feature_predictions_summary, top_feats = sfpu.get_summary_and_top_single_feature_data(10,
                                                                                                     phases, stride_numbers,
                                                                                                     feature_names,
                                                                                                     single_feature_predictions)

        print("Plotting single feature predictions...")
        sfpp.plot_featureXruns_heatmap(phases, stride_numbers, feature_names,
                                       single_feature_predictions_summary, 'RunPreds', SingleFeatPath)
        sfpp.plot_featureXruns_heatmap(phases, stride_numbers, feature_names,
                                       feature_data_average, 'RawFeats', SingleFeatPath)

        # Save the top features for each phase combo and stride
        with open(os.path.join(SingleFeatPath, filename_top_feats), 'wb') as f:
            pickle.dump(top_feats, f)
        # Save the single feature predictions
        with open(os.path.join(SingleFeatPath, filename_summary), 'wb') as f:
            pickle.dump(single_feature_predictions_summary, f)

    """
    ------------------------- PCA ----------------------
    """
    filename_pca = f'pca_{condition}.pkl'
    if os.path.exists(os.path.join(MultiFeatPath, filename_pca)):
        print("Loading PCA from file...")
        with open(os.path.join(MultiFeatPath, filename_pca), 'rb') as f:
            pca_all = pickle.load(f)
    else:
        print("Running PCA...")
        pca_all = pca.pca_main(feature_data, stride_data, phases, stride_numbers, condition, MultiFeatPath)

        # Plot how each feature loads onto the PCA components
        pcap.pca_plot_feature_loadings(pca_all, phases, MultiFeatPath)
        pcap.plot_top_features_per_PC(pca_all, feature_data, phases, stride_numbers, MultiFeatPath,
                                      n_top_features=5)

        # Save PCA results
        with open(os.path.join(MultiFeatPath, filename_pca), 'wb') as f:
            pickle.dump(pca_all, f)


    """
    -------------------- PCA/Multi Feature Predictions ----------------------
    """
    filename_pca_pred = f'pca_predictions_{condition}.pkl'
    if os.path.exists(os.path.join(MultiFeatPath, filename_pca_pred)):
        print("Loading PCA predictions from file...")
        with open(os.path.join(MultiFeatPath, filename_pca_pred), 'rb') as f:
            pca_pred = pickle.load(f)
    else:
        pca_pred = mfpu.run_pca_regressions(phases, stride_numbers, condition, pca_all, feature_data, stride_data, MultiFeatPath)
        # Save PCA predictions
        with open(os.path.join(MultiFeatPath, filename_pca_pred), 'wb') as f:
            pickle.dump(pca_pred, f)

    # Plot PCA predictions as heatmap
    regp.plot_PCA_pred_heatmap(pca_all, pca_pred, feature_data, stride_data, phases, stride_numbers,condition, MultiFeatPath)

    for s in stride_numbers:
        reg.calculate_PC_prediction_significances(pca_pred, s)






    print('Done')



if __name__ == "__main__":
    # add flattened LowHigh etc settings to global_settings for log
    global_settings["LowHigh_c"] = condition_specific_settings['APAChar_LowHigh']['c']
    global_settings["HighLow_c"] = condition_specific_settings['APAChar_HighLow']['c']
    global_settings["LowHigh_mice"] = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
    global_settings["HighLow_mice"] = condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']

    # Combine the settings in a single dict to log.
    settings_to_log = {
        "global_settings": global_settings,
        "instance_settings": instance_settings
    }

    # Run each instance.
    for inst in instance_settings:
        main(
            global_settings["mouse_ids"],
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log
        )
