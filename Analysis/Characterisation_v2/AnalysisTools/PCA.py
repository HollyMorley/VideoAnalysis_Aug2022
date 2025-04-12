import itertools
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import DataClasses as dc
from Analysis.Characterisation_v2.Plotting import PCA_plotting as pcap
from Analysis.Tools.config import condition_specific_settings, global_settings



def pca_main(feature_data, stride_data, phases, stride_numbers,
                    condition, base_save_dir_condition):
                    # feature_data_compare, global_stride_fs_results, phases, stride_numbers,
                    # condition, compare_condition, stride_data, stride_data_compare, base_save_dir_condition,
                    # select_feats=True, combine_conditions=False, combine_strides=False):
    pca_results = []
    for p1, p2 in itertools.combinations(phases, 2):
        pca, pcs, pca_loadings = compute_pca_allStrides(
            feature_data,
            condition_specific_settings[condition]['global_fs_mouse_ids'],
            stride_data, p1, p2, stride_numbers,
            n_components=global_settings["pcs_to_show"]
        )
        pca_class = dc.PCAData(phase=(p1,p2),
                               stride='all',
                               pca=pca,
                               pcs=pcs,
                               pca_loadings=pca_loadings)
        pca_results.append(pca_class)
        pcap.plot_scree(pca, p1, p2, "all_strides", condition, base_save_dir_condition)

        # todo average across mice, then re-run pca on averaged data and plot scree

    return pca_results

def compute_pca_allStrides(feature_data, mouseIDs_condition, stride_data, p1, p2, stride_numbers, n_components):
    aggregated_data = []
    data_mouse_pairs = [(feature_data, mouseIDs_condition, stride_data)]

    for data, mouseIDs, stride_d in data_mouse_pairs:
        for stride_number in stride_numbers:
            for mouse_id in mouseIDs:
                # Select runs data for the current mouse and stride number AND phases
                _, selected_data, _, _, _, _ = gu.select_runs_data(
                    mouse_id, stride_number, data, stride_d, p1, p2)
                aggregated_data.append(selected_data.T) # todo added .T as the function tranforms the data for some reason as we didnt here before

    global_data = pd.concat(aggregated_data)
    if n_components > global_data.shape[1]:
        n_components = global_data.shape[1]
    pca, pcs, loadings_df = perform_pca(global_data, n_components=n_components)
    return pca, pcs, loadings_df

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
