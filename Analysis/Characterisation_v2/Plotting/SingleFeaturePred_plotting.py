import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import os
from scipy.signal import medfilt

from Helpers.Config_23 import *

from Analysis.Characterisation_v2 import Plotting_utils as pu


def plot_featureXruns_heatmap(phases, stride_numbers, feature_names, data, data_name, save_path):
    p1, p2 = phases
    display_feat_names = [short_names[f] for f in feature_names]
    for s in stride_numbers:
        if data_name == 'RunPreds':
            feats_ypred, features = zip(*[(pred.y_pred, pred.feature)
                                          for pred in data
                                          if pred.phase == (p1, p2) and pred.stride == s])
            feats_dict = {feature: y_pred for feature, y_pred in zip(features, feats_ypred)}
            feats_df = pd.DataFrame(feats_dict, index=np.arange(0, 160))

            assert feats_df.columns.tolist() == feature_names, "Feature names do not match!"

            heatmap_data = feats_df.T

            filename = f"FeatureXruns_heatmap_RunPrediction"

            heatmap_runpred_or_rawfeat(p1, p2, s, feature_names, display_feat_names,
                                        heatmap_data, save_path, filename, cbar_lim=1)
        elif data_name == 'RawFeats':
            stride_raw_feats = data.loc(axis=0)[s]
            assert stride_raw_feats.columns.tolist() == feature_names, "Feature names do not match!"
            heatmap_data = stride_raw_feats.T

            filename = f"FeatureXruns_heatmap_RawFeats"

            heatmap_runpred_or_rawfeat(p1, p2, s, feature_names, display_feat_names,
                                      heatmap_data, save_path, filename, cbar_lim=1.25)




def heatmap_runpred_or_rawfeat(p1, p2, s, feat_names, display_feat_names, heatmap_data, save_path, filename, cbar_lim):
        # smooth with medfilt
        smooth = heatmap_data.apply(lambda x: medfilt(x, kernel_size=3), axis=1)
        # Convert the Series of NumPy arrays into a DataFrame:
        smooth_df = pd.DataFrame(smooth.tolist(), index=heatmap_data.index, columns=heatmap_data.columns)

        fig, ax = plt.subplots(figsize=(12, 8))
        h = sns.heatmap(smooth_df, cmap='coolwarm', cbar=True, yticklabels=display_feat_names,
                        vmin=-1*cbar_lim, vmax=cbar_lim, cbar_kws={'label': 'Prediction', 'orientation': 'vertical',
                                                   'shrink': 0.8})

        # Change x-axis tick labels
        ax.set_xticks(np.arange(0, 161, 10))
        ax.set_xticklabels(np.arange(0, 161, 10))
        ax.tick_params(axis='both', which='major', labelsize=7)

        # Change the colorbar tick labels font size
        cbar = h.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        # Optionally, change the colorbar label font size as well:
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=7)

        plt.axvline(x=10, color='black', linestyle='--')
        plt.axvline(x=110, color='black', linestyle='--')
        plt.ylabel('')
        plt.xlabel('Trial')

        # shift the plot to the right
        plt.subplots_adjust(left=0.3, right=0.99, top=0.95, bottom=0.15)

        pu.add_cluster_brackets_heatmap(manual_clusters, feat_names, ax,
                                        horizontal=False, vertical=True,
                                        base_line_num = 5, label_offset=5, fs=7,
                                        distance_from_plot=-60)

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{filename}_{p1}-{p2}_stride{s}.png'),
                    dpi=300, format='png')
        plt.savefig(os.path.join(save_path, f'{filename}_{p1}-{p2}_stride{s}.svg'),
                    dpi=300, format='svg')
        plt.close(fig)