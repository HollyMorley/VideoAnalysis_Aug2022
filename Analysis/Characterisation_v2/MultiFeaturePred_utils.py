import itertools
import os

import pandas as pd

from Helpers.Config_23 import *
from Analysis.Tools.config import global_settings, condition_specific_settings, instance_settings
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import DataClasses as dc
from Analysis.Characterisation_v2.Plotting import PCA_plotting as pcap
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg


def run_pca_regressions(phases, stride_numbers, condition, pca_data, feature_data, stride_data, save_dir):
    pca_predictions = []
    for p1, p2 in itertools.combinations(phases, 2):

        if not (len(pca_data) == 1 and pca_data[0].phase == (p1, p2)):
            raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")
        else:
            pca = pca_data[0].pca
            #pcs = pca_data[0].pcs
            pca_loadings = pca_data[0].pca_loadings

        for s in stride_numbers:
            print(f"Running single feature predictions for {p1}-{p2} on stride {s}...")
            for midx in condition_specific_settings[condition]['global_fs_mouse_ids']:
                save_path = gu.create_mouse_save_directory(save_dir, midx, s, p1, p2)

                # --------- Collect and organise PCA/feature data ---------
                # # Get mouse run data
                featsxruns, featsxruns_phaseruns, run_ns, stepping_limbs, mask_p1, mask_p2 = gu.select_runs_data(
                    midx, s, feature_data, stride_data, p1, p2)

                pcs = pca.transform(featsxruns)

                pcs_p1 = pcs[mask_p1]
                pcs_p2 = pcs[mask_p2]
                pcs_p1p2 = np.vstack([pcs_p1, pcs_p2])

                labels_phase1 = np.array([p1] * pcs_p1.shape[0])
                labels_phase2 = np.array([p2] * pcs_p2.shape[0])
                labels = np.concatenate([labels_phase1, labels_phase2])


                # --------- Plot PCA projections for mouse ---------
                pcap.plot_pca(pca, pcs_p1p2, labels, p1, p2, s, stepping_limbs, run_ns, midx,
                         condition, save_path)

                # ----------- Run Regression on PCA data and use to predict full run set for mouse -----------
                results = reg.run_regression_on_PCA_and_predict(pca_loadings, pcs_p1p2, featsxruns,
                                                                featsxruns_phaseruns,
                                                                mask_p1, mask_p2,
                                                                midx, p1, p2, s,
                                                                condition, save_path)
                (y_pred, smoothed_y_pred, feature_weights, w_PC, normalize_mean_pc, normalize_std_pc,
                 acc, cv_acc, w_folds, pc_acc, pc_y_preds, null_acc) = results

                # Skipping w/in vs b/wn mice comparison

                # --------- Find significance of PC predictions ---------

                # Save the regression results
                pca_pred_class = dc.PCAPredictionData(phase=(p1, p2),
                                                      stride=s,
                                                      mouse_id=midx,
                                                      x_vals= featsxruns.index,
                                                      y_pred=y_pred,
                                                      y_pred_smoothed=smoothed_y_pred,
                                                      feature_weights=feature_weights,
                                                      pc_weights=w_PC,
                                                      accuracy=acc,
                                                      cv_acc=cv_acc,
                                                      w_folds=w_folds,
                                                      y_preds_PCwise=pc_y_preds,
                                                      pc_acc=pc_acc,
                                                      null_acc=null_acc
                                                      # y_preds_PCwise=y_preds_PCwise,
                                                      # cv_acc_PCwise=cv_acc_PCwise,
                                                      # shuffle_acc=cv_acc_shuffle_PCwise,
                                                      # mean_acc_PCwise=mean_cv_acc_PCwise,
                                                      # mean_acc_shuffle_PCwise=mean_cv_acc_shuffle_PCwise,
                                                      )
                pca_predictions.append(pca_pred_class)
    return pca_predictions





















