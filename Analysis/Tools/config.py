import os
from Helpers.Config_23 import paths
"""
Description of global_settings:
- stride_numbers: List of stride numbers to filter data
- phases: List of phases to compare
- script_version: this just represents the name of the script
- allmice: if True, features will be selected based on data from all specified mice, if False, features will be selected based on data from each mouse separately
- method: regression (change within to switch knn, reg, lasso), rfecv, rf
- overwrite_FeatureSelection: if True, will overwrite existing feature selection results
- n_iterations_selection: number of iterations for recursive feature elimination
- nFolds_selection: number of folds for cross-validation
- mouse_pool_thresh: similarity threshold for pooling mice
- mouse_ids: list of mouse IDs to analyze
------------------------------------------------
- c (condition specific): inverse regularization strength for logistic regression 
- global_fs_mouse_ids (condition specific): list of mouse IDs to train the feature selection model
"""
############################################################################################################
# -------------------------------------------- To change: --------------------------------------------------
############################################################################################################

# ----------------------------- Save path: -----------------------------------
base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\Round22-9mice_descriptives--Lh-featselectPCA-LhHl-reg')
#base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'FeatureReduction\\test')


# ----------------------- Individual running settings: -----------------------
instance_settings = [
    {
        "condition": 'APAChar_LowHigh',
        "exp": 'Extended',
        "day": None,
        "compare_condition": 'APAChar_HighLow',
    },
    {
        "condition": 'APAChar_HighLow',
        "exp": 'Extended',
        "day": None,
        "compare_condition": 'APAChar_LowHigh',
    }
]

# --------------------------- Global settings: -------------------------------
global_settings = {
    "stride_numbers": [-3,-2,-1],
    "phases": ['APA2','Wash2'],
    "script_version": 'SingleCondition', # todo if combine MultiCondition and SingleCondition this will determine which of those is run
    "allmice": True,
    "method": 'rfecv',
    "cluster_method": "kmeans",
    "cluster_all_strides": True,
    "select_features": True,
    "pool_mice": False,
    "multi_strides": True,
    "overwrite_FeatureSelection": False,
    "overwrite_data_collection": True,
    "combine_stride_features": False,
    "plot_raw_features": False,
    # less frequently changed settings:
    "n_iterations_selection": 100,
    "nFolds_selection": 5,
    "mouse_pool_thresh": 0.85,
    "mouse_ids": ['1035243', '1035244', '1035245', '1035246',
                '1035249', '1035250', '1035297', '1035298',
                '1035299', '1035301', '1035302'],
}

# ----------------------- To mostly ignore: -----------------------
condition_specific_settings = {
    'APAChar_LowHigh': {
        'c': 1,
        'global_fs_mouse_ids': ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035298', '1035299', '1035301'], # and 298 again same day # removed 297 again 5/3/25 # just added back in 297 and 298 03/03/2025 # excluding 249 + 302 as missing data in APA2 and Wash2
    },
    'APAChar_HighLow': {
        'c': 0.5,
        'global_fs_mouse_ids': ['1035243', '1035244', '1035245', '1035246','1035250','1035301', '1035302'],
    },
}