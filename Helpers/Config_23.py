import numpy as np
########################################################################################################################
################################################### User-set values ####################################################
########################################################################################################################

vidstuff = {
    'cams': ['side','front','overhead'],
    'scorers': {
        'side': "DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000",
        'side_new': "DLC_resnet50_DLC_DualBeltJul25shuffle1_1020000",
        'overhead': "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000",
        'front': "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000"
    }
}
pcutoff = 0.9
fps = 247
expstuff = {
    'exp_chunks': {
        'RunStages': ['TrialStart', 'RunStart', 'Transition', 'RunEnd', 'TrialEnd'],
        'ExpPhases': ['Baseline', 'APA', 'Washout']
    },
    'condition_exp_lengths': {
        'APACharRuns': [10,20,10],
        'APAPerRuns': [10,20,10], #### warning if change i have sometimes used the apachar variable to define perception runs too
        'APAVmtRuns': [10,15,10]
    },
    'setup': {
        'distancescm': [13,12,11.5,11.5] # goes 0: wall1-wall0,  1: wall2-wall1, 3: wall3-wall2, 4: wall4-wall3
    },
    'speeds': {'Low': 6, 'Mid': 18, 'High': 30},
    'preprun_nos': {
        'preruns_CharLow': 2,
        'preruns_CharMidHigh': 5
    }
}
structural_stuff = {
    'belt_width': 52, # mm
    'belt_length_sideviewrange': 600, # mm
    'belt_length_sideviewend': 130 # mm
}
locostuff = {
    'swst_vals': {
        'st': 0,
        'sw': 1,
        'st_bkwd': 2,
        'sw_bkwd': 3
    }
}
settings = {
    'analysis_chunks': {
        'APA_lengthruns': 25, #100#25
        'after_lengthruns': 25,
        'literature_apa_length': 100 #ms ## this isnt acutally correct, check this
    }
}
# paths = {
#         'filtereddata_folder': r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round2_Jan23",
#         'plotting_destfolder': r'M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\Nov23'
# }
paths = {
        'filtereddata_folder': r'H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round2_Jan23',
        'plotting_destfolder': r'H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\Nov23'
}

micestuff = {
    'mice_ID': [
        '1034976',
        '1034978',
        '1034979',
        '1034980',
        '1034982',
        '1034983'
    ],
    'mice_name': [
        'MNone',
        'MR',
        'MLR',
        'MNone',
        'MR',
        'MLR'
    ],
    'LR': {
        'ForepawToeL': 1,
        'ForepawToeR': 2
    }
}

label_list = {
    'sideXfront': ['ForepawToeR', 'ForepawAnkleR', 'HindpawToeR', 'HindpawAnkleR','ForepawToeL', 'ForepawAnkleL',
                   'HindpawToeL', 'HindpawAnkleL'],
    'sideXoverhead': ['Nose', 'EarL', 'EarR', 'Back1', 'Back2', 'Back3','Back4', 'Back5', 'Back6', 'Back7', 'Back8',
                      'Back9', 'Back10','Back11', 'Back12', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5','Tail6',
                      'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12', 'StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL']
}

measure_list = {
    'limb': ['walking_speed','stride_duration','cadence','swing_velocity','stride_length','stance_duration','duty_factor','coo_x','coo_y'], ## for different plot add swing z trajectory & instantaneous swing vel,
    'interlimb': ['bos_ref_stance', 'double_support'], # stance phase, supports, 'bos_hom_stance', 'tail1_displacement'
    'whole_body': ['tail1_ptp_amplitude_stride',
                   'tail1_speed',
                   'body_length_stance', 'body_length_swing',
                   'back_skew_stance', 'back_skew_swing',
                   'neck_z_stance', 'neck_z_swing',
                   'midback_z_stance', 'midback_z_swing',
                   'tail1_z_stance', 'tail1_z_swing',
                   'stepping_limb_z_stance', 'stepping_limb_z_swing',
                   'contra_limb_z_stance', 'contra_limb_z_swing',
                   'neck_x_displacement_stance', 'neck_x_displacement_swing',
                   'midback_x_displacement_stance', 'midback_x_displacement_swing',
                   'tail1_x_displacement_stance', 'tail1_x_displacement_swing',
                   'stepping_limb_x_displacement_stance', 'stepping_limb_x_displacement_swing',
                   'contra_limb_x_displacement_stance', 'contra_limb_x_displacement_swing',
                   'stepping_limb_x_displacement_stance', 'stepping_limb_x_displacement_swing',
                   'contra_limb_x_displacement_stance', 'contra_limb_x_displacement_swing'
                   'head_tilt_stance', 'head_tilt_swing',
                   'body_tilt_stance', 'body_tilt_swing',
                   'tail_tilt_stance', 'tail_tilt_swing',
                   'limb_rel_to_body_stance'], #, 'back_curvature_stance', 'back_curvature_swing'], # x/y of paws during swing and stance, body sway (overhead), back curvature, , 'nose_ptp_amplitude_stride'
    #'behavioural': ['wait_time','no_rbs','transitioning_limb']
}
#
# measure_list = {
#      'limb': ['walking_speed','stride_duration'] #,'cadence','swing_velocity','stride_length','stance_duration','duty_factor','coo_x','coo_y'], ## for different plot add swing z trajectory & instantaneous swing vel,
# #     'interlimb': ['bos_ref_stance', 'double_support'], # stance phase, supports, 'bos_hom_stance', 'tail1_displacement'
# #     'whole_body': ['tail1_ptp_amplitude_stride', 'tail1_speed', 'body_length_stance', 'body_length_swing'] #,'back_skew_stance', 'back_skew_swing','neck_height_stance', 'neck_height_swing', 'midback_height_stance', 'midback_height_swing', 'tail1_height_stance', 'tail1_height_swing','head_tilt_stance', 'head_tilt_swing', 'body_tilt_stance', 'body_tilt_swing', 'tail_angle_stance', 'tail_angle_swing'], # x/y of paws during swing and stance, body sway (overhead), back curvature, , 'nose_ptp_amplitude_stride'
# #     #'behavioural': ['wait_time','no_rbs','transitioning_limb']
# }
def measures_list(buffer): ## add in displacement??
    measures = {
        'multi_val_measure_list' : {
            'instantaneous_swing_velocity': {
                'speed_correct': [True,False],
                'xyz': ['x','y','z'],
                'smooth': [False]
            },
            'x': {
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'speed_correct': [True,False],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'y': {
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'z' : {
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'traj': {
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'coord': ['x','y','z'],
                'step_phase': [None],
                'full_stride': [True],
                'speed_correct': [True, False],
                'aligned': [True], # False
                'buffer_size': [0, buffer]
            },
            'body_distance': {
                'bodyparts': [['Back1','Back12']],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                       'Back11', 'Back12'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'back_skew': {
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'limb_rel_to_body': {
                'time': [None],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
             },
            'signed_angle': {
                'ToeAnkleL_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawToeL','ForepawAnkleL'], buffer],
                'ToeAnkleL_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawToeL', 'ForepawAnkleL'], 0],
                'ToeAnkleR_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawToeR', 'ForepawAnkleR'],buffer],
                'ToeAnkleR_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]),['ForepawToeR', 'ForepawAnkleR'], 0],
                'Back1Back12_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],buffer],
                'Back1Back12_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],0],
                'Tail1Tail12_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],buffer],
                'Tail1Tail12_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],0],
                'NoseBack1_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],buffer],
                'NoseBack1_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],0],
                'Back1Back12_overhead_xref_buff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],buffer],
                'Back1Back12_overhead_xref_nobuff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],0],
                'Tail1Tail12_overhead_xref_buff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'], buffer],
                'Tail1Tail12_overhead_xref_nobuff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'], 0],
                'NoseBack1_overhead_xref_buff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'], buffer],
                'NoseBack1_overhead_xref_nobuff': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'], 0],
            }
        },
        'single_val_measure_list': {
            'stride_duration': [],
            'stance_duration': [],
            'swing_duration': [],
            'cadence': [],
            'duty_factor': [],
            'walking_speed': {
                'bodypart': ['Back6','Tail1'],
                'speed_correct': [True,False]
            },
            'swing_velocity': {
                'speed_correct': [True,False]
            },
            'stride_length': {
                'speed_correct': [True,False]
            },
            'x': {
                # this is displacement
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'speed_correct': [True,False],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'y': {
                # this is displacement
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'z': {
                # this is displacement
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'coo_xyz': {
                'xyz': ['x','y','z']
            },
            'coo_euclidean': [],
            'bos_stancestart': {
                'ref_or_contr': ['ref','contr'],
                'y_or_euc': ['y','euc']
            },
            'ptp_amplitude_stride': {
                'bodypart': ['Tail1','Back6']
            },
            'body_distance': {
                'bodyparts': [['Back1','Back12']],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'double_support': [],
            'back_skew': {
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            },
            'limb_rel_to_body': {
                'time': ['start','end'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, 0.25]
            }
        }
    }
    return measures
