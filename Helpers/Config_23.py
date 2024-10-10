import numpy as np
########################################################################################################################
################################################### User-set values ####################################################
########################################################################################################################

vidstuff = {
    'cams': ['side','front','overhead'],
    'scorers': {
        'side': "DLC_resnet50_DLC_DualBeltAug2shuffle1_1000000",
        'overhead': "DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000",
        'front': "DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000"
    }
}

# vidstuff = {
#     'cams': ['side','front','overhead'],
#     'scorers': {
#         'side': "DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000",
#         'side_new': "DLC_resnet50_DLC_DualBeltJul25shuffle1_1020000",
#         'overhead': "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000",
#         'front': "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000"
#     }
# }
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
    'condition_exp_runs': {
        'APACharRuns': {
            'Short': {
                'Baseline': np.arange(1, 11),
                'APA1': np.arange(11, 21),
                'APA2': np.arange(21, 31),
                'Washout': np.arange(31, 41)
            },
            'Extended': []
        }
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

exp_cats = {
    'APAChar_LowHigh': {
        'Repeats': {
            'Wash': {
                'Exp': {
                    'Day1': {'A': ['20230306'], 'B': ['20230309']},
                    'Day2': {'A': ['20230308'], 'B': ['20230312']},
                    'Day3': {'A': ['20230310'], 'B': ['20230314']},
                },
                'Washout': {
                    'Day1': {'A': ['20230307'], 'B': ['20230310']},
                    'Day2': {'A': ['20230309'], 'B': ['20230313']},
                    'Day3': {'A': ['20230312'], 'B': ['20230315']},
                }
            }
        },
        'Extended': {
            'Day1': {'A': ['20230316'], 'B': ['20230403']},
            'Day2': {'A': ['20230317'], 'B': ['20230404']},
            'Day3': {'A': ['20230318'], 'B': ['20230405']},
            'Day4': {'A': ['20230319'], 'B': ['20230406']}
        }
    },
    'APAChar_HighLow': {
        'Extended': {
            'Day1': {'A': ['20230325'], 'B': ['20230412']},
            'Day2': {'A': ['20230326'], 'B': ['20230413']},
            'Day3': {'A': ['20230327'], 'B': ['20230414']},
            'Day4': {'A': ['20230328'], 'B': ['20230415']}
        }
    },
    'APAChar_LowMid': {
        'Extended': {
            'Day1': {'A': ['20230407'], 'B': ['20230320']},
            'Day2': {'A': ['20230408'], 'B': ['20230321']},
            'Day3': {'A': ['20230409'], 'B': ['20230322']},
            'Day4': {'A': ['20230410'], 'B': ['20230323']}
        }
    },
    'PerceptionTest': {'A': ['20230411'], 'B': ['20230416']},
    'VMT_LowHigh': {
        'pd': {
            'Basic': {
                'Prep': {'A': ['20230420'], 'B': ['20230427']},
                'Experiment': {'A': ['20230421'], 'B': ['20230428']}
            },
            'Extreme': {
                'Prep': {'A': ['20230513'], 'B': ['20230514']},
                'Experiment': {'A': ['20230514'], 'B': ['20230515']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': ['20230426'], 'B': ['20230503']},
                'Experiment': {'A': ['20230427'], 'B': ['20230504']}
            },
            'Extreme': {
                'Prep': {'A': ['20230515'], 'B': ['20230516']},
                'Experiment': {'A': ['20230516'], 'B': ['20230517']}
            }
        }
    },
    'VMT_HighLow': {
        'pd': {
            'Basic': {
                'Prep': {'A': ['20230504'], 'B': ['20230421']},
                'Experiment': {'A': ['20230505'], 'B': ['20230422']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': ['20230428'], 'B': ['20230505']},
                'Experiment': {'A': ['20230429'], 'B': ['20230506']}
            }
        }
    }
}

structural_stuff = {
    'belt_width': 53.5, # mm
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
        'filtereddata_folder': r'H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24',
        'plotting_destfolder': r'H:\Dual-belt_APAs\Plots\Mar24',
        'video_folder': r"X:\hmorley\Dual-belt_APAs\videos\Round_3",
        'data_folder': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles" #todo maybe change this to Margrie ceph dir
}

micestuff = {
    'mice_IDs': {'A': ['1035243', '1035244', '1035245', '1035246', '1035249', '1035250'],
            'B': ['1035297', '1035298', '1035299', '1035301', '1035302']
            },
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
        'ForepawToeR': 2,
    },
    'skeleton':[
        ('Nose', 'Back1'), ('EarL', 'Back1'), ('EarR', 'Back1'),
        ('Back1', 'Back2'), ('Back2', 'Back3'), ('Back3', 'Back4'), ('Back4', 'Back5'), ('Back5', 'Back6'),
        ('Back6', 'Back7'), ('Back7', 'Back8'), ('Back8', 'Back9'), ('Back9', 'Back10'), ('Back10', 'Back11'),
        ('Back11', 'Back12'), ('Back12', 'Tail1'),
        ('Tail1', 'Tail2'), ('Tail2', 'Tail3'), ('Tail3', 'Tail4'), ('Tail4', 'Tail5'), ('Tail5', 'Tail6'),
        ('Tail6', 'Tail7'), ('Tail7', 'Tail8'), ('Tail8', 'Tail9'), ('Tail9', 'Tail10'), ('Tail10', 'Tail11'),
        ('Tail11', 'Tail12'),
        ('Back3', 'ForepawKneeL'), ('Back3', 'ForepawKneeR'), ('Back9', 'HindpawKneeL'),('Back9', 'HindpawKneeR'),
        ('ForepawKneeL', 'ForepawAnkleL'), ('ForepawKneeR', 'ForepawAnkleR'),
        ('HindpawKneeL', 'HindpawAnkleL'), ('HindpawKneeR', 'HindpawAnkleR'),
        ('ForepawAnkleL', 'ForepawKnuckleL'), ('ForepawAnkleR', 'ForepawKnuckleR'),
        ('HindpawAnkleL', 'HindpawKnuckleL'), ('HindpawAnkleR', 'HindpawKnuckleR'),
        ('ForepawKnuckleL', 'ForepawToeL'), ('ForepawKnuckleR', 'ForepawToeR'),
        ('HindpawKnuckleL', 'HindpawToeL'), ('HindpawKnuckleR', 'HindpawToeR')
        ]
}

label_list_World = ['Door','StartPlatL','StartPlatR','TransitionL','TransitionR', 'StepR', 'StepL',
                    'Nose','EarL','EarR',
                    'Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12',
                    'Tail1','Tail2','Tail3','Tail4','Tail5','Tail6','Tail7','Tail8','Tail9','Tail10','Tail11','Tail12',
                    'ForepawToeL','ForepawToeR','ForepawKnuckleL','ForepawKnuckleR','ForepawAnkleL','ForepawAnkleR','ForepawKneeL','ForepawKneeR',
                    'HindpawToeL','HindpawToeR', 'HindpawKnuckleL','HindpawKnuckleR','HindpawAnkleL','HindpawAnkleR','HindpawKneeL','HindpawKneeR']

label_list = {
    'sideXfront': ['ForepawToeR', 'ForepawAnkleR', 'HindpawToeR', 'HindpawAnkleR','ForepawToeL', 'ForepawAnkleL',
                   'HindpawToeL', 'HindpawAnkleL'],
    'sideXoverhead': ['Nose', 'EarL', 'EarR', 'Back1', 'Back2', 'Back3','Back4', 'Back5', 'Back6', 'Back7', 'Back8',
                      'Back9', 'Back10','Back11', 'Back12', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5','Tail6',
                      'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12', 'StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL']
}

measure_type_names = ['single_kinematics_runXstride', 'multi_kinematics_runXstride', 'behaviour_run']

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
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'tail_height': {
                'tail_label': ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10','Tail11', 'Tail12'],
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
                'ToeAnkleL_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawAnkleL','ForepawToeL'], buffer],
                'ToeAnkleL_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawAnkleL', 'ForepawToeL'], 0],
                'ToeAnkleR_side_zref_buff': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['ForepawAnkleR', 'ForepawToeR'],buffer],
                'ToeAnkleR_side_zref_nobuff': [np.array([0, 0, 1]), np.array([0, 1, 0]),['ForepawAnkleR', 'ForepawToeR'], 0],
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
                'buffer_size': [0, buffer]
            },
            'y': {
                # this is displacement
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
            },
            'z': {
                # this is displacement
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','ForepawToeR','ForepawToeL','ForepawAnkleR','ForepawAnkleL'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
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
                'buffer_size': [0, buffer]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
            },
            'tail_height': {
                'tail_label': ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
            },
            'double_support': [],
            'back_skew': {
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
            },
            'limb_rel_to_body': {
                'time': ['start','end'],
                'step_phase': [0, 1, None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0, buffer]
            }
        }
    }
    return measures

