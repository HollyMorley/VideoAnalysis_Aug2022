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
paths = {
        'filtereddata_folder': r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round2_Jan23",
        'plotting_destfolder': r'M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\Nov23'
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
    ]
}

measure_list = {
    'limb': ['walking_speed','stride_duration','cadence','swing_velocity','stride_length','stance_duration','duty_factor','coo_x','coo_y'], ## for different plot add swing z trajectory & instantaneous swing vel,
    'interlimb': ['bos_ref_stance', 'double_support'], # stance phase, supports, 'bos_hom_stance', 'tail1_displacement'
    'whole_body': ['tail1_ptp_amplitude_stride', 'tail1_speed', 'body_length_stance', 'body_length_swing', 'back_skew_stance', 'back_skew_swing','neck_z_stance', 'neck_z_swing', 'midback_z_stance', 'midback_z_swing', 'tail1_z_stance', 'tail1_z_swing', 'stepping_limb_z_stance', 'stepping_limb_z_swing', 'contra_limb_z_stance', 'contra_limb_z_swing','head_tilt_stance', 'head_tilt_swing', 'body_tilt_stance', 'body_tilt_swing', 'tail_tilt_stance', 'tail_tilt_swing', 'limb_rel_to_body_stance'], #, 'back_curvature_stance', 'back_curvature_swing'], # x/y of paws during swing and stance, body sway (overhead), back curvature, , 'nose_ptp_amplitude_stride'
    #'behavioural': ['wait_time','no_rbs','transitioning_limb']
}
#
# measure_list = {
#      'limb': ['walking_speed','stride_duration'] #,'cadence','swing_velocity','stride_length','stance_duration','duty_factor','coo_x','coo_y'], ## for different plot add swing z trajectory & instantaneous swing vel,
# #     'interlimb': ['bos_ref_stance', 'double_support'], # stance phase, supports, 'bos_hom_stance', 'tail1_displacement'
# #     'whole_body': ['tail1_ptp_amplitude_stride', 'tail1_speed', 'body_length_stance', 'body_length_swing'] #,'back_skew_stance', 'back_skew_swing','neck_height_stance', 'neck_height_swing', 'midback_height_stance', 'midback_height_swing', 'tail1_height_stance', 'tail1_height_swing','head_tilt_stance', 'head_tilt_swing', 'body_tilt_stance', 'body_tilt_swing', 'tail_angle_stance', 'tail_angle_swing'], # x/y of paws during swing and stance, body sway (overhead), back curvature, , 'nose_ptp_amplitude_stride'
# #     #'behavioural': ['wait_time','no_rbs','transitioning_limb']
# }