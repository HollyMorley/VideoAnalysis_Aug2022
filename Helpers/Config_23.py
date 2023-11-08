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
    'belt_length_sideviewrange': 600 # mm
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
        'plotting_destfolder': r'M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\July23'
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