########################################################################################################################
################################################### User-set values ####################################################
########################################################################################################################

cams = ['side','front','overhead']
scorer_side = "DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000"
scorer_side_new = "DLC_resnet50_DLC_DualBeltJul25shuffle1_1020000"
scorer_overhead = "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000"
scorer_front = "DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000"
pcutoff = 0.9
mice_ID = [
    '1034976',
    '1034978',
    '1034979',
    '1034980',
    '1034982',
    '1034983'
]
mice_name = [
    'MNone',
    'MR',
    'MLR',
    'MNone',
    'MR',
    'MLR'
]
RunStages = ['TrialStart', 'RunStart', 'Transition', 'RunEnd', 'TrialEnd']
ExpPhases = ['Baseline', 'APA', 'Washout']
distancescm = [13,12,11.5,11.5] # goes 0: wall1-wall0,  1: wall2-wall1, 3: wall3-wall2, 4: wall4-wall3
APACharRuns = [10,20,10]
APAPerRuns = [10,20,10] #### warning if change i have sometimes used the apachar variable to define perception runs too
APAVmtRuns = [10,15,10]
APA_lengthruns = 25#100#25
after_lengthruns = 25
literature_apa_length = 100 #ms ## this isnt acutally correct, check this
preruns_CharLow = 2
preruns_CharMidHigh = 5
filtereddata_folder = r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round2_Jan23"
plotting_destfolder = r'M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\Plots\July23'
fps = 247
speeds = {'Low': 6, 'Mid': 18, 'High': 30}
swst_vals = {
    'st': 0,
    'sw': 1,
    'st_bkwd': 2,
    'sw_bkwd': 3
}

