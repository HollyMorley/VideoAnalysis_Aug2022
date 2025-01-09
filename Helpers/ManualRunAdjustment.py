'''
Specifies which files and runs need to be dropped or adjusted for position
'''
runs_to_drop_placeholder = { # first = 0, last = 40 + 2/5
    '20230306': {
        '1035245': [40,41], # failed last 2 due to getting stuck (and jerry attempted + 2 attempts for destress/attempt)
    },
    '20230309': {
        '1035298': [27], # mouse climbed out and back on to belt mid-way so not normal run
    },
    '20230310': {
        '1035245': [28,29,30,31], # mouse got distressed and all these runs were a mess (extras have also been removed)
    },
    '20230317': {
        '1035244': [2,3], # belt was off for the first 2 runs
    },
}

runs_to_drop_completely = {
    '20230306': {
        '1035245': [42,43], # (failed last 2 due to getting stuck) and jerry attempted + 2 attempts for destress/attempt
        '1035249': [24], # belt was accidentally off on r24 so jerry re-attempted
        '1035250': [21], # mouse sat across the transition so jerry re-attempted r21
    },
    '20230308': {
        '1035244': [24], # belt was off at 0
        '1035249': [9], # belt was off at 5:32 (this was the 'extra')
    },
    '20230309': {
        '1035299': [42], # mouse was being difficult, seems like an accidental extra run
        '1035301': [6], # belt was off at 4:42 (this was the 'extra')
    },
    '20230310': {
        '1035243': [12], # jerry forgot to change the belt speed at 6:02 (this was the 'extra')
        '1035245': [32,33,34,35,36,37,38,39,40,41,42,43,44,45], # mouse got distressed so there were several extra runs in APA phase. Also removing the rest (>= 36) as it was too much disruption
        '1035249': [31], # !!!*** not sure if this door open was detected but mouse stood up and climbed (this was the 'extra') !!!*** todo: check if this was detected
    },
    '20230312': {
        '1035297': [32,41], # 1) extra run as messed up the first run, 2) failed run attempt but had runbacks so was captured on belt (!! possible this wasnt registered though)
        '1035298': [34], # mouse climbed out and back on to belt mid-way so not normal run, so jerry re-attempted
    },
    '20230314': {
        '1035297': [16,40], # 1) belt was off at 7:09 (this was the 'extra'), 2) belt was off at 20:55 (this was the 'extra')
        '1035298': [2], # belt was off for the 1st baseline run (this was the 'extra')
    },
    '20230316': {
        '1035245': [33,34,35], # mouse failed 3 times in a row so jerry re-attempted
    },
    '20230317': {
        '1035243': [14,15,16,17], # 4 fails where mouse was on belt so jerry re-attempted
        '1035244': [4,5,6,7,8,17,18,19,20], # mouse failed at [4,5,6,7,8] so jerry re-attempted, then 1 failed at 27:01, then 3 habituation runs mid-way through
    },
    '20230318': {
        '1035249': [2], # first real run was a fail so jerry re-attempted
    },
    '20230319': {
        '1035244': [42], # jerry knocked the door out at r29 so he did an extra but i think its fine so have just removed the final run
    },

}

missing_runs = {
    '20230306': {
        '1035246': [27,28,29,30,31], # missing the last 5 runs of APA phase
        '1035249': [32,33,34,35,36,37,38,39,40,41] # missing last 10 runs of washout phase (missing side video_2)
    },
    '20230308': {
        '1035246': [0, 1],  # missing the 2 habituation runs
    },
    '20230309': {
            '1035298': [0,1], # missing the 2 habituation runs
    },
    '20230319': {
        '1035249': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], # 28 runs missing from start
    },
}

'''
## Possible issues:

20230308 - 1035244: escaped under moving door at 4:51
20230309 - 1035299: slid under door at 3:31/after r6, also slight incorrect door close at 9:40/after r16 and 14:46/after
 r23, door close and mouse slip under at 20:43/after r31
20230309 - 1035302: slid under door at 3:07/after r5
20230316 - 1035250: did a extra run at 17:39 with a door open/close but dont think this would have been registered


'''




### Shifting run numbers
# old code:
# if 'HM_20230306_APACharRepeat_FAA-1035246_LR_side_1DLC_resnet50_DLC_DualBeltAug10shuffle1_1030000' in filename:  # 5 missing APA from end of APA phase (accidentally quit video)
#     print('Five trials missing from APA phase. Shifting subsequent washout run labels...')
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=27,
#                          max=37, shifted=5)
# if "HM_20230308_APACharRepeat_FAA-1035246_LR_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=0,
#                          max=40, shifted=2)
# if "HM_20230316_APACharExt_FAA-1035243_None_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=1,
#                          max=41, shifted=1)
# if "HM_20230317_APACharExt_FAA-1035244_L_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=4,
#                          max=39, shifted=1)
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=2,
#                          max=40, shifted=2)
# if "HM_20230319_APACharExt_FAA-1035249_R_2_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000" in filename:
#     self.shiftRunNumbers(side=DataframeCoor_side, front=DataframeCoor_front, overhead=DataframeCoor_overhead, min=0,
#                          max=15, shifted=2)
