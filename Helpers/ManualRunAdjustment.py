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
        '1035245': [30,33,34,35, 36,37,38,39,40,41,42,43,44,45], # mouse got distressed and all these runs were a mess (extras have also been removed), plus washout removed as so messy
    },
    '20230314': {
        '1035297': [16],  # belt was off at 7:09
    },
    '20230317': {
        '1035244': [2,3], # belt was off for the first 2 runs
    },
    '20230325': {
        '1035244': [34], # mouse got stuck
        '1035245': [33], # mouse got stuck
    },
    '20230412': {
        '1035297': [18],  # got stuck in middle of run at r18
        '1035298': [24], # got stuck in middle of run at r24
    },
    '20230413': {
        '1035297': [43],  # got stuck in middle of run at r43
    },
}

runs_to_drop_completely = {
    '20230306': {
        '1035245': [42,43], # (failed last 2 due to getting stuck) and jerry attempted + 2 attempts for destress/attempt
        '1035249': [24], # belt was accidentally off on r24 so jerry re-attempted
        '1035250': [21], # mouse sat across the transition so jerry re-attempted r21
    },
    '20230308': {
        '1035244': [24], # belt was off at 11:27
        '1035245': [32], # door opened and closed after mouse had slipped under and walked at 17:40
        '1035249': [9,36], # belt was off at 5:32 (this was the 'extra'), mouse slipped under at 20:40/after r35
        '1035250': [11], # slipped under door at 5:01/after r10
    },
    '20230309': {
        '1035299': [42], # mouse was being difficult, seems like an accidental extra run
        '1035301': [6], # belt was off at 4:42 (this was the 'extra')
    },
    '20230310': {
        '1035243': [12], # jerry forgot to change the belt speed at 6:02 (this was the 'extra')
        '1035245': [28,29,31,32], # mouse got distressed so there were several extra runs in APA phase.
        '1035249': [31], # !!!*** not sure if this door open was detected but mouse stood up and climbed (this was the 'extra') !!!*** todo: check if this was detected
    },
    '20230312': {
        '1035297': [32,39,43], # 1) extra run as messed up the first run, 2) failed (didnt leave platform but seems was registered still), 3) failed run attempt but had runbacks so was captured on belt (!! possible this wasnt registered though)
        '1035298': [34], # mouse climbed out and back on to belt mid-way so not normal run, so jerry re-attempted
    },
    '20230314': {
        '1035297': [40], # belt was off at 20:55 (this was the 'extra')
        '1035298': [2], # belt was off for the 1st baseline run (this was the 'extra')
    },
    '20230316': {
        '1035244': [33,34,35], # mouse failed 3 times in a row so jerry re-attempted
    },
    '20230317': {
        '1035243': [14,15,16,17,18], # 4 fails where mouse was on belt so jerry re-attempted
        '1035244': [4,5,6,7,8,17,18,19,20,21], # mouse failed at [4,5,6,7,8] so jerry re-attempted, then 1 failed at 27:01, then 3 habituation runs mid-way through #todo added 21 as wont run with it
    },
    '20230318': {
        '1035249': [2], # first real run was a fail so jerry re-attempted
    },
    '20230319': {
        '1035244': [42], # jerry knocked the door out at r29 so he did an extra but i think its fine so have just removed the final run
    },
    '20230320': {
        '1035299': [20], # mouse climbed out so jerry counted this as an extra
    },
    '20230322': {
        '1035298': [14,33], # 1) mouse stood/sat across belt so jerry re-attempted, 2) same but the extra run is where mouse stood
    },
    '20230323': {
        '1035299': [37], # not sure why but jerry did an extra run at 20:21/after r36
        '1035301': [17,20,39], # all marked as extra runs todo check why when can access videos!!
    },
    '20230325': {
        '1035244': [35,36,37,38,39], # after mouse got stuck jerry re-attempted several times
        '1035245': [34,35,37,38,39,41], # after mouse got stuck jerry re-attempted several times
    },
    '20230326': {
        '1035250': [3], # mouse tried to turn back after half stepping but struggled so jerry stopped (this was the 'extra') after r2/@6:59
    },
    '20230327': {
        '1035245': [16], # mouse failed this run so jerry re-attempted
        '1035246': [42], # jerry wrote extra after r32/@20:20 but not sure why so have just taken from then end as looks fine to me
    },
    '20230328': {
        '1035245': [4], # belt was off for the first run
        '1035246': [42], # jerry wrote extra after r16/@12:47 but not sure why so have just taken from then end as looks fine to me
    },
    '20230403': {
        '1035302': [18,19,10], # mouse had been failing (in preserved runs), jerry did an extra run which failed and then 2 stationary runs
        '1035299': [25], # see notes at bottom (slipped under door?)
        '1035301': [27], #todo have deleted as this run is stopping running, maybe it is not a real run??
    },
    '20230404': {
        '1035302': [2], # belt was off for the first run
    },
    '20230406': {
        '1035299': [36], # not sure why but jerry did an extra run at 18:27/after r35
        '1035302': [37,39], # not sire why but seems like when mouse slid under door jerry re-attempted the run
    },
    '20230408': {
        '1035246': [2,4,17], # belt 1 kept turning off so jerry re-attempted
    },
    '20230409': {
        '1035246': [4], # belt 1 was off so jerry re-attempted (this was the extra)
    },
    '20230410': {
        '1035243': [2], # belt was off for the first run
        '1035250': [25,39], # belt was off so jerry re-attempted (these were the extra runs)
    },
    '20230412': {
        '1035297': [19], # got stuck in middle of run at r18 so gave 1 stationery run after
        '1035298': [20,25,40,44,46], # stood up, extra after got stuck, climbed out x 3
        '1035299': [25],  # got stuck in middle of run at r25
        '1035302': [5,19,21], # fail so jerry re-attempted,
    },
    '20230413': {
        '1035297': [44,45], # todo i think 2x extra as stationary runs after got stuck
        '1035302': [51] # fail so jerry re-attempted (this is the mouse we staggered, hence the extra runs!)
    },
    '20230414': {
        '1035297': [6], # jerry seemed to count this as an extra run due to RB but im not sure
    },
    '20230415': {
        '1035301': [36], # 'retro speed'?
    },


}

missing_runs = {
    '20230306': {
        '1035246': [27,28,29,30,31], # missing the last 5 runs of APA phase
        '1035249': [32,33,34,35,36,37,38,39,40,41] # missing last 10 runs of washout phase (missing side video_2)
    },
    '20230308': {
        '1035244': [0], # missing door opening of 1st habituation run so not registered
        '1035246': [0, 1],  # missing the 2 habituation runs
    },
    '20230309': {
            '1035298': [0,1], # missing the 2 habituation runs
    },
    '20230310': {
        '1035250': [9], # not been detected in my code !!!!WARNING!!! Changes to code could change this
    },
    '20230312':{
        '1035299': [0,1], # missing the 2 habituation runs
    },
    '20230316': {
        '1035243': [1], # missing the 2nd habituation run as obscured from view
    },
    '20230319': {
        '1035249': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], # 28 runs missing from start
    },
    '20230325': {
        '1035244': [0,1], # missing the 2 stationary habituation runs (3 slow incld)
        '1035245': [41], # video cut out before last run
    },
    '20230328': {
        '1035243': [0],  # missed a habituation run according to jerrys notes (4 not 5)
    },
    '20230403': {
        '1035297': [1, 23],  # moused slipped under door for prep run and havent detected run 23 for unknown reason (think it was a bad door close)
    },
    '20230412': {
        '1035299': [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45], # no notes for rest of this file but not if no runs recorded after stuck at r25
        # 302 would be here but we staggered the runs across the days to make up for the lost 20 here
    }
}

'''
## Possible issues:

20230308 - 1035244: escaped under moving door at 4:51
20230309 - 1035299: slid under door at 3:31/after r6, also slight incorrect door close at 9:40/after r16 and 14:46/after
 r23, door close and mouse slip under at 20:43/after r31
20230309 - 1035302: slid under door at 3:07/after r5
20230316 - 1035250: did a extra run at 17:39 with a door open/close but dont think this would have been registered
20230403 - 1035299: slipped under door at 12:25/after r24 but dont think this would have been registered - has been acted on!
20230404 - 1035297: slipped under door at 4:00/after r8 but dont think this would have been registered


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
#                          max=15, shifted=2) ## this is because i thought it was start of the vid
