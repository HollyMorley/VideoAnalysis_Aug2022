'''
Specifies which files and runs need to be dropped or adjusted for position
'''
runs_to_drop = { # first = 0, last = 40 + 2/5
    '20230306': {
        '1035245': [40,41,42,43], # failed last 2 due to getting stuck and jerry attempted + 2 attempts for destress/attempt
    }
}






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
