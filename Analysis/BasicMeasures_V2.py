import pandas as pd
import re
import os
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from Helpers.ConditionsFinder import BaseConditionFiles
from Helpers import utils
from Helpers.Config_23 import *

from Analysis.MeasuresByStride import CalculateMeasuresByStride, RunMeasures
from Analysis.MeasuresByRun import CalculateMeasuresByRun

class Save():
    def __init__(self, conditions, buffer_size=0.25):
        self.conditions, self.buffer_size = conditions, buffer_size
        #self.data = utils.Utils().GetDFs(conditions,reindexed_loco=True)
        self.XYZw = utils.Utils().Get_XYZw_DFs(conditions)
        self.CalculateMeasuresByStride = CalculateMeasuresByStride

    def find_pre_post_transition_strides(self, con, mouseID, r, numstrides=3):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        pre_frame = self.XYZw[con][mouseID].loc(axis=0)[r, 'RunStart'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[-1]
        post_frame = self.XYZw[con][mouseID].loc(axis=0)[r, 'Transition'].loc(axis=1)[
            ['ForepawToeL', 'ForepawToeR'], 'StepCycleFill'].iloc[0]
        trans_limb_mask = post_frame - pre_frame == -1
        stepping_limb = np.array(['ForepawToeL', 'ForepawToeR'])[trans_limb_mask]
        if len(stepping_limb) == 1:
            stepping_limb = stepping_limb[0]
        else:
            raise ValueError('wrong number of stepping limbs identified')

        # limbs_mask_post = (self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition', 'RunEnd']].loc(axis=1)[['ForepawToeR','ForepawToeL'], 'likelihood'] > pcutoff).any(axis=1)
        #
        stance_mask_pre = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_pre = self.XYZw[con][mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1
        stance_mask_post = self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 0
        swing_mask_post = self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'] == 1

        stance_idx_pre = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][stance_mask_pre].tail(numstrides))
        swing_idx_pre = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'StepCycle'][swing_mask_pre].tail(numstrides))
        stance_idx_post = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][stance_mask_post].head(2))
        swing_idx_post = pd.DataFrame(self.XYZw[con][mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'StepCycle'][swing_mask_post].head(2))

        stance_idx_pre['Stride_no'] = np.sort(np.arange(1,len(stance_idx_pre)+1)*-1)
        swing_idx_pre['Stride_no'] = np.sort(np.arange(1,len(swing_idx_pre)+1)*-1)
        stance_idx_post['Stride_no'] = np.arange(0,len(stance_idx_post))
        swing_idx_post['Stride_no'] = np.arange(0,len(swing_idx_post))


        # Combine pre and post DataFrames
        combined_df = pd.concat([stance_idx_pre,swing_idx_pre, stance_idx_post, swing_idx_post]).sort_index(level='FrameIdx')

        return combined_df

    def find_pre_post_transition_strides_ALL_RUNS(self, con, mouseID):
        #view = 'Side'
        SwSt = []
        for r in self.XYZw[con][mouseID].index.get_level_values(level='Run').unique().astype(int):
            try:
                stsw = self.find_pre_post_transition_strides(con=con,mouseID=mouseID,r=r)
                SwSt.append(stsw)
            except:
                pass
                #print('Cant get stsw for run %s' %r)
        SwSt_df = pd.concat(SwSt)

        return SwSt_df

    def get_measures_byrun_bystride(self, SwSt, con, mouseID):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        st_mask = (SwSt.loc(axis=1)[['ForepawToeR', 'ForepawToeL'],'StepCycle'] == 0).any(axis=1)
        stride_borders = SwSt[st_mask]

        temp_single_list = []
        temp_multi_list = []
        temp_run_list = []

        for r in tqdm(stride_borders.index.get_level_values(level='Run').unique(), desc=f"Processing: {mouseID}"):
            stepping_limb = np.array(['ForepawToeR','ForepawToeL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawToeR','ForepawToeL']].count() > 1).values][0]
            for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no']):#[:-1]):
                # if len(stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')) <= sidx + 1:
                #     print("Can't calculate s: %s" %s)
                try:
                    stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                    stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1] - 1 # todo check i am right to consider the previous frame the end frame

                    #class_instance = self.CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
                    calc_obj = CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
                    measures_dict = measures_list(buffer=self.buffer_size)

                    runXstride_measures = RunMeasures(measures_dict, calc_obj, buffer_size=self.buffer_size, stride=s)
                    single_val, multi_val = runXstride_measures.get_all_results()
                    temp_single_list.append(single_val)
                    temp_multi_list.append(multi_val)
                except:
                    pass
                    #print("Cant calculate s: %s" %s)

            run_measures = CalculateMeasuresByRun(XYZw=self.XYZw,con=con,mouseID=mouseID,r=r,stepping_limb=stepping_limb)
            run_val = run_measures.run()
            temp_run_list.append(run_val)
        measures_bystride_single = pd.concat(temp_single_list)
        measures_bystride_multi = pd.concat(temp_multi_list)
        measures_byrun = pd.concat(temp_run_list)

        return measures_bystride_single, measures_bystride_multi, measures_byrun

    def process_mouse_data(self, mouseID, con):
        try:
            # Process data for the given mouseID
            SwSt = self.find_pre_post_transition_strides_ALL_RUNS(con=con, mouseID=mouseID)
            single_byStride, multi_byStride, byRun = self.get_measures_byrun_bystride(SwSt=SwSt, con=con, mouseID=mouseID)

            # add mouseID to SwSt index
            index = SwSt.index
            multi_idx_tuples = [(mouseID, a[0], a[1], a[2]) for a in index]
            multi_idx = pd.MultiIndex.from_tuples(multi_idx_tuples, names=['MouseID', 'Run', 'Stride', 'FrameIdx'])
            SwSt.set_index(multi_idx, inplace=True)

            return single_byStride, multi_byStride, byRun, SwSt
        except Exception as e:
            print(f"Error processing mouse {mouseID}: {e}")
            return None, None, None, None

    def process_mouse_data_wrapper(self, args):
        mouseID, con = args
        return self.process_mouse_data(mouseID, con)

    def save_all_measures_parallel(self):
        pool = Pool(cpu_count())
        for con in self.conditions:
            # Initialize multiprocessing Pool with number of CPU cores
            results = []

            # Process data for each mouseID in parallel
            for mouseID in self.XYZw[con].keys():
                result = pool.apply_async(self.process_mouse_data_wrapper, args=((mouseID, con),))
                results.append(result)

            # Aggregate results
            single_byStride_all, multi_byStride_all, byRun_all, SwSt_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for result in results:
                single_byStride, multi_byStride, byRun, SwSt = result.get()
                if single_byStride is not None:
                    single_byStride_all = pd.concat([single_byStride_all, single_byStride])
                if multi_byStride is not None:
                    multi_byStride_all = pd.concat([multi_byStride_all, multi_byStride])
                if byRun is not None:
                    byRun_all = pd.concat([byRun_all, byRun])
                if SwSt is not None:
                    SwSt_all = pd.concat([SwSt_all, SwSt])

            single_byStride_all = single_byStride_all.apply(pd.to_numeric, errors='coerce', downcast='float')
            multi_byStride_all = multi_byStride_all.apply(pd.to_numeric, errors='coerce', downcast='float')
            byRun_all = byRun_all.apply(pd.to_numeric, errors='coerce', downcast='float')
            #SwSt_all = SwSt_all.apply(pd.to_numeric, errors='coerce', downcast='float')

            # Write to HDF files
            if 'Day' not in con:
                dir = os.path.join(paths['filtereddata_folder'], con)
            else:
                dir = utils.Utils().Get_processed_data_locations(con)

            single_byStride_all.to_hdf(os.path.join(dir, "MEASURES_single_kinematics_runXstride.h5"),
                                       key='single_kinematics', mode='w')
            multi_byStride_all.to_hdf(os.path.join(dir, "MEASURES_multi_kinematics_runXstride.h5"), key='multi_kinematics',
                                      mode='w')
            byRun_all.to_hdf(os.path.join(dir, "MEASURES_behaviour_run.h5"), key='behaviour', mode='w')
            SwSt_all.to_hdf(os.path.join(dir, "MEASURES_StrideInfo.h5"), key='stride_info', mode='w')

        # Wait for all processes to complete and collect results
        pool.close()
        pool.join()


class GetAllFiles():
    def __init__(self, directory=None,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None):
        self.directory = directory
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

    def get_files(self):
        """
        If 'Extended', gather .h5 from all Day subdirectories so we can plot them all at once.
        Otherwise, do the original logic for 'Repeats' or other cases.
        """
        if self.repeat_extend == 'Extended':
            parent_dir = os.path.dirname(self.directory)
            if not os.path.isdir(parent_dir):
                print(f"Parent dir not found for extended condition: {parent_dir}")
                return

            # Gather .h5 from subdirs named 'Day...'
            all_files = []
            all_mouseIDs = []
            all_dates = []
            for subd in os.listdir(parent_dir):
                if not subd.lower().startswith('day'):
                    continue
                sub_path = os.path.join(parent_dir, subd)
                if os.path.isdir(sub_path):
                    these_files = utils.Utils().GetListofRunFiles(sub_path)
                    for f in these_files:
                        match = re.search(r'FAA-(\d+)', f)
                        if not match:
                            continue
                        all_files.append(f)
                        all_mouseIDs.append(match.group(1))
                        date_part = f.split(os.sep)[-1].split('_')[1]
                        all_dates.append(date_part)

            if not all_files:
                print(f"No day-based .h5 files found under {parent_dir}")
                return

            save = Save(
                files=all_files,
                mouseIDs=all_mouseIDs,
                dates=all_dates,
                exp=self.exp,
                speed=self.speed,
                repeat_extend=self.repeat_extend,
                exp_wash=self.exp_wash,
                day=self.day,
                vmt_type=self.vmt_type,
                vmt_level=self.vmt_level,
                prep=self.prep
            )
            save.save_all_measures_parallel()

        else:
            # Original logic for Repeats
            files = utils.Utils().GetListofRunFiles(self.directory)
            if not files:
                print(f"No run files found in directory: {self.directory}")
                return

            mouseIDs = []
            dates = []
            for f in files:
                match = re.search(r'FAA-(\d+)', f)
                if match:
                    mouseIDs.append(match.group(1))
                else:
                    mouseIDs.append(None)
                try:
                    date = f.split(os.sep)[-1].split('_')[1]
                    dates.append(date)
                except IndexError:
                    dates.append(None)

            valid_indices = [
                i for i, (m, d) in enumerate(zip(mouseIDs, dates))
                if m is not None and d is not None
            ]
            filtered_files = [files[i] for i in valid_indices]
            filtered_mouseIDs = [mouseIDs[i] for i in valid_indices]
            filtered_dates = [dates[i] for i in valid_indices]

            if not filtered_files:
                print("No valid run files to process after filtering.")
                return

            save = Save(
                files=filtered_files,
                mouseIDs=filtered_mouseIDs,
                dates=filtered_dates,
                exp=self.exp,
                speed=self.speed,
                repeat_extend=self.repeat_extend,
                exp_wash=self.exp_wash,
                day=self.day,
                vmt_type=self.vmt_type,
                vmt_level=self.vmt_level,
                prep=self.prep
            )
            save.save_all_measures_parallel()

class GetConditionFiles(BaseConditionFiles):
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None):
        super().__init__(
            exp=exp, speed=speed, repeat_extend=repeat_extend, exp_wash=exp_wash,
            day=day, vmt_type=vmt_type, vmt_level=vmt_level, prep=prep
        )

    def process_final_directory(self, directory):
        GetAllFiles(
            directory=directory,
            exp=self.exp,
            speed=self.speed,
            repeat_extend=self.repeat_extend,
            exp_wash=self.exp_wash,
            day=self.day,
            vmt_type=self.vmt_type,
            vmt_level=self.vmt_level,
            prep=self.prep,
        ).get_files()

def main():
    # Repeats
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day2').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day3').get_dirs()

    # Extended
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended').get_dirs()

if __name__ == '__main__':
    main()
