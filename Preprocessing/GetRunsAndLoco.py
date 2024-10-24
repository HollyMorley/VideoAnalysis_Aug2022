import numpy as np
import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import threading
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
import cv2
import Helpers.utils as utils
from Helpers.Config_23 import *

class GetRuns:
    def __init__(self, file):
        self.file = file
        self.data = self.get_data()

        self.trial_starts, self.trial_ends = [], []
        self.run_starts, self.run_ends = [], []

    def visualise_video_frames(self, view, start_frame, end_frame):
        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        else:
            video_file = video_files[0]

        # Open the video file
        cap = cv2.VideoCapture(video_file)

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure end_frame does not exceed total_frames
        if end_frame > total_frames:
            end_frame = total_frames

        # Get the width and height of the video frame
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Buffer to hold the frames from start_frame to end_frame
        frames = []

        # Read through the video to buffer the frames between start_frame and end_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib

        # Release the video capture after buffering the frames
        cap.release()

        # Initialize Matplotlib figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Display the first frame
        frame_image = ax.imshow(frames[0])
        ax.set_xticks([])
        ax.set_yticks([])

        # Create the slider for frame selection
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        # Update the displayed frame when the slider is changed
        def update(val):
            frame_num = int(slider.val) - start_frame
            frame_image.set_data(frames[frame_num])  # Update the image data with the selected frame
            plt.draw()

        slider.on_changed(update)

        # Show the Matplotlib window
        plt.show()

    def get_data(self):
        # Get the data from the h5 file
        data = pd.read_hdf(self.file, key='real_world_coords')
        # label multiindex columns as 'bodyparts' and 'coords'
        data.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in data.columns], names=['bodyparts', 'coords'])

        return data

    def get_runs(self):
        self.find_trials()
        self.find_run_stages()
        # self.get_run_ends()
        # self.get_run_starts()
        return runs

    def find_trials(self):
        door_open, door_close = self.door_opn_cls()
        mouse_present = self.find_mouse_on_belt()

        door_open_checked = []
        door_close_checked = []
        # check if the mouse is present between all the door open and close frames
        for open_frame, close_frame in zip(door_open, door_close):            # Check if any mouse_present frames fall between open and close frame
            present_frames = mouse_present[(mouse_present >= open_frame) & (mouse_present <= close_frame)]

            # Add the result for this door open-close pair
            if len(present_frames) > 0:
                door_open_checked.append(open_frame)
                door_close_checked.append(close_frame)

        self.trial_starts = door_open
        self.trial_ends = door_close

    def door_opn_cls(self):
        # Extract the rolling window
        rolling_window = self.data.loc(axis=1)['Door', 'z'].rolling(window=10, center=True, min_periods=0)
        # Apply NumPy for faster computation
        door_movement = rolling_window.apply(lambda x: np.subtract(x[-1], x[0]) if len(x) >= 2 else 0, raw=True)
        door_closed = door_movement.index[door_movement.abs() < 1]
        door_closed_chunks = utils.Utils().find_blocks(door_closed, gap_threshold=50, block_min_size=fps*2)

        # check if after the end of each closed chunk there is a door opening, where the door marker is not visible

        door_open = []
        door_close = []
        for cidx, chunk in enumerate(door_closed_chunks):
            chunk_end = chunk[-1]

            # After a closed chunk ends, check for the absence of the door marker (NaNs or missing values in 'z')
            window_after_chunk = self.data.loc[chunk_end:chunk_end + 2000, ('Door', 'z')]

            # Check if there is a long period where the door marker is missing (NaNs)
            door_out_of_frame = window_after_chunk.isna()
            if door_out_of_frame.sum() > 1500:  # if there is any NaN in the window after the chunk
                # Retrieve the chunk value to denote the beginnning of the trial
                door_open.append(chunk[-1])
                if cidx + 1 < len(door_closed_chunks):
                    door_close.append(door_closed_chunks[cidx + 1][0])
        if door_close[0] < door_open[0]:
            door_open.append(0)
        if door_close[-1] < door_open[-1]:
            door_close.append(self.data.index[-1])

        # sort and convert to np array
        door_open = np.sort(np.array(door_open))
        door_close = np.sort(np.array(door_close))

        if len(door_open) != len(door_close):
            raise ValueError("Number of trial starts and ends do not match!!!")
        if sum((door_close - door_open) < 0) > 0:
            raise ValueError("Miscalculated trial start and end times!!!")

        return door_open, door_close

    def find_forward_facing_bool(self, data, xthreshold, zthreshold):
        # filter by when mouse facing forward
        back_median = data.loc(axis=1)[
            ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11',
             'Back12'], 'x'].median(axis=1)
        tail_median = data.loc(axis=1)[
            ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11',
             'Tail12'], 'x'].median(axis=1)
        #smooth both medians and put back into series
        back_median = pd.Series(gaussian_filter1d(back_median, sigma=1), index=back_median.index)
        tail_median = pd.Series(gaussian_filter1d(tail_median, sigma=1), index=tail_median.index)
        back_tail_mask = back_median - tail_median > xthreshold
        nose_tail_x_mask = data.loc[:, ('Nose', 'x')] > data.loc[:, ('Tail1', 'x')]
        nose_tail_z_mask = data.loc[:, ('Nose', 'z')] - data.loc[:, ('Tail1', 'z')] < zthreshold
        #belt1_mask = data.loc[:, ('Nose', 'x')] < 470
        facing_forward_mask = back_tail_mask & nose_tail_x_mask & nose_tail_z_mask #& belt1_mask
        return facing_forward_mask


    def find_mouse_on_belt(self):
        # find the frame where the mouse crosses the finish line (x=600?) for the first time after the trial start
        facing_forward_mask = self.find_forward_facing_bool(self.data, xthreshold=0, zthreshold=40)
        mouse_on_belt = facing_forward_mask & (self.data.loc[:, ('Nose', 'x')] > 200) & (self.data.loc[:, ('Nose', 'x')] < 500)
        mouse_on_belt_index = mouse_on_belt.index[mouse_on_belt]
        return mouse_on_belt_index

    def check_post_runs(self, forward_data, forward_chunks):
        post_transition_mask = forward_data.loc[:, ('Nose', 'x')] > 470
        post_transition = forward_data[post_transition_mask]
        first_transition = post_transition.index[0]
        # drop runs that occur after the first transition
        correct_forward_chunks = [chunk for chunk in forward_chunks if chunk[0] < first_transition]
        return correct_forward_chunks

    def check_run_backs(self, data, forward_chunks):
        runbacks = []
        true_run = []
        i = 0
        while i < len(forward_chunks) - 1:
            run = forward_chunks[i]
            next_run = forward_chunks[i + 1]
            next_run_start = next_run[0]
            run_data = data.loc[run[0]:next_run_start]

            # check if mice run backwards between this run and the next
            runback_mask = run_data.loc(axis=1)['Nose', 'x'] < run_data.loc(axis=1)['Tail1', 'x']
            runback_data = run_data[runback_mask]

            # check if mice step off the platform between this run and the next
            if len(runback_data) > 0:
                step_off_mask = run_data.loc(axis=1)['Tail1', 'x'] < 0
                step_off_data = run_data[step_off_mask]

                # if mice meet these conditions, add this run to the runbacks list
                if len(step_off_data) > 0:
                    runbacks.append(run)
                else:
                    true_run.append(run)
            else:
                # No backwards running detected in this snippet
                raise ValueError("no runbacks or real runs detected, using old logic") #todo remove this
                # # Check if 'Tail1' is invalid between current run end and next run start
                # tail1_data = data.loc[run[1] + 1:next_run_start - 1, ('Tail1', 'x')] # check from frame after the chunk end to frame before next chunk start
                #
                # if tail1_data.empty or not tail1_data.notna().any():
                #     # The current run is part of the next one; merge them
                #     real_run_start = run[0]
                #     real_run_end = next_run[1]
                #     merged_run = (real_run_start, real_run_end)
                #     # Update the upcoming run to the merged run
                #     forward_chunks[i + 1] = merged_run
                # else:
                #     # Check for 'EarR' validity
                #     earR_data = data.loc[run[1] + 1:next_run_start - 1, ('EarR', 'x')]
                #
                #     if earR_data.empty or not earR_data.notna().any():
                #         # Merge runs
                #         real_run_start = run[0]
                #         real_run_end = next_run[1]
                #         merged_run = (real_run_start, real_run_end)
                #         # Update both runs to the merged run
                #         forward_chunks[i] = merged_run
                #         forward_chunks[i + 1] = merged_run
                #     else:
                #         # Valid data present; include the run
                #         true_run.append(run)

            i += 1

        true_run.append(forward_chunks[-1])

        return runbacks, true_run

    def find_real_run_vs_rbs(self, r):
        run_data = self.data.loc(axis=0)[self.trial_starts[r]:self.trial_ends[r]]

        # filter by when mouse facing forward
        facing_forward_mask = self.find_forward_facing_bool(run_data, xthreshold=20, zthreshold=40)
        facing_forward = run_data[facing_forward_mask]
        forward_chunks = utils.Utils().find_blocks(facing_forward.index, gap_threshold=10, block_min_size=25)
        if len(forward_chunks) > 1:
            forward_chunks = self.check_post_runs(facing_forward, forward_chunks)
            if len(forward_chunks) > 1:
                runbacks, forward_chunks = self.check_run_backs(run_data, forward_chunks)
                if len(forward_chunks) > 1:
                    raise ValueError("More than one run detected in a trial")
            else:
                runbacks = []
        elif len(forward_chunks) == 0:
            raise ValueError("No runs detected in a trial")
        else:
            runbacks = []
        return forward_chunks, runbacks

    def find_run_stages(self):
        for r in range(len(self.trial_starts)):
            run_bounds, runbacks = self.find_real_run_vs_rbs(r)
            if len(run_bounds) == 1:
                run_bounds = run_bounds[0]
                runstart = self.find_run_start(r, run_bounds)
            else:
                raise ValueError("More than one run detected in a trial (in find_run_stages)")


    def find_run_ends(self):
        # find the frame where the mouse crosses the finish line (x=600?) for the first time after the trial start
        nose_in_front_of_tail = self.data.loc[:, ('Nose', 'x')] > self.data.loc[:, ('TailBase', 'x')]



    # def find_run_start(self, r, run_bounds):
    #     # check back from run_bounds[0] to find the first frame where Nose appears (with a gap threshold of 40 frames)
    #     nose_present_mask = self.data.loc[self.trial_starts[r]:run_bounds[0], ('Nose', 'x')].notna()
    #     nose_present = self.data.loc[self.trial_starts[r]:run_bounds[0], ('Nose', 'x')][nose_present_mask]
    #     nose_present_chunks = utils.Utils().find_blocks(nose_present.index, gap_threshold=15, block_min_size=10)
    #     if len(nose_present_chunks) > 0:
    #         run_bound_start = nose_present_chunks[-1][0]
    #     else:
    #         run_bound_start = self.trial_starts[r]
    #         print(f"No nose detected before run start in trial {r}")
    #
    #
    #
    #     # Extract the right and left forepaw data into separate DataFrames
    #     right = self.data.loc[run_bound_start:run_bounds[1], (['ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR'])]
    #     left = self.data.loc[run_bound_start:run_bounds[1], (['ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL'])]
    #
    #     for coord in ['z', 'x']:
    #         # Combine coordinates
    #         right_combined = self.combine_paw_data(right.loc(axis=1)[:, coord], sigma=1)
    #         left_combined = self.combine_paw_data(left.loc(axis=1)[:, coord], sigma=1)
    #
    #         # Add the combined coordinates to the right and left DataFrames
    #         right[f'Combined{coord.upper()}'] = right_combined
    #         left[f'Combined{coord.upper()}'] = left_combined
    #     paw_data = pd.concat([right, left], axis=1, keys=['Right', 'Left'])
    #
    #     """
    #     criteria to meet:
    #     1. left or right paw is first to set down to belt level - ie z ~= 0 for several frames
    #     2. this paw is beyond the start platform and in the belt area x > 0, x < ~200
    #     3. this touchdown is followed by the other paw touching down and the whole body passing the start line
    #     proposed approach:
    #     1. look for where nose in front of tail and tail base is 0 < x < 10
    #     2. look backwards from this frame to find the first frame where the paw is at belt level
    #     """
    #
    #     # refine paw data by CombinedX > 0
    #     x0_idx = paw_data.index[np.all(paw_data.loc(axis=1)[:, 'CombinedX'] > 0, axis=1)] # ensures paw data is only > start platform
    #     paw_data_x0 = paw_data.loc[x0_idx]
    #
    #     # find z data where frame to frame difference is less than 0.1 and z is less than 1.5
    #     stance_r = paw_data_x0.index[(abs(paw_data_x0.loc(axis=1)['Right', 'CombinedZ'].diff()) < 0.2) & (paw_data_x0.loc(axis=1)['Right', 'CombinedZ'] < 1.5)]
    #     stance_l = paw_data_x0.index[(abs(paw_data_x0.loc(axis=1)['Left', 'CombinedZ'].diff()) < 0.2) & (paw_data_x0.loc(axis=1)['Left', 'CombinedZ'] < 1.5)]
    #     swingend_r = paw_data_x0.index[(paw_data_x0.loc(axis=1)['Right', 'CombinedZ'].diff() < -0.5) & (paw_data_x0.loc(axis=1)['Right', 'CombinedZ'] > 1)]
    #     swingend_l = paw_data_x0.index[(paw_data_x0.loc(axis=1)['Left', 'CombinedZ'].diff() < -0.5) & (paw_data_x0.loc(axis=1)['Left', 'CombinedZ'] > 1)]
    #
    #     stance_r_chunks = utils.Utils().find_blocks(stance_r, gap_threshold=2, block_min_size=4)
    #     stance_l_chunks = utils.Utils().find_blocks(stance_l, gap_threshold=2, block_min_size=4)
    #     swingend_r_chunks = utils.Utils().find_blocks(swingend_r, gap_threshold=3, block_min_size=1)
    #     swingend_l_chunks = utils.Utils().find_blocks(swingend_l, gap_threshold=3, block_min_size=1)
    #
    #     # check alternating stance and swingend chunks (based on middle index for each chunk) for each paw
    #     stance_r_middle = np.mean(stance_r_chunks, axis=1).astype(int)
    #     swing_r_middle = np.mean(swingend_r_chunks, axis=1).astype(int)
    #     stance_l_middle = np.mean(stance_l_chunks, axis=1).astype(int)
    #     swing_l_middle = np.mean(swingend_l_chunks, axis=1).astype(int)
    #
    #     # Paw events
    #     events_r = np.concatenate((
    #         np.column_stack((stance_r_middle, np.full(len(stance_r_middle), 'stance'))),
    #         np.column_stack((swing_r_middle, np.full(len(swing_r_middle), 'swing')))
    #     ))
    #     events_l = np.concatenate((
    #         np.column_stack((stance_l_middle, np.full(len(stance_l_middle), 'stance'))),
    #         np.column_stack((swing_l_middle, np.full(len(swing_l_middle), 'swing')))
    #     ))
    #
    #     # Sort paw events
    #     events_r_sorted = events_r[events_r[:, 0].astype(int).argsort()]
    #     events_l_sorted = events_l[events_l[:, 0].astype(int).argsort()]
    #
    #     is_valid_r = self.check_alternating(events_r_sorted)
    #     is_valid_l = self.check_alternating(events_l_sorted)
    #
    #     if is_valid_r and is_valid_l:
    #         # Get the starting frame of the first stance chunk for each paw
    #         first_stance_r_frame = stance_r_chunks[0][0] if len(stance_r_chunks) > 0 else None
    #         first_stance_l_frame = stance_l_chunks[0][0] if len(stance_l_chunks) > 0 else None
    #
    #         # Determine which paw touches down first
    #         if first_stance_r_frame is not None and first_stance_l_frame is not None:
    #             if first_stance_r_frame <= first_stance_l_frame:
    #                 runstart_frame = first_stance_r_frame
    #                 runstart_side = 'r'
    #             else:
    #                 runstart_frame = first_stance_l_frame
    #                 runstart_side = 'l'
    #         elif first_stance_r_frame is not None:
    #             runstart_frame = first_stance_r_frame
    #             runstart_side = 'r'
    #         elif first_stance_l_frame is not None:
    #             runstart_frame = first_stance_l_frame
    #             runstart_side = 'l'
    #         else:
    #             raise ValueError("No stance events found for either paw")
    #
    #         # Verify that the other 3 paws touch down shortly after
    #
    #
    #     else:
    #         raise ValueError("Swing stance detection failed, alternating pattern not detected")
    #
    #
    #
    #     return runstarts

    def find_run_start(self, r, run_bounds):
        # Step 1: Determine the initial run bound start based on nose detection
        run_bound_start = self.determine_initial_run_bound_start(r, run_bounds)

        # Step 2 & 3: Extract and combine data for all four paws
        paw_data_combined = self.extract_and_combine_paw_data(run_bound_start, run_bounds[1])

        # Step 4: Filter data where both forepaw 'x' positions are greater than 0
        paw_data_x0 = self.filter_paw_data_beyond_start_platform(paw_data_combined)

        # Step 5: Detect stance and swing events for each paw
        stance_events = self.detect_stance_events(paw_data_x0)

        # Step 6: Check for alternating patterns for each paw
        is_valid_paws = self.check_real_stance(stance_events)#, swing_events)
        #is_valid_paws = self.check_alternating_patterns(stance_events)#, swing_events)

        # Step 6.1: Filter stance events to include only valid paws
        valid_stance_events = {
            key: stances for key, stances in stance_events.items() if is_valid_paws.get(key, False)
        }

        if not valid_stance_events:
            raise ValueError("No valid paws found after alternating pattern check.")

        # Step 7: Determine run start frame based on the earliest stance event among all valid paws
        runstart_frame, runstart_key, runstart_paw = self.determine_run_start_frame(valid_stance_events)

        # Step 8: Verify that all valid paws touch down shortly after
        self.verify_paws_touch_down(valid_stance_events, runstart_frame, runstart_key)

        # Step 9: Additional checks (nose ahead of tail, etc.)
        self.perform_additional_checks(runstart_frame)

        # Step 10: Set the run start frame
        self.run_start_frame = runstart_frame
        print(f"Run starts at frame {runstart_frame}, paw: {runstart_paw}")

        return runstart_frame

    def determine_initial_run_bound_start(self, r, run_bounds):
        utils_obj = utils.Utils()

        nose_present_mask = self.data.loc[self.trial_starts[r]:run_bounds[0], ('Nose', 'x')].notna()
        nose_present = self.data.loc[self.trial_starts[r]:run_bounds[0], ('Nose', 'x')][nose_present_mask]
        nose_present_chunks = utils_obj.find_blocks(nose_present.index, gap_threshold=15, block_min_size=10)

        if len(nose_present_chunks) > 0:
            run_bound_start = nose_present_chunks[-1][0]
        else:
            run_bound_start = self.trial_starts[r]
            print(f"No nose detected before run start in trial {r}")

        return run_bound_start

    def extract_and_combine_paw_data(self, run_bound_start, run_bound_end):
        paw_sides = ['Right', 'Left']
        paw_types = ['Forepaw', 'Hindpaw']
        combined_coords = {}

        for side in paw_sides:
            for paw in paw_types:
                # Define paw markers
                if paw == 'Forepaw':
                    paw_markers = [f'{paw}Toe{side[0]}', f'{paw}Knuckle{side[0]}', f'{paw}Ankle{side[0]}']
                else:
                    paw_markers = [f'{paw}Toe{side[0]}', f'{paw}Knuckle{side[0]}']
                # Extract data for the paw
                paw_data = self.data.loc[run_bound_start:run_bound_end, paw_markers]
                # Combine coordinates for 'x' and 'z'
                for coord in ['x', 'z']:
                    combined_coord = self.combine_paw_data(paw_data.loc(axis=1)[:, coord], sigma=1)
                    # Store in a dictionary for easy access
                    combined_coords[(paw, side, coord)] = combined_coord

        # Create a MultiIndex DataFrame for all paws
        paw_data_combined = pd.DataFrame(index=self.data.loc[run_bound_start:run_bound_end].index)
        for (paw, side, coord), data in combined_coords.items():
            paw_data_combined[(paw, side, coord)] = data

        return paw_data_combined

    def filter_paw_data_beyond_start_platform(self, paw_data_combined):
        paw_sides = ['Right', 'Left']
        x0_idx = paw_data_combined.index[
            np.all(
                [paw_data_combined[('Forepaw', side, 'x')] > 0 for side in paw_sides],
                axis=0
            )
        ]
        paw_data_x0 = paw_data_combined.loc[x0_idx]

        return paw_data_x0

    def detect_stance(self, paw_data_x0):
        utils_obj = utils.Utils()
        paw_sides = ['Right', 'Left']
        paw_types = ['Forepaw', 'Hindpaw']
        stance_events = {}
        #swing_events = {}

        for paw in paw_types:
            for side in paw_sides:
                key = (paw, side)
                combined_z = paw_data_x0[(paw, side, 'z')]
                combined_z_diff = combined_z.diff()

                combined_x = paw_data_x0[(paw, side, 'x')]
                combined_x_diff = combined_x.diff()

                # Detect stance: when z is low and stable
                stance_idx = paw_data_x0.index[
                    (combined_z < 1.5) & (combined_z_diff.abs() < 0.2)
                    ]

                # # Detect swing end: when z decreases significantly
                # swingend_idx = paw_data_x0.index[
                #     (combined_z > 1) & (combined_z_diff < -0.5)
                #     ]

                # Find chunks of stance and swing events
                if paw == 'Forepaw':
                    stance_chunks = utils_obj.find_blocks(stance_idx, gap_threshold=2, block_min_size=4)
                    #swing_chunks = utils_obj.find_blocks(swingend_idx, gap_threshold=3, block_min_size=1)
                else:
                    stance_chunks = utils_obj.find_blocks(stance_idx, gap_threshold=5, block_min_size=2)
                   # swing_chunks = utils_obj.find_blocks(swingend_idx, gap_threshold=4, block_min_size=4)

                # Calculate middle indices
                stance_middle = np.mean(stance_chunks, axis=1).astype(int)
                #swing_middle = np.mean(swing_chunks, axis=1).astype(int)

                # Store events
                stance_events[key] = stance_middle
                #swing_events[key] = swing_middle

        return stance_events #, swing_events

    def check_real_stance(self, stance_events):
        is_valid_paws = {}
        paw_sides = ['Right', 'Left']
        paw_types = ['Forepaw', 'Hindpaw']
        for paw in paw_types:
            for side in paw_sides:
                key = (paw, side)
                stances = stance_events.get(key, [])

                if len(stances) == 0:
                    is_valid_paws[key] = False
                    continue

                # check

                is_valid_paws[key] = is_valid

    def check_alternating_patterns(self, stance_events): #, swing_events):
        is_valid_paws = {}
        paw_sides = ['Right', 'Left']
        paw_types = ['Forepaw', 'Hindpaw']

        for paw in paw_types:
            for side in paw_sides:
                key = (paw, side)
                stances = stance_events.get(key, [])
                #swings = swing_events.get(key, [])

                if len(stances) == 0: # and len(swings) == 0:
                    is_valid_paws[key] = False
                    continue

                events = np.concatenate((
                    np.column_stack((stances, np.full(len(stances), 'stance'))),
                    #np.column_stack((swings, np.full(len(swings), 'swing')))
                ))

                # Sort events chronologically
                events_sorted = events[events[:, 0].astype(int).argsort()]
                # Check for alternating patterns
                is_valid = self.check_alternating(events_sorted)
                is_valid_paws[key] = is_valid

        return is_valid_paws

    def determine_run_start_frame(self, stance_events):
        # Collect the first stance frames from valid paws
        first_stance_frames = {key: stances[0] for key, stances in stance_events.items() if len(stances) > 0}

        if not first_stance_frames:
            raise ValueError("No stance events found for any valid paw")

        # Find the earliest stance event
        runstart_key = min(first_stance_frames, key=first_stance_frames.get)
        runstart_frame = first_stance_frames[runstart_key]
        runstart_paw = runstart_key

        return runstart_frame, runstart_key, runstart_paw

    def verify_paws_touch_down(self, stance_events, runstart_frame, runstart_key, max_allowed_gap=10):
        first_stance_frames = {key: stances[0] for key, stances in stance_events.items() if len(stances) > 0}
        for key, first_frame in first_stance_frames.items():
            if key != runstart_key:
                gap = first_frame - runstart_frame
                if gap < 0 or gap > max_allowed_gap:
                    raise ValueError(f"Paw {key} did not touch down shortly after the run start frame.")

    def perform_additional_checks(self, runstart_frame):
        nose_x = self.data.loc[runstart_frame, ('Nose', 'x')]
        tail_base_x = self.data.loc[runstart_frame, ('TailBase', 'x')]

        if np.isnan(nose_x) or np.isnan(tail_base_x):
            raise ValueError(f"Nose or tail base data missing at run start frame {runstart_frame}")

        if nose_x <= tail_base_x:
            raise ValueError("Nose is not ahead of tail base at run start frame.")

        if not (0 < tail_base_x < 10):
            raise ValueError(
                f"Tail base x position at run start frame {runstart_frame} is out of bounds: {tail_base_x}"
            )
    # def check_alternating(self, events):
    #     states = events[:, 1]
    #     # Check starting with 'swing'
    #     is_alternating_swing = all(
    #         states[i] != states[i + 1] for i in range(len(states) - 1)
    #     ) and states[0] == 'swing'
    #
    #     # Check starting with 'stance'
    #     is_alternating_stance = all(
    #         states[i] != states[i + 1] for i in range(len(states) - 1)
    #     ) and states[0] == 'stance'
    #
    #     return is_alternating_swing or is_alternating_stance

    # def combine_paw_data(self, paw_df, sigma=2, interpolation_method='spline', limit=10): #limit=100, interpolated well but big effect pre data
    #     # Identify rows where all values are NaN
    #     all_nan_rows = np.isnan(paw_df.values).all(axis=1)
    #     # Prepare an array to hold combined values
    #     combined_z = np.full(paw_df.shape[0], np.nan)
    #     # Compute nanmedian only for rows that are not all NaN
    #     if not all_nan_rows.all():
    #         combined_z[~all_nan_rows] = np.nanmedian(paw_df.values[~all_nan_rows], axis=1)
    #     # Create a pandas Series to handle interpolation easily
    #     combined_z_series = pd.Series(combined_z, index=paw_df.index)
    #     # Interpolate missing values (NaNs)
    #     interpolated_z = combined_z_series.interpolate(
    #         method=interpolation_method,
    #         order=3,
    #         limit=limit,
    #         limit_direction='backward'
    #     ).values
    #     # Apply Gaussian smoothing with the customizable sigma
    #     smoothed_z = gaussian_filter1d(interpolated_z, sigma=sigma)
    #     return smoothed_z

    ################################################# Helper functions #################################################
    def detect_paw_touchdown(data, paw_labels, x_min, x_max, z_tolerance):
        # Returns frames where paw touches down within x-range and z ≈ 0
        paw_z = data.loc[:, (paw_labels, 'z')]
        paw_x = data.loc[:, (paw_labels, 'x')]
        touchdown_frames = (paw_z.abs() < z_tolerance) & (paw_x > x_min) & (paw_x < x_max)
        return touchdown_frames.any(axis=1)

    def confirm_body_movement(data, frame_idx, frames_ahead, x_threshold):
        # Checks if nose and tail base move onto the belt within frames_ahead
        future_frames = data.index[frame_idx:frame_idx + frames_ahead]
        nose_x = data.loc[future_frames, ('Nose', 'x')]
        tail_base_x = data.loc[future_frames, ('TailBase', 'x')]
        return (nose_x > x_threshold).any() and (tail_base_x > x_threshold).any()

    def detect_run_backs(data, start_frame, end_frame, x_threshold):
        # Detects significant decreases in x-position indicating run backs
        x_positions = data.loc[start_frame:end_frame, ('Nose', 'x')]
        return ((x_positions.diff() < -x_threshold).any())










class GetAllFiles:
    def __init__(self, directory=None, overwrite=False):
        self.directory = directory
        self.overwrite = overwrite

    def GetFiles(self):
        files = utils.Utils().GetListofMappedFiles(self.directory)  # gets dictionary of side, front and overhead 3D files

        for j in range(0, len(files)):
            match = re.search(r'FAA-(\d+)', files[j])
            mouseID = match.group(1)
            pattern = "*%s*_Runs.h5" % mouseID
            dir = os.path.dirname(files[j])

            if not glob.glob(os.path.join(dir, pattern)) or self.overwrite:
                print(f"###############################################################"
                      f"\nFinding runs and extracting gait for {mouseID}...\n###############################################################")
                get_runs = GetRuns(files[j])
                runs = get_runs.get_runs() # todo update this to real end function
                # gait = GetGait(runs)??
            else:
                print(f"Data for {mouseID} already exists. Skipping...")

        print('All experiments have been mapped to real-world coordinates and saved.')


class GetConditionFiles:
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None, vmt_type=None,
                 vmt_level=None, prep=None, overwrite=False):
        self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep, self.overwrite = (
            exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level, prep, overwrite)

    def get_dirs(self):
        if self.speed:
            exp_speed_name = f"{self.exp}_{self.speed}"
        else:
            exp_speed_name = self.exp
        base_path = os.path.join(paths['filtereddata_folder'], exp_speed_name)

        # join any of the conditions that are not None in the order they appear in the function as individual directories
        conditions = [self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep]
        conditions = [c for c in conditions if c is not None]

        # if Repeats in conditions, add 'Wash' directory in the next position in the list
        if 'Repeats' in conditions:
            idx = conditions.index('Repeats')
            conditions.insert(idx + 1, 'Wash')
        condition_path = os.path.join(base_path, *conditions)

        if os.path.exists(condition_path):
            print(f"Directory found: {condition_path}")
        else:
            raise FileNotFoundError(f"No path found {condition_path}")

        # Recursively find and process the final data directories
        self._process_subdirectories(condition_path)

    def _process_subdirectories(self, current_path):
        """
        Recursively process directories and get to the final data directories.
        """
        subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]

        # If subdirectories exist, traverse deeper
        if len(subdirs) > 0:
            print(f"Subdirectories found in {current_path}: {subdirs}")
            for subdir in subdirs:
                full_subdir_path = os.path.join(current_path, subdir)
                # Recursively process subdirectory
                self._process_subdirectories(full_subdir_path)
        else:
            # No more subdirectories, assume this is the final directory with data
            print(f"Final directory: {current_path}")
            try:
                GetAllFiles(directory=current_path, overwrite=self.overwrite).GetFiles()
            except Exception as e:
                print(f"Error processing directory {current_path}: {e}")


def main():
    # Get all data
    # GetALLRuns(directory=directory).GetFiles()
    ### maybe instantiate first to protect entry point of my script
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1',
                          overwrite=False).get_dirs()

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # main(directory)
    main()