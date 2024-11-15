import numpy as np
import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import threading
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from scipy.signal import find_peaks
import joblib
from joblib import Parallel, delayed
import cv2
import time
import Helpers.utils as utils
from Helpers.Config_23 import *
from Preprocessing import GaitFeatureExtraction as gfe

class GetRuns:
    def __init__(self, file, debug_steps=False):
        self.file = file
        self.debug_steps = debug_steps
        self.model, self.label_encoders, self.feature_columns = self.load_model()
        self.trial_starts, self.trial_ends = [], []
        self.run_starts, self.run_ends_steps, self.run_ends, self.transitions = [], [], [], []
        self.buffer = 250

        if self.debug_steps:
            try:
                # Try to load self.data from a saved file
                filename = self.get_saved_data_filename()
                self.data = pd.read_hdf(filename, key='real_world_coords_steps')
                self.data_loaded_from_file = True
                print("Loaded self.data from saved file.")
            except FileNotFoundError:
                print("Saved data file not found. Recalculating...")
                self.data = self.get_data()
                self.data_loaded_from_file = False
        else:
            self.data = self.get_data()
            self.data_loaded_from_file = False

    def load_model(self):
        model_filename = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'limb_classification_model.pkl')
        model = joblib.load(model_filename)

        # Load label encoders
        label_encoders_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'label_encoders.pkl')
        label_encoders = joblib.load(label_encoders_path)

        # Load feature columns
        feature_columns_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'feature_columns.pkl')
        feature_columns = joblib.load(feature_columns_path)

        # Return model, label_encoders, and feature_columns
        return model, label_encoders, feature_columns

    def get_saved_data_filename(self):
        # Generate a consistent filename for saving/loading self.data
        base_dir = os.path.dirname(self.file)
        filename = os.path.basename(self.file).replace('.h5', '_steps.h5')
        return os.path.join(base_dir, filename)

    def save_data(self):
        filename = self.get_saved_data_filename()
        self.data.to_hdf(filename, key='real_world_coords_steps')
        print(f"Saved self.data to {filename}")

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
        self.find_steps()
        self.index_by_run()

        self.find_run_stages()

        return runs

    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------- Finding trials -------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------

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

    def index_by_run(self):
        # create multiindex for self.data with run number as level 0 and frame number as level 1
        run_idx = []
        for r, (start, end) in enumerate(zip(self.trial_starts, self.trial_ends)):
            run_idx.extend([r] * (end - start + 1))
        frame_idx = np.concatenate([np.arange(start, end + 1) for start, end in zip(self.trial_starts, self.trial_ends)])
        new_data_idx = pd.MultiIndex.from_arrays([run_idx, frame_idx], names=['Run', 'FrameIdx'])
        data_snippet = self.data.loc[frame_idx]
        data_snippet.index = new_data_idx
        self.data = data_snippet



    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------- Finding steps --------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    def find_steps(self):
        # Process each run in parallel
        Steps = []
        RunBounds = []
        Runbacks = []
        for r in range(len(self.trial_starts)):
            steps, run_bounds, runbacks = self.process_run(r)
            Steps.append(steps)
            RunBounds.append(run_bounds)
            Runbacks.append(runbacks)

        # Combine the steps from all runs
        StepsALL = pd.concat(Steps)
        StepsALL = StepsALL.reindex(self.data.index)
        for paw in StepsALL.columns:
            self.data[(paw, 'SwSt')] = StepsALL[paw].values

        # fill in 'run' column in self.data between run start and end values with 1's
        self.data['run'] = False
        for start, end in RunBounds:
                self.data.loc[start:end, 'running'] = True

        # fill in 'rb' column in self.data between runback start and end values with 1's
        self.data['rb'] = False
        for rb in Runbacks:
            for start, end in rb:
                self.data.loc[start:end, 'rb'] = True


    def process_run(self, r):
        try:
            # Create a copy of the data relevant to this trial
            trial_start = self.trial_starts[r]
            trial_end = self.trial_ends[r]
            run_data = self.data.loc[trial_start:trial_end].copy()

            # Pass run_data to methods that should operate within trial bounds
            run_bounds, runbacks = self.find_real_run_vs_rbs(r, run_data)

            if len(run_bounds) == 1:
                run_bounds = run_bounds[0]
                steps = self.classify_steps_in_run(r, run_bounds)
                print(f"Run {r} completed")
                return steps, run_bounds, runbacks
            else:
                raise ValueError("More than one run detected in a trial (in find_steps)")
        except Exception as e:
            print(f"Error processing run {r}: {e}")
            return pd.DataFrame()

    # def find_steps(self):
    #     Steps = []
    #     for r in range(len(self.trial_starts)):
    #         #steps = self.classify_steps_in_run(r, [self.trial_starts[r], self.trial_ends[r]])
    #         run_bounds, runbacks = self.find_real_run_vs_rbs(r) #todo#### CLASSIFY STEPS BEFORE THIS!!!! ####
    #         if len(run_bounds) == 1:
    #             run_bounds = run_bounds[0]
    #             steps = self.classify_steps_in_run(r, run_bounds)
    #             Steps.append(steps)
    #             print(f"Run {r} completed")
    #         else:
    #             raise ValueError("More than one run detected in a trial (in find_steps)")
    #     StepsALL = pd.concat(Steps)
    #     # transfer StepsALL to a df with the same index as self.data and nans for empty rows not included in StepsALL
    #     StepsALL = StepsALL.reindex(self.data.index)
    #     # add the steps to self.data
    #     self.data['Steps'] = StepsALL

    def show_steps_in_videoframes(self, view='Side'):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Create a list of runs
        num_runs = len(self.trial_starts)
        if num_runs == 0:
            raise ValueError("No runs available to display.")

        # Prepare run numbers for the dropdown menu
        run_numbers = [f"Run {i}" for i in range(num_runs)]  # Labels for runs

        # Load the video file
        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        elif len(video_files) == 0:
            raise ValueError("No video file found for the specified view.")
        else:
            video_file = video_files[0]

            # Initialize Tkinter window
        root = tk.Tk()
        root.title("Run Steps Visualization")

        # Create a frame for the dropdown menu
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)  # Use fill=tk.X to prevent vertical expansion

        # Create the dropdown menu
        selected_run = tk.StringVar(value=run_numbers[0])
        run_dropdown = ttk.OptionMenu(top_frame, selected_run, run_numbers[0], *run_numbers,
                                      command=lambda _: update_run())
        run_dropdown.pack(side=tk.LEFT, padx=1, pady=2)

        # Create a frame for the Matplotlib figure
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create Matplotlib figure and canvas
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=bottom_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize variables
        run_index = 0
        start_frame = int(self.trial_starts[run_index])
        end_frame = int(self.trial_ends[run_index])

        # Load frames for the initial run
        frames = self.load_frames(video_file, start_frame, end_frame)

        if not frames:
            raise ValueError("No frames were loaded. Cannot proceed with visualization.")

        # Set initial frame and index setup
        frame_index = 0
        actual_frame_number = start_frame + frame_index

        # Display the first frame
        frame_image = ax.imshow(frames[frame_index])
        ax.axis('off')

        # Add frame number text at the top
        frame_text = ax.text(0.5, 1.02, f'Run: {run_index}, Frame: {actual_frame_number}',
                             transform=ax.transAxes, ha='center', fontsize=12)

        # Define paws and limb boxes
        paws = ['HindpawR', 'HindpawL', 'ForepawR', 'ForepawL']
        limb_boxes = {}

        # Set limb box positions
        box_positions = {
            'HindpawR': [0.1, 0.05, 0.35, 0.08],
            'HindpawL': [0.1, 0.15, 0.35, 0.08],
            'ForepawR': [0.55, 0.05, 0.35, 0.08],
            'ForepawL': [0.55, 0.15, 0.35, 0.08],
        }

        for paw in paws:
            left, bottom, width, height = box_positions[paw]
            ax_box = fig.add_axes([left, bottom, width, height])
            rect = patches.Rectangle((0, 0), 1, 1, facecolor='grey')
            ax_box.add_patch(rect)
            ax_box.axis('off')
            ax_box.set_xlim(0, 1)
            ax_box.set_ylim(0, 1)
            ax_box.set_title(paw, fontsize=10)
            limb_boxes[paw] = rect

        # Create frame slider
        slider_ax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
        frame_slider = Slider(slider_ax, 'Frame', 0, len(frames) - 1, valinit=0, valfmt='%d')

        # Function to display the frame
        def display_frame():
            nonlocal frame_index, actual_frame_number
            frame_image.set_data(frames[frame_index])
            actual_frame_number = start_frame + frame_index
            frame_text.set_text(f'Run: {run_index}, Frame: {actual_frame_number}')
            self.update_limb_boxes(actual_frame_number, limb_boxes, paws)
            canvas.draw_idle()

        # Update functions
        def update_frame(val):
            nonlocal frame_index
            frame_index = int(frame_slider.val)
            display_frame()

        def update_run():
            nonlocal run_index, start_frame, end_frame, frames, frame_index, actual_frame_number
            run_index = run_numbers.index(selected_run.get())
            start_frame = int(self.trial_starts[run_index])
            end_frame = int(self.trial_ends[run_index])

            # Load frames for the selected run
            frames_new = self.load_frames(video_file, start_frame, end_frame)
            if not frames_new:
                print(f"No frames loaded for run {run_index}")
                return

            frames.clear()
            frames.extend(frames_new)
            frame_index = 0

            # Update frame slider
            frame_slider.valmin = 0
            frame_slider.valmax = len(frames) - 1
            frame_slider.set_val(0)
            frame_slider.ax.set_xlim(frame_slider.valmin, frame_slider.valmax)
            display_frame()

        # Connect slider to update function
        frame_slider.on_changed(update_frame)

        # Keyboard event handling
        def on_key_press(event):
            nonlocal frame_index
            if event.key == 'right':
                if frame_index < len(frames) - 1:
                    frame_index += 1
                    frame_slider.set_val(frame_index)
                    display_frame()
            elif event.key == 'left':
                if frame_index > 0:
                    frame_index -= 1
                    frame_slider.set_val(frame_index)
                    display_frame()

        # Connect the key press event
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Initial display
        display_frame()

        # Start Tkinter main loop
        root.mainloop()

    def load_frames(self, video_file, start_frame, end_frame):
        # Load frames from video file between start_frame and end_frame
        cap = cv2.VideoCapture(video_file)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def update_limb_boxes(self, actual_frame_number, limb_boxes, paws):
        if actual_frame_number in self.data.index:
            for paw in paws:
                try:
                    SwSt = self.data.loc[actual_frame_number, (paw, 'SwSt')]
                    # Map labels to colors
                    color_mapping = {'1.0': 'green', '0.0': 'red', 'unknown': 'grey'}
                    limb_boxes[paw].set_facecolor(color_mapping.get(SwSt, 'grey'))
                except KeyError:
                    limb_boxes[paw].set_facecolor('grey')
        else:
            for paw in paws:
                limb_boxes[paw].set_facecolor('grey')

    def classify_steps_in_run(self, r, run_bounds):
        # Compute start_frame and end_frame with buffer
        start_frame = run_bounds[0] - self.buffer
        end_frame = run_bounds[1] + self.buffer

        # Ensure frames are within self.data index range
        start_frame = max(start_frame, self.data.index.min())
        end_frame = min(end_frame, self.data.index.max())

        # Verify that frames exist in self.data
        if not set(range(start_frame, end_frame + 1)).issubset(self.data.index):
            print(f"Warning: Some frames from {start_frame} to {end_frame} are not in self.data index.")

        # Extract features from the paws directly from self.data
        paw_columns = [
            'ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
            'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
            'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
            'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL',
            'Nose', 'Tail1', 'Tail12'
        ]
        coords = ['x', 'z']

        # Fetch data from self.data to include buffer frames
        paw_data = self.data.loc[start_frame:end_frame, (paw_columns, coords)]
        paw_data.columns = ['{}_{}'.format(bp, coord) for bp, coord in paw_data.columns]

        # Interpolation
        interpolated = paw_data.interpolate(
            method='spline',
            order=3,
            limit=20,
            limit_direction='both',
            axis=0
        )

        # Smoothing
        data_array = interpolated.values
        all_nan_cols = np.isnan(data_array).all(axis=0)
        smoothed_array = np.empty_like(data_array)
        smoothed_array[:, all_nan_cols] = np.nan
        valid_cols = ~all_nan_cols
        smoothed_array[:, valid_cols] = gaussian_filter1d(
            data_array[:, valid_cols], sigma=2, axis=0, mode='nearest'
        )

        # Convert back to DataFrame
        smoothed_paw_data = pd.DataFrame(smoothed_array, index=paw_data.index, columns=paw_data.columns)

        # Restore MultiIndex columns
        smoothed_paw_data.columns = pd.MultiIndex.from_tuples(
            [tuple(col.rsplit('_', 1)) for col in smoothed_paw_data.columns],
            names=['bodyparts', 'coords']
        )

        # Proceed with feature extraction using smoothed_paw_data
        feature_extractor = gfe.FeatureExtractor(data=smoothed_paw_data, fps=fps)
        frames_to_process = smoothed_paw_data.index
        feature_extractor.extract_features(frames_to_process)
        features_df = feature_extractor.features_df

        # Ensure that features_df contains the same features used during training
        features_df = features_df[self.feature_columns]

        # Predict stance/swing/unknown
        stance_pred = self.model.predict(features_df)

        # Decode predictions using label_encoders
        decoded_predictions = {}
        paw_labels = ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']
        for idx, paw in enumerate(paw_labels):
            y_pred_col = stance_pred[:, idx]
            label_enc = self.label_encoders[paw]
            decoded_labels = label_enc.inverse_transform(y_pred_col)
            decoded_predictions[paw] = decoded_labels

        # Convert decoded predictions to DataFrame
        stance_pred_df = pd.DataFrame(decoded_predictions, index=features_df.index)

        # Plot stances using decoded labels
        self.plot_stances(r, stance_pred_df)

        # Return the DataFrame with decoded predictions
        return stance_pred_df

    def plot_stances(self, r, stance_pred_df):
        # Map labels to numerical values for plotting
        label_to_num = {'0.0': 0, '1.0': 1, 'unknown': np.nan}
        frames = np.arange(stance_pred_df.shape[0])  # X-axis: Frames or time steps

        # Create a figure with two subplots: one for forepaws and one for hindpaws
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot for Forepaws (ForepawR and ForepawL)
        for i, paw in enumerate(['ForepawR', 'ForepawL']):  # Loop through Forepaws
            y_values = stance_pred_df[paw].map(label_to_num)
            ax1.plot(frames, y_values, label=paw, marker='o')

        # Add labels, title, and legend for Forepaws
        ax1.set_ylabel('Stance/Swing')
        ax1.set_title('Stance/Swing Periods for Forepaws')
        ax1.legend()
        ax1.grid(True)

        # Plot for Hindpaws (HindpawR and HindpawL)
        for i, paw in enumerate(['HindpawR', 'HindpawL']):  # Loop through Hindpaws
            y_values = stance_pred_df[paw].map(label_to_num)
            ax2.plot(frames, y_values, label=paw, marker='o')

        # Add labels, title, and legend for Hindpaws
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Stance/Swing')
        ax2.set_title('Stance/Swing Periods for Hindpaws')
        ax2.legend()
        ax2.grid(True)

        # Show the plot
        plt.tight_layout()
        # Save plot to file
        plot_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'RunStances')
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f'stance_periods_run{r}.png'))
        # Close the plot
        plt.close()

    def visualize_run_steps(self, run_number, view='Front'):
        import matplotlib.pyplot as plt

        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        elif len(video_files) == 0:
            raise ValueError("No video file found for the specified view.")
        else:
            video_file = video_files[0]

        # Get start and end frames for the specified run
        start_frame = int(self.trial_starts[run_number])
        end_frame = int(self.trial_ends[run_number])

        # Open the video file and preload frames
        cap = cv2.VideoCapture(video_file)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError("No frames were loaded. Cannot proceed with visualization.")

        # Set initial frame and index setup
        frame_index = 0
        actual_frame_number = start_frame + frame_index

        # Initialize figure and axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.25)

        # Display the first frame
        frame_image = ax.imshow(frames[frame_index])
        ax.axis('off')

        # Add frame number text at the top
        frame_text = ax.text(0.5, 1.02, f'Frame: {actual_frame_number}', transform=ax.transAxes, ha='center', fontsize=12)

        # Define paws and limb boxes
        paws = ['ForepawL', 'HindpawL', 'ForepawR', 'HindpawR']
        limb_boxes = {}

        # Set limb box positions
        box_positions = {
            'ForepawL': [0.1, 0.05, 0.35, 0.08],
            'HindpawL': [0.1, 0.15, 0.35, 0.08],
            'ForepawR': [0.55, 0.05, 0.35, 0.08],
            'HindpawR': [0.55, 0.15, 0.35, 0.08],
        }

        for paw in paws:
            left, bottom, width, height = box_positions[paw]
            ax_box = fig.add_axes([left, bottom, width, height])
            rect = patches.Rectangle((0, 0), 1, 1, facecolor='grey')
            ax_box.add_patch(rect)
            ax_box.axis('off')
            ax_box.set_xlim(0, 1)
            ax_box.set_ylim(0, 1)
            ax_box.set_title(paw, fontsize=10)
            limb_boxes[paw] = rect

        # Create skip buttons with fixed delta values, using partial for callbacks
        button_positions = [(0.05 + i * 0.09, 0.9, 0.08, 0.04) for i in range(8)]
        button_labels = ['-1000', '-100', '-10', '-1', '+1', '+10', '+100', '+1000']
        button_deltas = [-1000, -100, -10, -1, 1, 10, 100, 1000]

        for pos, label, delta in zip(button_positions, button_labels, button_deltas):
            ax_button = fig.add_axes(pos)
            button = Button(ax_button, label)
            button.on_clicked(partial(self._update_frame_via_button, delta=delta))

        # Create slider above limb boxes
        ax_slider = plt.axes([0.15, 0.86, 0.7, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=frame_index, valfmt='%d')

        def update_frame(delta=None, frame_number=None):
            nonlocal frame_index, actual_frame_number
            if delta is not None:
                frame_index = max(0, min(frame_index + delta, len(frames) - 1))
            elif frame_number is not None:
                frame_index = frame_number
            actual_frame_number = start_frame + frame_index

            # Update the frame image
            frame_image.set_data(frames[frame_index])

            # Update limb boxes based on stance or swing
            if actual_frame_number in self.data.index:
                for paw in paws:
                    SwSt = self.data.loc[actual_frame_number, (paw, 'SwSt')]
                    if SwSt == 1:
                        limb_boxes[paw].set_facecolor('green')
                    elif SwSt == 0:
                        limb_boxes[paw].set_facecolor('red')
                    else:
                        limb_boxes[paw].set_facecolor('grey')
            else:
                for paw in paws:
                    limb_boxes[paw].set_facecolor('grey')

            # Update frame number display
            frame_text.set_text(f'Frame: {actual_frame_number}')

            # Synchronize slider
            slider.eventson = False
            slider.set_val(frame_index)
            slider.eventson = True

            # Refresh canvas
            fig.canvas.draw_idle()
            plt.pause(0.001)

        # Slider event to move frames
        def slider_update(val):
            frame_number = int(slider.val)
            update_frame(frame_number=frame_number)
        slider.on_changed(slider_update)

        # Set update_frame function to be accessible within _update_frame_via_button
        self.update_frame = update_frame

        plt.show()

    def _update_frame_via_button(self, delta, event=None):
        # Call update_frame with the delta value
        self.update_frame(delta=delta)

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
                raise ValueError("No backwards running detected in this snippet")
            i += 1

        true_run.append(forward_chunks[-1])

        return runbacks, true_run

    def find_real_run_vs_rbs(self, r, run_data):
        #run_data = self.data.loc(axis=0)[self.trial_starts[r]:self.trial_ends[r]]

        # Use run_data instead of self.data
        trial_start = self.trial_starts[r]
        trial_end = self.trial_ends[r]
        run_data = run_data.loc[trial_start:trial_end]

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

    #-------------------------------------------------------------------------------------------------------------------
    #------------------------------------------ Finding runstages ------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------


    def find_run_stages(self):
        runs = self.data.index.get_level_values('Run').unique()
        for r in runs:
            step_data, limb_data, paw_touchdown = self.get_run_data(r)
            self.find_run_start(r, paw_touchdown)
            self.find_run_end(r, paw_touchdown)
            self.find_transition(r, paw_touchdown, limb_data)
        self.create_runstage_index()
        self.find_taps(paw_touchdown, limb_data)

    def get_run_data(self, r):
        step_data = self.data.loc(axis=0)[r].loc(axis=1)[['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']]
        limb_data = self.data.loc(axis=0)[r].loc(axis=1)[['ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
                                                            'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
                                                            'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
                                                            'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL']]
        step_data = step_data.replace(['unknown','1', '0'], [np.nan, 1, 0])

        paw_labels=['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']
        # check when all 4 paws are first on the belt - need x postion of all paws to be > 0 and paws to be in stance
        paw_touchdown = self.detect_paw_touchdown(step_data, limb_data, paw_labels, x_threshold=0)

        return step_data, limb_data, paw_touchdown

    def find_touchdown_all4s(self, paw_touchdown, paw_labels, r, time):
        # check all touchdown during running phase
        # For each paw, find the first frame where touchdown occurs
        touchdownx4 = {}
        for paw in paw_labels:
            touchdown_series = paw_touchdown[paw]
            if touchdown_series.any():
                timed_touchdown_frame_mask = touchdown_series == True
                timed_touchdown_frame = []
                if time == 'first':
                    timed_touchdown_frame = touchdown_series[timed_touchdown_frame_mask].index[0]
                elif time == 'last':
                    timed_touchdown_frame = touchdown_series[timed_touchdown_frame_mask].index[-1]
                elif time == 'transition':
                    pass #todo for now
                touchdownx4[paw] = timed_touchdown_frame
            else:
                raise ValueError(f"No touchdown detected for {paw} in run {r}")
        if time == 'first':
            # Find last paw to touch down for first time
            timed_touchdown = max(touchdownx4.values())
        elif time == 'last':
            # Find first paw to touch down for last time
            timed_touchdown = min(touchdownx4.values())

        if self.data.loc[(r, timed_touchdown + 1),'running'] == False:
            raise ValueError(f"First touchdown for all paws in run {r} is not during running phase")

        return timed_touchdown

    def find_run_start(self, r, paw_touchdown):
        paw_labels = paw_touchdown.columns

        first_touchdown_x4 = self.find_touchdown_all4s(paw_touchdown, paw_labels, r, time='first')

        # Find the first touchdown frame for each paw leading up to first_touchdown_x4
        paw_first_touchdowns = {}
        for paw in paw_labels:
            # Get the touchdown Series for the paw
            touchdown_series = paw_touchdown[paw]

            # Extract frames where the paw is touching down before first_touchdown_x4
            paw_touchdown_frames = touchdown_series[touchdown_series.index < first_touchdown_x4]
            paw_touchdown_frames = paw_touchdown_frames[
                paw_touchdown_frames]  # Only frames where the paw is touching down

            # If there are no touchdown frames before first_touchdown_x4, skip
            if paw_touchdown_frames.empty:
                #print(f"No touchdown frames found for {paw} before frame {first_touchdown_x4}")
                continue

            # Use find_blocks to find continuous blocks of touchdown frames
            blocks = utils.Utils().find_blocks(paw_touchdown_frames.index, gap_threshold=5, block_min_size=0)

            # Find the block that ends at or just before first_touchdown_x4 for this paw #todo need something more sophisticated here, e.g. instead of looking for when there are all 4 paws in touchdown, look for 3 in touchdown and 1 (ie hindpaw) in swing and THEN look for the first touchdown of the forepaw preceding this
            block_found = False
            for block in reversed(blocks):
                if block[0] <= first_touchdown_x4:
                    # This block leads up to first_touchdown_x4
                    first_touchdown_frame = block[0]
                    paw_first_touchdowns[paw] = first_touchdown_frame
                    block_found = True
                    break  # Exit the loop once the block is found
            if not block_found:
                raise ValueError(f"No stepping sequence found leading up to {first_touchdown_x4} for {paw}, run {r}")

        if paw_first_touchdowns:
            # Find the earliest first touchdown frame among all paws
            earliest_first_touchdown_frame = min(paw_first_touchdowns.values())
            # Identify which paw(s) have this earliest frame
            initiating_paws = [paw for paw, frame in paw_first_touchdowns.items() if
                               frame == earliest_first_touchdown_frame]
            if np.logical_and(len(initiating_paws) == 1, 'Hind' not in initiating_paws[0]):
                self.run_starts.append(earliest_first_touchdown_frame)
                # adjust 'running' column to start at the first touchdown frame
                current_bound0 = self.data.loc(axis=0)[r].loc(axis=1)['running'].index[self.data.loc(axis=0)[r].loc(axis=1)['running'] == True][0]
                for f in range(earliest_first_touchdown_frame,current_bound0):
                    self.data.loc[(r,f),'running'] = True
        else:
            raise ValueError("No valid stepping sequences found leading up to the all-paws touchdown in run {r}")

    def find_run_end(self, r, paw_touchdown):
        paw_labels = paw_touchdown.columns

        # find frame where first paw touches down for the last time (end of stance)
        last_touchdown_x4 = self.find_touchdown_all4s(paw_touchdown, paw_labels, r, time='last')

        # find frame where mouse exits frame
        tail_data = self.data.loc(axis=0)[r, last_touchdown_x4:].loc(axis=1)[['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7',
                                                          'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12']]
        tail_present = tail_data[tail_data.any(axis=1)]
        tail_blocks = utils.Utils().find_blocks(tail_present.index.get_level_values('FrameIdx'), gap_threshold=100, block_min_size=20)
        run_end = tail_blocks[0][-1]

        # adjust 'running' column to end at the run end frame
        current_bound1 = self.data.loc(axis=0)[r].loc(axis=1)['running'].index[self.data.loc(axis=0)[r].loc(axis=1)['running'] == True][-1]
        for f in range(current_bound1,run_end):
            self.data.loc[(r,f),'running'] = True

        self.run_ends.append(run_end)
        self.run_ends_steps.append(last_touchdown_x4)

    def find_transition(self, r, paw_touchdown, limb_data):
        # find first forepaw touchdown frame on to second belt, which is after the gap at 470mm x position
        forepaw_labels = pd.unique([paw for paw in limb_data.columns.get_level_values(level=0) if 'Forepaw' in paw])

        paw_touchdown_snippet = paw_touchdown.loc[self.run_starts[r]:self.run_ends[r], ['ForepawR', 'ForepawL']]
        limb_data_snippet = limb_data.loc[self.run_starts[r]:self.run_ends[r], forepaw_labels]

        # get mean of each paw in paw_labels (comprised of toe, knuckle, ankle)
        limb_x = limb_data_snippet.loc(axis=1)[:, 'x'].droplevel('coords', axis=1)
        limb_x = limb_x.drop(columns=['ForepawKneeR', 'ForepawKneeL'])
        limb_x.columns = pd.MultiIndex.from_tuples([(col, col[-1]) for col in limb_x.columns], names=['bodyparts', 'side'])
        limb_x_mean = limb_x.groupby(axis=1,level='side').mean()

        post_transition_mask = limb_x_mean > 470
        transition_mask = np.logical_and(post_transition_mask.values, paw_touchdown_snippet.values)
        transition_frame = paw_touchdown_snippet[transition_mask].index[0]
        self.transitions.append(transition_frame)

    def create_runstage_index(self):
        # instantiate RunStage column
        self.data['RunStage'] = 'None'
        # Fill in the stages 'RunStart', 'Transition', 'RunEnd' within the running == True period:
        # runstart >= 'RunStart' > transition
        # transition >= 'Transition' > runend_steps
        # runend_steps > 'RunEnd'
        for r in self.data.index.get_level_values('Run').unique():
            run_start = self.run_starts[r]
            transition = self.transitions[r]
            run_end = self.run_ends[r]
            run_end_steps = self.run_ends_steps[r]
            for f in range(run_start, transition):
                self.data.loc[(r,f),'RunStage'] = 'RunStart'
            for f in range(transition, run_end_steps + 1):
                self.data.loc[(r,f),'RunStage'] = 'Transition'
            for f in range(run_end_steps + 1, run_end):
                self.data.loc[(r,f),'RunStage'] = 'RunEnd'
            # drop the rows after run_end for this run
            self.data.drop(index=self.data.loc[(r,run_end+1):].index, inplace=True)
        # make 'RunStage' part of index, ['Run', 'RunStage', 'FrameIdx']
        self.data.set_index('RunStage', append=True, inplace=True)
        self.data = self.data.reorder_levels(['Run', 'RunStage', 'FrameIdx'])

    def find_taps(self, paw_touchdown, limb_data):
        # find taps and can measure this as a duration of time where mouse has a paw either hovering or touching the belt without stepping
        pass







    ################################################# Helper functions #################################################
    def detect_paw_touchdown(self, step_data, limb_data, paw_labels, x_threshold):
        """
        Detects paw touchdown frames for each paw individually based on:
        - Paw is in stance phase (step_data).
        - Paw's x-position is greater than x_threshold (limb_data).

        Parameters:
        - step_data: DataFrame with paw stance/swing data for each paw (values are 1.0 for stance, 0.0 for swing).
        - limb_data: DataFrame with limb position data, must have x positions for paws.
        - paw_labels: List of paw labels to check (e.g., ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']).
        - x_threshold: The x-position threshold (e.g., 0) to check if paw has crossed the start line.

        Returns:
        - touchdown_frames: DataFrame with index as frames and columns as paw_labels, values are True/False indicating if conditions are met.
        """
        # Ensure indices are aligned
        common_index = step_data.index.intersection(limb_data.index)
        step_data = step_data.loc[common_index]
        limb_data = limb_data.loc[common_index]

        # Initialize a DataFrame to hold the touchdown status for each paw
        touchdown_frames = pd.DataFrame(index=common_index, columns=paw_labels)

        for paw in paw_labels:
            # Extract paw prefix and side
            if 'Forepaw' in paw:
                paw_prefix = 'Forepaw'
            elif 'Hindpaw' in paw:
                paw_prefix = 'Hindpaw'
            else:
                raise ValueError(f"Unknown paw label: {paw}")

            side = paw[-1]  # 'R' or 'L'

            # Check if paw is in stance (assuming stance is labeled as 1.0)
            in_stance = step_data[paw] == 1

            # Get markers for this paw
            markers = [col for col in limb_data.columns if col[0].startswith(paw_prefix) and col[0].endswith(side)]

            # Get x positions for the paw's markers
            x_positions = limb_data.loc[:, markers]

            # Calculate the mean x position for the paw
            x_mean = x_positions.xs('x', level='coords', axis=1).mean(axis=1)

            # Check if x position is greater than the threshold
            x_condition = x_mean > x_threshold

            # Combine conditions
            touchdown_frames[paw] = np.logical_and(in_stance.values.flatten(), x_condition.values.flatten())

        return touchdown_frames

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
    def __init__(self, directory=None, overwrite=False, debug_steps=False):
        self.directory = directory
        self.overwrite = overwrite
        self.debug_steps = debug_steps

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
                get_runs = GetRuns(files[j], self.debug_steps)
                runs = get_runs.get_runs() # todo update this to real end function
                # gait = GetGait(runs)??
            else:
                print(f"Data for {mouseID} already exists. Skipping...")

        print('All experiments have been mapped to real-world coordinates and saved.')


class GetConditionFiles:
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None, vmt_type=None,
                 vmt_level=None, prep=None, overwrite=False, debug_steps=False):
        self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep, self.overwrite, self.debug_steps = (
            exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level, prep, overwrite, debug_steps)

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
                GetAllFiles(directory=current_path, overwrite=self.overwrite, debug_steps=self.debug_steps).GetFiles()
            except Exception as e:
                print(f"Error processing directory {current_path}: {e}")


def main():
    # Get all data
    # GetALLRuns(directory=directory).GetFiles()
    ### maybe instantiate first to protect entry point of my script
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1',
                          overwrite=False, debug_steps=False).get_dirs()

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # main(directory)
    main()