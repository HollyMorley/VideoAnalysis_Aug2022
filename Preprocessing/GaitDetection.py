import numpy as np
import pandas as pd
import os, cv2, re
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button, RectangleSelector
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import tkinter for message boxes
import tkinter as tk
from tkinter import messagebox

# Custom Button class that doesn't change color on hover or click
class NoFlashButton(Button):
    def __init__(self, *args, **kwargs):
        super(NoFlashButton, self).__init__(*args, **kwargs)

    def on_enter(self, event):
        pass  # Do nothing to prevent hover color change

    def on_leave(self, event):
        pass  # Do nothing to prevent hover color change

    def on_press(self, event):
        pass  # Do nothing to prevent color change on press

    def on_release(self, event):
        if not self.eventson:
            return
        if self.ignore(event):
            return
        contains, attrd = self.ax.contains(event)
        if not contains:
            return
        for cid, func in self._clickobservers.items():
            func(event)

class ImageLabeler:
    def __init__(self, base_dir, subdirs_to_include, output_file):
        self.base_dir = base_dir
        self.subdirs_to_include = subdirs_to_include
        self.output_file = output_file
        self.image_files = self.load_image_files()
        self.current_index = 0
        self.labels = {}  # Store labels with (frame_num, subdir) as keys
        self.num_images = len(self.image_files)
        self.zoom_rect_selector = None  # For zoom functionality
        self.ax = None  # Will be set in label_images
        self.image_display = None # Will be set in label_images

        # Initialize Tkinter root (needed for message boxes)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

        # Load existing labels if the output file exists
        if os.path.exists(self.output_file):
            self.load_existing_labels()

    def load_image_files(self):
        image_files = []
        for subdir in self.subdirs_to_include:
            dir_path = os.path.join(self.base_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"Warning: Subdirectory {dir_path} does not exist.")
                continue
            files = [f for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.jpg')]
            for f in files:
                match = re.search(r'img(\d+)', f)  # Extract frame numbers from filenames
                if match:
                    frame_num = int(match.group(1))
                    image_files.append({
                        'frame_num': frame_num,
                        'filepath': os.path.join(dir_path, f),
                        'subdir': subdir  # Include subdirectory
                    })
        # Sort the images by frame number and subdirectory
        sorted_files = sorted(image_files, key=lambda x: (x['subdir'], x['frame_num']))
        return sorted_files

    def load_existing_labels(self):
        try:
            labels_df = pd.read_csv(self.output_file)
            for _, row in labels_df.iterrows():
                frame_num = row['Frame']
                subdir = row['Subdirectory']
                key = (frame_num, subdir)
                self.labels[key] = {
                    'HindpawL': bool(row['HindpawL']),
                    'ForepawL': bool(row['ForepawL']),
                    'HindpawR': bool(row['HindpawR']),
                    'ForepawR': bool(row['ForepawR']),
                }
            print(f"Loaded existing labels from {self.output_file}")
        except Exception as e:
            print(f"Failed to load existing labels: {e}")

    def load_image(self, index):
        image_info = self.image_files[index]
        image_path = image_info['filepath']
        frame_num = image_info['frame_num']
        image = plt.imread(image_path)  # Assuming the image is in a supported format
        return image, frame_num

    def load_current_labels(self):
        frame_info = self.image_files[self.current_index]
        frame_num = frame_info['frame_num']
        subdir = frame_info['subdir']
        key = (frame_num, subdir)
        if key in self.labels:
            self.limb_states = self.labels[key].copy()
        else:
            self.limb_states = {
                'HindpawL': False,
                'ForepawL': False,
                'HindpawR': False,
                'ForepawR': False
            }
        self.update_button_colors()

    def update_button_colors(self):
        # Update the text color to indicate selection
        self.hindpawL_button.label.set_color('green' if self.limb_states['HindpawL'] else 'red')
        self.forepawL_button.label.set_color('green' if self.limb_states['ForepawL'] else 'red')
        self.hindpawR_button.label.set_color('green' if self.limb_states['HindpawR'] else 'red')
        self.forepawR_button.label.set_color('green' if self.limb_states['ForepawR'] else 'red')
        # Redraw the buttons
        self.hindpawL_button.ax.figure.canvas.draw_idle()
        self.forepawL_button.ax.figure.canvas.draw_idle()
        self.hindpawR_button.ax.figure.canvas.draw_idle()
        self.forepawR_button.ax.figure.canvas.draw_idle()

    def label_images(self):
        # Before initializing Matplotlib figure, ask the user
        if self.labels:
            answer = messagebox.askyesno("Start Position", "Do you want to start from the first frame?")
            if answer:
                self.current_index = 0
            else:
                # Find the index of the last labeled frame in self.image_files
                labeled_indices = []
                for idx, image_info in enumerate(self.image_files):
                    key = (image_info['frame_num'], image_info['subdir'])
                    if key in self.labels:
                        labeled_indices.append(idx)
                if labeled_indices:
                    self.current_index = max(labeled_indices) + 1  # Start from the next frame after the last labeled
                    if self.current_index >= self.num_images:
                        self.current_index = self.num_images - 1  # Ensure index is within bounds
                else:
                    self.current_index = 0  # No labeled frames found
        else:
            self.current_index = 0

        # Initialize Matplotlib figure and axis
        fig, self.ax = plt.subplots(figsize=(16, 9))
        # Adjust the subplot to make room at the bottom
        plt.subplots_adjust(left=0, bottom=0.35, right=1, top=1, wspace=0, hspace=0)

        # Initialize limb states for the current image
        self.limb_states = {
            'HindpawL': False,
            'ForepawL': False,
            'HindpawR': False,
            'ForepawR': False
        }

        # Create navigation buttons at the bottom
        bprev = Button(plt.axes([0.3, 0.05, 0.1, 0.05]), 'Prev')
        bnext = Button(plt.axes([0.6, 0.05, 0.1, 0.05]), 'Next')

        # Add zoom and reset view buttons
        zoom_button = Button(plt.axes([0.05, 0.05, 0.1, 0.05]), 'Zoom')
        reset_button = Button(plt.axes([0.15, 0.05, 0.1, 0.05]), 'Reset View')

        # Add save button
        save_button = Button(plt.axes([0.85, 0.05, 0.1, 0.05]), 'Save')

        # Create paw buttons below the image in specified positions
        button_width = 0.15
        button_height = 0.05
        button_spacing_x = 0.05
        button_spacing_y = 0.05

        # Calculate starting positions to center the buttons
        start_x = (1 - (button_width * 2 + button_spacing_x)) / 2
        start_y_top = 0.35  # Adjusted Y-position for the top row
        start_y_bottom = start_y_top - button_height - button_spacing_y  # Y-position for the bottom row

        # Define button positions
        hindpawL_button_ax = plt.axes([start_x, start_y_top, button_width, button_height])
        forepawL_button_ax = plt.axes([start_x + button_width + button_spacing_x, start_y_top, button_width, button_height])
        hindpawR_button_ax = plt.axes([start_x, start_y_bottom, button_width, button_height])
        forepawR_button_ax = plt.axes([start_x + button_width + button_spacing_x, start_y_bottom, button_width, button_height])

        # Create custom buttons that do not change color on click or hover and assign to self
        self.hindpawL_button = NoFlashButton(hindpawL_button_ax, 'HindpawL', color='lightgrey', hovercolor='grey')
        self.forepawL_button = NoFlashButton(forepawL_button_ax, 'ForepawL', color='lightgrey', hovercolor='grey')
        self.hindpawR_button = NoFlashButton(hindpawR_button_ax, 'HindpawR', color='lightgrey', hovercolor='grey')
        self.forepawR_button = NoFlashButton(forepawR_button_ax, 'ForepawR', color='lightgrey', hovercolor='grey')

        # Now, load labels for the current image if available
        self.load_current_labels()

        # Load the current image
        image, frame_num = self.load_image(self.current_index)
        self.image_display = self.ax.imshow(image)  # Use self.image_display here
        self.ax.axis('off')  # Hide axes ticks

        # Function to update the image display
        def update_image():
            image, frame_num = self.load_image(self.current_index)
            if self.image_display is None:
                # This should not happen, but handle it just in case
                self.image_display = self.ax.imshow(image)
            else:
                self.image_display.set_data(image)
            self.ax.set_xlim(0, image.shape[1])
            self.ax.set_ylim(image.shape[0], 0)
            frame_info = self.image_files[self.current_index]
            subdir = frame_info['subdir']
            self.ax.set_title(f"Frame {frame_num} ({self.current_index + 1}/{self.num_images})\nSubdirectory: {subdir}", fontsize=12)

            # Update button colors based on limb states
            self.update_button_colors()
            plt.draw()

        # Handlers for paw buttons
        def toggle_hindpawL(event):
            self.limb_states['HindpawL'] = not self.limb_states['HindpawL']
            self.update_button_colors()

        def toggle_forepawL(event):
            self.limb_states['ForepawL'] = not self.limb_states['ForepawL']
            self.update_button_colors()

        def toggle_hindpawR(event):
            self.limb_states['HindpawR'] = not self.limb_states['HindpawR']
            self.update_button_colors()

        def toggle_forepawR(event):
            self.limb_states['ForepawR'] = not self.limb_states['ForepawR']
            self.update_button_colors()

        self.hindpawL_button.on_clicked(toggle_hindpawL)
        self.forepawL_button.on_clicked(toggle_forepawL)
        self.hindpawR_button.on_clicked(toggle_hindpawR)
        self.forepawR_button.on_clicked(toggle_forepawR)

        # Handlers for next and previous image buttons
        def next_image(event):
            self.save_current_labels()
            if self.current_index < self.num_images - 1:
                self.current_index += 1
                self.load_current_labels()
                update_image()

        def prev_image(event):
            self.save_current_labels()
            if self.current_index > 0:
                self.current_index -= 1
                self.load_current_labels()
                update_image()

        bnext.on_clicked(next_image)
        bprev.on_clicked(prev_image)

        # Save labels based on the limb states
        def save_labels_callback(event):
            self.save_current_labels()
            try:
                self.save_labels()
                # Show success message
                messagebox.showinfo("Save Successful", f"Labels saved to {self.output_file}")
            except Exception as e:
                # Show error message
                messagebox.showerror("Save Failed", f"Failed to save labels:\n{str(e)}")

        save_button.on_clicked(save_labels_callback)

        # Zoom functionality
        def zoom_callback(event):
            if self.zoom_rect_selector is None:
                self.zoom_rect_selector = RectangleSelector(self.ax, onselect, drawtype='box',
                                                            useblit=True, button=[1],
                                                            minspanx=5, minspany=5, spancoords='pixels',
                                                            interactive=True)
            else:
                self.zoom_rect_selector.set_active(True)

        zoom_button.on_clicked(zoom_callback)

        # Reset view functionality

        def reset_callback(event):
            if self.zoom_rect_selector is not None:
                self.zoom_rect_selector.set_active(False)
            if self.image_display is not None:
                self.ax.set_xlim(0, self.image_display.get_array().shape[1])
                self.ax.set_ylim(self.image_display.get_array().shape[0], 0)
                plt.draw()

        reset_button.on_clicked(reset_callback)

        # Function to handle zoom area selection
        def onselect(eclick, erelease):
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_max, y_min)
            plt.draw()
            self.zoom_rect_selector.set_active(False)

        # Function to save current limb states
        def save_current_labels():
            frame_info = self.image_files[self.current_index]
            frame_num = frame_info['frame_num']
            subdir = frame_info['subdir']
            key = (frame_num, subdir)
            self.labels[key] = self.limb_states.copy()

        self.save_current_labels = save_current_labels  # Make accessible outside label_images

        # Initial display
        update_image()

        plt.show()
        # After the GUI is closed, destroy the Tkinter root
        self.root.destroy()

    def save_labels(self):
        # Convert self.labels dict to DataFrame
        labels_list = []
        for key, limb_states in self.labels.items():
            frame_num, subdir = key
            row = {
                'Frame': frame_num,
                'Subdirectory': subdir  # Include subdirectory
            }
            # Convert booleans to integers (1 for True, 0 for False)
            row.update({k: int(v) for k, v in limb_states.items()})
            labels_list.append(row)
        labels_df = pd.DataFrame(labels_list)
        # Remove duplicates, keeping the last entry
        labels_df = labels_df.drop_duplicates(subset=['Frame', 'Subdirectory'], keep='last')
        # Sort labels by subdir and frame number
        labels_df = labels_df.sort_values(['Subdirectory', 'Frame'])
        # Save to CSV
        labels_df.to_csv(self.output_file, index=False)
        print(f"Labels saved to {self.output_file}")


class FeatureExtractor:
    def __init__(self, data_file, fps=30):
        self.data_file = data_file
        self.fps = fps  # Frames per second of the video/data
        self.data = self.load_data()
        self.features_df = None

    def load_data(self):
        # Load data from HDF5 file or any other format
        data = pd.read_hdf(self.data_file, key='real_world_coords')
        data.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in data.columns],
                                                 names=['bodyparts', 'coords'])
        return data

    def extract_features(self, frames_to_process):
        # Define joints and paws
        joints = ['Toe', 'Knuckle', 'Ankle', 'Knee']
        paws = {
            'ForepawR': {'Toe': 'ForepawToeR', 'Knuckle': 'ForepawKnuckleR', 'Ankle': 'ForepawAnkleR', 'Knee': 'ForepawKneeR'},
            'ForepawL': {'Toe': 'ForepawToeL', 'Knuckle': 'ForepawKnuckleL', 'Ankle': 'ForepawAnkleL', 'Knee': 'ForepawKneeL'},
            'HindpawR': {'Toe': 'HindpawToeR', 'Knuckle': 'HindpawKnuckleR', 'Ankle': 'HindpawAnkleR', 'Knee': 'HindpawKneeR'},
            'HindpawL': {'Toe': 'HindpawToeL', 'Knuckle': 'HindpawKnuckleL', 'Ankle': 'HindpawAnkleL', 'Knee': 'HindpawKneeL'}
        }
        coords = ['x', 'z']
        time_offsets = [-5, 0, 5]
        delta_t = 1 / self.fps  # Time difference between consecutive frames

        features_list = []
        indices = self.data.index

        # Process only the frames that have been labeled
        frames_set = set()
        for frame in frames_to_process:
            frames_set.update([frame + offset for offset in time_offsets])

        frames_to_extract = sorted(frames_set)

        for idx in frames_to_extract:
            frame_features = {'Frame': idx}
            for offset in time_offsets:
                t = idx + offset
                if t in indices:
                    # For velocity calculation at time t + offset
                    t_minus = t - 1
                    t_plus = t + 1
                    for paw_name, paw_joints in paws.items():
                        for joint in joints:
                            joint_label = paw_joints.get(joint)
                            if joint_label is None:
                                continue  # Skip if joint is not available

                            # Position features at time t + offset
                            for coord in coords:
                                pos = self.data.loc[t, (joint_label, coord)]
                                feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
                                frame_features[feature_name] = pos

                            # Velocity features at time t + offset
                            if t_minus in indices and t_plus in indices:
                                pos_minus = self.data.loc[t_minus, (joint_label, coord)]
                                pos_plus = self.data.loc[t_plus, (joint_label, coord)]
                                velocity = (pos_plus - pos_minus) / (2 * delta_t)
                                velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                frame_features[velocity_feature_name] = velocity
                            else:
                                # Assign NaN if neighboring frames are not available
                                velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                frame_features[velocity_feature_name] = np.nan

                        # Angle features at time t + offset
                        for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
                            joint1_label = paw_joints.get(angle_joints[0])
                            joint2_label = paw_joints.get(angle_joints[1])
                            joint3_label = paw_joints.get(angle_joints[2])

                            if joint1_label and joint2_label and joint3_label:
                                coord1 = self.data.loc[t, (joint1_label, ['x', 'z'])].values.astype(float)
                                coord2 = self.data.loc[t, (joint2_label, ['x', 'z'])].values.astype(float)
                                coord3 = self.data.loc[t, (joint3_label, ['x', 'z'])].values.astype(float)

                                angle = self.calculate_angle(coord1, coord2, coord3)
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = angle
                            else:
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = np.nan
                else:
                    # Assign NaN for all features at time t + offset if t + offset is out of bounds
                    for paw_name, paw_joints in paws.items():
                        for joint in joints:
                            if paw_joints.get(joint) is None:
                                continue
                            for coord in coords:
                                feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
                                frame_features[feature_name] = np.nan
                                velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
                                frame_features[velocity_feature_name] = np.nan
                            for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
                                angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
                                frame_features[angle_feature_name] = np.nan

            features_list.append(frame_features)

        # Convert list of dicts to DataFrame
        self.features_df = pd.DataFrame(features_list).set_index('Frame')

    def calculate_angle(self, point1, point2, point3):
        # Calculate angle at point2 between point1 and point3
        vector1 = point1 - point2
        vector2 = point3 - point2
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            return np.nan
        cos_theta = np.dot(vector1, vector2) / norm_product
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

def main():
    base_directory = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side"
    # List of subdirectories to include
    subdirectories_to_include = [
        "HM_20230309_APACharRepeat_FAA-1035297_R_side_1",
        "HM_20230306_APACharRepeat_FAA-1035244_L_side_1",
        "HM_20230307_APAChar_FAA-1035302_LR_side_1", # no files present, not a real experiment *** Have added in to analyse temporarily
        "HM_20230308_APACharRepeat_FAA-1035244_L_side_1",
        "HM_20230319_APACharExt_FAA-1035245_R_side_1",
        "HM_20230326_APACharExt_FAA-1035246_LR_side_1",
        "HM_20230404_APACharExt_FAA-1035299_None_side_1" # files present but not been mapped
    ]
    output_file = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff\limb_labels.csv"

    # Pass the output_file to the ImageLabeler instance
    labeler = ImageLabeler(base_directory, subdirectories_to_include, output_file)
    if labeler.num_images == 0:
        print("No images found in the specified subdirectories.")
        return

    labeler.label_images()


if __name__ == '__main__':
    main()
