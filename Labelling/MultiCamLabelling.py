'''
Obj: make labelling gui for labelling body parts in three camera views simultaneously and using the camera positions to
guide labelling across the three views

Skeleton:
-  Create a main menu GUI with option to a) extract frames from videos, and b) label frames
-  Frame extraction:
    -  Select video file (always the side view)
    -  Create GUI with all three camera views (for each selected side view) visible and a single slider for moving through frames
    -  Buttons to a) calibrate camera positions, b) extract frames
    -  Calibrate camera positions:
        -  Label the points: "StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR", "Nose"
            -  Can slide between frames while labelling and these labeled points will remain static to check they fit across the video
        -  Save the labeled points
            -  Path: "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration\[video_name]\calibration_labels.csv"
        -  Use the labeled points to calculate the camera positions
            -  Using CalibrateCams.py
                - calib = BasicCalibration(labeled_coordinates)
                - cameras_extrinsics = calib.estimate_cams_pose()
                - cameras_intrisinics = calib.cameras_intrinsics
            -  Save the camera calibration data
                -  Path: "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration\[video_name]\calibration_matrices.csv"
    -  Extract frames:
        -  Choose frames using a slider and extract button
        -  Save extracted frames to a folder (predefined directory in the code)
            - Paths:
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side\[video_name]\img[frame number].png"
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Front\[video_name]\img[frame number].png"
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Overhead\[video_name]\img[frame number].png"
-  Frame labelling:
    -  Select the extracted frames folder
        -  Raise an error if there is no corresponding camera calibration data
    -  With all three camera views visible, label body parts one by one in each extracted frame
        -  Toggle through frames with 'Next frame' and 'Previous frame' buttons
        -  Have a list/menu of body parts in the GUI window with you can toggle between
        -  Bodyparts: "Nose", "EarL", "EarR", "Back1",..., "Back12", "Tail1", ..., "Tail12", "ForepawToeR",
            "ForepawAnkleR", "ForepawToeL", "ForepawAnkleL", "HindpawToeR", "HindpawAnkleR", "HindpawToeL", "HindpawAnkleL"
    -  As label is placed in one view, a red line is plotted in the other two views representing the projection of
        that label in the other views to guide labelling
        - The red line should be updated as the label is moved
        - Reprojection on to each other view is done using the camera calibration data
        - Have toggle buttons to signify which view to base the red line off of, e.g. if side is selected, this shows the
            line of sight from the side camera to the current labeled point in the front and overhead views
            - Buttons: "Side", "Front", "Overhead", "None" (for no red line)
    -  Controls for labelling are:
        -  Right-click to place a point
        -  Shift + Right-click to delete the nearest point
        -  Left-click and drag to move a point
        -  Buttons to open/save images and close the labelling tool
    -  Save the labelled frames to a folder (predefined directory in the code)
        - Paths:
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side\[video_name]\CollectedData_Holly.csv"
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Front\[video_name]\CollectedData_Holly.csv"
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Overhead\[video_name]\CollectedData_Holly.csv"
'''

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageTk, ImageEnhance

# import configurations
import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration

class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Parts Labeling Tool")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.calibration_points_static = {}

        self.crosshair_lines = []
        self.dragging_point = None
        self.panning = False
        self.pan_start = None

        self.marker_size = config.DEFAULT_MARKER_SIZE
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)

        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        extract_button = tk.Button(main_frame, text="Extract Frames from Videos", command=self.extract_frames_menu)
        extract_button.pack(pady=5)

        label_button = tk.Button(main_frame, text="Label Frames", command=self.label_frames_menu)
        label_button.pack(pady=5)

    def extract_frames_menu(self):
        self.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        self.calibration_file_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)

        extract_frame = tk.Frame(self.root)
        extract_frame.pack(pady=20)

        calibration_button = tk.Button(extract_frame, text="Calibrate Camera Positions", command=self.calibrate_cameras)
        calibration_button.pack(pady=5)

        extract_button = tk.Button(extract_frame, text="Extract Frames", command=self.extract_frames)
        extract_button.pack(pady=5)

        back_button = tk.Button(extract_frame, text="Back to Main Menu", command=self.main_menu)
        back_button.pack(pady=5)

    def load_calibration_points(self):
        try:
            df = pd.read_csv(self.calibration_file_path)
            df.set_index(["bodyparts", "coords"], inplace=True)
            for label in df.index.levels[0]:
                for view in ["side", "front", "overhead"]:
                    if not pd.isna(df.loc[(label, 'x'), view]):
                        x, y = df.loc[(label, 'x'), view], df.loc[(label, 'y'), view]
                        self.calibration_points_static[label][view] = self.axs[
                            ["side", "front", "overhead"].index(view)].scatter(
                            x, y, c=self.label_colors[label], s=self.marker_size * 10, label=label
                        )
                        self.calibration_points[label][view] = (x, y)
            messagebox.showinfo("Info", "Calibration points loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration points: {e}")

    def calibrate_cameras(self):
        self.clear_root()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(frame_control, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=400,
                               command=self.show_frames)
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Calibration Points", command=self.save_calibration_points)
        save_button.pack(pady=5)

        back_button = tk.Button(button_frame, text="Back to Extract Frames Menu", command=self.extract_frames_menu)
        back_button.pack(pady=5)

        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.labels = config.CALIBRATION_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.calibration_points = {label: {"side": None, "front": None, "overhead": None} for label in self.labels}

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            color_box = tk.Label(label_frame, bg=color, width=2)
            color_box.pack(side=tk.LEFT, padx=5)
            label_button = tk.Radiobutton(label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20)
            label_button.pack(side=tk.LEFT)

        self.current_view = tk.StringVar(value="side")
        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(control_frame_right, text=view.capitalize(), variable=self.current_view, value=view).pack(
                pady=2)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in
                                          self.labels}

        if os.path.exists(self.calibration_file_path):
            if messagebox.askyesno("Calibration Found", "Calibration labels found. Do you want to load them?"):
                self.load_calibration_points()
        else:
            default_calibration_file = config.DEFAULT_CALIBRATION_FILE_PATH
            if os.path.exists(default_calibration_file):
                if messagebox.askyesno("Default Calibration",
                                       "No specific calibration file found. Do you want to load the default calibration labels?"):
                    self.calibration_file_path = default_calibration_file
                    self.load_calibration_points()

        self.show_frames()

    def on_scroll(self, event):
        ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        zoom_factor = 0.9 if event.button == 'up' else 1.1  # Reverse the zoom direction

        new_xlim = [xdata + (x - xdata) * zoom_factor for x in xlim]
        new_ylim = [ydata + (y - ydata) * zoom_factor for y in ylim]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == MouseButton.MIDDLE:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == MouseButton.MIDDLE:
            self.panning = False
            self.pan_start = None

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)

            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]

            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)  # Reverse the panning direction for y-axis

            self.canvas.draw_idle()

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def update_frame_label(self, val):
        frame_number = int(val)
        self.frame_label.config(text=f"Frame: {frame_number}")
        self.show_frames()

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts)
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view)
        return os.path.join(base_path, corresponding_file)

    def update_marker_size(self, val):
        self.marker_size = self.marker_size_var.get()
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    point.set_sizes([self.marker_size * 10])
        self.canvas.draw()

    def update_contrast_brightness(self, val):
        self.show_frames()

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        label = self.current_label.get()
        color = self.label_colors[label]
        marker_size = self.marker_size_var.get()

        if event.button == MouseButton.RIGHT:
            if event.key == 'shift':
                self.delete_closest_point(ax, event)
            else:
                if self.calibration_points_static[label][view] is not None:
                    self.calibration_points_static[label][view].remove()
                self.calibration_points_static[label][view] = ax.scatter(event.xdata, event.ydata, c=color,
                                                                         s=marker_size * 10, label=label)
                self.canvas.draw()
                self.advance_label()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event)

    def find_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point_label = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_label = label

        if closest_point_label:
            self.calibration_points_static[closest_point_label][self.current_view.get()].remove()
            self.calibration_points_static[closest_point_label][self.current_view.get()] = None
            self.canvas.draw()

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]

        if event.button == MouseButton.LEFT:
            self.dragging_point.set_offsets((event.xdata, event.ydata))
            self.canvas.draw()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []

        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color='cyan', linestyle='--', linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color='cyan', linestyle='--', linewidth=0.5))
            self.canvas.draw_idle()

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.labels) == 1:
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set('')  # No more labels to advance to

    def save_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            for coord in ['x', 'y']:
                data["bodyparts"].append(label)
                data["coords"].append(coord)
                for view in ["side", "front", "overhead"]:
                    if coords[view] is not None:
                        x, y = coords[view].get_offsets()[0]
                        if coord == 'x':
                            data[view].append(x)
                        else:
                            data[view].append(y)
                    else:
                        data[view].append(None)

        df = pd.DataFrame(data)
        df.to_csv(calibration_path, index=False)

        messagebox.showinfo("Info", "Calibration points saved successfully")

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: self.rgb_to_hex(color) for label, color in zip(labels, colors)}

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    def extract_frames(self):
        self.clear_root()

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.frame_label = tk.Label(control_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        self.slider = tk.Scale(control_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=600,
                               command=self.show_frames_extraction)
        self.slider.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(side=tk.TOP, pady=10)
        self.add_skip_buttons(skip_frame)

        control_frame_right = tk.Frame(self.root)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        extract_button = tk.Button(control_frame_right, text="Extract Frames", command=self.save_extracted_frames)
        extract_button.pack(pady=5)

        back_button = tk.Button(control_frame_right, text="Back to Extract Frames Menu",
                                command=self.extract_frames_menu)
        back_button.pack(pady=5)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.show_frames_extraction()

    def save_extracted_frames(self):
        frame_number = self.slider.get()
        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            side_path = config.FRAME_SAVE_PATH_TEMPLATE["side"].format(video_name=self.video_name,
                                                                       frame_number=frame_number)
            front_path = config.FRAME_SAVE_PATH_TEMPLATE["front"].format(video_name=self.video_name,
                                                                         frame_number=frame_number)
            overhead_path = config.FRAME_SAVE_PATH_TEMPLATE["overhead"].format(video_name=self.video_name,
                                                                               frame_number=frame_number)

            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            os.makedirs(os.path.dirname(front_path), exist_ok=True)
            os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

            cv2.imwrite(side_path, frame_side)
            cv2.imwrite(front_path, frame_front)
            cv2.imwrite(overhead_path, frame_overhead)

    def show_frames_extraction(self, val=None):
        frame_number = self.slider.get()

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.canvas.draw()

    def show_frames(self, val=None):
        frame_number = self.slider.get()

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side = self.apply_contrast_brightness(frame_side)
            frame_front = self.apply_contrast_brightness(frame_front)
            frame_overhead = self.apply_contrast_brightness(frame_overhead)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.show_static_points()

            self.canvas.draw()

    def show_static_points(self):
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size * 10])
        self.canvas.draw()

    def scroll_y(self, ax, *args):
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - float(args[1]) * 0.1, ylim[1] - float(args[1]) * 0.1)
        self.canvas.draw_idle()

    def scroll_x(self, ax, *args):
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] + float(args[1]) * 0.1, xlim[1] + float(args[1]) * 0.1)
        self.canvas.draw_idle()

    def update_scrollregion(self, event):
        for ax in self.axs:
            ax.set_xlim(0, self.canvas.get_width_height()[0])
            ax.set_ylim(0, self.canvas.get_width_height()[1])
        self.canvas.get_tk_widget().configure(scrollregion=self.canvas.get_tk_widget().bbox("all"))

    def zoom_in(self):
        ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([xlim[0] + (xlim[1] - xlim[0]) * 0.1, xlim[1] - (xlim[1] - xlim[0]) * 0.1])
        ax.set_ylim([ylim[0] + (ylim[1] - ylim[0]) * 0.1, ylim[1] - (ylim[1] - ylim[0]) * 0.1])
        self.canvas.draw()

    def zoom_out(self):
        ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([xlim[0] - (xlim[1] - xlim[0]) * 0.1, xlim[1] + (xlim[1] - xlim[0]) * 0.1])
        ax.set_ylim([ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1] + (ylim[1] - ylim[0]) * 0.1])
        self.canvas.draw()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.show_frames()

    def skip_frames(self, step):
        new_frame_number = self.slider.get() + step
        new_frame_number = max(0, min(new_frame_number, self.total_frames - 1))
        self.slider.set(new_frame_number)
        self.show_frames()

    def label_frames_menu(self):
        self.clear_root()
        calibration_folder_path = filedialog.askdirectory(title="Select Calibration Folder")

        if not calibration_folder_path:
            return

        self.video_name = os.path.basename(calibration_folder_path)
        self.video_date = self.extract_date_from_folder_path(calibration_folder_path)
        self.calibration_file_path = os.path.join(calibration_folder_path, "calibration_labels.csv")

        if not os.path.exists(self.calibration_file_path):
            messagebox.showerror("Error", "No corresponding camera calibration data found.")
            return

        self.extracted_frames_path = {
            'side': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{self.video_name}",
            'front': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{self.video_name}",
            'overhead': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{self.video_name}"
        }

        if not all(os.path.exists(path) for path in self.extracted_frames_path.values()):
            messagebox.showerror("Error", "One or more corresponding extracted frames folders not found.")
            return

        self.current_frame_index = 0

        # prompt to inform user that the frames will be loaded
        messagebox.showinfo("Info", "Loading frames...")
        self.load_frames()
        messagebox.showinfo("Info", "Frames loaded successfully")
        self.load_calibration_data(self.calibration_file_path)
        self.setup_labeling_ui()

    def extract_date_from_folder_path(self, folder_path):
        parts = folder_path.split(os.sep)
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return part
        return None

    def load_frames(self):
        self.frames = {'side': [], 'front': [], 'overhead': []}
        for view in self.frames.keys():
            frame_files = sorted(os.listdir(self.extracted_frames_path[view]))
            self.frames[view] = [cv2.imread(os.path.join(self.extracted_frames_path[view], file)) for file in
                                 frame_files]

        # Ensure all views have the same number of frames
        min_frame_count = min(len(self.frames[view]) for view in self.frames)
        for view in self.frames:
            self.frames[view] = self.frames[view][:min_frame_count]

    def load_calibration_data(self, calibration_data_path):
        try:
            calibration_coordinates = pd.read_csv(calibration_data_path)

            calib = BasicCalibration(calibration_coordinates)
            cameras_extrinsics = calib.estimate_cams_pose()
            cameras_intrisinics = calib.cameras_intrinsics

            self.calibration_data = {
                'extrinsics': cameras_extrinsics,
                'intrinsics': cameras_intrisinics
            }

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")

    def setup_labeling_ui(self):
        self.clear_root()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text=f"Frame: 1/{len(self.frames['side'])}")
        self.frame_label.pack()

        self.slider = tk.Scale(frame_control, from_=0, to=len(self.frames['side']) - 1, orient=tk.HORIZONTAL,
                               length=400,
                               command=self.update_frame_label)
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Labels", command=self.save_labels)
        save_button.pack(pady=5)

        back_button = tk.Button(button_frame, text="Back to Main Menu", command=self.main_menu)
        back_button.pack(pady=5)

        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        body_parts = config.BODY_PART_LABELS
        self.body_part_var = tk.StringVar(value=body_parts[0])
        for body_part in body_parts:
            tk.Radiobutton(control_frame_right, text=body_part, variable=self.body_part_var, value=body_part).pack(
                pady=2)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.display_frame()

    # def change_frame(self, step):
    #     self.current_frame_index += step
    #     self.current_frame_index = max(0, min(self.current_frame_index, len(self.frames['side']) - 1))
    #     self.frame_count_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")
    #     self.display_frame()

    def display_frame(self):
        for i, (canvas, view) in enumerate(
                zip([self.canvas_side, self.canvas_front, self.canvas_overhead], ['side', 'front', 'overhead'])):
            frame = cv2.cvtColor(self.frames[view][self.current_frame_index], cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=frame)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk

    def save_labels(self):
        side_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{self.video_name}/CollectedData_Holly.csv"
        front_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{self.video_name}/CollectedData_Holly.csv"
        overhead_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{self.video_name}/CollectedData_Holly.csv"

        os.makedirs(os.path.dirname(side_path), exist_ok=True)
        os.makedirs(os.path.dirname(front_path), exist_ok=True)
        os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

        # Placeholder for actual saving logic
        messagebox.showinfo("Info", "Labels saved successfully")

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()
