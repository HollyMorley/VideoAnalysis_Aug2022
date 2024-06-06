import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageEnhance

import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration

class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Labeling Tool")

        self.video_path = None
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0

        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.marker_size = 10
        self.marker_size_var = tk.DoubleVar(value=self.marker_size)

        self.labels = config.BODY_PART_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])
        self.current_view = tk.StringVar(value="side")

        self.body_part_points = {
            frame_idx: {label: {"side": None, "front": None, "overhead": None} for label in self.labels} for frame_idx in range(1)
        }

        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        label_button = tk.Button(main_frame, text="Label Frames", command=self.label_frames_menu)
        label_button.pack(pady=5)

    def label_frames_menu(self):
        self.clear_root()

        folder_path = filedialog.askdirectory(title="Select Extracted Frames Folder")
        if not folder_path:
            return

        self.video_name = os.path.basename(folder_path)
        self.extracted_frames_path = {
            'side': f"Side/{self.video_name}",
            'front': f"Front/{self.video_name}",
            'overhead': f"Overhead/{self.video_name}"
        }

        self.calibration_data_path = f"Calibration/{self.video_name}/calibration_labels.csv"
        if not os.path.exists(self.calibration_data_path):
            messagebox.showerror("Error", "No corresponding camera calibration data found.")
            return

        self.current_frame_index = 0
        self.load_frames()
        self.load_calibration_data()
        self.setup_labeling_ui()

        self.cap_side = cv2.VideoCapture(self.get_corresponding_video_path('side'))
        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

    def load_frames(self):
        self.frames = {'side': [], 'front': [], 'overhead': []}
        for view in self.frames.keys():
            frame_files = sorted(os.listdir(self.extracted_frames_path[view]),
                                 key=lambda x: os.path.getctime(os.path.join(self.extracted_frames_path[view], x)))
            self.frames[view] = [cv2.imread(os.path.join(self.extracted_frames_path[view], file)) for file in frame_files]

        min_frame_count = min(len(self.frames[view]) for view in self.frames)
        for view in self.frames:
            self.frames[view] = self.frames[view][:min_frame_count]
        self.total_frames = min_frame_count

    def load_calibration_data(self):
        try:
            calibration_coordinates = pd.read_csv(self.calibration_data_path)

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

        self.frame_label = tk.Label(frame_control,
                                    text=f"Frame: {self.current_frame_index + 1}/{self.total_frames}")
        self.frame_label.pack()

        self.prev_button = tk.Button(frame_control, text="<<", command=lambda: self.skip_labeling_frames(-1))
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(frame_control, text=">>", command=lambda: self.skip_labeling_frames(1))
        self.next_button.pack(side=tk.LEFT, padx=5)

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

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            color_box = tk.Label(label_frame, bg=color, width=2)
            color_box.pack(side=tk.LEFT, padx=5)
            label_button = tk.Radiobutton(label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20)
            label_button.pack(side=tk.LEFT)

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

        self.show_frames()

    def skip_labeling_frames(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(self.current_frame_index, self.total_frames - 1))
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{self.total_frames}")
        self.show_frames()

    def show_frames(self):
        frame_number = self.current_frame_index

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

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def show_static_points(self):
        current_points = self.body_part_points[self.current_frame_index]
        for label, views in current_points.items():
            for view, point in views.items():
                if point is not None:
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size * 10])
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        label = self.current_label.get()
        color = self.label_colors[label]
        marker_size = self.marker_size_var.get()

        frame_points = self.body_part_points[self.current_frame_index]

        if event.button == MouseButton.RIGHT:
            if event.key == 'shift':
                self.delete_closest_point(ax, event, frame_points)
            else:
                if frame_points[label][view] is not None:
                    frame_points[label][view].remove()
                frame_points[label][view] = ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)
                self.canvas.draw()
                self.advance_label()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event, frame_points)

    def delete_closest_point(self, ax, event, points_static):
        min_dist = float('inf')
        closest_point_label = None
        for label, points in points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_label = label

        if closest_point_label:
            points_static[closest_point_label][self.current_view.get()].remove()
            points_static[closest_point_label][self.current_view.get()] = None
            self.canvas.draw()

    def find_closest_point(self, ax, event, points_static):
        min_dist = float('inf')
        closest_point = None
        for label, points in points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.labels) == 1:
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set('')

    def update_marker_size(self, val):
        self.marker_size = self.marker_size_var.get()
        if hasattr(self, 'body_part_points') and self.body_part_points:
            current_points = self.body_part_points[self.current_frame_index]
            for label, views in current_points.items():
                for view, point in views.items():
                    if point is not None:
                        point.set_sizes([self.marker_size * 10])
            self.canvas.draw()

    def update_contrast_brightness(self, val):
        if hasattr(self, 'frames'):
            self.show_frames()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.show_frames()

    def save_labels(self):
        side_path = f"Side/{self.video_name}/CollectedData_Holly.csv"
        front_path = f"Front/{self.video_name}/CollectedData_Holly.csv"
        overhead_path = f"Overhead/{self.video_name}/CollectedData_Holly.csv"

        os.makedirs(os.path.dirname(side_path), exist_ok=True)
        os.makedirs(os.path.dirname(front_path), exist_ok=True)
        os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

        # Placeholder for actual saving logic
        messagebox.showinfo("Info", "Labels saved successfully")

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace('side', view)
        return os.path.join(base_path, corresponding_file)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()
