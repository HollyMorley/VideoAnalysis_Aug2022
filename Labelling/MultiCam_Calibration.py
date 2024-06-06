import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageEnhance

import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration

class CalibrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration Tool")

        self.video_path = None
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0

        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)

        self.labels = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR", "Nose"]
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])
        self.current_view = tk.StringVar(value="side")

        self.calibration_points = {label: {"side": None, "front": None, "overhead": None} for label in self.labels}
        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in self.labels}

        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        calibrate_button = tk.Button(main_frame, text="Calibrate Camera Positions", command=self.calibrate_menu)
        calibrate_button.pack(pady=5)

    def calibrate_menu(self):
        self.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            return

        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        self.calibration_file_path = f"Calibration/{os.path.basename(self.video_path)}/calibration_labels.csv"

        calibrate_frame = tk.Frame(self.root)
        calibrate_frame.pack(pady=20)

        save_button = tk.Button(calibrate_frame, text="Save Calibration Points", command=self.save_calibration_points)
        save_button.pack(pady=5)

        back_button = tk.Button(calibrate_frame, text="Back to Main Menu", command=self.main_menu)
        back_button.pack(pady=5)

        self.slider = tk.Scale(calibrate_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=600, command=self.update_frame_label)
        self.slider.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(calibrate_frame)
        skip_frame.pack(side=tk.TOP, pady=10)
        self.add_skip_buttons(skip_frame)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.show_frames()

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames()

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def skip_frames(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(self.current_frame_index, self.total_frames - 1))
        self.slider.set(self.current_frame_index)
        self.show_frames()

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace('side', view)
        return os.path.join(base_path, corresponding_file)

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def show_frames(self, val=None):
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

            self.canvas.draw()

    def save_calibration_points(self):
        calibration_path = f"Calibration/{os.path.basename(self.video_path)}/calibration_labels.csv"
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
        return {label: "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for label, color in zip(labels, colors)}

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationTool(root)
    root.mainloop()
