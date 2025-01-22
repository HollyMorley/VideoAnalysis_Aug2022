"""
Label the lost TransitionL markers manually for the cases where front cam shifted. For use with MappingRealWorld_V3.py
"""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
import os

from Helpers.Config_23 import *


###############################################################################
# 1) DATA STRUCTURES
###############################################################################
videos_info = [
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230405\HM_20230405_APACharExt_FAA-1035302_LR_front_1.avi",
        'name': "20230405_302",
        'extracted_frames': [],
        'labels': {}  # frame_number -> (x, y)
    },
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230408\HM_20230408_APACharExt_FAA-1035243_None_front_1.avi",
        'name': "20230408_243",
        'extracted_frames': [],
        'labels': {}
    },
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230408\HM_20230408_APACharExt_FAA-1035246_LR_front_1.avi",
        'name': "20230408_246",
        'extracted_frames': [],
        'labels': {}
    },
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230409\HM_20230409_APACharExt_FAA-1035244_L_front_1.avi",
        'name': "20230409_244",
        'extracted_frames': [],
        'labels': {}
    },
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230409\HM_20230409_APACharExt_FAA-1035250_LR_front_1.avi",
        'name': "20230409_250",
        'extracted_frames': [],
        'labels': {}
    },
    {
        'path': r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230411\HM_20230411_APAPer_FAA-1035244_L_front_1.avi",
        'name': "20230411_244",
        'extracted_frames': [],
        'labels': {}
    }
]

output_extracted_csv = os.path.join(paths['filtereddata_folder'], 'extracted_bad_frontcam_frames.csv')
output_labels_csv    = os.path.join(paths['filtereddata_folder'], 'bad_frontcam_labels.csv')

###############################################################################
# 2) THE MAIN APP CLASS
###############################################################################
class VideoLabelerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter Video Extract+Label")
        self.geometry("1200x800")

        # Current state
        self.current_video_idx = 0
        self.cap = None
        self.total_frames = 0

        # For label mode, we track the index within 'extracted_frames'
        self.label_extracted_idx = 0

        # For dragging
        self.dragging = False
        self.drag_offset = (0, 0)  # how far mouse is from label center
        self.drag_frame = None     # which frame's label is being dragged

        # Are we in extract mode or label mode
        self.extract_mode = True  # start in Extract mode by default

        # Setup UI
        self.create_widgets()
        self.load_video(0)

    ###########################################################################
    # 2A) CREATE ALL WIDGETS
    ###########################################################################
    def create_widgets(self):
        # Top frame: list of videos, mode toggles
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # List of videos
        tk.Label(top_frame, text="Videos:").pack(side=tk.LEFT, padx=5)
        self.video_listbox = tk.Listbox(top_frame, height=4, exportselection=False)
        self.video_listbox.pack(side=tk.LEFT, padx=5)
        for vinfo in videos_info:
            self.video_listbox.insert(tk.END, vinfo['name'])
        self.video_listbox.bind("<<ListboxSelect>>", self.on_video_select)

        # Buttons for modes
        self.mode_btn = tk.Button(top_frame, text="Switch to Label Mode", command=self.toggle_mode)
        self.mode_btn.pack(side=tk.LEFT, padx=10)

        # Save CSVs
        tk.Button(top_frame, text="Save Extracted CSV", command=self.save_extracted_csv).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Save Labels CSV", command=self.save_labels_csv).pack(side=tk.LEFT, padx=10)

        # Middle frame: video display & right panel
        mid_frame = tk.Frame(self)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for video
        self.canvas = tk.Canvas(mid_frame, bg="black", width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse events for labeling
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # right-click

        # Right panel for controls
        right_panel = tk.Frame(mid_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame slider (for EXTRACT mode)
        tk.Label(right_panel, text="Frame Slider:").pack(pady=5)
        self.frame_slider = tk.Scale(right_panel, from_=0, to=0, orient=tk.HORIZONTAL,
                                     command=self.on_frame_slider)
        self.frame_slider.pack(pady=5)

        # Extract button
        self.extract_btn = tk.Button(right_panel, text="Extract Frame", command=self.extract_frame)
        self.extract_btn.pack(pady=10)

        # Buttons for label navigation
        tk.Button(right_panel, text="Prev Extracted", command=self.prev_extracted).pack(pady=5)
        tk.Button(right_panel, text="Next Extracted", command=self.next_extracted).pack(pady=5)

        # Quit
        tk.Button(right_panel, text="Quit", command=self.quit).pack(side=tk.BOTTOM, pady=10)

        # Info label
        self.info_label = tk.Label(right_panel, text="", bg="white")
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    ###########################################################################
    # 2B) VIDEO LOADING
    ###########################################################################
    def load_video(self, idx):
        """
        Open the video file at idx, set up self.cap and frame_slider range.
        """
        if self.cap is not None:
            self.cap.release()
        self.current_video_idx = idx
        path = videos_info[idx]['path']
        if not os.path.isfile(path):
            self.info_label.config(text=f"Video not found: {path}")
            return

        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.config(to=(self.total_frames - 1))
        self.frame_slider.set(0)

        # If we are in label mode, we ignore the slider's actual range
        # and show only extracted frames.  We'll handle that in update_frame().
        self.update_frame()
        self.update_info_label()

    ###########################################################################
    # 3) MODE HANDLING
    ###########################################################################
    def toggle_mode(self):
        self.extract_mode = not self.extract_mode
        if self.extract_mode:
            self.mode_btn.config(text="Switch to Label Mode")
            self.extract_btn.config(state=tk.NORMAL)
        else:
            self.mode_btn.config(text="Switch to Extract Mode")
            self.extract_btn.config(state=tk.DISABLED)
            self.label_extracted_idx = 0
        self.update_frame()
        self.update_info_label()

    ###########################################################################
    # 4) EXTRACT MODE
    ###########################################################################
    def on_frame_slider(self, val):
        """
        Called when user drags the scale in EXTRACT mode.
        """
        if self.extract_mode:
            self.update_frame()

    def extract_frame(self):
        """
        Adds the current frame number to 'extracted_frames'.
        """
        if not self.extract_mode:
            return
        vinfo = videos_info[self.current_video_idx]
        frame_num = int(self.frame_slider.get())
        if frame_num not in vinfo['extracted_frames']:
            vinfo['extracted_frames'].append(frame_num)
            vinfo['extracted_frames'].sort()
            self.info_label.config(text=f"Extracted frame {frame_num} for {vinfo['name']}.")

    ###########################################################################
    # 5) LABEL MODE
    ###########################################################################
    def prev_extracted(self):
        """Move to previous extracted frame in label mode."""
        if self.extract_mode:
            return
        vinfo = videos_info[self.current_video_idx]
        if not vinfo['extracted_frames']:
            return
        self.label_extracted_idx = max(0, self.label_extracted_idx - 1)
        self.update_frame()

    def next_extracted(self):
        """Move to next extracted frame in label mode."""
        if self.extract_mode:
            return
        vinfo = videos_info[self.current_video_idx]
        if not vinfo['extracted_frames']:
            return
        self.label_extracted_idx = min(self.label_extracted_idx + 1, len(vinfo['extracted_frames']) - 1)
        self.update_frame()

    ###########################################################################
    # 6) FRAME DISPLAY
    ###########################################################################
    def update_frame(self):
        """
        Reads and displays the correct frame from self.cap, either from:
        - self.frame_slider (extract mode), or
        - extracted_frames[label_extracted_idx] (label mode).
        """
        if self.cap is None:
            return

        vinfo = videos_info[self.current_video_idx]
        if self.extract_mode:
            # Show whatever frame the slider is on
            fnum = int(self.frame_slider.get())
        else:
            # Show only extracted frames
            if not vinfo['extracted_frames']:
                # no frames extracted
                self.show_blank("No extracted frames to label.")
                return
            fnum = vinfo['extracted_frames'][self.label_extracted_idx]

        # Ensure valid range
        if fnum < 0 or fnum >= self.total_frames:
            self.show_blank("Frame out of range.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.show_blank("Failed to read frame.")
            return

        # Convert BGR to RGB, then to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Optionally draw the label for label-mode
        if not self.extract_mode:
            label_xy = vinfo['labels'].get(fnum, None)
            if label_xy is not None:
                # Draw a small circle in PIL. One approach is ImageDraw:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                x, y = label_xy
                r = 10
                draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255,0,0,128))

        # Convert to ImageTk
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.image = tk_img  # keep reference!
        self.canvas.config(width=pil_img.width, height=pil_img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def show_blank(self, message):
        self.canvas.delete("all")
        self.canvas.config(width=800, height=600)
        self.canvas.create_text(400, 300, text=message, fill="white", font=("Arial", 20))

    ###########################################################################
    # 7) MOUSE EVENTS FOR LABELING
    ###########################################################################
    def on_left_click(self, event):
        """
        Start dragging if near existing label (within 20 px).
        """
        if self.extract_mode:
            return  # no labeling in extract mode

        vinfo = videos_info[self.current_video_idx]
        if not vinfo['extracted_frames']:
            return
        fnum = vinfo['extracted_frames'][self.label_extracted_idx]
        label_xy = vinfo['labels'].get(fnum, None)
        if label_xy is not None:
            lx, ly = label_xy
            dist_sq = (event.x - lx)**2 + (event.y - ly)**2
            if dist_sq < 20**2:  # within 20 px
                self.dragging = True
                self.drag_frame = fnum
                self.drag_offset = (lx - event.x, ly - event.y)

    def on_left_drag(self, event):
        """
        If dragging, update label's position.
        """
        if self.dragging and not self.extract_mode:
            vinfo = videos_info[self.current_video_idx]
            if self.drag_frame in vinfo['labels']:
                new_x = event.x + self.drag_offset[0]
                new_y = event.y + self.drag_offset[1]
                vinfo['labels'][self.drag_frame] = (new_x, new_y)
                self.update_frame()  # redraw circle

    def on_left_release(self, event):
        """
        Stop dragging.
        """
        self.dragging = False
        self.drag_frame = None

    def on_right_click(self, event):
        """
        Right-click = place or move label. SHIFT+Right-click = delete label.
        """
        if self.extract_mode:
            return
        vinfo = videos_info[self.current_video_idx]
        if not vinfo['extracted_frames']:
            return
        fnum = vinfo['extracted_frames'][self.label_extracted_idx]

        # Check if SHIFT is pressed. On Windows, SHIFT sets event.state bits.
        # This can vary by OS. 0x0001 is SHIFT, or might be 0x0004, etc.
        shift_pressed = (event.state & 0x0001) != 0 or (event.state & 0x0004) != 0

        if shift_pressed:
            # SHIFT + right-click -> delete label
            if fnum in vinfo['labels']:
                del vinfo['labels'][fnum]
                self.info_label.config(text=f"Deleted label for frame {fnum}")
                self.update_frame()
        else:
            # Normal right-click -> place or move label
            vinfo['labels'][fnum] = (event.x, event.y)
            self.info_label.config(text=f"Labeled frame {fnum} at ({event.x}, {event.y})")
            self.update_frame()

    ###########################################################################
    # 8) VIDEO SELECTION
    ###########################################################################
    def on_video_select(self, event):
        # If user selects a different video from the list
        idxs = self.video_listbox.curselection()
        if idxs:
            self.load_video(idxs[0])
            self.extract_mode = True
            self.mode_btn.config(text="Switch to Label Mode")
            self.extract_btn.config(state=tk.NORMAL)
            self.label_extracted_idx = 0
            self.update_info_label()

    ###########################################################################
    # 9) INFO LABEL UPDATES
    ###########################################################################
    def update_info_label(self):
        mode = "Extract" if self.extract_mode else "Label"
        vinfo = videos_info[self.current_video_idx]
        self.info_label.config(text=f"Mode: {mode} | Video: {vinfo['name']}")

    ###########################################################################
    # 10) CSV SAVE
    ###########################################################################
    def save_extracted_csv(self):
        with open(output_extracted_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "frame_number"])
            for v in videos_info:
                for fn in v['extracted_frames']:
                    writer.writerow([v['name'], fn])
        self.info_label.config(text=f"Saved extracted frames to {output_extracted_csv}")

    def save_labels_csv(self):
        with open(output_labels_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_name","frame_number","x","y"])
            for v in videos_info:
                for fn, (x, y) in v['labels'].items():
                    writer.writerow([v['name'], fn, x, y])
        self.info_label.config(text=f"Saved labels to {output_labels_csv}")

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    app = VideoLabelerApp()
    # Pre-select the first video in the listbox
    app.video_listbox.selection_set(0)
    app.mainloop()
