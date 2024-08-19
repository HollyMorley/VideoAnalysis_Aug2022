import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import Button as MplButton
import os
import mplcursors
from matplotlib import cm
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to select video paths via file dialog
def select_video_paths():
    root = Tk()
    root.withdraw()
    chosen_video_path = askopenfilename(title="Select a Video (Side, Front, or Overhead)")
    if not chosen_video_path:
        raise ValueError("No video selected.")

    if '_side_' in chosen_video_path:
        chosen_cam = 'side'
    elif '_front_' in chosen_video_path:
        chosen_cam = 'front'
    elif '_overhead_' in chosen_video_path:
        chosen_cam = 'overhead'
    else:
        raise ValueError("Unknown camera view selected. Please select a side, front, or overhead video.")

    video_base = chosen_video_path.replace(f'_{chosen_cam}_1.avi', '')  # Base path without view-specific suffix
    video_paths = {
        'side': video_base + '_side_1.avi',
        'front': video_base + '_front_1.avi',
        'overhead': video_base + '_overhead_1.avi'
    }

    # Find the corresponding coordinate files in the same directories as the videos
    coord_paths = {
        'side': find_matching_coord_file(video_paths['side'], 'side'),
        'front': find_matching_coord_file(video_paths['front'], 'front'),
        'overhead': find_matching_coord_file(video_paths['overhead'], 'overhead')
    }

    return chosen_cam, video_paths, coord_paths

def find_matching_coord_file(video_path, view):
    video_dir = os.path.dirname(video_path)
    video_core_name = os.path.basename(video_path).replace(f'_{view}_1.avi', '')  # Extract the core part of the name

    # List all .h5 files in the directory
    coord_files = [f for f in os.listdir(video_dir) if f.endswith('.h5') and view in f]

    # Search for any .h5 file that contains the core video name and the specific view
    matching_files = [f for f in coord_files if video_core_name in f and f'_{view}_' in f]

    if not matching_files:
        raise FileNotFoundError(f"No matching .h5 file found for {video_path} with view {view}")

    # If multiple files match, choose the one with the longest common prefix
    matching_files.sort(key=lambda f: len(os.path.commonprefix([video_core_name, f])), reverse=True)

    return os.path.join(video_dir, matching_files[0])

# Use the function to get video and coordinate paths
chosen_cam, video_paths, coord_paths = select_video_paths()

# Confirm the paths being used
print(f"Using {chosen_cam} video path: {video_paths[chosen_cam]}")
print(f"Using side coordinate path: {coord_paths['side']}")
print(f"Using front coordinate path: {coord_paths['front']}")
print(f"Using overhead coordinate path: {coord_paths['overhead']}")

# Construct the extracted frames directory path
video_base_name = os.path.basename(video_paths[chosen_cam]).replace(f'_{chosen_cam}_1.avi', '')
extracted_frames_dir = f"H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/{chosen_cam.capitalize()}/{video_base_name}"

# Create the folder for extracted frames if it doesn't exist
extracted_frames_dir_1 = extracted_frames_dir + '_1'
if not os.path.exists(extracted_frames_dir_1):
    os.makedirs(extracted_frames_dir_1)

extracted_coords = pd.DataFrame()

print("Reading coordinates...")
coords = pd.read_hdf(coord_paths[chosen_cam])
print("Finished")

# Flatten the multi-index columns and rename them
coords = coords.droplevel(0, axis=1)
coords.columns = ['_'.join(col).strip() for col in coords.columns.values]
coords['frame'] = coords.index

print("Coords DataFrame:")
print(coords.head())
print(coords.columns)

print("Reading video...")
cap = cv2.VideoCapture(video_paths[chosen_cam])
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Finished")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)
else:
    print(f"Total number of frames: {frame_count}")

# Set initial values
current_frame = 0
pcutoff = 0.9  # initial cutoff value

# Extract body parts in the order they appear in the DataFrame
bodyparts = []
for col in coords.columns:
    if '_x' in col:
        bodypart = col.split('_')[0]
        if bodypart not in bodyparts:
            bodyparts.append(bodypart)

# Create the color map based on the original order
cmap = cm.get_cmap('viridis', len(bodyparts))
color_map = {bodypart: cmap(i) for i, bodypart in enumerate(bodyparts)}

# Initial scatter point size
scatter_size = 50

def plot_frame(frame_idx):
    global current_frame, scatter_points
    current_frame = frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_coords = coords[coords['frame'] == frame_idx]

    scatter_points = []  # Clear scatter points list for each frame

    # Create a new annotation object to handle hovering labels
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind, scatter, label):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        annot.set_text(label)
        annot.get_bbox_patch().set_facecolor(color_map[label])
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for label, scatter in scatter_points:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind, scatter, label)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    for col in frame_coords.columns:
        if 'likelihood' in col:
            bodypart = col.split('_likelihood')[0]
            likelihood = frame_coords[f'{bodypart}_likelihood'].values[0]
            x = frame_coords[f'{bodypart}_x'].values[0]
            y = frame_coords[f'{bodypart}_y'].values[0]
            color = color_map[bodypart]  # Get the color for this bodypart

            if likelihood > pcutoff:
                scatter = ax.scatter(
                    x, y,
                    s=scatter_size,
                    color=color,
                    edgecolors='k',
                    marker='o'
                )
            else:
                scatter = ax.scatter(
                    x, y,
                    s=scatter_size,
                    color=color,
                    edgecolors='k',
                    marker='x'
                )

            scatter_points.append((bodypart, scatter))

    ax.set_title(f'Frame {frame_idx}')
    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

# Update function for scatter size slider
def update_scatter_size(val):
    global scatter_size
    scatter_size = val
    plot_frame(current_frame)

# Scrollbar update function
def update(val):
    frame_idx = int(slider.val)
    plot_frame(frame_idx)

# Function to skip frames
def skip_frames(skip):
    new_frame = current_frame + skip
    if new_frame < 0:
        new_frame = 0
    elif new_frame >= frame_count:
        new_frame = frame_count - 1
    slider.set_val(new_frame)

# Zoom, pan, and home button functions
def zoom(event):
    fig.canvas.manager.toolbar.zoom()

def pan(event):
    fig.canvas.manager.toolbar.pan()

def home(event):
    fig.canvas.manager.toolbar.home()

def restore_scorer_level(df):
    df.columns = pd.MultiIndex.from_tuples([('Holly', *col.split('_')) for col in df.columns])
    # df.columns.names = ['scorer', 'bodyparts', 'coords']
    return df

def extract_frame():
    global extracted_coords

    # Extract and save the current frame as an image
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        extracted_frames_dir_1 = extracted_frames_dir + '_1'
        img_filename = os.path.join(extracted_frames_dir_1, f"img{current_frame}.png")
        cv2.imwrite(img_filename, frame)
        print(f"Saved: {img_filename}")

    # Extract and save the corresponding row from coords
    frame_coords = coords[coords['frame'] == current_frame]

    # Restore scorer level before saving
    frame_coords_with_scorer = restore_scorer_level(frame_coords)

    # add multiindex
    frames_dir = extracted_frames_dir_1.split('/')[-1]
    frame_coords_with_scorer.index = pd.MultiIndex.from_tuples([('labeled_data', frames_dir, img_filename)])

    # Append to the extracted coordinates DataFrame
    extracted_coords = pd.concat([extracted_coords, frame_coords_with_scorer])
    extracted_coords.columns.names = ['scorer', 'bodyparts', 'coords']

    # Save to CSV and HDF5
    csv_filename = os.path.join(extracted_frames_dir, f"extracted_coords_{chosen_cam}.csv")
    h5_filename = os.path.join(extracted_frames_dir, f"extracted_coords_{chosen_cam}.h5")

    extracted_coords.to_csv(csv_filename, index=False)
    extracted_coords.to_hdf(h5_filename, key='df', mode='w')

    print(f"Saved: {csv_filename} and {h5_filename}")

# Create Matplotlib figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Create scrollbar
ax_slider = plt.axes([0.1, 0.25, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, frame_count - 1, valinit=current_frame, valfmt='%0.0f')
slider.on_changed(update)

# Scatter size slider next to skip buttons
ax_scatter_slider = plt.axes([0.8, 0.1, 0.15, 0.03])
scatter_slider = Slider(ax_scatter_slider, 'Size', 5, 100, valinit=scatter_size, valfmt='%0.0f')
scatter_slider.on_changed(update_scatter_size)

# Create skip buttons, zoom, pan, and home buttons
button_positions = [0.1, 0.18, 0.26, 0.34, 0.42, 0.5, 0.58, 0.66]
skip_values = [-1000, -100, -10, -1, 1, 10, 100, 1000]

buttons = []
for pos, skip in zip(button_positions, skip_values):
    ax_btn = plt.axes([pos, 0.1, 0.08, 0.03])
    btn = MplButton(ax_btn, f'{skip:+}')
    btn.on_clicked(lambda event, s=skip: skip_frames(s))
    buttons.append(btn)

ax_zoom = plt.axes([0.7, 0.05, 0.08, 0.03])
btn_zoom = MplButton(ax_zoom, 'Zoom')
btn_zoom.on_clicked(zoom)

ax_pan = plt.axes([0.8, 0.05, 0.08, 0.03])
btn_pan = MplButton(ax_pan, 'Pan')
btn_pan.on_clicked(pan)

ax_home = plt.axes([0.9, 0.05, 0.08, 0.03])
btn_home = MplButton(ax_home, 'Home')
btn_home.on_clicked(home)

ax_extract = plt.axes([0.44, 0.05, 0.12, 0.04])  # Position the button appropriately
btn_extract = MplButton(ax_extract, 'Extract Frame')
btn_extract.on_clicked(lambda event: extract_frame())

# Show the initial frame
plot_frame(current_frame)

plt.show()
