import cv2
import matplotlib.pyplot as plt
from Archive import Locomotion
from Helpers import GetRuns
from Helpers import utils
import numpy as np

#######################################################################################################################
#######################################################################################################################
######################### CONFIGURATIONS ##############################
video_file = r"M:\TEMP\HM_20230306_APACharRepeat_FAA-1035243_None_side_1.avi"
conditions = ['APAChar_LowHigh_Repeats_Wash_Day1']
con = conditions[0]
mouseID = 'FAA-1035243'
view = 'Side'
already_loco_analysed = True

# Open the video file
cap = cv2.VideoCapture(video_file)
clear_points = True
run_num = 0

def getTransitionFrameForVid(df):
    trans_blocks = utils.Utils().find_blocks(
        df.xs('Transition', axis=0, level='RunStage').index.get_level_values(level='FrameIdx'), 200, 2)
    trans_frames = trans_blocks[:,0]
    return trans_frames

def findTransitioningLimb(df,r,trans_frames):
    limbs = []
    # for ridx, r in enumerate(df.index.get_level_values(level='Run').unique()):
    st_limb_mask = df.loc(axis=0)[r, 'Transition', trans_frames].loc(axis=0)[['ForepawToeL', 'ForepawToeR'], 'StepCycle'] == 0
    st_limb = df.loc(axis=0)[r, 'Transition', trans_frames].loc(axis=0)[['ForepawToeL', 'ForepawToeR'], 'StepCycle'].index.get_level_values(level='bodyparts')[st_limb_mask]
    try:
        limbs.append(st_limb[0])
    except:
        limbs.append(np.nan)
    return limbs

def plotLimb(df,trans_frame,trans_limb):
    try:
        x = df.xs(trans_frame, axis=0, level='FrameIdx').loc(axis=1)[trans_limb, 'x']
        y = df.xs(trans_frame, axis=0, level='FrameIdx').loc(axis=1)[trans_limb, 'y']
        plt.scatter(x, y, color='r', marker='o', s=10)
    except:
        print('Cant plot position for frame: %s, limb: %s' %(trans_frame, trans_limb))

def updateImage(df, frame_num):
    limb = findTransitioningLimb(df,r=run_num, trans_frames=frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Set the frame number
    ret, frame = cap.read()
    if ret:
        # Convert BGR to RGB color space for Matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Clear the current plot
        plt.clf()
        # Display the frame in the Matplotlib window
        plt.imshow(frame)
        # Update the frame number display
        plt.title('Run %s\nFrame %d\nTransitioning Limb: %s' %(run_num,frame_num, limb))

        # Update the scatter plots
        plotLimb(df=df,trans_frame=frame_num,trans_limb=limb)

        # Refresh the plot
        plt.draw()

# Define a function to handle key presses
def on_key_press(event):
    global run_num
    if event.key == 'right':
        # Move to the next frame
        run_num += 1
    elif event.key == 'left' and run_num > 0:
        # Move to the previous frame
        run_num -= 1
    elif event.key == 'enter':
        # Get a new frame number from the user
        new_run_num = input("Enter a new frame number:")
        try:
            frame_num = int(new_run_num)
        except ValueError:
            # If the input is not a valid integer, print an error message
            print("Invalid input. Please enter a valid integer.")
            return
    elif event.key == 'c':
        # Toggle clearing of plotted points
        global clear_points
        clear_points = not clear_points
    else:
        # Ignore other key presses
        return
    # Update the image and frame number display
    updateImage(df, frame_num=frame_nums[run_num])


if not already_loco_analysed:
    # Get data for stance and swing
    data = utils.Utils().GetDFs(conditions)
    markerstuff = GetRuns.GetRuns().findMarkers(data[con][mouseID]['Side'])
    data[con][mouseID] = Locomotion.Locomotion().getLocoPeriods(data, con, mouseID, markerstuff, fillvalues=False)
else:
    data = utils.Utils().GetDFs(conditions, reindexed_loco=True)
df = data[con][mouseID][view]

# get frames for transition across all runs
frame_nums = getTransitionFrameForVid(df)

# Start at the first frame
updateImage(df, frame_num=frame_nums[run_num])

# Connect the key press event to the figure
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Show the Matplotlib window
plt.show()

# Release the video file when finished
cap.release()

