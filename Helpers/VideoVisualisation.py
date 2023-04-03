import cv2
import matplotlib.pyplot as plt
import Locomotion
import Plot
import numpy as np

video_file = r"Z:\Holly\Data\Behaviour\Dual-belt_APAs\videos\Raw_videos\Round_2\20220826\HM_20220826_APAChar_FAA-1034976_MNone_side_1.avi"
conditions = ['APAChar_LowHigh_Repeats_NoWash_Day3']
con = conditions[0]
mouseID = 'FAA-1034976'
view = 'Side'

frame_num = 140377

# Open the video file
cap = cv2.VideoCapture(video_file)

# def plotlimbdot():
#

# Define a function to update the image when a key is pressed
def update_image():
    global frame_num
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Set the frame number
    ret, frame = cap.read()  # Read the frame
    if ret:
        # Convert BGR to RGB color space for Matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame in the Matplotlib window
        plt.imshow(frame)
        # Update the frame number display
        plt.title('Frame %d' % frame_num)
        # Clear the previously plotted scatter points
        for artist in plt.gca().collections:
            artist.remove()


# Define a function to handle key presses
def on_key_press(event):
    global frame_num
    if event.key == 'right':
        # Move to the next frame
        frame_num += 1
    elif event.key == 'left' and frame_num > 0:
        # Move to the previous frame
        frame_num -= 1
    elif event.key == 'enter':
        # get a new frame number from the user
        new_frame_num = input("Enter a new frame number:")
        try:
            frame_num = int(new_frame_num)
        except ValueError:
            # If the input is not a valid integer, print an error message
            print("Invalid input. Please enter a valid integer.")
            return
    else:
        # Ignore other key presses
        return
    # Update the image and frame number display
    update_image()
    plotSwSt(stsw)
    plotTail()
    plt.draw()


def getStSwFramesForVid(data, con, mouseID, view):
    stsw_FR = Locomotion.Locomotion().getStanceSwingFrames(data, con, mouseID, view, 'ForepawToeR')
    stsw_FL = Locomotion.Locomotion().getStanceSwingFrames(data, con, mouseID, view, 'ForepawToeL')
    stsw_HR = Locomotion.Locomotion().getStanceSwingFrames(data, con, mouseID, view, 'HindpawToeR')
    stsw_HL = Locomotion.Locomotion().getStanceSwingFrames(data, con, mouseID, view, 'HindpawToeL')

    stsw = {
        'FR': stsw_FR,
        'FL': stsw_FL,
        'HR': stsw_HR,
        'HL': stsw_HL
    }

    return stsw

def plotSwSt(stsw):
    colors = {'FR': 'darkblue', 'FL': 'lightblue', 'HR': 'darkgreen', 'HL': 'lightgreen'}
    markersize = 10
    for l in stsw.keys():
        if frame_num in stsw[l]['Stance']['idx']:
            position = np.where(stsw[l]['Stance']['idx'] == frame_num)
            plt.scatter(stsw[l]['Stance']['x'][position], stsw[l]['Stance']['y'][position], color=colors[l], marker='o', s=markersize) ##### put in here the same colours as plot for graphs
        if frame_num in stsw[l]['Swing']['idx']:
            position = np.where(stsw[l]['Swing']['idx'] == frame_num)
            plt.scatter(stsw[l]['Swing']['x'][position], stsw[l]['Swing']['y'][position] ,color=colors[l], marker='s', s=markersize)  ##### put in here the same colours as plot for graphs

def plotTail():
    tailx = df[con][mouseID]['Side'].loc(axis=0)[:, ['TrialStart','RunStart','Transition','RunEnd'], frame_num].loc(axis=1)['Tail1','x'].values[0]
    taily = df[con][mouseID]['Side'].loc(axis=0)[:, ['TrialStart','RunStart','Transition','RunEnd'], frame_num].loc(axis=1)['Tail1','y'].values[0]
    markersize = 10
    plt.scatter(tailx, taily, color='red', marker='s', s=markersize)  ##### put in here the same colours as plot for graphs

# Get data for stance and swing
df = Plot.Plot().GetDFs(conditions)
df[con][mouseID] = Locomotion.Locomotion().getLocoPeriods(df, con, mouseID)
stsw = getStSwFramesForVid(df, con, mouseID, view)

# Start at the first frame
update_image()
plotSwSt(stsw)
plotTail()


# Connect the key press event to the figure
fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Show the Matplotlib window
plt.show()

# Release the video file when finished
cap.release()


