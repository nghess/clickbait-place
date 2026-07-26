import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random
import System
from System import Array

# Function to extract dims from image
def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]

def create_blank_canvas(width, height, channels=3, color=(0, 0, 0)):
    depth = IplDepth.U8
    img = IplImage(Size(width, height), depth, channels)
    if channels == 1:
        fill_color = Scalar.All(color[0])
    else:
        fill_color = Scalar.Rgb(color[0], color[1], color[2])
    img.Set(fill_color)

    return img

"""
Global variables
"""
# Initialize reward variables
global trial_count
global reward_left_count
global reward_right_count
global reward_state
global click
global click_start_time
global drinking
global reward_left
global reward_right
global reward_left_start_time
global reward_right_start_time
global trial_armed

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = True
click = False
click_start_time = 0
drinking = False
reward_left = False
reward_right = False
reward_left_start_time = 0
reward_right_start_time = 0
trial_armed = True  # Correct-trial trigger available (replaces active_target's per-trial gate)

# ITI Variables
global iti_start_time
global iti_duration
global in_iti
global withdrawal_start_time
global in_withdrawal_period
global prev_poke_left
global prev_poke_right
global prev_correct_trigger

iti_start_time = 0
iti_duration = 0
in_iti = False
withdrawal_start_time = 0
in_withdrawal_period = False
prev_poke_left = False
prev_poke_right = False
prev_correct_trigger = False  # For rising-edge detection on the incoming boolean

"""
# Visualization parameters
"""
centroid_color = Scalar.Rgb(255, 255, 255)
centroid_radius = 5

"""
Execute task
"""

@returns(tuple)
def process(value):
    # Declare global vars
    global trial_count
    global reward_left_count
    global reward_right_count
    global reward_state
    global click
    global click_start_time
    global drinking
    global reward_left
    global reward_right
    global reward_left_start_time
    global reward_right_start_time
    global trial_armed
    global iti_start_time
    global iti_duration
    global in_iti
    global withdrawal_start_time
    global in_withdrawal_period
    global prev_poke_left
    global prev_poke_right
    global prev_correct_trigger

    # Timing-related vars
    current_time = time.time()
    reward_duration_left = 0.032
    reward_duration_right = 0.032
    click_duration = 0.1
    iti_duration_min = 1.0
    iti_duration_max = 5.0
    withdrawal_duration = 0.5

    # Load realtime variables from Zip node
    centroid_x, centroid_y, image = value[0].Item1, value[0].Item2, value[0].Item3
    poke_left, poke_right = value[1][0], value[1][1]
    correct_trigger = bool(value[1][2])

    # Catch NaN centroid immediately and convert to (0, 0)
    if math.isnan(centroid_x) or math.isnan(centroid_y):
        centroid_x, centroid_y = 0.0, 0.0

    # Convert correct_trigger None to False
    if correct_trigger is None:
        correct_trigger = False

    # Process canvas
    img_dims = get_image_shape(image)
    canvas = create_blank_canvas(img_dims[0], img_dims[1])

    # Draw centroid (visualization is centroid-only)
    CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)

    # Rising-edge detection: fire only on a fresh False->True transition, so a
    # boolean held high across drinking/ITI cannot immediately re-trigger reward
    # the moment the trial re-arms at ITI end.
    correct_rising_edge = correct_trigger and not prev_correct_trigger

    # Correct-trial trigger: single bool replaces spatial target detection
    if correct_rising_edge and trial_armed and not reward_state:
        # Consume the trial (re-armed at ITI end)
        trial_armed = False

        # Trigger reward state
        reward_state = True
        click = True
        click_start_time = current_time

    # State machine logic
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False

            # Re-arm the trigger for the next trial (only if consumed)
            if not trial_armed:
                trial_armed = True

    elif in_withdrawal_period:
        if not (poke_left or poke_right):  # Mouse has withdrawn
            if current_time - withdrawal_start_time >= withdrawal_duration:
                in_withdrawal_period = False
                in_iti = True
                iti_start_time = current_time
                iti_duration = random.uniform(iti_duration_min, iti_duration_max)
        else:  # Mouse is still poking, reset withdrawal timer
            withdrawal_start_time = current_time

    elif reward_state:
        if reward_left and current_time - reward_left_start_time >= reward_duration_left:
            reward_left = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state = False
        elif reward_right and current_time - reward_right_start_time >= reward_duration_right:
            reward_right = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state = False
        elif poke_left and not reward_left and not reward_right:
            reward_left = True
            reward_left_count += 1
            reward_left_start_time = current_time
        elif poke_right and not reward_right and not reward_left:
            reward_right = True
            reward_right_count += 1
            reward_right_start_time = current_time

    # Handle click duration
    if click and current_time - click_start_time >= click_duration:
        click = False

    # Update previous poke states and set drinking state
    prev_poke_left, prev_poke_right = poke_left, poke_right
    prev_correct_trigger = correct_trigger
    drinking = poke_left or poke_right

    # Return values
    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right,
            poke_left, poke_right, drinking, in_iti, click,
            trial_count, reward_left_count, reward_right_count)