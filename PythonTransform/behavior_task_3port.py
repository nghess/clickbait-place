import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random

# Class to generate maze coordinates
class GridMaze:
    def __init__(self, maze_bounds, maze_dims):
        self.bounds = maze_bounds
        self.shape = maze_dims
        
        cellsize_x = maze_bounds[0] // maze_dims[0]
        cellsize_y = maze_bounds[1] // maze_dims[1]
        
        # Generate Grid
        self.cells = [
            (Point(x * cellsize_x, y * cellsize_y), 
             Point((x + 1) * cellsize_x, (y + 1) * cellsize_y))
            for y in range(self.shape[1])
            for x in range(self.shape[0])
        ]

def draw_grid(grid, img):
    for cell in grid.cells:
        CV.Rectangle(img, cell[0], cell[1], grid_color, thickness=2)

# Function to extract dims from image
def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]

# Modified get_grid_location to handle potential float inputs
def get_grid_location(grid, centroid_x, centroid_y, tgt_list, img):
    cell_width = grid.bounds[0] // grid.shape[0]
    cell_height = grid.bounds[1] // grid.shape[1]
    
    grid_x = int(centroid_x // cell_width)
    grid_y = int(centroid_y // cell_height)
    
    # Get current cell and draw it
    if 0 <= grid_x < grid.shape[0] and 0 <= grid_y < grid.shape[1]:
        cell = grid.cells[grid_y * grid.shape[0] + grid_x]
        CV.Rectangle(img, cell[0], cell[1], mouse_loc_color, thickness=-1)
    
    # Check if current cell is in target list. If so, remove from target list
    tgt_list = [tgt for tgt in tgt_list if tgt != grid_y * grid.shape[0] + grid_x]
    
    return grid_x, grid_y, tgt_list

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
Define targets
"""

def generate_targets(grid_cells_x, grid_cells_y, num_tgt):

  possible_targets = grid_cells_x * grid_cells_y
  tgt_list = []

  for ii in range(num_targets):
    tgt_cell = random.randint(0, possible_targets-1)
    tgt_list.append(tgt_cell)
  
  return grid_cells_x, grid_cells_y, tgt_list

def draw_targets(tgt_list, grid, img):
  for target in tgt_list:
    CV.Rectangle(img, grid.cells[target][0], grid.cells[target][1], target_color, thickness=-1)

# Here we define the number of grid cells to divide the arena by
grid_x = 6
grid_y = 3

# And the number of search targets
num_targets = 1

# Generate grid and targets
global target_list
grid_cells_x, grid_cells_y, target_list = generate_targets(grid_x, grid_y, num_targets)

"""
Global variables
"""
# Initialize reward variables
global trial_count
global reward_left_count
global reward_right_count
global reward_init_count
global reward_state
global click
global click_start_time
global drinking
global reward_left
global reward_right
global reward_init
global reward_left_start_time
global reward_right_start_time
global reward_init_start_time

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_init_count = 0
reward_state = False
click = False
click_start_time = 0
drinking = False
reward_left = False
reward_right = False
reward_init = False
reward_left_start_time = 0
reward_right_start_time = 0
reward_init_start_time = 0

# ITI Variables
global iti_start_time
global iti_duration
global in_iti
global withdrawal_start_time
global in_withdrawal_period
global prev_poke_left
global prev_poke_right
global prev_poke_init

iti_start_time = 0
iti_duration = 0
in_iti = False
withdrawal_start_time = 0
in_withdrawal_period = False
prev_poke_left = False
prev_poke_right = False
prev_poke_init = False

"""
# Visualization parameters
"""
centroid_color = Scalar.Rgb(255, 255, 255)
mouse_loc_color = Scalar.Rgb(255, 0, 0)
target_color = Scalar.Rgb(64, 64, 64)
grid_color = Scalar.Rgb(128, 128, 128)
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
    global reward_init_count
    global target_list
    global reward_state
    global click
    global click_start_time
    global drinking
    global reward_left
    global reward_right
    global reward_init
    global reward_left_start_time
    global reward_right_start_time
    global reward_init_start_time
    global iti_start_time
    global iti_duration
    global in_iti
    global withdrawal_start_time
    global in_withdrawal_period
    global prev_poke_left
    global prev_poke_right
    global prev_poke_init


    # Timing-related vars
    current_time = time.time()
    reward_duration_left = 0.046  # 0.046 seconds
    reward_duration_right = 0.05
    reward_duration_init = 0.05
    click_duration = 0.05
    iti_duration_min = 1.0
    iti_duration_max = 5.0
    withdrawal_duration = 0.5

    # Load realtime variables from Zip node
    centroid_x, centroid_y, image = value[0].Item1, value[0].Item2, value[0].Item3
    poke_left, poke_right = bool(value[1].Item1[0]), bool(value[1].Item1[1])
    poke_init = bool(value[1].Item2)

    # Process grid and canvas
    grid_loc_x, grid_loc_y = None, None
    img_dims = get_image_shape(image)
    grid = GridMaze(img_dims, [grid_cells_x, grid_cells_y])
    canvas = create_blank_canvas(img_dims[0], img_dims[1])
    draw_grid(grid, canvas)
    draw_targets(target_list, grid, canvas)

    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_x, grid_loc_y, target_list = get_grid_location(grid, centroid_x, centroid_y, target_list, canvas)
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)

    # State machine logic
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False
            target_list = generate_targets(grid_x, grid_y, num_targets)[2]
    elif in_withdrawal_period:
        if not (poke_left or poke_right or poke_init):  # Mouse has withdrawn from all ports
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
        elif reward_init and current_time - reward_init_start_time >= reward_duration_init:
            reward_init = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state = False
        elif poke_left and not reward_left and not reward_right and not reward_init:
            reward_left = True
            reward_left_count += 1
            reward_left_start_time = current_time
        elif poke_right and not reward_right and not reward_left and not reward_init:
            reward_right = True
            reward_right_count += 1
            reward_right_start_time = current_time
        elif poke_init and not reward_init and not reward_left and not reward_right:
            reward_init = True
            reward_init_count += 1
            reward_init_start_time = current_time
    else:
        if len(target_list) == 0:
            reward_state = True
            click = True
            click_start_time = current_time

    # Handle click duration
    if click and current_time - click_start_time >= click_duration:
        click = False

    # Update previous poke states and set drinking state
    prev_poke_left, prev_poke_right, prev_poke_init = poke_left, poke_right, poke_init
    drinking = poke_left or poke_right or poke_init

    return (canvas, (grid_loc_x, grid_loc_y), reward_state, reward_left, reward_right, reward_init,
            poke_left, poke_right, poke_init, drinking, in_iti, click, tuple(target_list),
            trial_count, reward_left_count, reward_right_count, reward_init_count)