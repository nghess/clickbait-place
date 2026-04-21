import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random
import System
from System import Array

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
            for y in range(1,self.shape[1]-1)
            for x in range(1,self.shape[0]-1)
        ]

def draw_grid(grid, img):
    for cell in grid.cells:
        CV.Rectangle(img, cell[0], cell[1], grid_color, thickness=2)

# Function to extract dims from image
def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]

def draw_future_targets(target_queue, grid, img):
    """Draw upcoming random targets from the queue. Each cell appears at most once."""
    all_targets = list(target_queue)

    if not all_targets:
        return img

    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)

    # With uniform distribution, each cell appears at most once — fixed intensity
    target_future_color = Scalar.Rgb(0, 150, 150)  # Cyan

    for cell_idx in all_targets:
        if cell_idx < len(grid.cells):
            cell = grid.cells[cell_idx]
            CV.Rectangle(overlay, cell[0], cell[1], target_future_color, thickness=-1)

    alpha = 0.5
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)

    return img

# Modified get_grid_location to handle the active target
def get_grid_location(grid, centroid_x, centroid_y, active_target, img):
    cell_width = grid.bounds[0] // grid.shape[0]
    cell_height = grid.bounds[1] // grid.shape[1]
    
    # Calculate grid position accounting for border
    grid_x = int(centroid_x // cell_width) - 1  # Subtract 1 to account for border
    grid_y = int(centroid_y // cell_height) - 1  # Subtract 1 to account for border
    
    target_found = False
    
    # Calculate the index in the cells list
    # Only consider valid inner grid cells
    if 0 <= grid_x < grid.shape[0]-2 and 0 <= grid_y < grid.shape[1]-2:
        # Calculate index in the flattened cells list
        cell_index = grid_y * (grid.shape[0]-2) + grid_x
        
        # Check if index is valid for the cells list
        if 0 <= cell_index < len(grid.cells):
            cell = grid.cells[cell_index]
            CV.Rectangle(img, cell[0], cell[1], mouse_loc_color, thickness=-1)
            
            # Check if this is the active target
            if cell_index == active_target:
                target_found = True
    
    return grid_x, grid_y, target_found

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

def compute_center_cell(grid_cells_x, grid_cells_y):
    """Compute the index of the centermost inner-grid cell.
    
    Inner grid has (grid_cells_x - 2) columns and (grid_cells_y - 2) rows.
    With odd inner dimensions the center is unique.
    """
    inner_cols = grid_cells_x - 2
    inner_rows = grid_cells_y - 2
    center_col = inner_cols // 2
    center_row = inner_rows // 2
    return center_row * inner_cols + center_col

def generate_targets(grid_cells_x, grid_cells_y):
    """Generate a shuffled queue of all non-center cell indices (one per cell).
    
    Returns (center_cell_index, target_queue, active_target).
    active_target is the first item popped from the queue.
    """
    inner_cols = grid_cells_x - 2
    inner_rows = grid_cells_y - 2
    total_cells = inner_cols * inner_rows
    center_cell = compute_center_cell(grid_cells_x, grid_cells_y)
    
    # All inner cells except the center, one visit each
    all_cells = [i for i in range(total_cells*2) if i != center_cell]
    random.shuffle(all_cells)
    
    # Pop the first target
    active_target = all_cells[0]
    target_queue = all_cells[1:]
    
    return center_cell, target_queue, active_target

def regenerate_queue(grid_cells_x, grid_cells_y, center_cell):
    """Re-shuffle a fresh full set of non-center targets and pop the first one."""
    inner_cols = grid_cells_x - 2
    inner_rows = grid_cells_y - 2
    total_cells = inner_cols * inner_rows
    
    all_cells = [i for i in range(total_cells) if i != center_cell]
    random.shuffle(all_cells)
    
    active_target = all_cells[0]
    target_queue = all_cells[1:]
    return target_queue, active_target

# Draw targets
def draw_targets(active_target, target_queue, grid, img, trial_phase, draw_future=False):
    # Draw future random target locations if enabled
    if draw_future:
        img = draw_future_targets(target_queue, grid, img)
    
    # Draw the active target
    if active_target is not None and active_target < len(grid.cells):
        cell = grid.cells[active_target]
        # Use different color for center target vs random target
        if trial_phase == "center":
            color = center_target_color
        else:
            color = target_color
        CV.Rectangle(img, cell[0], cell[1], color, thickness=-1)
    
    return img

# Here we define the number of grid cells to divide the arena by
grid_x = 5
grid_y = 11

# Initialize grid dimensions explicitly
global grid_cells_x
global grid_cells_y
grid_cells_x = grid_x
grid_cells_y = grid_y

# Initialize target landscape: uniform, one per non-center cell
global center_cell_index
global target_queue
global active_target
global trial_phase

center_cell_index, target_queue, active_target = generate_targets(grid_cells_x, grid_cells_y)
trial_phase = "random"  # Each trial starts seeking the random target

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

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = False
click = False
click_start_time = 0
drinking = False
reward_left = False
reward_right = False
reward_left_start_time = 0
reward_right_start_time = 0

# ITI Variables
global iti_start_time
global iti_duration
global in_iti
global withdrawal_start_time
global in_withdrawal_period
global prev_poke_left
global prev_poke_right

iti_start_time = 0
iti_duration = 0
in_iti = False
withdrawal_start_time = 0
in_withdrawal_period = False
prev_poke_left = False
prev_poke_right = False

"""
# Visualization parameters
"""
centroid_color = Scalar.Rgb(255, 255, 255)
mouse_loc_color = Scalar.Rgb(255, 0, 0)
target_color = Scalar.Rgb(255, 255, 255)          # Random target: white
center_target_color = Scalar.Rgb(255, 255, 0)     # Center target: yellow
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
    global target_queue
    global active_target
    global center_cell_index
    global trial_phase
    global reward_state
    global click
    global click_start_time
    global drinking
    global reward_left
    global reward_right
    global reward_left_start_time
    global reward_right_start_time
    global iti_start_time
    global iti_duration
    global in_iti
    global withdrawal_start_time
    global in_withdrawal_period
    global prev_poke_left
    global prev_poke_right
    global grid_cells_x
    global grid_cells_y

    # Timing-related vars
    current_time = time.time()
    reward_duration_left = 0.032
    reward_duration_right = 0.032
    click_duration = 0.1
    iti_duration_min = 1.0
    iti_duration_max = 5.0
    withdrawal_duration = 0.5
    
    # Flag to track if target was found in this frame
    target_found_this_frame = False

    # Load realtime variables from Zip node
    centroid_x, centroid_y, image = value[0].Item1, value[0].Item2, value[0].Item3
    poke_left, poke_right = bool(value[1][0]), bool(value[1][1])

    # Process grid and canvas
    grid_loc_x, grid_loc_y = None, None
    img_dims = get_image_shape(image)
    grid = GridMaze(img_dims, [grid_cells_x, grid_cells_y])
    canvas = create_blank_canvas(img_dims[0], img_dims[1])
    
    # Draw targets (future overlay only shown during random phase)
    draw_targets(active_target, target_queue, grid, canvas, trial_phase,
                 draw_future=(trial_phase == "random"))

    # Process mouse position and check for target
    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_x, grid_loc_y, target_found_this_frame = get_grid_location(grid, centroid_x, centroid_y, active_target, canvas)
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)
        
        # Two-target trial logic
        if target_found_this_frame and active_target is not None and not reward_state and not in_iti and not in_withdrawal_period:
            if trial_phase == "random":
                # Random target found -> click, switch to center target
                click = True
                click_start_time = current_time
                active_target = center_cell_index
                trial_phase = "center"
                
            elif trial_phase == "center":
                # Center target found -> click, enable reward
                click = True
                click_start_time = current_time
                active_target = None
                reward_state = True

    # State machine logic
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False
            
            # Start next trial: pop next random target from queue
            if target_queue:
                active_target = target_queue[0]
                target_queue = target_queue[1:]
            else:
                # Queue exhausted — re-shuffle full set
                target_queue, active_target = regenerate_queue(grid_cells_x, grid_cells_y, center_cell_index)
            
            trial_phase = "random"
            
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
    drinking = poke_left or poke_right

    # Return values
    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right, 
            poke_left, poke_right, drinking, in_iti, click, active_target, 
            trial_count, reward_left_count, reward_right_count, trial_phase)