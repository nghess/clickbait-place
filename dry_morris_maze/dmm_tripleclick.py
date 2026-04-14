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

# Platform dimensions in full-grid cell units
PLATFORM_COLS = 2
PLATFORM_ROWS = 4

def get_platform_rect(flip_state, grid_cells_x, grid_cells_y, img_dims):
    """Return the platform cell as a (Point, Point) pixel rectangle in the border ring.
    
    Platform is PLATFORM_COLS wide x PLATFORM_ROWS tall in full-grid cell units.
      flip_state 0 -> right edge: full-grid cols (grid_cells_x-2) to (grid_cells_x-1), rows 1-4
      flip_state 1 -> left edge:  full-grid cols 0 to 1, rows 11-14
    
    Returns (top_left_Point, bottom_right_Point).
    """
    cell_w = img_dims[0] // grid_cells_x
    cell_h = img_dims[1] // grid_cells_y
    
    if flip_state == 0:
        # Right edge, top-aligned at row 1
        col_start = grid_cells_x - PLATFORM_COLS
        row_start = 1
    else:
        # Left edge, bottom-aligned at row (grid_cells_y - 1)
        col_start = 0
        row_start = grid_cells_y - 1 - PLATFORM_ROWS
    
    tl = Point(col_start * cell_w, row_start * cell_h)
    br = Point((col_start + PLATFORM_COLS) * cell_w, (row_start + PLATFORM_ROWS) * cell_h)
    return (tl, br)

def get_platform_excluded_cells(flip_state, grid_cells_x, grid_cells_y):
    """Return the set of inner-grid cell indices that overlap with the platform rect.
    
    The platform spans full-grid cells; any that fall within the inner grid
    (full-grid cols 1 to grid_cells_x-2, rows 1 to grid_cells_y-2) are excluded.
    """
    inner_cols = grid_cells_x - 2
    inner_rows = grid_cells_y - 2
    
    if flip_state == 0:
        col_start = grid_cells_x - PLATFORM_COLS
        row_start = 1
    else:
        col_start = 0
        row_start = grid_cells_y - 1 - PLATFORM_ROWS
    
    excluded = set()
    for fg_row in range(row_start, row_start + PLATFORM_ROWS):
        for fg_col in range(col_start, col_start + PLATFORM_COLS):
            # Convert full-grid coords to inner-grid coords
            inner_col = fg_col - 1
            inner_row = fg_row - 1
            if 0 <= inner_col < inner_cols and 0 <= inner_row < inner_rows:
                idx = inner_row * inner_cols + inner_col
                excluded.add(idx)
    
    return excluded

def centroid_in_rect(cx, cy, rect):
    """Check whether centroid (cx, cy) falls within a (Point, Point) rectangle."""
    tl, br = rect
    return tl.X <= cx < br.X and tl.Y <= cy < br.Y

def generate_targets(grid_cells_x, grid_cells_y, excluded_cells):
    """Generate a shuffled queue of all inner-grid cell indices (one per cell),
    excluding cells that overlap with the platform.
    
    Returns (target_queue, active_target).
    active_target is the first item popped from the queue.
    """
    inner_cols = grid_cells_x - 2
    inner_rows = grid_cells_y - 2
    total_cells = inner_cols * inner_rows
    
    all_cells = [i for i in range(total_cells) if i not in excluded_cells]
    random.shuffle(all_cells)
    
    # Pop the first target
    active_target = all_cells[0]
    target_queue = all_cells[1:]
    
    return target_queue, active_target

def regenerate_queue(grid_cells_x, grid_cells_y, excluded_cells):
    """Re-shuffle a fresh full set of targets and pop the first one."""
    return generate_targets(grid_cells_x, grid_cells_y, excluded_cells)

# Draw targets
def draw_targets(active_target, target_queue, grid, img, trial_phase, platform_rect=None, draw_future=False):
    # Draw future random target locations if enabled
    if draw_future:
        img = draw_future_targets(target_queue, grid, img)
    
    if trial_phase == "platform" and platform_rect is not None:
        # Draw the platform target using its pixel rect (border cell)
        CV.Rectangle(img, platform_rect[0], platform_rect[1], platform_target_color, thickness=-1)
    elif trial_phase == "interim" and platform_rect is not None:
        # Platform found — show it in green
        CV.Rectangle(img, platform_rect[0], platform_rect[1], platform_found_color, thickness=-1)
    elif active_target is not None and active_target < len(grid.cells):
        # Draw the active random target from the inner grid
        cell = grid.cells[active_target]
        CV.Rectangle(img, cell[0], cell[1], target_color, thickness=-1)
    
    return img

# Here we define the number of grid cells to divide the arena by
grid_x = 7
grid_y = 15

# Flip state parameters
FLIP_STATE_CONFIG = {
    'min_trials_before_flip': 1000,    # Minimum trials before state flip
    'max_trials_before_flip': 10001,   # Maximum trials before state flip
}

# Initialize grid dimensions explicitly
global grid_cells_x
global grid_cells_y
grid_cells_x = grid_x
grid_cells_y = grid_y

# Initialize flip state
global flip_state
global trials_since_last_flip
global trials_until_next_flip

flip_state = 1
trials_since_last_flip = 0
trials_until_next_flip = random.randint(
    FLIP_STATE_CONFIG['min_trials_before_flip'],
    FLIP_STATE_CONFIG['max_trials_before_flip']
)

# Initialize target landscape: uniform, one per inner cell (excluding platform overlap)
global platform_rect
global platform_excluded
global target_queue
global active_target
global trial_phase

platform_rect = None  # Computed on first process() call when image dims are available
platform_excluded = get_platform_excluded_cells(flip_state, grid_cells_x, grid_cells_y)
target_queue, active_target = generate_targets(grid_cells_x, grid_cells_y, platform_excluded)
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
global click_count
global click_total
global drinking
global reward_left
global reward_right
global reward_left_start_time
global reward_right_start_time
global punishment_count

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = True  # Free reward on first trial
click = False
click_start_time = 0
click_count = 0     # How many clicks have been delivered so far in current sequence
click_total = 0     # How many clicks total in current sequence (1 for random, 3 for platform)
drinking = False
reward_left = False
reward_right = False
reward_left_start_time = 0
reward_right_start_time = 0
punishment_count = 0

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
platform_target_color = Scalar.Rgb(255, 255, 0)   # Platform target: yellow
platform_found_color = Scalar.Rgb(0, 255, 0)      # Platform found: green
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
    global platform_rect
    global platform_excluded
    global trial_phase
    global reward_state
    global click
    global click_start_time
    global click_count
    global click_total
    global drinking
    global reward_left
    global reward_right
    global reward_left_start_time
    global reward_right_start_time
    global punishment_count
    global iti_start_time
    global iti_duration
    global in_iti
    global withdrawal_start_time
    global in_withdrawal_period
    global prev_poke_left
    global prev_poke_right
    global grid_cells_x
    global grid_cells_y
    global flip_state
    global trials_since_last_flip
    global trials_until_next_flip

    # Timing-related vars
    current_time = time.time()
    reward_duration_left = 0.032
    reward_duration_right = 0.032
    click_duration = 0.1          # Duration of each click pulse
    click_gap = 0.1               # Gap between consecutive clicks
    iti_duration_min = 1.0
    iti_duration_max = 5.0
    withdrawal_duration = 0.5
    
    # Flag to track if target was found in this frame
    target_found_this_frame = False
    platform_found_this_frame = False

    # Load realtime variables from Zip node
    centroid_x, centroid_y, image = value[0].Item1, value[0].Item2, value[0].Item3
    poke_left, poke_right = bool(value[1][0]), bool(value[1][1])

    # Process grid and canvas
    grid_loc_x, grid_loc_y = None, None
    img_dims = get_image_shape(image)
    grid = GridMaze(img_dims, [grid_cells_x, grid_cells_y])
    canvas = create_blank_canvas(img_dims[0], img_dims[1])
    
    # Compute platform_rect on first call (needs image dims)
    if platform_rect is None:
        platform_rect = get_platform_rect(flip_state, grid_cells_x, grid_cells_y, img_dims)
    
    # Draw targets (future overlay only shown during random phase)
    draw_targets(active_target, target_queue, grid, canvas, trial_phase,
                 platform_rect=platform_rect, draw_future=(trial_phase == "random"))

    # Process mouse position and check for target
    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_x, grid_loc_y, target_found_this_frame = get_grid_location(grid, centroid_x, centroid_y, active_target, canvas)
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)
        
        # Check platform cell separately (border cell, not in grid.cells)
        if trial_phase == "platform":
            platform_found_this_frame = centroid_in_rect(centroid_x, centroid_y, platform_rect)
        
        # Two-target trial logic
        if not reward_state and not in_iti and not in_withdrawal_period:
            if trial_phase == "random" and target_found_this_frame and active_target is not None:
                # Random target found -> single click, switch to platform target
                click = True
                click_start_time = current_time
                click_count = 1
                click_total = 1
                active_target = None  # No inner-grid target active during platform phase
                trial_phase = "platform"
                
            elif trial_phase == "platform" and platform_found_this_frame:
                # Platform target found -> triple click, enable reward, enter interim phase
                click = True
                click_start_time = current_time
                click_count = 1
                click_total = 3
                reward_state = True
                trial_phase = "interim"

    # Punishment: poke during "platform" phase (before reaching platform) -> end trial, go to ITI
    if trial_phase == "platform" and not reward_state and not in_iti and not in_withdrawal_period:
        if (poke_left or poke_right):
            # Punish: burn this trial, skip to ITI
            punishment_count += 1
            active_target = None
            trial_phase = "random"  # Reset phase for next trial
            in_iti = True
            iti_start_time = current_time
            iti_duration = random.uniform(iti_duration_min, iti_duration_max)

    # State machine logic
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            trials_since_last_flip += 1
            in_iti = False
            
            # Check if we need to flip state
            if trials_since_last_flip >= trials_until_next_flip:
                # Flip the state
                flip_state = 1 - flip_state  # Toggle between 0 and 1
                trials_since_last_flip = 0
                trials_until_next_flip = random.randint(
                    FLIP_STATE_CONFIG['min_trials_before_flip'],
                    FLIP_STATE_CONFIG['max_trials_before_flip']
                )
                # Update platform rect and excluded cells for new state
                platform_rect = get_platform_rect(flip_state, grid_cells_x, grid_cells_y, img_dims)
                platform_excluded = get_platform_excluded_cells(flip_state, grid_cells_x, grid_cells_y)
                
                # Filter any newly-excluded cells out of existing queue
                target_queue = [c for c in target_queue if c not in platform_excluded]
            
            # Start next trial: pop next random target from queue
            if target_queue:
                active_target = target_queue[0]
                target_queue = target_queue[1:]
            else:
                # Queue exhausted — re-shuffle full set
                target_queue, active_target = regenerate_queue(grid_cells_x, grid_cells_y, platform_excluded)
            
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

    # Handle multi-click sequence
    # Each click cycle: click ON for click_duration, then OFF for click_gap, then next click
    if click_total > 0:
        elapsed = current_time - click_start_time
        cycle_duration = click_duration + click_gap
        current_cycle = int(elapsed / cycle_duration)
        time_in_cycle = elapsed - (current_cycle * cycle_duration)
        
        if current_cycle >= click_total:
            # All clicks delivered
            click = False
            click_count = 0
            click_total = 0
        else:
            # Within a click cycle: ON during first click_duration, OFF during gap
            click_count = current_cycle + 1
            if time_in_cycle < click_duration:
                click = True   # Pulse is ON
            else:
                click = False  # In the gap between clicks

    # Update previous poke states and set drinking state
    prev_poke_left, prev_poke_right = poke_left, poke_right
    drinking = poke_left or poke_right

    # Return values
    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right, 
            poke_left, poke_right, drinking, in_iti, click, active_target, 
            trial_count, reward_left_count, reward_right_count, punishment_count, 
            trial_phase, flip_state)