import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random
import System
from System import Array

# Enhanced 2D distribution generator for IronPython
def generate_2d_distribution(x_size, y_size, mean_x=None, mean_y=None, 
                           sigma_x=None, sigma_y=None, log_normal=False,
                           log_sigma_x=0.5, log_sigma_y=0.5, flip_log_y=False):
    """
    Generate a 2D normal or log-normal distribution for IronPython.
    
    Parameters:
    -----------
    x_size : int
        Width of the grid (number of cells in x direction)
    y_size : int
        Height of the grid (number of cells in y direction)
    mean_x : float, optional
        X-coordinate of the mean (default: center of x-axis)
    mean_y : float, optional
        Y-coordinate of the mean (default: center of y-axis)
    sigma_x : float, optional
        Standard deviation in x direction (default: x_size/6)
    sigma_y : float, optional
        Standard deviation in y direction (default: y_size/6)
    log_normal : bool, str, or list
        Control log-normal distribution:
        - False: normal distribution for both axes
        - True: log-normal for both axes  
        - 'x': log-normal for x-axis only
        - 'y': log-normal for y-axis only
    log_sigma_x : float
        Standard deviation in log-space for x-axis log-normal (default: 0.5)
    log_sigma_y : float
        Standard deviation in log-space for y-axis log-normal (default: 0.5)
    flip_log_y : bool
        If True and log_normal includes 'y', flips the y-axis log-normal distribution
        
    Returns:
    --------
    distribution : list
        1D list containing the flattened distribution values
    """
    # Set default parameters
    if mean_x is None:
        mean_x = x_size / 2.0
    if mean_y is None:
        mean_y = y_size / 2.0
        
    if sigma_x is None:
        sigma_x = x_size / 6.0
    if sigma_y is None:
        sigma_y = y_size / 6.0
    
    # Parse log_normal parameter
    if log_normal is True:
        log_x, log_y = True, True
    elif log_normal is False:
        log_x, log_y = False, False
    elif log_normal == 'x':
        log_x, log_y = True, False
    elif log_normal == 'y':
        log_x, log_y = False, True
    elif hasattr(log_normal, '__iter__'):  # Handle list/tuple in IronPython
        log_x = 'x' in log_normal
        log_y = 'y' in log_normal
    else:
        raise ValueError("log_normal must be True, False, 'x', 'y', or a list containing 'x' and/or 'y'")
    
    # Generate distribution
    distribution = []
    
    for y in range(y_size):
        for x in range(x_size):
            # Calculate distributions for each axis separately
            if log_x:
                # Log-normal for X axis
                log_mean_x = math.log(max(mean_x, 1e-10))
                x_pos = max(x + 1, 1e-10)  # Add 1 to avoid log(0), ensure positive
                z_x = math.exp(-((math.log(x_pos) - log_mean_x)**2 / (2 * log_sigma_x**2)))
            else:
                # Normal for X axis
                z_x = math.exp(-((x - mean_x)**2 / (2 * sigma_x**2)))
            
            if log_y:
                # Log-normal for Y axis
                log_mean_y = math.log(max(mean_y, 1e-10))
                if flip_log_y:
                    # Flip the log-normal by using (y_size - y - 1) instead of y
                    y_flipped = y_size - y - 1
                    y_pos = max(y_flipped + 1, 1e-10)
                else:
                    y_pos = max(y + 1, 1e-10)  # Add 1 to avoid log(0), ensure positive
                z_y = math.exp(-((math.log(y_pos) - log_mean_y)**2 / (2 * log_sigma_y**2)))
            else:
                # Normal for Y axis
                z_y = math.exp(-((y - mean_y)**2 / (2 * sigma_y**2)))
            
            # Combine the distributions (multiply since they're independent)
            z = z_x * z_y
            distribution.append(z)
    
    # Normalize distribution so it sums to 1
    total_prob = sum(distribution)
    if total_prob > 0:
        distribution = [p / total_prob for p in distribution]
    
    return distribution

# Class to generate maze coordinates (unchanged)
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

# Function to extract dims from image (unchanged)
def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]

# Enhanced target generation function
def generate_targets(grid_cells_x, grid_cells_y, max_targets_per_cell=5, shuffle=True,
                    mean_x=None, mean_y=None, sigma_x=None, sigma_y=None, 
                    log_normal=False, log_sigma_x=0.5, log_sigma_y=0.5, flip_log_y=False):
    """
    Generate targets with enhanced distribution control.
    
    Parameters:
    -----------
    grid_cells_x, grid_cells_y : int
        Grid dimensions
    max_targets_per_cell : int
        Maximum number of targets per cell
    shuffle : bool
        Whether to shuffle the target queue
    mean_x, mean_y : float, optional
        Mean position of the distribution
    sigma_x, sigma_y : float, optional
        Standard deviation for normal distribution
    log_normal : bool, str, or list
        Log-normal distribution control
    log_sigma_x, log_sigma_y : float
        Log-space standard deviations
        
    Returns:
    --------
    tuple : (grid_cells_x, grid_cells_y, target_queue, distribution, active_target)
    """
    # Calculate the actual grid size (excluding border cells)
    inner_x = grid_cells_x - 2
    inner_y = grid_cells_y - 2
    
    # Set default parameters if not provided
    if mean_x is None:
        mean_x = (inner_x - 1) / 2.0  # Center of 0-indexed grid
    if mean_y is None:
        mean_y = (inner_y - 1) / 2.0  # Center of 0-indexed grid
    if sigma_x is None:
        sigma_x = inner_x / 3.0
    if sigma_y is None:
        sigma_y = inner_y / 3.0
    
    # Generate the 2D distribution
    base_distribution = generate_2d_distribution(
        inner_x, inner_y, mean_x, mean_y, sigma_x, sigma_y,
        log_normal, log_sigma_x, log_sigma_y, flip_log_y
    )
    
    # Find max probability for scaling
    max_prob = max(base_distribution) if base_distribution else 1.0
    
    # Quantize probabilities into discrete target counts
    target_counts = {}
    total_targets = 0
    
    for i, prob in enumerate(base_distribution):
        # Scale to [1, max_targets_per_cell] range
        scaled_targets = 0 + int((prob / max_prob) * (max_targets_per_cell - 1))
        target_counts[i] = scaled_targets
        total_targets += scaled_targets
    
    # Create target queue
    target_queue = []
    for cell_idx, count in target_counts.items():
        for _ in range(count):
            target_queue.append(cell_idx)
    
    if shuffle:
        random.shuffle(target_queue)

    # Set the first active target
    active_target = None
    if target_queue:
        active_target = target_queue[0]
        target_queue = target_queue[1:]
    
    # Store initial cell counts for visualization
    global initial_cell_counts
    global max_initial_count
    initial_cell_counts = target_counts.copy()
    max_initial_count = max_targets_per_cell
    
    return grid_cells_x, grid_cells_y, target_queue, base_distribution, active_target

# Function to get current distribution config based on flip state
def get_current_distribution_config():
    """Return the appropriate distribution config based on current flip_state."""
    if flip_state == 0:
        return DISTRIBUTION_CONFIG_STATE_0
    else:
        return DISTRIBUTION_CONFIG_STATE_1

# Visualization functions (unchanged from your original)
def draw_target_distribution(target_distribution, grid, img, max_intensity=255):
    if sum(target_distribution) > 0:
        max_prob = max(target_distribution)
    else:
        max_prob = 1.0
    
    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)
    
    for i, prob in enumerate(target_distribution):
        if prob > 0 and i < len(grid.cells):
            norm_prob = prob / max_prob
            intensity = int(norm_prob * max_intensity)
            dist_color = Scalar.Rgb(intensity, 0, 0)
            cell = grid.cells[i]
            CV.Rectangle(overlay, cell[0], cell[1], dist_color, thickness=-1)
    
    alpha = 0.5
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)
    return img

def draw_future_targets(target_queue, grid, img):
    all_targets = list(target_queue)
    if not all_targets:
        return img

    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)
    cell_counts = {}
    for cell_idx in all_targets:
        if cell_idx in cell_counts:
            cell_counts[cell_idx] += 1
        else:
            cell_counts[cell_idx] = 1

    min_possible_count = 1
    max_possible_count = 5
    base_intensity = 50
    intensity_range = 205
    
    for cell_idx, count in cell_counts.items():
        if cell_idx < len(grid.cells):
            clamped_count = max(min_possible_count, min(count, max_possible_count))
            normalized_position = (clamped_count - min_possible_count) / float(max_possible_count - min_possible_count)
            intensity = base_intensity + int(normalized_position * intensity_range)
            target_future_color = Scalar.Rgb(0, intensity, intensity)
            cell = grid.cells[cell_idx]
            CV.Rectangle(overlay, cell[0], cell[1], target_future_color, thickness=-1)

    alpha = 0.5
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)
    return img

def get_grid_location(grid, centroid_x, centroid_y, active_target, img):
    cell_width = grid.bounds[0] // grid.shape[0]
    cell_height = grid.bounds[1] // grid.shape[1]
    
    grid_x = int(centroid_x // cell_width) - 1
    grid_y = int(centroid_y // cell_height) - 1
    
    target_found = False
    
    if 0 <= grid_x < grid.shape[0]-2 and 0 <= grid_y < grid.shape[1]-2:
        cell_index = grid_y * (grid.shape[0]-2) + grid_x
        
        if 0 <= cell_index < len(grid.cells):
            cell = grid.cells[cell_index]
            CV.Rectangle(img, cell[0], cell[1], mouse_loc_color, thickness=-1)
            
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

def draw_targets(active_target, target_queue, grid, img, draw_distribution=False, draw_future=False):
    if draw_distribution:
        img = draw_target_distribution(target_distribution, grid, img)
    
    if draw_future:
        img = draw_future_targets(target_queue, grid, img)
    
    if active_target is not None and active_target < len(grid.cells):
        cell = grid.cells[active_target]
        CV.Rectangle(img, cell[0], cell[1], target_color, thickness=-1)
    
    return img

# Configuration parameters - MODIFY THESE TO CONTROL YOUR DISTRIBUTION
grid_x = 7
grid_y = 15
max_targets_per_cell = 5

# Flip state parameters
FLIP_STATE_CONFIG = {
    'min_trials_before_flip': 5,    # Minimum trials before state flip
    'max_trials_before_flip': 10,   # Maximum trials before state flip
}

# Distribution configurations for each flip state
DISTRIBUTION_CONFIG_STATE_0 = {
    # Mean position (None = center)
    'mean_x': None,  # Try: 2.0 for left-shifted, 4.0 for right-shifted
    'mean_y': 3,     # Top-focused
    
    # Standard deviations for normal distribution
    'sigma_x': None,  # Try: 1.0 for narrow, 3.0 for wide
    'sigma_y': None,  # Try: 2.0 for narrow, 5.0 for wide
    
    # Log-normal control
    'log_normal': 'y',  # Options: False, True, 'x', 'y', ['x', 'y']
    
    # Log-space standard deviations (for log-normal axes)
    'log_sigma_x': 0.5,  # 0.2-0.4: tight, 0.5: moderate, 0.8-1.2: fat tails
    'log_sigma_y': 0.25,  # 0.2-0.4: tight, 0.5: moderate, 0.8-1.2: fat tails
    'flip_log_y': False,  # Normal log-normal for state 0
}

DISTRIBUTION_CONFIG_STATE_1 = {
    # Perfect mirror: bottom-focused distribution using flipped log-normal
    'mean_x': None,  # Keep centered
    'mean_y': 3,     # Same mean position as state 0
    
    'sigma_x': None,
    'sigma_y': None,
    
    'log_normal': 'y',
    'log_sigma_x': 0.5,
    'log_sigma_y': 0.25,   # Same parameters as state 0
    'flip_log_y': True,   # Flip the log-normal distribution
}

# Initialize global variables
global target_queue
global active_target
global target_distribution
global grid_cells_x
global grid_cells_y
global initial_cell_counts
global max_initial_count
global flip_state
global trials_since_last_flip
global trials_until_next_flip

grid_cells_x = grid_x
grid_cells_y = grid_y
initial_cell_counts = {}
max_initial_count = 1

# Initialize flip state variables
flip_state = 0
trials_since_last_flip = 0
trials_until_next_flip = random.randint(
    FLIP_STATE_CONFIG['min_trials_before_flip'],
    FLIP_STATE_CONFIG['max_trials_before_flip']
)

# Initialize target landscape with enhanced distribution (state 0)
current_config = get_current_distribution_config()
_, _, target_queue, target_distribution, active_target = generate_targets(
    grid_cells_x, grid_cells_y, max_targets_per_cell, **current_config
)

# Initialize other global variables (unchanged from your original)
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
target_color = Scalar.Rgb(255, 255, 255)
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
    global target_distribution
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
    global flip_state
    global trials_since_last_flip
    global trials_until_next_flip

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
    # Flag to track if we just completed a trial
    trial_completed_this_frame = False

    # Load realtime variables from Zip node
    centroid_x, centroid_y, image = value[0].Item1, value[0].Item2, value[0].Item3
    poke_left, poke_right = bool(value[1][0]), bool(value[1][1])

    # Process grid and canvas
    grid_loc_x, grid_loc_y = None, None
    img_dims = get_image_shape(image)
    grid = GridMaze(img_dims, [grid_cells_x, grid_cells_y])
    canvas = create_blank_canvas(img_dims[0], img_dims[1])
    
    # Draw targets and distribution
    draw_targets(active_target, target_queue, grid, canvas, draw_distribution=False, draw_future=True)

    # Process mouse position and check for target
    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_x, grid_loc_y, target_found_this_frame = get_grid_location(grid, centroid_x, centroid_y, active_target, canvas)
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)
        
        # If target found and we're not in reward state yet
        if target_found_this_frame and active_target is not None and not reward_state:
            # Remove the found target
            active_target = None
            
            # Trigger reward state
            reward_state = True
            click = True
            click_start_time = current_time

    # State machine logic
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            trials_since_last_flip += 1
            trial_completed_this_frame = True
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
                
                # Generate new targets with the new distribution
                current_config = get_current_distribution_config()
                _, _, target_queue, target_distribution, active_target = generate_targets(
                    grid_cells_x, grid_cells_y, max_targets_per_cell, **current_config
                )
            else:
                # Set next target if we need to (only if active target is None)
                if active_target is None and target_queue:
                    active_target = target_queue[0]
                    target_queue = target_queue[1:]
            
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

    # Convert target_queue to tuple for return
    queue_tuple = tuple(target_queue) if target_queue else tuple()
    
    # Return values (added flip_state at the end)
    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right, 
            poke_left, poke_right, drinking, in_iti, click, active_target, 
            trial_count, reward_left_count, reward_right_count, tuple(target_distribution), flip_state)