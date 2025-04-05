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

# Modified function to visualize the target distribution without using Rgba
def draw_target_distribution(target_distribution, grid, img, max_intensity=255):
    # First find the maximum probability in the distribution (for normalization)
    if sum(target_distribution) > 0:
        max_prob = max(target_distribution)
    else:
        max_prob = 1.0  # Default if distribution is empty
    
    # Create a temporary overlay image
    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)
    
    # Create a color for each cell based on its probability
    for i, prob in enumerate(target_distribution):
        if prob > 0 and i < len(grid.cells):
            # Calculate normalized probability (0-1)
            norm_prob = prob / max_prob
            
            # Calculate intensity based on probability (brighter = higher probability)
            intensity = int(norm_prob * max_intensity)
            
            # Use a red color with intensity proportional to probability
            dist_color = Scalar.Rgb(intensity, 0, 0)
            
            # Draw rectangle with color intensity showing probability
            cell = grid.cells[i]
            CV.Rectangle(overlay, cell[0], cell[1], dist_color, thickness=-1)
    
    # Blend the overlay with the original image
    # We'll use a simple blend - add the images and clip values
    alpha = 0.5  # Blend factor (adjust for transparency effect)
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)
    
    return img

# Global variable to store initial target counts per cell
global initial_cell_counts
global max_initial_count
initial_cell_counts = {}
max_initial_count = 1  # Default to 1 to avoid division by zero

def draw_future_targets(target_queue, grid, img):
    # Create a combined list with current queue and active target for visualization
    all_targets = list(target_queue)

    if not all_targets:
        return img

    # Create a temporary overlay image
    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)

    # Count occurrences of each cell in the combined list
    cell_counts = {}
    for cell_idx in all_targets:
        if cell_idx in cell_counts:
            cell_counts[cell_idx] += 1
        else:
            cell_counts[cell_idx] = 1

    # Hard-code the intensity range to match the expected counts (1-5)
    min_possible_count = 1
    max_possible_count = 5
    
    # Define base and range for intensity scaling
    base_intensity = 50
    intensity_range = 205  # from 50 to 255
    
    # Draw each cell with brightness proportional to count
    for cell_idx, count in cell_counts.items():
        if cell_idx < len(grid.cells):
            # Clamp count to our expected range just in case
            clamped_count = max(min_possible_count, min(count, max_possible_count))
            
            # Calculate normalized position in range [0-1]
            normalized_position = (clamped_count - min_possible_count) / float(max_possible_count - min_possible_count)
            
            # Calculate intensity based on position in range
            intensity = base_intensity + int(normalized_position * intensity_range)
            
            # Use a brighter, more distinct color
            target_future_color = Scalar.Rgb(0, intensity, intensity)  # Cyan

            cell = grid.cells[cell_idx]
            CV.Rectangle(overlay, cell[0], cell[1], target_future_color, thickness=-1)

    # Blend overlay with image
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
# Modified to generate targets deterministically based on probability distribution
def generate_targets(grid_cells_x, grid_cells_y, max_targets_per_cell=5, shuffle=True):
    # Initialize target distribution array
    possible_targets = (grid_cells_x-2) * (grid_cells_y-2)
    
    # Create a 2D array to store probability distribution
    base_distribution = [0] * possible_targets
    
    # Calculate centers (mean) of the grid
    center_x = (grid_cells_x - 3) / 2.0
    center_y = (grid_cells_y - 3) / 2.0
    
    # Standard deviation - adjust these values to control spread
    sigma_x = (grid_cells_x - 2) / 3.0
    sigma_y = (grid_cells_y - 2) / 3.0
    
    # Generate normal distribution of probabilities
    for y in range(grid_cells_y - 2):
        for x in range(grid_cells_x - 2):
            index = y * (grid_cells_x - 2) + x
            
            # Calculate distance from center using normal distribution
            px = math.exp(-0.5 * ((x - center_x) / sigma_x) ** 2)
            py = math.exp(-0.5 * ((y - center_y) / sigma_y) ** 2)
            
            # Combined probability
            base_distribution[index] = px * py
    
    # Normalize distribution so it sums to 1
    total_prob = sum(base_distribution)
    base_distribution = [p / total_prob for p in base_distribution]
    
    # Find max probability for scaling
    max_prob = max(base_distribution)
    
    # Quantize probabilities into discrete target counts (1 to max_targets_per_cell)
    target_counts = {}
    total_targets = 0
    
    for i, prob in enumerate(base_distribution):
        # Scale to [1, max_targets_per_cell] range
        # Cells with the highest probability get max_targets_per_cell targets
        # All cells get at least 1 target
        scaled_targets = 1 + int((prob / max_prob) * (max_targets_per_cell - 1))
        target_counts[i] = scaled_targets
        total_targets += scaled_targets
    
    # Create a flat target queue with the appropriate number of targets per cell
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
        target_queue = target_queue[1:]  # Remove active target from queue
    
    # Store initial cell counts for visualization
    global initial_cell_counts
    global max_initial_count
    initial_cell_counts = target_counts.copy()
    max_initial_count = max_targets_per_cell
    
    return grid_cells_x, grid_cells_y, target_queue, base_distribution, active_target

# Draw targets and distribution
def draw_targets(active_target, target_queue, grid, img, draw_distribution=False, draw_future=False):
    # First draw distribution if enabled
    if draw_distribution:
        img = draw_target_distribution(target_distribution, grid, img)
    
    # Draw future target locations if enabled
    if draw_future:
        img = draw_future_targets(target_queue, grid, img)
    
    # Then draw the active target
    if active_target is not None and active_target < len(grid.cells):
        # Draw the active target
        cell = grid.cells[active_target]
        CV.Rectangle(img, cell[0], cell[1], target_color, thickness=-1)
    
    return img

# Here we define the number of grid cells to divide the arena by
grid_x = 7
grid_y = 15

# Define maximum targets per cell (quantization steps)
max_targets_per_cell = 5

# Initialize global variables
global target_queue  # Queue of upcoming targets
global active_target  # Currently active target
global target_distribution  # Normalized distribution for visualization

# Initialize grid dimensions explicitly
global grid_cells_x
global grid_cells_y
grid_cells_x = grid_x
grid_cells_y = grid_y

# Initialize target landscape with deterministic distribution
_, _, target_queue, target_distribution, active_target = generate_targets(grid_cells_x, grid_cells_y, max_targets_per_cell)

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
reward_state = True
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

    # Timing-related vars
    current_time = time.time()
    reward_duration_left = 0.046  # 0.051 was satiating after ~60 trials
    reward_duration_right = 0.046
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
            in_iti = False
            
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
    
    # Return values
    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right, 
            poke_left, poke_right, drinking, in_iti, click, active_target, 
            trial_count, reward_left_count, reward_right_count, tuple(target_distribution))