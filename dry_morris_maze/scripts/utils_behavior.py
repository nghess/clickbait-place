import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# FILE I/O AND PATH UTILITIES
# ============================================================================

def natural_sort_key(path):
    """
    Create a key for sorting strings with numbers naturally.
    Converts "1", "2", "10" into proper numeric order instead of lexicographic order.
    """
    # Convert path to string if it's a Path object
    path_str = str(path)
    # Split string into chunks of numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', path_str)]

def get_file_paths(directory: str = '', extension: str = '', keyword: str = '', session_type: str = '', keyword_exact=False, keyword_bool=False, not_keyword='', print_paths=False, print_n=np.inf) -> list:
    """
    Get all files matching extension and keyword in a directory and its subdirectories.

    Parameters:
    -----------
    directory : str
        Directory to search in
    extension : str
        File extension to match (e.g., 'npy', 'csv')
    keyword : str
        Keyword to match in filename
    session_type : str
        Session type prefix to filter by (e.g., 'm' for mouse sessions)
    keyword_exact : bool
        If True, keyword must match filename exactly
    keyword_bool : bool
        If True, apply not_keyword filter
    not_keyword : str
        Keyword to exclude from results
    print_paths : bool
        If True, print found paths
    print_n : int
        Number of paths to print (if print_paths=True)

    Returns:
    --------
    list
        List of Path objects matching the criteria, sorted naturally
    """
    if keyword_bool:
        if keyword_exact:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/{keyword}.{extension}') if not_keyword not in str(f)]
        else:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name and not_keyword not in str(f)]
    else:
        if keyword_exact:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/{keyword}.{extension}')]
        else:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name]
    # Sort paths using natural sorting
    paths = sorted(paths, key=natural_sort_key)
    print(f'Found {len(paths)} {keyword}.{extension} files')
    if print_paths:
            show_paths(paths, print_n)
    return paths

def show_paths(data_paths, print_n=np.inf):
    """
    Print collected paths and their indices.

    Parameters:
    -----------
    data_paths : list
        List of paths to print
    print_n : int
        Maximum number of paths to print
    """
    for ii, path in enumerate(data_paths[:min(print_n, len(data_paths))]):
        print(f"{ii} {path}")

def filter_paths(paths_to_filter, paths_to_reference):
    """
    Paths are valid only if their final two parent directories (mouse and session ids)
    match the reference paths.

    Parameters:
    -----------
    paths_to_filter : list
        Paths to filter
    paths_to_reference : list
        Reference paths to match against

    Returns:
    --------
    list
        Filtered paths
    """
    filtered_paths = []
    for path in paths_to_filter:
        for ref_path in paths_to_reference:
            if (path.parents[1].name == ref_path.parents[1].name and
                path.parents[0].name == ref_path.parents[0].name):
                filtered_paths.append(path)
                break
    return filtered_paths

def filter_paths_numeric(paths, session_type: str, min_value):
    """
    Filter paths by checking if session number meets minimum threshold.

    Parameters:
    -----------
    paths : list
        Paths to filter
    session_type : str
        Session type prefix (e.g., 'm' for mouse)
    min_value : int
        Minimum session number to include

    Returns:
    --------
    list
        Filtered paths
    """
    return [
        p for p in paths
        if not any(
            part.startswith(session_type) and part[1:].isdigit() and int(part[1:]) < min_value
            for part in p.parts
        )
    ]

def filter_paths_by_session_id(data_paths, session_id_threshold: int, part=-2, greater=True):
    """
    Filter paths by checking if session id is greater/less than a given threshold.

    Parameters:
    -----------
    data_paths : list
        Paths to filter
    session_id_threshold : int
        Threshold value for comparison
    part : int
        Index of path part containing session_id (default: -2)
    greater : bool
        If True, keep sessions > threshold; if False, keep sessions < threshold

    Returns:
    --------
    list
        Filtered paths
    """
    filtered_paths = []
    for path in data_paths:
        session_id = path.parts[part] # Extract session_id
        numeric_value = int(re.search(r'\d+', session_id).group()) # Extract numeric part of session_id
        if greater:
            if int(numeric_value) > session_id_threshold:
                filtered_paths.append(path)
        else:
            if int(numeric_value) < session_id_threshold:
                filtered_paths.append(path)
    return filtered_paths

def get_savedirs(path):
    """
    Get the name of the last two directories in data path.

    Parameters:
    -----------
    path : str or Path
        Path to extract directories from

    Returns:
    --------
    str
        Last two directories joined by os separator
    """
    path = str(path)
    parts = path.split(os.path.sep)
    return os.path.sep.join(parts[-3:-1])

def get_ids_from_path(data_paths, part=3):
    """
    Function to extract session id and mouse id from Open Ephys save directory name.

    Parameters:
    -----------
    data_paths : list
        List of Path objects
    part : int
        Index of path part containing the Open Ephys save directory

    Returns:
    --------
    tuple
        (session_ids, mouse_ids) - lists of extracted IDs
    """
    session_ids = []
    mouse_ids = []

    for data_path in data_paths:
        open_ephys_savedir = data_path.parts[part] # Open Ephys save path is in index 3 of full path
        session_id = open_ephys_savedir.split('_')[0]  # Session ID is before the first underscore
        mouse_id = open_ephys_savedir.split('_')[-1]  # Mouse ID is after the last underscore
        session_ids.append(session_id)
        mouse_ids.append(mouse_id)

    return session_ids, mouse_ids

# ============================================================================
# BEHAVIORAL DATA PROCESSING
# ============================================================================

def process_events(idx, event_paths_a, event_paths_b, columns: dict):
    """
    Process event data from paired CSV files.

    Parameters:
    -----------
    idx : int
        Index of the session to process
    event_paths_a : list
        List of paths to eventsA CSV files
    event_paths_b : list
        List of paths to eventsB CSV files
    columns : dict
        Dictionary of variable names and datatypes

    Returns:
    --------
    DataFrame
        DataFrame containing processed event data
    """
    # Load events .csv part A (This .csv is always 7 cols wide, but we don't hardcode in case Bonsai WriteCsv improves.)
    event_data_a = pd.read_csv(event_paths_a[idx])
    col_names_a = list(columns.keys())[:len(event_data_a.columns)]
    event_data_a.columns = col_names_a
    pd.to_datetime(event_data_a['timestamp'])

    # Load events .csv part B (col number varies by experimental condition, so we need to load in col names flexibly.)
    event_data_b = pd.read_csv(event_paths_b[idx])
    col_names_b = list(columns.keys())[len(event_data_a.columns):]
    event_data_b.columns = col_names_b

    # Concatenate eventsA and eventsB dataframes
    if len(event_data_a) == len(event_data_b):
        event_data = pd.concat([event_data_a, event_data_b], axis=1)
    else:
        print("Event dataframes must contain same number of rows")
        min_length = min(len(event_data_a), len(event_data_b))
        max_length = max(len(event_data_a), len(event_data_b))
        print(f"Trimmed long dataframe by {max_length-min_length} rows.")
        event_data_a = event_data_a.iloc[:min_length]
        event_data_b = event_data_b.iloc[:min_length]
        event_data = pd.concat([event_data_a, event_data_b], axis=1)

    # Set columns to appropriate types as specified in columns dictionary
    event_data = event_data.astype(columns)

    # Calculate speed and direction columns
    event_data['speed'] = calculate_speed(event_data)
    event_data['direction'] = calculate_direction(event_data)  # Check that this is working

    # Rebuild 'reward_state' as periods between click onset and poke events
    # reward_state = True from when 'click' goes False->True until either 'poke_left' or 'poke_right' goes False->True
    reward_state = np.zeros(len(event_data), dtype=bool)

    # Find click transitions from False to True
    click_onsets = np.where((event_data['click'].shift(1) == False) &
                           (event_data['click'] == True))[0]

    # Find poke transitions from False to True (either left or right)
    poke_left_onsets = np.where((event_data['poke_left'].shift(1) == False) &
                               (event_data['poke_left'] == True))[0]
    poke_right_onsets = np.where((event_data['poke_right'].shift(1) == False) &
                                (event_data['poke_right'] == True))[0]

    # Combine and sort all poke onsets
    all_poke_onsets = np.sort(np.concatenate([poke_left_onsets, poke_right_onsets]))

    # For each click onset, find the next poke onset and mark the period as reward_state
    for click_idx in click_onsets:
        # Find the next poke onset after this click onset
        next_pokes = all_poke_onsets[all_poke_onsets > click_idx]
        if len(next_pokes) > 0:
            poke_idx = next_pokes[0]
            # Mark reward_state period from click onset to poke onset
            reward_state[click_idx:poke_idx] = True

    event_data['reward_state'] = reward_state

    # Rename centroids so that they don't get confused with SLEAP centroid
    event_data = event_data.rename(columns={'centroid_x' : 'bonsai_centroid_x'})
    event_data = event_data.rename(columns={'centroid_y' : 'bonsai_centroid_y'})

    # Add 'drinking' column (period between reward initiation poke and start of ITI)
    event_data['drinking'] = calculate_drinking(event_data)

    return event_data

def calculate_speed(data):
    """
    Calculate speed from x,y coordinates.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'centroid_x' and 'centroid_y' columns

    Returns:
    --------
    np.ndarray
        Speed values (0 prepended to match data length)
    """
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y'])
    speed = np.sqrt(dx**2 + dy**2)
    # Add 0 at start to match length
    return np.concatenate(([0], speed))

def calculate_direction(data):
    """
    Calculate movement direction in radians from x,y coordinates.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'centroid_x' and 'centroid_y' columns

    Returns:
    --------
    np.ndarray
        Direction values in radians (0 prepended to match data length)
    """
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y'])
    direction = np.arctan2(dy, dx)
    # Add 0 at start to match length
    return np.concatenate(([0], direction))

def calculate_drinking(data):
    """
    Calculate drinking periods between reward_state True->False and ITI False->True.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'reward_state' and 'iti' columns

    Returns:
    --------
    np.ndarray
        Boolean array indicating drinking periods
    """
    drinking = np.zeros(len(data), dtype=bool)

    # Find reward_state transitions from True to False
    reward_transitions = np.where((data['reward_state'].shift(1) == True) &
                                 (data['reward_state'] == False))[0]

    # Find iti transitions from False to True
    iti_transitions = np.where((data['iti'].shift(1) == False) &
                              (data['iti'] == True))[0]

    # For each reward_state True->False transition, find the next iti False->True transition
    for reward_idx in reward_transitions:
        # Find the next iti transition after this reward transition
        next_iti = iti_transitions[iti_transitions > reward_idx]
        if len(next_iti) > 0:
            iti_idx = next_iti[0]
            # Mark drinking period from reward transition to iti transition
            drinking[reward_idx:iti_idx] = True

    return drinking