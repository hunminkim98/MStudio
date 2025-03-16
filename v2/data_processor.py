"""
Data processing utilities for MarkerStudio v2
Handles transformation and formatting of marker data for visualization
"""

import numpy as np
import pandas as pd


def process_marker_data(data, marker_names):
    """
    Process marker data from pandas DataFrame to a numpy array suitable for vispy visualization
    
    Args:
        data: pandas DataFrame containing marker data
        marker_names: list of marker names
    
    Returns:
        numpy array of shape (num_frames, num_markers, 3) containing the 3D positions of markers
    """
    num_frames = len(data)
    num_markers = len(marker_names)
    
    # Initialize output array
    result = np.zeros((num_frames, num_markers, 3))
    
    # For each marker, extract X, Y, Z coordinates
    for i, marker in enumerate(marker_names):
        try:
            result[:, i, 0] = data[f'{marker}_X'].values
            result[:, i, 1] = data[f'{marker}_Y'].values
            result[:, i, 2] = data[f'{marker}_Z'].values
        except KeyError as e:
            print(f"Warning: Column not found: {e}")
    
    return result


def detect_missing_markers(marker_data):
    """
    Detect missing markers (NaN values) in the marker data
    
    Args:
        marker_data: numpy array of shape (num_frames, num_markers, 3)
    
    Returns:
        binary mask of shape (num_frames, num_markers) where True indicates missing marker
    """
    # For each marker at each frame, check if any coordinate is NaN
    return np.isnan(marker_data).any(axis=2)


def interpolate_gaps(marker_data, missing_mask, method='linear'):
    """
    Interpolate missing marker data
    
    Args:
        marker_data: numpy array of shape (num_frames, num_markers, 3)
        missing_mask: binary mask of shape (num_frames, num_markers) where True indicates missing marker
        method: interpolation method ('linear', 'cubic', etc.)
    
    Returns:
        numpy array with interpolated values
    """
    num_frames, num_markers, _ = marker_data.shape
    frames = np.arange(num_frames)
    
    # Make a copy to avoid modifying the original
    result = marker_data.copy()
    
    # Interpolate each marker separately
    for m in range(num_markers):
        marker_missing = missing_mask[:, m]
        
        # Skip if no missing data or all data is missing
        if not np.any(marker_missing) or np.all(marker_missing):
            continue
        
        # Get frames with valid data
        valid_frames = frames[~marker_missing]
        
        # For each coordinate (X, Y, Z)
        for c in range(3):
            # Get valid values
            valid_values = marker_data[valid_frames, m, c]
            
            # Interpolate
            interp_values = np.interp(frames, valid_frames, valid_values)
            
            # Update result
            result[:, m, c] = interp_values
    
    return result


def process_skeleton_definition(skeleton_def, marker_names):
    """
    Process a skeleton definition to get pairs of marker indices for drawing connections
    
    Args:
        skeleton_def: dictionary containing skeleton definition (from Pose2Sim)
        marker_names: list of marker names in the data
    
    Returns:
        list of tuples containing indices of markers to connect
    """
    if skeleton_def is None:
        return []
    
    # Create a mapping from marker names to indices
    marker_to_idx = {name: i for i, name in enumerate(marker_names)}
    
    # Extract links from skeleton definition
    links = skeleton_def.get('links', [])
    
    # Convert links to pairs of indices
    pairs = []
    for link in links:
        if len(link) >= 2:
            # Try to find both markers
            if link[0] in marker_to_idx and link[1] in marker_to_idx:
                pairs.append((marker_to_idx[link[0]], marker_to_idx[link[1]]))
    
    return pairs
