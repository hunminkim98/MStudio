"""
Data Manager for MStudio application.

This module handles all data-related operations including loading, processing,
and managing marker data and coordinate transformations.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages all data operations for the TRCViewer application.
    
    This class handles:
    - Data loading and saving
    - Coordinate system transformations
    - Data limits calculation
    - Marker name management
    - Original data backup
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.marker_names: List[str] = []
        self.num_frames: int = 0
        self.data_limits: Optional[Dict[str, Tuple[float, float]]] = None
        self.initial_limits: Optional[Dict[str, Tuple[float, float]]] = None
        self.coordinate_system: str = "y-up"  # Default coordinate system
        
    def set_data(self, data: pd.DataFrame, marker_names: List[str]) -> None:
        """
        Set the main data and marker names.
        
        Args:
            data: DataFrame containing marker coordinate data
            marker_names: List of marker names
        """
        self.data = data.copy() if data is not None else None
        self.original_data = data.copy() if data is not None else None
        self.marker_names = marker_names.copy() if marker_names else []
        self.num_frames = len(data) if data is not None else 0
        
        if self.data is not None:
            self.calculate_data_limits()
            
    def calculate_data_limits(self) -> None:
        """
        Calculate the spatial limits of the data for visualization.
        
        This method computes the min/max bounds for X, Y, Z coordinates
        with a margin for better visualization.
        """
        if self.data is None:
            self.data_limits = None
            self.initial_limits = None
            return
            
        try:
            # Get coordinate columns
            x_coords = [col for col in self.data.columns if col.endswith('_X')]
            y_coords = [col for col in self.data.columns if col.endswith('_Y')]
            z_coords = [col for col in self.data.columns if col.endswith('_Z')]
            
            if not x_coords or not y_coords or not z_coords:
                logger.warning("No coordinate columns found in data")
                return
                
            # Calculate min/max for each axis
            x_min, x_max = self.data[x_coords].min().min(), self.data[x_coords].max().max()
            y_min, y_max = self.data[y_coords].min().min(), self.data[y_coords].max().max()
            z_min, z_max = self.data[z_coords].min().min(), self.data[z_coords].max().max()
            
            # Add margin (10% of range)
            margin = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            self.data_limits = {
                'x': (x_min - x_range * margin, x_max + x_range * margin),
                'y': (y_min - y_range * margin, y_max + y_range * margin),
                'z': (z_min - z_range * margin, z_max + z_range * margin)
            }
            
            self.initial_limits = self.data_limits.copy()
            logger.info("Data limits calculated successfully")
            
        except Exception as e:
            logger.error("Error calculating data limits: %s", e, exc_info=True)
            self.data_limits = None
            self.initial_limits = None
            
    def update_keypoint_names(self, skeleton_model: Any) -> bool:
        """
        Update keypoint names based on the selected skeleton model.
        
        Args:
            skeleton_model: The skeleton model to use for naming
            
        Returns:
            bool: True if names were updated, False otherwise
        """
        if self.data is None or not self.marker_names:
            return False
            
        # Check if marker names are generic (indicating 2D data from JSON)
        is_generic_naming = all(name.startswith("Keypoint_") for name in self.marker_names)
        
        if not is_generic_naming or skeleton_model is None:
            return False
            
        try:
            # Get all nodes from the skeleton model
            skeleton_nodes = [skeleton_model]
            skeleton_nodes.extend(skeleton_model.descendants)
            
            # Create mapping from node ID to name
            id_to_name = {}
            for node in skeleton_nodes:
                if hasattr(node, 'id') and node.id is not None:
                    id_to_name[node.id] = node.name
                    
            # Update marker names and DataFrame columns
            new_data = self.data.copy()
            new_column_names = {}
            new_marker_names = []
            
            for i, old_name in enumerate(self.marker_names):
                # Use skeleton name if available, otherwise keep original
                new_name = id_to_name.get(i, old_name)
                new_marker_names.append(new_name)
                
                # Update column names
                for axis in ['X', 'Y', 'Z']:
                    old_col = f"{old_name}_{axis}"
                    new_col = f"{new_name}_{axis}"
                    if old_col in new_data.columns:
                        new_column_names[old_col] = new_col
                        
            # Apply changes
            new_data.rename(columns=new_column_names, inplace=True)
            self.data = new_data
            self.marker_names = new_marker_names
            
            # Update original data as well
            if self.original_data is not None:
                original_data_copy = self.original_data.copy()
                original_data_copy.rename(columns=new_column_names, inplace=True)
                self.original_data = original_data_copy
                
            logger.info("Keypoint names updated successfully")
            return True
            
        except Exception as e:
            logger.error("Error updating keypoint names: %s", e, exc_info=True)
            return False
            
    def restore_original_data(self) -> bool:
        """
        Restore data to its original state.
        
        Returns:
            bool: True if restoration was successful, False otherwise
        """
        if self.original_data is None:
            logger.warning("No original data to restore")
            return False
            
        try:
            self.data = self.original_data.copy(deep=True)
            logger.info("Data restored to original state")
            return True
        except Exception as e:
            logger.error("Error restoring original data: %s", e, exc_info=True)
            return False
            
    def get_marker_coordinates(self, marker_name: str, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        """
        Get the X, Y, Z coordinates for a specific marker at a specific frame.
        
        Args:
            marker_name: Name of the marker
            frame_idx: Frame index
            
        Returns:
            Tuple of (x, y, z) coordinates or None if not found
        """
        if self.data is None or frame_idx >= len(self.data):
            return None
            
        try:
            x = self.data.loc[frame_idx, f'{marker_name}_X']
            y = self.data.loc[frame_idx, f'{marker_name}_Y']
            z = self.data.loc[frame_idx, f'{marker_name}_Z']
            return (x, y, z)
        except KeyError:
            return None
            
    def clear_data(self) -> None:
        """Clear all data and reset to initial state."""
        self.data = None
        self.original_data = None
        self.marker_names = []
        self.num_frames = 0
        self.data_limits = None
        self.initial_limits = None
        logger.info("Data cleared")
        
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.data is not None and not self.data.empty
        
    def get_coordinate_columns(self, axis: str) -> List[str]:
        """
        Get all column names for a specific axis.
        
        Args:
            axis: The axis ('X', 'Y', or 'Z')
            
        Returns:
            List of column names for the specified axis
        """
        if self.data is None:
            return []
        return [col for col in self.data.columns if col.endswith(f'_{axis}')]
