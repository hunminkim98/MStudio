"""
State Manager for MStudio application.

This module manages the application state including view settings, 
selection states, and mode management.
"""

import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ViewState:
    """Represents the current view state of the application."""
    show_names: bool = False
    show_trajectory: bool = False
    show_skeleton: bool = False
    is_z_up: bool = False
    coordinate_system: str = "y-up"
    

@dataclass
class SelectionState:
    """Represents the current selection state."""
    current_marker: Optional[str] = None
    pattern_markers: Set[str] = field(default_factory=set)
    analysis_markers: List[str] = field(default_factory=list)
    selected_frames: Optional[tuple] = None  # (start_frame, end_frame)
    

@dataclass
class EditingState:
    """Represents the current editing state."""
    is_editing: bool = False
    is_analysis_mode: bool = False
    pattern_selection_mode: bool = False
    filter_type: str = "butterworth"
    interp_method: str = "linear"
    interp_order: int = 3
    reference_axis: str = "X"  # Reference axis for segment angle analysis (X, Y, or Z)


class StateManager:
    """
    Manages the application state and provides state change notifications.
    
    This class handles:
    - View state management
    - Selection state tracking
    - Editing mode management
    - State change notifications
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self.view_state = ViewState()
        self.selection_state = SelectionState()
        self.editing_state = EditingState()
        
        # State change callbacks
        self._view_callbacks: List[callable] = []
        self._selection_callbacks: List[callable] = []
        self._editing_callbacks: List[callable] = []
        
        # Skeleton model state
        self.current_skeleton_model: Any = None
        self.available_skeleton_models: Dict[str, Any] = {}
        self.skeleton_pairs: List[tuple] = []
        
        logger.info("State manager initialized")
        
    def register_view_callback(self, callback: callable) -> None:
        """Register a callback for view state changes."""
        self._view_callbacks.append(callback)
        
    def register_selection_callback(self, callback: callable) -> None:
        """Register a callback for selection state changes."""
        self._selection_callbacks.append(callback)
        
    def register_editing_callback(self, callback: callable) -> None:
        """Register a callback for editing state changes."""
        self._editing_callbacks.append(callback)
        
    # View State Management
    def toggle_marker_names(self) -> bool:
        """Toggle marker name visibility."""
        self.view_state.show_names = not self.view_state.show_names
        self._notify_view_change()
        logger.info(f"Marker names {'shown' if self.view_state.show_names else 'hidden'}")
        return self.view_state.show_names
        
    def toggle_trajectory(self) -> bool:
        """Toggle trajectory visibility."""
        self.view_state.show_trajectory = not self.view_state.show_trajectory
        self._notify_view_change()
        logger.info(f"Trajectory {'shown' if self.view_state.show_trajectory else 'hidden'}")
        return self.view_state.show_trajectory
        
    def toggle_skeleton(self) -> bool:
        """Toggle skeleton visibility."""
        self.view_state.show_skeleton = not self.view_state.show_skeleton
        self._notify_view_change()
        logger.info(f"Skeleton {'shown' if self.view_state.show_skeleton else 'hidden'}")
        return self.view_state.show_skeleton
        
    def toggle_coordinate_system(self) -> str:
        """Toggle between Y-up and Z-up coordinate systems."""
        self.view_state.is_z_up = not self.view_state.is_z_up
        self.view_state.coordinate_system = "z-up" if self.view_state.is_z_up else "y-up"
        self._notify_view_change()
        logger.info(f"Coordinate system changed to {self.view_state.coordinate_system}")
        return self.view_state.coordinate_system
        
    def set_coordinate_system(self, system: str) -> None:
        """Set the coordinate system explicitly."""
        if system in ["y-up", "z-up"]:
            self.view_state.coordinate_system = system
            self.view_state.is_z_up = (system == "z-up")
            self._notify_view_change()
            logger.info(f"Coordinate system set to {system}")
            
    # Selection State Management
    def set_current_marker(self, marker_name: Optional[str]) -> None:
        """Set the currently selected marker."""
        old_marker = self.selection_state.current_marker
        self.selection_state.current_marker = marker_name
        
        if old_marker != marker_name:
            self._notify_selection_change()
            if marker_name:
                logger.info(f"Marker selected: {marker_name}")
            else:
                logger.info("Marker deselected")
                
    def add_pattern_marker(self, marker_name: str) -> bool:
        """Add a marker to the pattern selection."""
        if marker_name not in self.selection_state.pattern_markers:
            self.selection_state.pattern_markers.add(marker_name)
            self._notify_selection_change()
            logger.info(f"Added {marker_name} to pattern markers")
            return True
        return False
        
    def remove_pattern_marker(self, marker_name: str) -> bool:
        """Remove a marker from the pattern selection."""
        if marker_name in self.selection_state.pattern_markers:
            self.selection_state.pattern_markers.remove(marker_name)
            self._notify_selection_change()
            logger.info(f"Removed {marker_name} from pattern markers")
            return True
        return False
        
    def toggle_pattern_marker(self, marker_name: str) -> bool:
        """Toggle a marker in the pattern selection."""
        if marker_name in self.selection_state.pattern_markers:
            return self.remove_pattern_marker(marker_name)
        else:
            return self.add_pattern_marker(marker_name)
            
    def clear_pattern_markers(self) -> None:
        """Clear all pattern markers."""
        if self.selection_state.pattern_markers:
            self.selection_state.pattern_markers.clear()
            self._notify_selection_change()
            logger.info("Pattern markers cleared")
            
    def add_analysis_marker(self, marker_name: str, max_markers: int = 3) -> bool:
        """Add a marker to the analysis selection."""
        if marker_name in self.selection_state.analysis_markers:
            return False
            
        if len(self.selection_state.analysis_markers) >= max_markers:
            logger.warning(f"Cannot add more than {max_markers} analysis markers")
            return False
            
        self.selection_state.analysis_markers.append(marker_name)
        self._notify_selection_change()
        logger.info(f"Added {marker_name} to analysis markers")
        return True
        
    def remove_analysis_marker(self, marker_name: str) -> bool:
        """Remove a marker from the analysis selection."""
        if marker_name in self.selection_state.analysis_markers:
            self.selection_state.analysis_markers.remove(marker_name)
            self._notify_selection_change()
            logger.info(f"Removed {marker_name} from analysis markers")
            return True
        return False
        
    def clear_analysis_markers(self) -> None:
        """Clear all analysis markers."""
        if self.selection_state.analysis_markers:
            self.selection_state.analysis_markers.clear()
            self._notify_selection_change()
            logger.info("Analysis markers cleared")
            
    def set_selected_frames(self, start_frame: Optional[int], end_frame: Optional[int]) -> None:
        """Set the selected frame range."""
        if start_frame is not None and end_frame is not None:
            self.selection_state.selected_frames = (min(start_frame, end_frame), 
                                                   max(start_frame, end_frame))
        else:
            self.selection_state.selected_frames = None
        self._notify_selection_change()
        
    # Editing State Management
    def set_editing_mode(self, enabled: bool) -> None:
        """Set the editing mode state."""
        if self.editing_state.is_editing != enabled:
            self.editing_state.is_editing = enabled
            self._notify_editing_change()
            logger.info(f"Editing mode {'enabled' if enabled else 'disabled'}")
            
    def set_analysis_mode(self, enabled: bool) -> None:
        """Set the analysis mode state."""
        if self.editing_state.is_analysis_mode != enabled:
            self.editing_state.is_analysis_mode = enabled
            if not enabled:
                self.clear_analysis_markers()
            self._notify_editing_change()
            logger.info(f"Analysis mode {'enabled' if enabled else 'disabled'}")

    def toggle_analysis_mode(self) -> bool:
        """Toggle analysis mode on/off."""
        new_state = not self.editing_state.is_analysis_mode
        self.set_analysis_mode(new_state)
        return new_state

    def cycle_reference_axis(self) -> str:
        """Cycle through reference axes: X → Y → Z → X."""
        axis_cycle = {"X": "Y", "Y": "Z", "Z": "X"}
        current_axis = self.editing_state.reference_axis
        new_axis = axis_cycle.get(current_axis, "X")
        self.editing_state.reference_axis = new_axis
        self._notify_editing_change()
        logger.info(f"Reference axis changed from {current_axis} to {new_axis}")
        return new_axis

    def get_reference_vector(self):
        """Get the reference vector for the current axis."""
        import numpy as np
        axis_vectors = {
            "X": np.array([1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0])
        }
        return axis_vectors.get(self.editing_state.reference_axis, np.array([1.0, 0.0, 0.0]))

    def set_pattern_selection_mode(self, enabled: bool) -> None:
        """Set the pattern selection mode state."""
        if self.editing_state.pattern_selection_mode != enabled:
            self.editing_state.pattern_selection_mode = enabled
            if not enabled:
                self.clear_pattern_markers()
            self._notify_editing_change()
            logger.info(f"Pattern selection mode {'enabled' if enabled else 'disabled'}")
            
    def set_filter_type(self, filter_type: str) -> None:
        """Set the current filter type."""
        self.editing_state.filter_type = filter_type
        self._notify_editing_change()
        
    def set_interpolation_method(self, method: str) -> None:
        """Set the current interpolation method."""
        self.editing_state.interp_method = method
        self._notify_editing_change()
        
    # Skeleton Model Management
    def set_skeleton_model(self, model_name: str, model: Any) -> None:
        """Set the current skeleton model."""
        self.current_skeleton_model = model
        if model is None:
            self.skeleton_pairs = []
            self.view_state.show_skeleton = False
        else:
            self.view_state.show_skeleton = True
        self._notify_view_change()
        logger.info(f"Skeleton model set to: {model_name}")
        
    def update_skeleton_pairs(self, pairs: List[tuple]) -> None:
        """Update the skeleton pairs."""
        self.skeleton_pairs = pairs.copy()
        logger.info(f"Updated skeleton pairs: {len(pairs)} pairs")
        
    # State Reset and Cleanup
    def reset_all_states(self) -> None:
        """Reset all states to default values."""
        self.view_state = ViewState()
        self.selection_state = SelectionState()
        self.editing_state = EditingState()
        self.current_skeleton_model = None
        self.skeleton_pairs = []
        
        # Notify all callbacks
        self._notify_view_change()
        self._notify_selection_change()
        self._notify_editing_change()
        
        logger.info("All states reset to defaults")
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of all current states."""
        return {
            'view': {
                'show_names': self.view_state.show_names,
                'show_trajectory': self.view_state.show_trajectory,
                'show_skeleton': self.view_state.show_skeleton,
                'coordinate_system': self.view_state.coordinate_system
            },
            'selection': {
                'current_marker': self.selection_state.current_marker,
                'pattern_markers': list(self.selection_state.pattern_markers),
                'analysis_markers': self.selection_state.analysis_markers.copy(),
                'selected_frames': self.selection_state.selected_frames
            },
            'editing': {
                'is_editing': self.editing_state.is_editing,
                'is_analysis_mode': self.editing_state.is_analysis_mode,
                'pattern_selection_mode': self.editing_state.pattern_selection_mode,
                'filter_type': self.editing_state.filter_type,
                'interp_method': self.editing_state.interp_method
            }
        }
        
    # Private notification methods
    def _notify_view_change(self) -> None:
        """Notify all view state callbacks."""
        for callback in self._view_callbacks:
            try:
                callback(self.view_state)
            except Exception as e:
                logger.error(f"Error in view callback: {e}")
                
    def _notify_selection_change(self) -> None:
        """Notify all selection state callbacks."""
        for callback in self._selection_callbacks:
            try:
                callback(self.selection_state)
            except Exception as e:
                logger.error(f"Error in selection callback: {e}")
                
    def _notify_editing_change(self) -> None:
        """Notify all editing state callbacks."""
        for callback in self._editing_callbacks:
            try:
                callback(self.editing_state)
            except Exception as e:
                logger.error(f"Error in editing callback: {e}")
