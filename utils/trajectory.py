import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
import pandas as pd

class MarkerTrajectory:
    def __init__(self):
        self.show_trajectory = False
        self.trajectory_length = 10
        self.trajectory_line = None
        self.marker_lines = []
        self.marker_last_pos = None
        self.current_marker = None  # Track the currently selected marker

    def toggle_trajectory(self):
        """Toggle the visibility of marker trajectories"""
        self.show_trajectory = not self.show_trajectory
        return self.show_trajectory

    def set_current_marker(self, marker_name):
        """Set the currently selected marker
        
        Args:
            marker_name: Name of the selected marker
        """
        self.current_marker = marker_name

    def update_trajectory(self, data, frame_idx, marker_names, axes):
        """Update the trajectory visualization for the selected marker
        
        Args:
            data: pandas DataFrame containing marker position data
            frame_idx: Current frame index
            marker_names: List of marker names
            axes: The matplotlib 3D axes object
        """
        if not self.show_trajectory or self.current_marker is None:
            self._clear_trajectories(axes)
            return

        start_frame = max(0, frame_idx - self.trajectory_length)
        end_frame = frame_idx + 1

        for line in self.marker_lines:
            if line in axes.lines:
                axes.lines.remove(line)
        self.marker_lines.clear()

        try:
            x_col = f'{self.current_marker}_X'
            y_col = f'{self.current_marker}_Y'
            z_col = f'{self.current_marker}_Z'
            
            x = data.loc[start_frame:end_frame, x_col].values
            y = data.loc[start_frame:end_frame, y_col].values
            z = data.loc[start_frame:end_frame, z_col].values
            
            # Skip if any NaN values are present
            if not (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any()):
                line = Line3D(x, y, z, color='yellow', alpha=0.7, linewidth=1)
                axes.add_line(line)
                self.marker_lines.append(line)
        except KeyError:
            pass  # Skip if marker columns don't exist

    def _clear_trajectories(self, axes):
        """Clear all trajectory lines from the plot"""
        for line in self.marker_lines:
            if line in axes.lines:
                axes.lines.remove(line)
        self.marker_lines.clear()

    def set_trajectory_length(self, length):
        """Set the length of the trajectory trail
        
        Args:
            length: Number of frames to show in the trajectory
        """
        self.trajectory_length = max(1, int(length)) 