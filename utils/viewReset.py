"""
This module contains functions for resetting view states in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

def reset_main_view(self):
    """
    Resets the main 3D view to show all data within the calculated data limits.
    """
    if self.data_limits:
        self.ax.set_xlim(self.data_limits['x'])
        self.ax.set_ylim(self.data_limits['y'])
        self.ax.set_zlim(self.data_limits['z'])
        self.ax.set_box_aspect([1,1,1])  # Force equal aspect ratio
        self.canvas.draw()

def reset_graph_view(self):
    """
    Resets the marker graph view to its initial limits.
    """
    if hasattr(self, 'marker_axes') and hasattr(self, 'initial_graph_limits'):
        for ax, limits in zip(self.marker_axes, self.initial_graph_limits):
            ax.set_xlim(limits['x'])
            ax.set_ylim(limits['y'])
        self.marker_canvas.draw()
