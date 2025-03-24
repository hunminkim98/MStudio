"""
This module contains toggle functions for various UI elements in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

from gui.editWindow import EditWindow

def toggle_marker_names(self):
    """
    Toggles the visibility of marker names in the 3D view.
    """
    self.show_names = not self.show_names
    self.names_button.configure(text="Show Names" if not self.show_names else "Hide Names")
    self.update_plot()

def toggle_coordinates(self):
    """
    Toggles between Z-up and Y-up coordinate systems.
    """
    if self.data is None:
        return

    self.is_z_up = not self.is_z_up
    self.coord_button.configure(text="Switch to Y-up" if self.is_z_up else "Switch to Z-up")

    # Redraw static elements and coordinate axes
    self._draw_static_elements()
    self._update_coordinate_axes()

    # Update the plot with new data
    self.update_plot()
    self._draw_static_elements()
    self._update_coordinate_axes()

    # Update the plot with new data
    self.update_plot()

def toggle_trajectory(self):
    """
    Toggles the visibility of marker trajectories.
    """
    self.show_trajectory = self.trajectory_handler.toggle_trajectory()
    self.trajectory_button.configure(text="Hide Trajectory" if self.show_trajectory else "Show Trajectory")
    self.update_plot()

def toggle_edit_window(self):
    """
    Toggles the visibility of the edit window.
    """
    try:
        # focus on existing edit_window if it exists
        if hasattr(self, 'edit_window') and self.edit_window:
            self.edit_window.focus()
        else:
            # create new EditWindow
            self.edit_window = EditWindow(self)
            self.edit_window.focus()
            
    except Exception as e:
        print(f"Error in toggle_edit_window: {e}")
        import traceback
        traceback.print_exc()

def toggle_animation(self):
    """
    Toggles the animation playback between play and pause.
    """
    if not self.data is None:
        if self.is_playing:
            self.pause_animation()
        else:
            self.play_animation()
