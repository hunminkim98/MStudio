"""
This module contains toggle functions for various UI elements in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MEditor')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"

def toggle_marker_names(self):
    """
    Toggles the visibility of marker names in the 3D view.
    """
    self.show_names = not self.show_names
    self.names_button.configure(text="Show Names" if not self.show_names else "Hide Names")
    
    # pass the display setting to the OpenGL renderer
    if hasattr(self, 'gl_renderer'):
        self.gl_renderer.set_show_marker_names(self.show_names)
        
    self.update_plot()


def toggle_trajectory(self):
    """Toggle the visibility of marker trajectories"""
    # use the previous trajectory_handler to directly switch the state
    self.show_trajectory = not self.show_trajectory
    
    # if using the OpenGL renderer, pass the state to the renderer
    if hasattr(self, 'gl_renderer') and self.gl_renderer:
        self.gl_renderer.set_show_trajectory(self.show_trajectory)
    
    # update the screen
    self.update_plot()
    
    # update the toggle button text
    if hasattr(self, 'trajectory_button'):
        text = "Hide Trajectory" if self.show_trajectory else "Show Trajectory"
        self.trajectory_button.configure(text=text)
    
    return self.show_trajectory

def toggle_edit_window(self):
    """
    Toggles the edit mode for the marker plot.
    This now uses the integrated edit UI rather than a separate window.
    """
    try:
        # Use the new toggle_edit_mode method
        self.toggle_edit_mode()
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



 # TODO for coordinate system manager:
 # 1. Add a X-up coordinate system
 # 2. Add a left-handed coordinate system
def toggle_coordinates(self):
    
    """Toggle between Z-up and Y-up coordinate systems"""
    # save the previous state
    previous_state = self.is_z_up

    # switch the coordinate system state
    self.is_z_up = not self.is_z_up
    
    # update the button text
    button_text = "Switch to Y-up" if self.is_z_up else "Switch to Z-up"
    if hasattr(self, 'coord_button'):
        self.coord_button.configure(text=button_text)
        self.update_idletasks()  # update the UI immediately
    
    # change the coordinate system setting
    self.coordinate_system = "z-up" if self.is_z_up else "y-up"
    
    # pass the coordinate system change to the OpenGL renderer
    if hasattr(self, 'gl_renderer'):
        if hasattr(self.gl_renderer, 'set_coordinate_system'):
            self.gl_renderer.set_coordinate_system(self.is_z_up)
        
        # request the screen to be forcefully updated - with a slight delay
        self.after(50, lambda: _force_update_opengl(self))


def _force_update_opengl(self):
    """Forcefully update the OpenGL renderer's screen."""
    if not hasattr(self, 'gl_renderer'):
        return
        
    try:
        # reset the current frame to force the update
        if self.data is not None:
            self.gl_renderer.set_frame_data(
                data=self.data,
                frame_idx=self.frame_idx,
                marker_names=self.marker_names,
                current_marker=getattr(self, 'current_marker', None), # Use getattr for safety
                show_marker_names=getattr(self, 'show_names', False), # Use getattr
                show_trajectory=getattr(self, 'show_trajectory', False), # Use getattr
                coordinate_system="z-up" if self.is_z_up else "y-up",
                skeleton_pairs=self.skeleton_pairs if hasattr(self, 'skeleton_pairs') else None
            )
            
            # update the screen command
            self.gl_renderer._force_redraw()
            
            # request the screen to be forcefully updated again (for safety)
            self.after(100, lambda: self.gl_renderer.redraw())
            
    except Exception as e:
        print(f"Error in _force_update_opengl within viewToggles: {e}") # Add error logging
        import traceback
        traceback.print_exc()


# TODO for analysis mode:
# 1. Distance (and dotted line?) visualization between two selected markers
# 2. Joint angle (and arc)visualization for three selected markers
def toggle_analysis_mode(self):
    """Toggles the analysis mode on and off."""
    self.is_analysis_mode = not self.is_analysis_mode
    if self.is_analysis_mode:
        print("Analysis mode activated.")
        # Potentially change button appearance or disable other interactions
        self.analysis_button.configure(fg_color="#00A6FF") # Example: Highlight button
    else:
        print("Analysis mode deactivated.")
        # Restore button appearance and re-enable other interactions
        button_style = {
            "fg_color": "#333333",
            "hover_color": "#444444"
        }
        self.analysis_button.configure(**button_style) # Example: Restore default style
