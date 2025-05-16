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
# __version__ = version('MStudio')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"

import logging
import numpy as np

logger = logging.getLogger(__name__)

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
        logger.error("Error in toggle_edit_window: %s", e, exc_info=True)

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
    
    # --- Step 1: Ensure skeleton state is correct FIRST --- 
    if hasattr(self, 'on_model_change') and hasattr(self, 'model_var'):
        current_model = self.model_var.get()
        if current_model != 'No skeleton': # Avoid unnecessary calls if no skeleton is selected
            self.on_model_change(current_model) 
    # ---------------------------------------------------------

    # --- Step 2: Update renderer data with correct skeleton info ---
    # Call update_data on the renderer with the expected arguments
    if hasattr(self.gl_renderer, 'update_data') and self.data is not None:
        self.gl_renderer.update_data(
            data=self.data, 
            frame_idx=self.frame_idx
        )

    # --- Step 3: Trigger redraw ---
    # Existing redraw logic
    if hasattr(self.gl_renderer, '_force_redraw'):
        self.gl_renderer._force_redraw()

    # Existing delayed redraw as fallback
    if hasattr(self.gl_renderer, 'redraw'):
        self.after(100, lambda: self.gl_renderer.redraw())


# TODO for analysis mode:
# 1. Distance (and dotted line?) visualization between two selected markers
# 2. Joint angle (and arc)visualization for three selected markers
def toggle_analysis_mode(self):
    """Toggles the analysis mode on/off."""
    self.is_analysis_mode = not self.is_analysis_mode
    logger.info(f"Analysis mode {'enabled' if self.is_analysis_mode else 'disabled'}.")

    # --- Analysis mode: add/remove Neck and Hip keypoints ---
    if self.data is not None:
        # Define keypoints and their corresponding left/right markers
        pairs = [('Neck','RShoulder','LShoulder'),('Hip','RHip','LHip')]
        for name, left, right in pairs:
            xyz = ['X','Y','Z']
            cols = [f"{name}_{ax}" for ax in xyz]
            if self.is_analysis_mode:
                # Add averaged keypoint columns if left/right exist
                if all(f"{m}_{ax}" in self.data.columns for m in (left, right) for ax in xyz):
                    for col, ax in zip(cols, xyz):
                        if col not in self.data.columns:
                            self.data[col] = (self.data[f"{left}_{ax}"] + self.data[f"{right}_{ax}"])/2
                    if name not in self.marker_names:
                        self.marker_names.append(name)
            else:
                # Remove keypoint columns and marker name
                to_drop = [c for c in cols if c in self.data.columns]
                if to_drop:
                    self.data.drop(columns=to_drop, inplace=True)
                if name in self.marker_names:
                    self.marker_names.remove(name)
    # ----------------------------------------------------------

    # Clear analysis markers when exiting the mode
    if not self.is_analysis_mode:
        self.analysis_markers.clear()
        # Update renderer state if needed (e.g., remove highlights)
        if hasattr(self, 'gl_renderer'):
            # We need a way to tell the renderer about the mode change and selected markers
            # Let's assume a method `set_analysis_state` exists or will be added
            self.gl_renderer.set_analysis_state(self.is_analysis_mode, self.analysis_markers)
            self.gl_renderer.redraw() # Redraw to remove highlights/text
    else:
        # Inform the user how to use the mode (Optional)
        # messagebox.showinfo("Analysis Mode", "Analysis mode enabled. Left-click markers in the 3D view to select for analysis (up to 3).")
        # Ensure renderer is aware of the mode change
         if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_analysis_state(self.is_analysis_mode, self.analysis_markers)
            self.gl_renderer.redraw()

    # Visually update the button state
    if hasattr(self, 'analysis_button'):
        if self.is_analysis_mode:
            # Indicate active state (e.g., change color, text)
            self.analysis_button.configure(fg_color="#00529B") # Example: Blue color when active
        else:
            # Indicate inactive state (e.g., default color)
            self.analysis_button.configure(fg_color=["#3B3B3B", "#3B3B3B"]) # Example: Default button color
