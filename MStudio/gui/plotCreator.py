import logging

logger = logging.getLogger(__name__)

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

def create_plot(self):
    """
    Creates the main 3D plot for displaying marker data.
    This function was extracted from the main class to improve code organization.
    
    Now supports only OpenGL rendering mode.
    """
    try:
        from MStudio.gui.opengl.GLMarkerRenderer import MarkerGLRenderer
        
        # If there is an existing canvas, destroy it
        if hasattr(self, 'canvas') and self.canvas:
            if hasattr(self.canvas, 'get_tk_widget'):
                try:
                    # Attempt to destroy the widget if it exists
                    widget = self.canvas.get_tk_widget()
                    if widget.winfo_exists():
                        widget.destroy()
                except Exception as destroy_error:
                    logger.error(f"Error destroying previous canvas widget: {destroy_error}")
            elif hasattr(self.canvas, 'destroy') and callable(self.canvas.destroy):
                 # Handle cases where canvas might be a direct Tk widget or similar
                 try:
                    if self.canvas.winfo_exists():
                        self.canvas.destroy()
                 except Exception as destroy_error:
                     logger.error(f"Error destroying previous canvas object: {destroy_error}")
            self.canvas = None
        
        # Initialize OpenGL renderer
        self.gl_renderer = MarkerGLRenderer(self, bg='black')
        self.gl_renderer.pack(in_=self.canvas_frame, fill='both', expand=True)
        
        # Set marker data
        if self.data_manager.has_data():
            # Set data limits
            data_limits = self.data_manager.data_limits
            if data_limits is not None:
                if 'x' in data_limits and 'y' in data_limits and 'z' in data_limits:
                    x_range = data_limits['x']
                    y_range = data_limits['y']
                    z_range = data_limits['z']

                    if hasattr(self.gl_renderer, 'set_data_limits'):
                        self.gl_renderer.set_data_limits(x_range, y_range, z_range)
        
        # Initialize renderer and draw
        self.gl_renderer.initialize()
        
        # Set skeleton-related information
        skeleton_pairs = self.state_manager.skeleton_pairs
        self.gl_renderer.set_skeleton_pairs(skeleton_pairs)

        show_skeleton = self.state_manager.view_state.show_skeleton
        self.gl_renderer.set_show_skeleton(show_skeleton)

        # Set coordinate system (Y-up or Z-up)
        is_z_up = self.state_manager.view_state.is_z_up
        self.gl_renderer.set_coordinate_system(is_z_up)
        
        # Set outlier information
        if hasattr(self, 'outliers') and self.outliers:
            self.gl_renderer.set_outliers(self.outliers)

        # Set marker visual settings
        if hasattr(self, 'marker_visual_settings'):
            self.gl_renderer.set_marker_visual_settings(self.marker_visual_settings)

        # Save canvas reference
        self.canvas = self.gl_renderer
        
    except ImportError as ie:
        logger.error(f"OpenGL renderer import error: {ie}")
        logger.error("To use OpenGL features, please install the necessary libraries.")
        # Optionally, you could disable OpenGL features or show a message to the user here
        raise # Re-raise to indicate failure
    except Exception as e:
        logger.error(f"OpenGL renderer initialization error: {type(e).__name__} - {e}")
        logger.exception("Exception occurred during OpenGL renderer initialization:")
        raise
