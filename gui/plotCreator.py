import traceback

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

def create_plot(self):
    """
    Creates the main 3D plot for displaying marker data.
    This function was extracted from the main class to improve code organization.
    
    Now supports only OpenGL rendering mode.
    """
    try:
        from gui.opengl.GLMarkerRenderer import MarkerGLRenderer
        
        # If there is an existing canvas, destroy it
        if hasattr(self, 'canvas') and self.canvas:
            if hasattr(self.canvas, 'get_tk_widget'):
                try:
                    # Attempt to destroy the widget if it exists
                    widget = self.canvas.get_tk_widget()
                    if widget.winfo_exists():
                        widget.destroy()
                except Exception as destroy_error:
                    print(f"Error destroying previous canvas widget: {destroy_error}")
            elif hasattr(self.canvas, 'destroy') and callable(self.canvas.destroy):
                 # Handle cases where canvas might be a direct Tk widget or similar
                 try:
                    if self.canvas.winfo_exists():
                        self.canvas.destroy()
                 except Exception as destroy_error:
                     print(f"Error destroying previous canvas object: {destroy_error}")
            self.canvas = None
        
        # Initialize OpenGL renderer
        self.gl_renderer = MarkerGLRenderer(self, bg='black')
        self.gl_renderer.pack(in_=self.canvas_frame, fill='both', expand=True)
        
        # Set marker data
        if self.data is not None:
            # Set data limits
            if hasattr(self, 'data_limits') and self.data_limits is not None:
                if 'x' in self.data_limits and 'y' in self.data_limits and 'z' in self.data_limits:
                    x_range = self.data_limits['x']
                    y_range = self.data_limits['y']
                    z_range = self.data_limits['z']
                    
                    if hasattr(self.gl_renderer, 'set_data_limits'):
                        self.gl_renderer.set_data_limits(x_range, y_range, z_range)
        
        # Initialize renderer and draw
        self.gl_renderer.initialize()
        
        # Set skeleton-related information
        if hasattr(self, 'skeleton_pairs'):
            self.gl_renderer.set_skeleton_pairs(self.skeleton_pairs)
        if hasattr(self, 'show_skeleton'):
            self.gl_renderer.set_show_skeleton(self.show_skeleton)
        else:
            self.gl_renderer.set_show_skeleton(False)
        
        # Set coordinate system (Y-up or Z-up)
        if hasattr(self, 'is_z_up'):
            self.gl_renderer.set_coordinate_system(self.is_z_up)
        else:
            self.gl_renderer.set_coordinate_system(False)
        
        # Set outlier information
        if hasattr(self, 'outliers') and self.outliers:
            self.gl_renderer.set_outliers(self.outliers)
            
        # Save canvas reference
        self.canvas = self.gl_renderer
        
    except ImportError as ie:
        print(f"OpenGL renderer import error: {ie}")
        print("To use OpenGL features, please install the necessary libraries.")
        # Optionally, you could disable OpenGL features or show a message to the user here
        raise # Re-raise to indicate failure
    except Exception as e:
        print(f"OpenGL renderer initialization error: {type(e).__name__} - {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        raise
