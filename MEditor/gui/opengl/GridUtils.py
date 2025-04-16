from OpenGL import GL

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


DEFAULT_COORDINATE_SYSTEM = False  # False: Y-up, True: Z-up

def create_opengl_grid(grid_size=2.0, grid_divisions=20, color=(0.3, 0.3, 0.3), is_z_up=DEFAULT_COORDINATE_SYSTEM):
    """
    Utility function to create an OpenGL grid based on the current coordinate system.
    
    This function creates a bottom grid based on the current coordinate system.
    - Y-up coordinate system: Creates a grid on the X-Z plane (Y=0)
    - Z-up coordinate system: Creates a grid on the X-Y plane (Z=0)
    
    Args:
        grid_size: Grid size (default: 2.0)
        grid_divisions: Number of grid divisions (default: 20)
        color: Grid color (R, G, B) (default: dark gray)
        is_z_up: Use Z-up coordinate system (True: Z-up, False: Y-up)
    
    Returns:
        grid_list: Created OpenGL display list ID
    
    Note:
        The actual data coordinates are not affected by the coordinate system.
        This function is only used for visualization purposes.
    Args:
        grid_size: Grid size (default: 2.0)
        grid_divisions: Grid divisions (default: 20)
        color: Grid color (R, G, B) (default: dark gray)
        is_z_up: Use Z-up coordinate system (True: Z-up, False: Y-up)
    
    Returns:
        grid_list: Created OpenGL display list ID
    """
    # Create grid list
    grid_list = GL.glGenLists(1)
    GL.glNewList(grid_list, GL.GL_COMPILE)
    
    # Enable two-sided rendering and disable back-face culling
    GL.glDisable(GL.GL_CULL_FACE)
    
    # Set grid line width
    GL.glLineWidth(1.0)
    
    # Set grid color
    GL.glColor3f(*color)
    GL.glBegin(GL.GL_LINES)
    
    # Calculate grid spacing
    step = (grid_size * 2) / grid_divisions
    
    if is_z_up:
        # Draw grid based on Z-up coordinate system (X-Y plane, Z=0)
        for i in range(grid_divisions + 1):
            x = -grid_size + i * step
            # X-axis line (Y-axis change)
            GL.glVertex3f(x, -grid_size, 0)
            GL.glVertex3f(x, grid_size, 0)
            # Y-axis line (X-axis change)
            GL.glVertex3f(-grid_size, x, 0)
            GL.glVertex3f(grid_size, x, 0)
    else:
        # Draw grid based on Y-up coordinate system (X-Z plane, Y=0)
        for i in range(grid_divisions + 1):
            x = -grid_size + i * step
            # X-axis line (Z-axis change)
            GL.glVertex3f(x, 0, -grid_size)
            GL.glVertex3f(x, 0, grid_size)
            # Z-axis line (X-axis change)
            GL.glVertex3f(-grid_size, 0, x)
            GL.glVertex3f(grid_size, 0, x)
    
    GL.glEnd()
    
    # Restore original settings
    GL.glEnable(GL.GL_CULL_FACE)
    
    GL.glEndList()
    
    return grid_list 