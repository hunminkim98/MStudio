from pyopengltk import OpenGLFrame
from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import sys
from MEditor.gui.opengl.GridUtils import create_opengl_grid

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

# Coordinate system rotation constants
COORDINATE_X_ROTATION_Y_UP = -270.0  # X-axis rotation angle in Y-up coordinate system (-270 degrees)
COORDINATE_X_ROTATION_Z_UP = -90.0   # X-axis rotation angle in Z-up coordinate system (-90 degrees)

class MarkerGLFrame(OpenGLFrame):
    """OpenGL based 3D marker visualization frame"""
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the OpenGL based 3D marker visualization frame.
        
        Coordinate Systems:
        - Y-up: Default coordinate system, Y-axis points upwards.
        - Z-up: Z-axis points upwards, X-Y forms the ground plane.
        """
        super().__init__(parent, **kwargs)
        
        # Bring in necessary attributes from the existing plotCreator.py
        self.data = None
        self.marker_names = []
        self.is_z_up = False  # Set Y-up as the default coordinate system
        self.show_skeleton = True
        self.skeleton_pairs = []
        self.skeleton_lines = []
        self.current_marker = None
        self.marker_labels = []
        self.coordinate_axes = []
        self.axis_labels = []
        self.grid_lines = []
        
        # Mouse control related
        self.rot_x = COORDINATE_X_ROTATION_Y_UP  # Default rotation angle suitable for Y-up coordinate system
        self.rot_y = 45.0   # Default rotation angle (Y-axis)
        self.zoom = -4.0    # Default zoom level
        self.last_x = 0
        self.last_y = 0
        
        # Initialization flag
        self.gl_initialized = False
        self.grid_list = None
        self.axes_list = None
        
        # Connect mouse events
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<B1-Motion>", self.on_mouse_move)
        self.bind("<MouseWheel>", self.on_scroll)
    
    def initgl(self):
        """Initialize OpenGL (automatically called by pyopengltk)"""
        try:
            # Add GLUT initialization
            GLUT.glutInit(sys.argv)

            # Set background color (black)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            
            # Enable depth testing
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            # Set point size and line width
            GL.glPointSize(5.0)
            GL.glLineWidth(2.0)
            
            # Set lighting (basic lighting)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
            
            # Set object material properties
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            
            # Set anti-aliasing
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
            
            # Initialize viewport
            width, height = self.winfo_width(), self.winfo_height()
            if width > 1 and height > 1:  # Check if the size is valid
                GL.glViewport(0, 0, width, height)
                
            # Initial projection setup
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(45, float(width)/float(height) if width > 0 and height > 0 else 1.0, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            
            # Set initialization flag
            self.gl_initialized = True
            
            # Create display lists - execute after initialization
            self.after(100, self.create_display_lists)
            
        except Exception as e:
            print(f"OpenGL initialization error: {str(e)}")
            self.gl_initialized = False
    
    def create_display_lists(self):
        """Create OpenGL display lists"""
        try:
            self.create_grid()
            self.create_axes()
        except Exception as e:
            print(f"Display list creation error: {str(e)}")
    
    def reshape(self, width, height):
        """Handle window resizing"""
        if not self.gl_initialized:
            return
            
        try:
            if width > 1 and height > 1:  # Check if the size is valid
                GL.glViewport(0, 0, width, height)
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                GLU.gluPerspective(45, float(width)/float(height), 0.1, 100.0)
                GL.glMatrixMode(GL.GL_MODELVIEW)
        except Exception as e:
            print(f"Window resize error: {str(e)}")
    
    def create_grid(self):
        """
        Create the ground grid.
        
        Generates an appropriate grid based on the current coordinate system (Y-up or Z-up).
        """
        # Use the centralized utility function
        self.grid_list = create_opengl_grid(
            grid_size=2.0,
            grid_divisions=20,
            color=(0.3, 0.3, 0.3),
            is_z_up=getattr(self, 'is_z_up', False)  # Default Y-up
        )
    
    def create_axes(self):
        """Create coordinate axes"""
        # Delete the list if it exists
        if hasattr(self, 'axes_list') and self.axes_list is not None:
            GL.glDeleteLists(self.axes_list, 1)

        self.axes_list = GL.glGenLists(1)
        GL.glNewList(self.axes_list, GL.GL_COMPILE)

        # Reduce axis length
        axis_length = 0.2

        # Offset value to lift the axes above the grid
        offset_y = 0.001

        # Set axis thickness
        original_line_width = GL.glGetFloatv(GL.GL_LINE_WIDTH)
        GL.glLineWidth(3.0)

        # X-axis (red)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(1.0, 0.0, 0.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(axis_length, offset_y, 0)

        # Y-axis (yellow)
        GL.glColor3f(1.0, 1.0, 0.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(0, axis_length + offset_y, 0)

        # Z-axis (blue)
        GL.glColor3f(0.0, 0.0, 1.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(0, offset_y, axis_length)
        GL.glEnd()

        # Draw axis label text (using GLUT)
        text_offset = 0.06 # Distance to offset text from the end of the axis

        # Disable lighting (to ensure text color appears correctly)
        lighting_enabled = GL.glIsEnabled(GL.GL_LIGHTING)
        if lighting_enabled:
            GL.glDisable(GL.GL_LIGHTING)

        # X Label
        GL.glColor3f(1.0, 0.0, 0.0) # Red
        GL.glRasterPos3f(axis_length + text_offset, offset_y, 0)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('X'))

        # Y Label
        GL.glColor3f(1.0, 1.0, 0.0) # Yellow
        GL.glRasterPos3f(0, axis_length + text_offset + offset_y, 0)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Y'))

        # Z Label
        GL.glColor3f(0.0, 0.0, 1.0) # Blue
        GL.glRasterPos3f(0, offset_y, axis_length + text_offset)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Z'))

        # Restore original state
        GL.glLineWidth(original_line_width)
        if lighting_enabled:
            GL.glEnable(GL.GL_LIGHTING) # Re-enable lighting

        GL.glEndList()
    
    def setup_view(self):
        """Set up the basic view (replaces plotCreator._setup_plot_style)"""
        # Automatically called during initialization, no additional work needed here
        pass
    
    # Mouse event handlers (replace functionality of the previous mouse_handler class)
    def on_mouse_press(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def on_mouse_release(self, event):
        pass
    
    def on_mouse_move(self, event):
        dx, dy = event.x - self.last_x, event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        
        self.rot_y += dx * 0.5
        self.rot_x += dy * 0.5
        self.redraw()
    
    def on_scroll(self, event):
        # On Windows: event.delta, other platforms may need different approaches
        self.zoom += event.delta * 0.001
        self.redraw()
        