from MStudio.gui.opengl.GLPlotCreator import MarkerGLFrame
from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import numpy as np
import pandas as pd
from MStudio.gui.opengl.GridUtils import create_opengl_grid
from MStudio.utils.analysisMode import calculate_distance, calculate_angle, calculate_arc_points, calculate_velocity, calculate_acceleration
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

# Coordinate system rotation constants
COORDINATE_X_ROTATION_Y_UP = 45  # X-axis rotation angle in Y-up coordinate system (-270 degrees)
COORDINATE_X_ROTATION_Z_UP = -90  # X-axis rotation angle in Z-up coordinate system (-90 degrees)

# Coordinate system string constants
COORDINATE_SYSTEM_Y_UP = "y-up"
COORDINATE_SYSTEM_Z_UP = "z-up"

# Scale factor for reference line length in analysis mode (deprecated)
REF_LINE_SCALE = 0.33

# Fixed reference line length in world units (meters)
REF_LINE_FIXED_LENGTH = 0.15

# Font constants for text rendering (smaller sizes)
SMALL_FONT = GLUT.GLUT_BITMAP_HELVETICA_12
LARGE_FONT = GLUT.GLUT_BITMAP_HELVETICA_18

# Picking Texture Class
class PickingTexture:
    """Picking texture class for marker selection"""
    
    def __init__(self):
        """Initialize picking texture"""
        self.fbo = 0
        self.texture = 0
        self.depth_texture = 0
        self.width = 0
        self.height = 0
        self.initialized = False
        
    def init(self, width, height):
        """
        Initialize picking texture
        
        Args:
            width: Texture width
            height: Texture height
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        self.width = width
        self.height = height
        
        try:
            # Create FBO with proper type handling
            fbo_raw = GL.glGenFramebuffers(1)
            self.fbo = int(fbo_raw)  # Convert to standard int to avoid numpy type issues
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

            # Create texture for ID information with proper type handling
            texture_raw = GL.glGenTextures(1)
            self.texture = int(texture_raw)  # Convert to standard int to avoid numpy type issues
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, width, height,
                           0, GL.GL_RGB, GL.GL_FLOAT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                    GL.GL_TEXTURE_2D, self.texture, 0)

            # Create texture for depth information with proper type handling
            depth_texture_raw = GL.glGenTextures(1)
            self.depth_texture = int(depth_texture_raw)  # Convert to standard int to avoid numpy type issues
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, width, height,
                           0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                    GL.GL_TEXTURE_2D, self.depth_texture, 0)
            
            # Disable read buffer (for older GPU compatibility)
            GL.glReadBuffer(GL.GL_NONE)
            
            # Set draw buffer
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # Check FBO status
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                logger.error(f"FBO creation error, status: {status:x}")
                self.cleanup()
                return False
            
            # Restore default framebuffer
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Picking texture initialization error: {e}")
            self.cleanup()
            return False
    
    def cleanup(self):
        """Clean up resources with proper OpenGL type handling"""
        try:
            # Ensure we have a valid OpenGL context before cleanup
            if not hasattr(GL, 'glGetError'):
                logger.warning("OpenGL context not available during cleanup")
                return

            # Delete textures with proper parameter format
            if self.texture != 0:
                # Convert to proper format for glDeleteTextures (count, array)
                texture_id = int(self.texture)  # Ensure it's a standard int
                GL.glDeleteTextures(1, [texture_id])
                self.texture = 0

            if self.depth_texture != 0:
                # Convert to proper format for glDeleteTextures (count, array)
                depth_texture_id = int(self.depth_texture)  # Ensure it's a standard int
                GL.glDeleteTextures(1, [depth_texture_id])
                self.depth_texture = 0

            if self.fbo != 0:
                # Convert to proper format for glDeleteFramebuffers (count, array)
                fbo_id = int(self.fbo)  # Ensure it's a standard int
                GL.glDeleteFramebuffers(1, [fbo_id])
                self.fbo = 0

            self.initialized = False

        except Exception as e:
            logger.error(f"Picking texture cleanup error: {e}")
            # Force reset of IDs even if cleanup failed to prevent further issues
            self.texture = 0
            self.depth_texture = 0
            self.fbo = 0
            self.initialized = False
    
    def enable_writing(self):
        """Enable writing to the picking texture"""
        if not self.initialized:
            return False
            
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        return True
    
    def disable_writing(self):
        """Disable writing to the picking texture"""
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
    
    def read_pixel(self, x, y):
        """
        Read pixel information at the given position
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
            
        Returns:
            tuple: (ObjectID, PrimID) or None (if no object is selected)
        """
        if not self.initialized:
            return None
        
        try:
            # Bind FBO as read framebuffer
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # Check if pixel coordinates are within texture bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return None
            
            # Read pixel information
            data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGB, GL.GL_FLOAT)
            pixel_info = np.frombuffer(data, dtype=np.float32)
            
            # Restore default settings
            GL.glReadBuffer(GL.GL_NONE)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            
            # Check for background pixel (ID 0 means background)
            if pixel_info[0] == 0.0:
                return None
                
            return (pixel_info[0], pixel_info[1], pixel_info[2])
            
        except Exception as e:
            logger.error(f"Pixel read error: {e}")
            return None

class MarkerGLRenderer(MarkerGLFrame):
    """Complete marker visualization OpenGL renderer"""
    
    def __init__(self, parent, **kwargs):
        """
        Initialize the frame for rendering marker data with OpenGL
        
        Coordinate Systems:
        - Y-up: Default coordinate system, Y-axis points upwards
        - Z-up: Z-axis points upwards, X-Y forms the ground plane
        """
        super().__init__(parent, **kwargs)
        self.parent = parent # Store the parent reference
        
        # Default coordinate system setting (Y-up)
        self.is_z_up = False
        self.coordinate_system = COORDINATE_SYSTEM_Y_UP
        
        # Variables for internal state and data storage
        self.frame_idx = 0
        self.outliers = {}
        self.num_frames = 0
        self.pattern_markers = []
        self.pattern_selection_mode = False
        self.show_trajectory = False
        self.marker_names = []
        self.current_marker = None
        self.show_marker_names = False
        self.skeleton_pairs = None
        self.show_skeleton = False

        # Marker visual settings (will be set from parent)
        self.marker_visual_settings = None

        # --- Analysis Mode State (internal to renderer) ---
        self.analysis_mode_active = False
        self.analysis_selection = [] # Store names of selected markers for analysis highlight

        # --- initial view state ---
        self.rot_x = 45
        self.rot_y = 45.0
        self.zoom = -4.0
        
        # Initialization completion flag
        self.initialized = False
        self.gl_initialized = False
        
        # Add variables for screen translation
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.last_x = 0
        self.last_y = 0
        
        # Add mouse event bindings
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<B1-Motion>", self.on_mouse_move)
        self.bind("<ButtonPress-3>", self.on_right_mouse_press)
        self.bind("<ButtonRelease-3>", self.on_right_mouse_release)
        self.bind("<B3-Motion>", self.on_right_mouse_move)
        self.bind("<MouseWheel>", self.on_scroll)
        self.bind("<Motion>", self.on_mouse_motion)  # For hover detection
        self.bind("<Configure>", self.on_configure) # Add binding for Configure event

        # Resize handling optimization variables
        self._resize_timer = None
        self._last_resize_time = 0
        self._resize_throttle_ms = 16  # ~60 FPS throttling
        self._last_viewport_size = (0, 0)
        self._resize_in_progress = False
        
        # Marker picking related variables
        self.picking_texture = PickingTexture()
        self.dragging = False

        # Reference line interaction variables
        self.ref_line_hover = False
        self.ref_line_click_tolerance = 15  # pixels
        self.ref_line_start = None  # Will store reference line start point
        self.ref_line_end = None    # Will store reference line end point

        # Performance optimization for animation
        self._context_active = False
        self._pending_redraw = False
        self._last_render_time = 0.0

        # Skeleton rendering optimization
        self._skeleton_cache = {}
        self._skeleton_display_list = None
        self._cached_frame_idx = -1
        self._skeleton_cache_valid = False

        # Cleanup flag to prevent multiple cleanup attempts
        self._cleanup_performed = False
        
    def initialize(self):
        """
        Initialize the OpenGL renderer - called from gui/plotCreator.py
        pyopengltk initializes OpenGL via the initgl method,
        so here we just set the initialization flag.
        """
        
        # Mark initialization as complete - actual OpenGL initialization happens in initgl
        self.initialized = True
        
        # Refresh the screen
        self.update()  # Automatically calls initgl and redraw
        
    def initgl(self):
        """Initialize OpenGL (automatically called by pyopengltk)"""
        try:
            # Call the parent class's initgl
            super().initgl()
            
            # Set background color (black)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            
            # Enable depth testing
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            # Set point size and line width
            GL.glPointSize(5.0)
            GL.glLineWidth(2.0)
            
            # Disable lighting - for consistent color from all angles
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_LIGHT0)
            
            # Remove existing display lists (if any)
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glDeleteLists(self.grid_list, 1)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glDeleteLists(self.axes_list, 1)
            if hasattr(self, '_skeleton_display_list') and self._skeleton_display_list is not None:
                GL.glDeleteLists(self._skeleton_display_list, 1)
                self._skeleton_display_list = None
                self._skeleton_cache_valid = False
            
            # Now create display lists after the OpenGL context is fully initialized
            self._create_grid_display_list()
            self._create_axes_display_list()
            
            # Initialize picking texture
            width, height = self.winfo_width(), self.winfo_height()
            if width > 0 and height > 0:
                self.picking_texture.init(width, height)
            
            # Set OpenGL initialization complete flag
            self.gl_initialized = True
        except Exception as e:
            logger.error(f"OpenGL initialization error: {e}")
            self.gl_initialized = False
        
    def _create_grid_display_list(self):
        """Create a display list for grid rendering"""
        if hasattr(self, 'grid_list') and self.grid_list is not None:
            GL.glDeleteLists(self.grid_list, 1)
            
        # Use the centralized utility function
        is_z_up = getattr(self, 'is_z_up', True)
        self.grid_list = create_opengl_grid(
            grid_size=2.0, 
            grid_divisions=20, 
            color=(0.3, 0.3, 0.3),
            is_z_up=is_z_up
        )
        
    def _create_axes_display_list(self):
        """Create a display list for coordinate axis rendering"""
        if hasattr(self, 'axes_list') and self.axes_list is not None:
            GL.glDeleteLists(self.axes_list, 1)
            
        # Create display list with proper type handling
        axes_list_raw = GL.glGenLists(1)
        self.axes_list = int(axes_list_raw)  # Convert to standard int to avoid numpy type issues
        GL.glNewList(self.axes_list, GL.GL_COMPILE)
        
        # Disable backface culling
        GL.glDisable(GL.GL_CULL_FACE)
        
        # Axis length (maintain original style)
        axis_length = 0.2
        
        # Move the origin to be clearly distinguished from the grid - float above the grid
        offset_y = 0.001
        
        # Set axis thickness (maintain original style)
        original_line_width = GL.glGetFloatv(GL.GL_LINE_WIDTH)
        GL.glLineWidth(3.0)
        
        # Draw axes suitable for Z-up coordinate system (rotation matrix is applied)
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
        
        # Draw axis label text (using GLUT - maintain original style)
        text_offset = 0.06  # Distance to offset text from the end of the axis
        
        # Disable lighting (to ensure text color appears correctly)
        lighting_enabled = GL.glIsEnabled(GL.GL_LIGHTING)
        if lighting_enabled:
            GL.glDisable(GL.GL_LIGHTING)
        
        # X Label
        GL.glColor3f(1.0, 0.0, 0.0)  # Red
        GL.glRasterPos3f(axis_length + text_offset, offset_y, 0)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord('X'))
        except:
            pass  # Skip label rendering if GLUT is unavailable
        
        # Y Label
        GL.glColor3f(1.0, 1.0, 0.0)  # Yellow
        GL.glRasterPos3f(0, axis_length + text_offset + offset_y, 0)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord('Y'))
        except:
            pass
        
        # Z Label
        GL.glColor3f(0.0, 0.0, 1.0)  # Blue
        GL.glRasterPos3f(0, offset_y, axis_length + text_offset)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord('Z'))
        except:
            pass
        
        # Restore original state
        GL.glLineWidth(original_line_width)
        if lighting_enabled:
            GL.glEnable(GL.GL_LIGHTING)  # Re-enable lighting
        
        GL.glEnable(GL.GL_CULL_FACE)
        
        GL.glEndList()
    
    def redraw(self):
        """
        Redraw the OpenGL screen.
        This is the main drawing method.
        """
        if not self.gl_initialized:
            return
            
        # Call the internal _update_plot method
        self._update_plot()
        
    def _update_plot(self):
        """
        Update the 3D marker visualization with OpenGL
        Integrates functionality previously in a separate file gui/opengl/GLPlotUpdater.py into the class
        
        Coordinate Systems:
        - Y-up: Default coordinate system, Y-axis points upwards
        - Z-up: Z-axis points upwards, X-Y forms the ground plane
        """
        # Check OpenGL initialization
        if not self.gl_initialized:
            return
        
        # Check current coordinate system state (default: Y-up)
        is_z_up_local = getattr(self, 'is_z_up', False)
        
        try:
            # OPTIMIZATION: Minimize context switching during animation
            if not self._context_active:
                try:
                    self.tkMakeCurrent()
                    self._context_active = True
                except Exception as context_error:
                    logger.error(f"Error setting OpenGL context: {context_error}")
                    return # Cannot proceed without context
            
            # --- Explicitly Reset Key OpenGL States --- START
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_LIGHTING) # Ensure lighting is off
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
            # --- Explicitly Reset Key OpenGL States --- END
            
            # --- Viewport and Projection Setup --- START
            width = self.winfo_width()
            height = self.winfo_height()
            if width <= 0 or height <= 0:
                 # Avoid division by zero or invalid viewport
                 return 

            # 1. Set the viewport to the entire widget area
            GL.glViewport(0, 0, width, height)

            # 2. Set up the projection matrix
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height)
            # Use perspective projection (like in pick_marker)
            GLU.gluPerspective(45, aspect, 0.1, 100.0) # fov, aspect, near, far

            # 3. Switch back to the modelview matrix for camera/object transformations
            GL.glMatrixMode(GL.GL_MODELVIEW)
            # --- Viewport and Projection Setup --- END
            
            # Initialize frame (Clear after setting viewport/projection)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0) # Ensure clear color is set
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity() # Reset modelview matrix before camera setup
            
            # Set camera position (zoom, rotation, translation)
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            
            # Apply additional rotation based on the coordinate system
            # - Y-up: No additional rotation needed as the default camera setup is already aligned for Y-up (-270 degrees)
            # - Z-up: Add -90 degrees rotation around the X-axis to view the opposite direction of the Y-up plane
            if is_z_up_local:
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)
            
            # Display grid and axes (only if display lists exist)
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)
            
            # If no data, display only the basic view and exit
            if self.data is None:
                self.tkSwapBuffers()
                return
            
            # Collect marker position data
            positions = []
            colors = []
            selected_position = None
            marker_positions = {}
            valid_markers = []
            
            # OPTIMIZATION: Batch data access for better performance during animation
            if hasattr(self.data, 'iloc'):
                # Use iloc for faster integer-based indexing
                frame_data = self.data.iloc[self.frame_idx]

                # Collect valid marker data for the current frame
                for marker in self.marker_names:
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if x_col in frame_data.index and y_col in frame_data.index and z_col in frame_data.index:
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                                continue
                        else:
                            continue
                    except (KeyError, IndexError):
                        continue
                    
                    # Adjust position according to the coordinate system
                    pos = [x, y, z]
                        
                    marker_positions[marker] = pos
                    
                    # Set color
                    marker_str = str(marker)
                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""
                    
                    if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                        if marker in self.pattern_markers:
                            color = self.marker_visual_settings.get_pattern_color() if self.marker_visual_settings else [1.0, 0.0, 0.0]
                            colors.append(color)
                        else:
                            color = self.marker_visual_settings.get_normal_color() if self.marker_visual_settings else [1.0, 1.0, 1.0]
                            colors.append(color)
                    elif marker_str == current_marker_str:
                        color = self.marker_visual_settings.get_selected_color() if self.marker_visual_settings else [1.0, 0.9, 0.4]
                        colors.append(color)
                    else:
                        color = self.marker_visual_settings.get_normal_color() if self.marker_visual_settings else [1.0, 1.0, 1.0]
                        colors.append(color)
                        
                    positions.append(pos)
                    valid_markers.append(marker)
                    
                    if marker_str == current_marker_str:
                        selected_position = pos
            
            # Marker rendering - optimized with pre-computed pattern selection
            if positions:
                # Pre-compute pattern selection status for better performance
                pattern_selected_set = set(self.pattern_markers) if self.pattern_selection_mode else set()

                # Stage 1: Normal markers (unselected markers or when not in pattern mode)
                marker_size = self.marker_visual_settings.get_marker_size() if self.marker_visual_settings else 5.0
                GL.glPointSize(marker_size)

                # Enable blending for opacity support
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

                GL.glBegin(GL.GL_POINTS)
                for i, pos in enumerate(positions):
                    marker = valid_markers[i]
                    if marker not in pattern_selected_set:
                        color = colors[i]
                        if self.marker_visual_settings:
                            # Apply opacity
                            opacity = self.marker_visual_settings.get_opacity()
                            GL.glColor4f(color[0], color[1], color[2], opacity)
                        else:
                            GL.glColor3fv(color)
                        GL.glVertex3fv(pos)
                GL.glEnd()

                # Disable blending
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glDisable(GL.GL_BLEND)
                
                # Stage 2: Selected pattern markers (when in pattern mode)
                if self.pattern_selection_mode and any(m in self.pattern_markers for m in valid_markers):
                    GL.glPointSize(8.0) # Larger size
                    GL.glBegin(GL.GL_POINTS)
                    for i, pos in enumerate(positions):
                        marker = valid_markers[i]
                        if marker in self.pattern_markers:
                            # Color is already set to red in the colors list
                            GL.glColor3fv(colors[i]) 
                            GL.glVertex3fv(pos)
                    GL.glEnd()
            
            # Highlight selected marker
            if selected_position:
                GL.glPointSize(8.0)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor3f(1.0, 0.9, 0.4)  # Light yellow
                GL.glVertex3fv(selected_position)
                GL.glEnd()
            
            # Skeleton line rendering - OPTIMIZED with caching
            if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs'):
                # Cache skeleton geometry for current frame to optimize camera interactions
                self._cache_skeleton_geometry()
                # --- Enable Blending and Smoothing (needed for normal lines) ---
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                # ---------------------------------------------------------------
                
                # Pass 1: Draw Normal Skeleton Lines with customized settings
                line_width = self.marker_visual_settings.get_skeleton_line_width() if self.marker_visual_settings else 2.0
                normal_color = self.marker_visual_settings.get_skeleton_normal_color() if self.marker_visual_settings else (0.7, 0.7, 0.7)
                opacity = self.marker_visual_settings.get_skeleton_opacity() if self.marker_visual_settings else 0.8

                GL.glLineWidth(line_width)
                GL.glColor4f(normal_color[0], normal_color[1], normal_color[2], opacity)
                GL.glBegin(GL.GL_LINES)
                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        is_outlier = outlier_status1 or outlier_status2
                        if not is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)
                GL.glEnd()
                
                # Pass 2: Draw Outlier Skeleton Lines with customized settings
                # Blending is already enabled, just change width and color
                outlier_line_width = (line_width + 1.5) if self.marker_visual_settings else 3.5  # Slightly thicker than normal
                outlier_color = self.marker_visual_settings.get_skeleton_outlier_color() if self.marker_visual_settings else (1.0, 0.0, 0.0)

                GL.glLineWidth(outlier_line_width)
                GL.glColor4f(outlier_color[0], outlier_color[1], outlier_color[2], 1.0)  # Full opacity for outliers
                GL.glBegin(GL.GL_LINES)
                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        is_outlier = outlier_status1 or outlier_status2
                        if is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)
                GL.glEnd()
                
                # --- Reset LineWidth and Disable Blending for standard skeleton--- 
                # GL.glLineWidth(1.0) # Resetting here might interfere if torso lines need different width
                # GL.glDisable(GL.GL_BLEND) # Keep blend enabled for torso potentially
                # ------------------------------------------
                
                # --- Draw additional explicit torso lines (Now inside the skeleton check) --- 
                explicit_torso_pairs = [
                    ("RHip", "RShoulder"),
                    ("LHip", "LShoulder"),
                    ("RHip", "LHip"),
                    ("RShoulder", "LShoulder")
                ]
                # Use the same style as normal skeleton lines with customized settings
                # Ensure Blend is enabled if needed
                # GL.glEnable(GL.GL_BLEND) # Already enabled from standard skeleton drawing
                GL.glLineWidth(line_width)  # Match normal skeleton line width
                GL.glColor4f(normal_color[0], normal_color[1], normal_color[2], opacity)  # Match normal skeleton color
                GL.glBegin(GL.GL_LINES)
                for pair in explicit_torso_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        GL.glVertex3fv(p1)
                        GL.glVertex3fv(p2)
                GL.glEnd()
                
                # --- Final Reset after all skeleton + torso lines --- 
                GL.glLineWidth(1.0) # Reset to OpenGL default
                GL.glDisable(GL.GL_BLEND) # Disable blending after all skeleton/torso lines
                # --- End additional torso lines ---
            
            # --- Analysis Mode Visualization ---
            if self.analysis_mode_active and len(self.analysis_selection) >= 1: 
                try:
                    # Highlight selected analysis markers (Green, larger size based on customization)
                    base_marker_size = self.marker_visual_settings.get_marker_size() if self.marker_visual_settings else 5.0
                    analysis_marker_size = base_marker_size + 5.0  # Always 5 pixels larger than base size
                    GL.glPointSize(analysis_marker_size)
                    GL.glColor3f(0.0, 1.0, 0.0) # Green color
                    GL.glBegin(GL.GL_POINTS)
                    analysis_positions_raw = {}
                    valid_analysis_markers = []
                    for marker_name in self.analysis_selection:
                        if marker_name in marker_positions:
                            pos = marker_positions[marker_name]
                            analysis_positions_raw[marker_name] = np.array(pos) # Store as numpy array
                            GL.glVertex3fv(pos)
                            valid_analysis_markers.append(marker_name)
                    GL.glEnd()
                    GL.glPointSize(base_marker_size) # Reset to base marker size

                    # --- Calculations and Visualizations based on selection count --- 
                    num_valid_analysis = len(valid_analysis_markers)
                    
                    # -- Velocity and Acceleration (1 Marker Selected) --
                    if num_valid_analysis == 1:
                        marker_name = valid_analysis_markers[0]
                        current_pos = analysis_positions_raw[marker_name]
                        frame_idx = self.frame_idx
                        frame_rate = float(self.parent.fps_var.get()) # Get fps from parent
                        
                        # --- Get positions for velocity and acceleration calculation --- 
                        pos_data = {}
                        valid_indices = True
                        for i in range(frame_idx - 2, frame_idx + 3): # Need i-2 to i+2 for accel calc
                            if 0 <= i < self.num_frames:
                                try:
                                    pos_data[i] = self.data.loc[i, [f'{marker_name}_{c}' for c in 'XYZ']].values
                                    if np.isnan(pos_data[i]).any():
                                         # If any needed position is NaN, cannot proceed reliably
                                         valid_indices = False
                                         logger.debug(f"NaN found at frame {i} for {marker_name}, skipping vel/accel.")
                                         break 
                                except KeyError:
                                    valid_indices = False
                                    logger.debug(f"KeyError at frame {i} for {marker_name}, skipping vel/accel.")
                                    break # Stop if data is missing
                            else:
                                # Frame index out of bounds
                                valid_indices = False
                                logger.debug(f"Frame index {i} out of bounds, skipping vel/accel.")
                                break
                                
                        # --- Calculate Velocity and Acceleration (if data is valid) --- 
                        velocity = None
                        acceleration = None
                        if valid_indices:
                            # Calculate velocity at current frame (i)
                            velocity = calculate_velocity(pos_data[frame_idx-1], current_pos, pos_data[frame_idx+1], frame_rate)
                            
                            # Calculate velocities at previous (i-1) and next (i+1) frames for acceleration
                            vel_prev = calculate_velocity(pos_data[frame_idx-2], pos_data[frame_idx-1], current_pos, frame_rate)
                            vel_next = calculate_velocity(current_pos, pos_data[frame_idx+1], pos_data[frame_idx+2], frame_rate)
                            
                            if vel_prev is not None and vel_next is not None:
                                acceleration = calculate_acceleration(vel_prev, vel_next, frame_rate)

                        # --- Display Text --- 
                        analysis_text_lines = [] # Initialize empty list
                        
                        if velocity is not None:
                            speed = np.linalg.norm(velocity)
                            # Only add if speed is meaningful (optional threshold check can be added)
                            analysis_text_lines.append(f"{speed:.2f} m/s") 
                            
                        if acceleration is not None:
                            accel_mag = np.linalg.norm(acceleration)
                            # Only add if acceleration is meaningful
                            analysis_text_lines.append(f"{accel_mag:.2f} m/s²")

                        # Only proceed with rendering if there is text to display
                        if analysis_text_lines:
                            text_color = (1.0, 1.0, 0.0) # Yellow
                            text_base_pos = [current_pos[0], current_pos[1] + 0.04, current_pos[2]] # Base position above marker
                            line_height_offset = 0.02 # Adjust for line spacing
                            
                            GL.glPushMatrix()
                            GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_DEPTH_BUFFER_BIT)
                            GL.glDisable(GL.GL_DEPTH_TEST)
                            GL.glColor3fv(text_color) 
                            
                            for i, line in enumerate(analysis_text_lines):
                                # Adjust Y position for each line
                                current_text_pos = [text_base_pos[0], text_base_pos[1] - i * line_height_offset, text_base_pos[2]]
                                GL.glRasterPos3f(current_text_pos[0], current_text_pos[1], current_text_pos[2])
                                for char in line:
                                    try:
                                        GLUT.glutBitmapCharacter(SMALL_FONT, ord(char)) # Smaller font
                                    except Exception:
                                        pass
                                    
                            GL.glPopAttrib()
                            GL.glPopMatrix()

                    # -- Distance / Angle (2 or 3 Markers Selected) --
                    elif num_valid_analysis >= 2:
                        # Get positions in the selection order
                        analysis_positions_ordered = [analysis_positions_raw[m] for m in valid_analysis_markers]

                        # Draw thicker lines between selected markers based on customization
                        base_line_width = self.marker_visual_settings.get_skeleton_line_width() if self.marker_visual_settings else 2.0
                        analysis_line_width = base_line_width + 1.0  # Always 1 pixel thicker than base skeleton width
                        GL.glLineWidth(analysis_line_width)
                        GL.glColor3f(0.0, 1.0, 0.0) # Green color for analysis lines
                        if len(analysis_positions_ordered) == 2:
                            GL.glBegin(GL.GL_LINES)
                            GL.glVertex3fv(analysis_positions_ordered[0])
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            GL.glEnd()
                            # Draw reference horizontal line and arc for segment angle
                            pA = analysis_positions_ordered[0]
                            pB = analysis_positions_ordered[1]
                            v = pA - pB
                            norm_v = np.linalg.norm(v)
                            if norm_v > 0:
                                # Get reference vector from state manager
                                if hasattr(self.parent, 'state_manager'):
                                    u = self.parent.state_manager.get_reference_vector()
                                else:
                                    u = np.array([1.0, 0.0, 0.0])  # Default to X-axis

                                # Store reference line endpoints for click detection (fixed length)
                                self.ref_line_start = pB.copy()
                                self.ref_line_end = pB + u * REF_LINE_FIXED_LENGTH

                                # reference line with hover effect
                                line_width = 2.5 if self.ref_line_hover else 1.5
                                line_color = [0.5, 0.9, 0.5] if self.ref_line_hover else [0.3, 0.7, 0.3]

                                GL.glLineWidth(line_width)
                                GL.glColor3fv(line_color)
                                GL.glBegin(GL.GL_LINES)
                                GL.glVertex3fv(self.ref_line_start)
                                GL.glVertex3fv(self.ref_line_end)
                                GL.glEnd()

                                # Add axis label at the end of reference line
                                if hasattr(self.parent, 'state_manager'):
                                    current_axis = self.parent.state_manager.editing_state.reference_axis
                                    self._render_axis_label(self.ref_line_end, current_axis)
                                # arc
                                radius = norm_v * 0.2
                                p_ref = pB + u * radius
                                p_seg = pB + (v / norm_v) * radius
                                pts = calculate_arc_points(vertex=pB, p1=p_ref, p3=p_seg, radius=radius, num_segments=20)
                                if pts:
                                    GL.glEnable(GL.GL_BLEND)
                                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                                    GL.glDepthMask(GL.GL_FALSE)
                                    GL.glDisable(GL.GL_CULL_FACE)
                                    GL.glColor4f(1.0, 0.6, 0.0, 0.5)
                                    GL.glBegin(GL.GL_TRIANGLE_FAN)
                                    GL.glVertex3fv(pB)
                                    for pt in pts:
                                        GL.glVertex3fv(pt)
                                    GL.glEnd()
                                    GL.glEnable(GL.GL_CULL_FACE)
                                    GL.glLineWidth(1.0)
                                    GL.glColor4f(1.0, 0.6, 0.0, 0.8)
                                    GL.glBegin(GL.GL_LINE_STRIP)
                                    for pt in pts:
                                        GL.glVertex3fv(pt)
                                    GL.glEnd()
                                    GL.glDepthMask(GL.GL_TRUE)
                            GL.glLineWidth(1.0)
                        elif len(analysis_positions_ordered) == 3:
                            GL.glBegin(GL.GL_LINES)
                            # Draw lines based on selection order: 0->1 and 1->2
                            GL.glVertex3fv(analysis_positions_ordered[0])
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            GL.glVertex3fv(analysis_positions_ordered[2])
                            GL.glEnd()
                        GL.glLineWidth(base_line_width) # Reset to base line width

                        # Calculate and prepare text for display
                        dist_text = None
                        dist_pos = None
                        angle_text = None
                        angle_pos = None
                        
                        if len(analysis_positions_ordered) == 2:
                            distance = calculate_distance(analysis_positions_ordered[0], analysis_positions_ordered[1])
                            if distance is not None:
                                pA = analysis_positions_ordered[0]
                                pB = analysis_positions_ordered[1]
                                # distance text at midpoint
                                mid_pt = (pA + pB) / 2
                                dist_text = f"{distance:.3f} m"
                                dist_pos = [mid_pt[0], mid_pt[1] + 0.02, mid_pt[2]]
                                # compute angle relative to reference axis
                                if hasattr(self.parent, 'state_manager'):
                                    u = self.parent.state_manager.get_reference_vector()
                                else:
                                    u = np.array([1.0, 0.0, 0.0])  # Default to X-axis
                                ref_point = pB + u
                                angle_val = calculate_angle(ref_point, pB, pA)
                                if angle_val is not None:
                                    angle_text = f"{angle_val:.1f}\u00B0"
                                    angle_pos = [pB[0], pB[1] + 0.03, pB[2]]
                        elif len(analysis_positions_ordered) == 3:
                            # Angle at the vertex (second selected marker)
                            angle = calculate_angle(analysis_positions_ordered[0], analysis_positions_ordered[1], analysis_positions_ordered[2])
                            if angle is not None:
                                angle_text = f"{angle:.1f}\u00B0"
                                angle_pos = [analysis_positions_ordered[1][0], analysis_positions_ordered[1][1] + 0.03, analysis_positions_ordered[1][2]]
                                
                                # Calculate and draw the arc
                                arc_radius = min(np.linalg.norm(analysis_positions_ordered[0]-analysis_positions_ordered[1]), 
                                                 np.linalg.norm(analysis_positions_ordered[2]-analysis_positions_ordered[1])) * 0.2 # Radius as 20% of shorter arm
                                arc_points = calculate_arc_points(vertex=analysis_positions_ordered[1], 
                                                                  p1=analysis_positions_ordered[0], 
                                                                  p3=analysis_positions_ordered[2], 
                                                                  radius=max(0.01, arc_radius), # Ensure minimum radius
                                                                  num_segments=20)
                                
                                if arc_points:
                                    # Draw the filled, semi-transparent arc using TRIANGLE_FAN
                                    GL.glEnable(GL.GL_BLEND) # Ensure blend is enabled
                                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA) 
                                    GL.glDepthMask(GL.GL_FALSE) # Disable depth writing for transparency
                                    
                                    # Disable face culling to ensure the arc is visible from both sides
                                    GL.glDisable(GL.GL_CULL_FACE)
                                    
                                    # Set color to semi-transparent green (match highlight color)
                                    GL.glColor4f(1.0, 1.0, 0.0, 0.3) # yellow with 30% alpha
                                    
                                    GL.glBegin(GL.GL_TRIANGLE_FAN)
                                    # Center vertex of the fan is the angle vertex
                                    GL.glVertex3fv(analysis_positions_ordered[1]) 
                                    # Outer vertices are the points along the arc
                                    for point in arc_points:
                                        GL.glVertex3fv(point)
                                    GL.glEnd()
                                    
                                    # Re-enable face culling
                                    GL.glEnable(GL.GL_CULL_FACE)
                                    
                                    # Optional: Draw outline of the arc (customized line width)
                                    arc_outline_width = max(1.0, base_line_width * 0.5)  # Half of base line width, minimum 1.0
                                    GL.glLineWidth(arc_outline_width)
                                    GL.glColor4f(1.0, 1.0, 0.0, 0.7) # Slightly more opaque outline
                                    GL.glBegin(GL.GL_LINE_STRIP)
                                    for point in arc_points:
                                        GL.glVertex3fv(point)
                                    GL.glEnd()

                                    GL.glDepthMask(GL.GL_TRUE) # Re-enable depth writing
                                    # GL.glDisable(GL.GL_BLEND) # Optionally disable blend if not needed afterwards
                                    
                        # Render the analysis text if available
                        if (dist_text and dist_pos) or (angle_text and angle_pos):
                            GL.glPushMatrix()
                            GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_DEPTH_BUFFER_BIT)
                            GL.glDisable(GL.GL_DEPTH_TEST)
                            GL.glColor3f(0.0, 1.0, 0.0)
                            if dist_text and dist_pos:
                                GL.glRasterPos3f(dist_pos[0], dist_pos[1], dist_pos[2])
                                for ch in dist_text:
                                    try:
                                        GLUT.glutBitmapCharacter(LARGE_FONT, ord(ch))
                                    except:
                                        pass
                            if angle_text and angle_pos:
                                GL.glRasterPos3f(angle_pos[0], angle_pos[1], angle_pos[2])
                                for ch in angle_text:
                                    try:
                                        GLUT.glutBitmapCharacter(LARGE_FONT, ord(ch))
                                    except:
                                        pass
                            GL.glPopAttrib()
                            GL.glPopMatrix()
                            
                except Exception as analysis_error:
                     logger.error(f"Error during analysis visualization: {analysis_error}", exc_info=True)
            # --- Analysis Mode Visualization End ---

            # Trajectory rendering
            if hasattr(self, 'show_trajectory') and self.show_trajectory:
                # Choose marker for trajectory: override current_marker in analysis mode
                marker_to_trace = self.current_marker
                if getattr(self, 'analysis_mode_active', False):
                    sel = self.analysis_selection
                    if len(sel) == 1:
                        marker_to_trace = sel[0]
                    elif len(sel) == 2:
                        marker_to_trace = sel[1]
                    elif len(sel) >= 3:
                        marker_to_trace = sel[1]
                if marker_to_trace is not None:
                    trajectory_points = []
                    
                    for i in range(0, self.frame_idx + 1):
                        try:
                            x = self.data.loc[i, f'{marker_to_trace}_X']
                            y = self.data.loc[i, f'{marker_to_trace}_Y']
                            z = self.data.loc[i, f'{marker_to_trace}_Z']
                            
                            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                                continue
                            
                            # Use original data directly (regardless of Y-up/Z-up)
                            trajectory_points.append([x, y, z])
                                
                        except KeyError:
                            continue
                    
                    if trajectory_points:
                        GL.glLineWidth(0.8)
                        GL.glColor3f(1.0, 0.9, 0.4)  # Light yellow
                        GL.glBegin(GL.GL_LINE_STRIP)
                        
                        for point in trajectory_points:
                            GL.glVertex3fv(point)
                        
                        GL.glEnd()
            
            # Marker name rendering
            if self.show_marker_names and valid_markers:
                # GLUT is required for text rendering
                try:
                    # Save current projection and modelview matrices
                    GL.glPushMatrix()
                    
                    # Initialize and save OpenGL rendering state
                    GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT)
                    
                    # Stringify current marker
                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""
                    
                    # First render all normal marker names (white)
                    for marker in valid_markers:
                        marker_str = str(marker)
                        if marker_str == current_marker_str:
                            continue  # Render selected marker later
                            
                        pos = marker_positions[marker]
                        
                        # Render normal marker names in white
                        GL.glColor3f(1.0, 1.0, 1.0)  # White
                        GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])
                        
                        # Render marker name
                        for c in marker_str:
                            try:
                                GLUT.glutBitmapCharacter(SMALL_FONT, ord(c))
                            except:
                                pass
                    
                    # Render only the selected marker name in yellow (separate pass)
                    GL.glFlush()  # Ensure previous rendering commands are executed
                    
                    if self.current_marker is not None:
                        # Find and render only the selected marker
                        for marker in valid_markers:
                            marker_str = str(marker)
                            if marker_str == current_marker_str:
                                pos = marker_positions[marker]
                                
                                # Render selected marker name in yellow
                                GL.glColor3f(1.0, 0.9, 0.4)  # Light yellow
                                GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])
                                
                                # Render marker name
                                for c in marker_str:
                                    try:
                                        GLUT.glutBitmapCharacter(SMALL_FONT, ord(c))
                                    except:
                                        pass
                                
                                GL.glFlush()  # Execute rendering command immediately
                                break
                    
                    # Restore OpenGL rendering state
                    GL.glPopAttrib()
                    
                    # Restore matrices
                    GL.glPopMatrix()
                    
                except Exception as e:
                    logger.error(f"Text rendering error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # OPTIMIZATION: Non-blocking buffer swap for smooth animation
            # Use frame rate limiting to prevent excessive rendering
            import time
            current_time = time.time()
            if current_time - self._last_render_time >= 1.0/120.0:  # Max 120 FPS
                self.tkSwapBuffers()
                self._last_render_time = current_time
        
        except Exception as e:
            # Log error for debugging
            logger.error(f"OpenGL rendering error: {e}")
        
    def update_data(self, data, frame_idx):
        """Update data called from external sources (backward compatibility)"""
        self.data = data
        self.frame_idx = frame_idx
        if data is not None:
            self.num_frames = len(data)
        
        # Keep the current_marker attribute unchanged
        
        self.initialized = True
        self.redraw()
    
    def set_frame_data(self, data, frame_idx, marker_names, current_marker=None,
                       show_marker_names=False, show_trajectory=False, show_skeleton=False,
                       coordinate_system="z-up", skeleton_pairs=None):
        """
        Integrated data update method called from TRCViewer

        Args:
            data: Full marker data
            frame_idx: Current frame index
            marker_names: List of marker names
            current_marker: Currently selected marker name
            show_marker_names: Whether to display marker names
            show_trajectory: Whether to display trajectory
            show_skeleton: Whether to display the skeleton
            coordinate_system: Coordinate system ("z-up" or "y-up")
            skeleton_pairs: List of skeleton pairs
        """
        # OPTIMIZATION: Invalidate skeleton cache if frame changes
        if hasattr(self, '_cached_frame_idx') and self._cached_frame_idx != frame_idx:
            self._skeleton_cache_valid = False

        self.data = data
        self.frame_idx = frame_idx
        self.marker_names = marker_names

        # Maintain selected marker information - update only if current_marker is not None
        # Or update if there is no current marker (self.current_marker is None)
        if current_marker is not None:
            self.current_marker = current_marker

        self.show_marker_names = show_marker_names
        self.show_trajectory = show_trajectory
        self.show_skeleton = show_skeleton
        self.coordinate_system = coordinate_system
        self.skeleton_pairs = skeleton_pairs

        # Update frame count if data exists
        if data is not None:
            self.num_frames = len(data)
            
        # Check OpenGL initialization
        self.initialized = True
        
        # Redraw immediately
        self.redraw()
        
    def set_current_marker(self, marker_name):
        """Set the currently selected marker name"""
        self.current_marker = marker_name
        # Update display only if necessary (e.g., marker highlight color)
        # self.redraw() # Removed redundant redraw call due to unnecessary re-rendering
    
    def set_show_skeleton(self, show):
        """
        Set whether to display the skeleton
        
        Args:
            show: True to display the skeleton, False otherwise
        """
        self.show_skeleton = show
        self.redraw()
    
    def set_show_trajectory(self, show):
        """Set trajectory display"""
        logger.debug(f"Setting show_trajectory to {show}")
        self.show_trajectory = show
        # Force complete redraw to ensure state change is reflected
        self._force_complete_redraw()

    def set_marker_visual_settings(self, settings):
        """Set marker visual settings"""
        self.marker_visual_settings = settings
        # Invalidate skeleton cache when visual settings change
        self._skeleton_cache_valid = False

        # Add callback to invalidate skeleton cache when settings change
        if hasattr(settings, 'add_change_callback'):
            settings.add_change_callback(self._on_visual_settings_change)

        logger.debug("Marker visual settings updated")

    def _on_visual_settings_change(self):
        """Called when visual settings change - invalidate caches and redraw"""
        self._skeleton_cache_valid = False
        if self.gl_initialized:
            self.redraw()
        
    def update_plot(self):
        """
        Screen update method called externally
        Previously called update_plot in an external module, now calls the internal method
        """
        if self.gl_initialized:
            self.redraw()
        
    def set_pattern_selection_mode(self, mode, pattern_markers=None):
        """Set pattern selection mode"""
        self.pattern_selection_mode = mode
        if pattern_markers is not None:
            self.pattern_markers = pattern_markers
        self.redraw()
    
    def set_coordinate_system(self, is_z_up):
        """
        Change coordinate system setting
        
        Args:
            is_z_up: True to use Z-up coordinate system, False for Y-up
        
        Note:
        Changing the coordinate system only changes the display method, not the actual coordinates of the markers.
        The data always retains its original coordinate system.
        """
        # Do not perform unnecessary processing if there is no change
        if self.is_z_up == is_z_up:
            return
        
        # Update coordinate system state
        self.is_z_up = is_z_up
        
        # Update coordinate system string
        self.coordinate_system = COORDINATE_SYSTEM_Z_UP if is_z_up else COORDINATE_SYSTEM_Y_UP
        
        # Regenerate axis display list according to the coordinate system
        if self.gl_initialized:
            try:
                # Activate OpenGL context - essential
                self.tkMakeCurrent()
                
                # Delete existing axis and grid display lists
                if hasattr(self, 'axes_list') and self.axes_list is not None:
                    GL.glDeleteLists(self.axes_list, 1)
                if hasattr(self, 'grid_list') and self.grid_list is not None:
                    GL.glDeleteLists(self.grid_list, 1)
                
                # Create axes and grid suitable for the new coordinate system
                self._create_axes_display_list()
                self._create_grid_display_list()
                
                # Force screen refresh
                self.redraw()
                # Request update more leisurely in the main event loop
                self.after(20, self._force_redraw)
            except Exception as e:
                logger.error(f"Error occurred during coordinate system change: {e}")
    
    def _force_redraw(self):
        """Force redraw the screen"""
        try:
            # Check OpenGL state
            if not self.gl_initialized:
                return

            # Activate context
            self.tkMakeCurrent()

            # Clear and redraw the entire screen
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity()

            # Set up 3D scene
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)

            # Call display lists
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)

            # Complete scene update
            self.redraw()

            # Force buffer swap
            self.tkSwapBuffers()

            # TK update
            self.update()
            self.update_idletasks()

        except Exception as e:
            pass

    def _force_complete_redraw(self):
        """Force complete redraw with state validation for toggle operations."""
        try:
            # Check OpenGL state
            if not self.gl_initialized:
                return

            # Activate context
            self.tkMakeCurrent()

            # Clear all buffers completely
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Force immediate redraw with current state
            self.redraw()

            # Force buffer swap to ensure changes are visible
            self.tkSwapBuffers()

            # Schedule additional redraw to ensure state changes are fully applied
            self.after_idle(self.redraw)

        except Exception as e:
            logger.error(f"Error in force complete redraw: {e}")
            # Fallback to regular redraw
            self.redraw()
    
    def reset_view(self):
        """
        Reset view - reset to default camera position and angle
        """
        # Use X-axis rotation angle suitable for the current coordinate system
        self.rot_x = COORDINATE_X_ROTATION_Y_UP  # Y-up is the default setting
        self.rot_y = 45.0
        self.zoom = -4.0
        self.trans_x = 0.0  # Additional: reset translation value too
        self.trans_y = 0.0  # Additional: reset translation value too
        self.redraw()
        
    def set_marker_names(self, marker_names):
        """Set the list of marker names"""
        self.marker_names = marker_names
        self.redraw()
        
    def set_skeleton_pairs(self, skeleton_pairs):
        """Set skeleton configuration pairs"""
        self.skeleton_pairs = skeleton_pairs
        self.redraw()
        
    def set_outliers(self, outliers):
        """Set outlier data"""
        self.outliers = outliers
        self.redraw()
        
    def set_show_marker_names(self, show):
        """
        Set whether to display marker names

        Args:
            show: True to display marker names, False otherwise
        """
        logger.debug(f"Setting show_marker_names to {show}")
        self.show_marker_names = show
        # Force complete redraw to ensure state change is reflected
        self._force_complete_redraw()
        
    def set_data_limits(self, x_range, y_range, z_range):
        """
        Sets the range of the data.

        Args:
            x_range: X-axis range (min, max)
            y_range: Y-axis range (min, max)
            z_range: Z-axis range (min, max)
        """
        self.data_limits = {
            'x': x_range,
            'y': y_range,
            'z': z_range
        }

    # Add mouse event handler methods
    def on_mouse_press(self, event):
        """Called when the left mouse button is pressed"""
        self.last_x, self.last_y = event.x, event.y
        self.dragging = False

        # Check for reference line click first (in analysis mode with 2 markers)
        if (self.analysis_mode_active and len(self.analysis_selection) == 2 and
            self._is_mouse_over_reference_line(event.x, event.y)):
            self._handle_reference_line_click()
            return

        # Perform marker picking
        if self.data is not None and len(self.marker_names) > 0:
            self.pick_marker(event.x, event.y)

    def on_mouse_release(self, _event):
        """Called when the left mouse button is released"""
        # Reset context state when mouse interaction ends
        self._context_active = False

        # Consider it a click if not in dragging state
        if not self.dragging and self.data is not None:
            pass  # Picking is handled in press

    def on_mouse_move(self, event):
        """Called when dragging with the left mouse button (rotation)"""
        dx, dy = event.x - self.last_x, event.y - self.last_y

        # Switch to dragging state only when significant drag occurs
        if abs(dx) > 3 or abs(dy) > 3:
            self.dragging = True

        # Perform only rotation during drag
        if self.dragging:
            self.last_x, self.last_y = event.x, event.y
            self.rot_y += dx * 0.5
            self.rot_x += dy * 0.5

            # OPTIMIZATION: Use immediate redraw for smooth camera controls
            # Skip frame rate limiting for mouse interactions
            self._immediate_redraw()

    def on_right_mouse_press(self, event):
        """Handle right mouse button press event (start view translation or pattern selection mode)"""
        if not self.pattern_selection_mode: # Start view translation only when not in pattern selection mode
            self.dragging = True
            self.last_x = event.x
            self.last_y = event.y
            
    def on_right_mouse_release(self, event):
        """Handle right mouse button release event (end view translation or select pattern marker)"""
        # Reset context state when mouse interaction ends
        self._context_active = False

        if self.pattern_selection_mode:
             # Pattern selection mode: Attempt marker picking
            self.pick_marker(event.x, event.y)
        elif self.dragging:
            # End view translation mode
            self.dragging = False

    def on_right_mouse_move(self, event):
        """Called when dragging with the right mouse button (translation)"""
        dx, dy = event.x - self.last_x, event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y

        # Calculate screen translation (move as a ratio of screen size)
        self.trans_x += dx * 0.005
        self.trans_y -= dy * 0.005  # Invert coordinate system direction (screen y increases downwards)

        # OPTIMIZATION: Use immediate redraw for smooth camera controls
        self._immediate_redraw()

    def on_scroll(self, event):
        """Called when scrolling the mouse wheel (zoom)"""
        # On Windows: event.delta, other platforms may need different approaches
        self.zoom += event.delta * 0.001

        # OPTIMIZATION: Use immediate redraw for smooth camera controls
        self._immediate_redraw()

    def on_mouse_motion(self, event):
        """Handle mouse motion for hover detection on reference line"""
        if not self.analysis_mode_active or len(self.analysis_selection) != 2:
            if self.ref_line_hover:
                self.ref_line_hover = False
                self.redraw()
            return

        # Check if mouse is hovering over reference line
        was_hovering = self.ref_line_hover
        self.ref_line_hover = self._is_mouse_over_reference_line(event.x, event.y)

        # Redraw only if hover state changed
        if was_hovering != self.ref_line_hover:
            self.redraw()

    def _immediate_redraw(self):
        """Immediate redraw without frame rate limiting for mouse interactions."""
        if not self.gl_initialized:
            return

        try:
            # Force immediate context activation and rendering
            self.tkMakeCurrent()
            self._context_active = True

            # Call the main rendering pipeline
            self._update_plot_immediate()

        except Exception as e:
            logger.error(f"Immediate redraw error: {e}")

    def _update_plot_immediate(self):
        """Immediate plot update for mouse interactions - includes all scene elements."""
        if not self.gl_initialized:
            return

        try:
            # Basic viewport and projection setup
            width = self.winfo_width()
            height = self.winfo_height()
            if width <= 0 or height <= 0:
                return

            # Set OpenGL states for immediate rendering
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height)
            GLU.gluPerspective(45, aspect, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)

            # Clear and setup camera
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity()
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)

            # Apply coordinate system rotation
            if getattr(self, 'is_z_up', False):
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)

            # Draw grid and axes
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)

            # CRITICAL FIX: Render markers during immediate camera updates
            self._render_markers_immediate()

            # ENHANCEMENT: Render skeleton if enabled for complete scene visibility
            self._render_skeleton_immediate()

            # BUG FIX: Render additional visual elements during camera interactions
            self._render_trajectories_immediate()
            self._render_marker_names_immediate()
            self._render_analysis_immediate()

            # Force immediate buffer swap for responsive camera controls
            self.tkSwapBuffers()

        except Exception as e:
            logger.error(f"Immediate plot update error: {e}")

    def _render_markers_immediate(self):
        """Render markers immediately for camera interactions - optimized version."""
        # Skip if no data available
        if self.data is None or not self.marker_names:
            return

        try:
            # Quick marker data collection for immediate rendering
            positions = []
            colors = []
            selected_position = None

            # Use optimized data access
            if hasattr(self.data, 'iloc'):
                frame_data = self.data.iloc[self.frame_idx]
                current_marker_str = str(self.current_marker) if self.current_marker is not None else ""

                for marker in self.marker_names:
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if x_col in frame_data.index and y_col in frame_data.index and z_col in frame_data.index:
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                                continue

                            pos = [x, y, z]
                            marker_str = str(marker)

                            # Set color based on selection state using visual settings
                            if marker_str == current_marker_str:
                                color = self.marker_visual_settings.get_selected_color() if self.marker_visual_settings else [1.0, 0.9, 0.4]
                                colors.append(color)
                                selected_position = pos
                            else:
                                color = self.marker_visual_settings.get_normal_color() if self.marker_visual_settings else [1.0, 1.0, 1.0]
                                colors.append(color)

                            positions.append(pos)

                    except (KeyError, IndexError):
                        continue

            # Render normal markers with customized visual settings
            if positions:
                marker_size = self.marker_visual_settings.get_marker_size() if self.marker_visual_settings else 5.0
                GL.glPointSize(marker_size)

                # Enable blending for opacity support
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

                GL.glBegin(GL.GL_POINTS)
                for i, pos in enumerate(positions):
                    color = colors[i]
                    if self.marker_visual_settings:
                        # Apply opacity
                        opacity = self.marker_visual_settings.get_opacity()
                        GL.glColor4f(color[0], color[1], color[2], opacity)
                    else:
                        GL.glColor3fv(color)
                    GL.glVertex3fv(pos)
                GL.glEnd()

                # Disable blending
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glDisable(GL.GL_BLEND)

            # Highlight selected marker with customized settings
            if selected_position:
                # Use larger size for selected marker
                selected_size = (self.marker_visual_settings.get_marker_size() + 3.0) if self.marker_visual_settings else 8.0
                GL.glPointSize(selected_size)

                # Enable blending for opacity support
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

                GL.glBegin(GL.GL_POINTS)
                selected_color = self.marker_visual_settings.get_selected_color() if self.marker_visual_settings else [1.0, 0.9, 0.4]
                if self.marker_visual_settings:
                    opacity = self.marker_visual_settings.get_opacity()
                    GL.glColor4f(selected_color[0], selected_color[1], selected_color[2], opacity)
                else:
                    GL.glColor3fv(selected_color)
                GL.glVertex3fv(selected_position)
                GL.glEnd()

                # Disable blending
                if self.marker_visual_settings and self.marker_visual_settings.get_opacity() < 1.0:
                    GL.glDisable(GL.GL_BLEND)

        except Exception as e:
            logger.error(f"Immediate marker rendering error: {e}")

    def _render_skeleton_immediate(self):
        """Render skeleton immediately for camera interactions - OPTIMIZED with caching."""
        # Skip if skeleton is not enabled or no data available
        if not (hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs')):
            return

        if self.data is None or not self.marker_names:
            return

        try:
            # OPTIMIZATION: Use cached display list if available and valid
            if (self._skeleton_cache_valid and
                self._cached_frame_idx == self.frame_idx and
                self._skeleton_display_list is not None):

                # Simply call the cached display list - MUCH faster than recalculating
                GL.glCallList(self._skeleton_display_list)
                return

            # Fallback: If cache is invalid, use simplified immediate rendering
            # This should rarely happen during camera interactions
            marker_positions = {}

            # Use optimized data access
            if hasattr(self.data, 'iloc'):
                frame_data = self.data.iloc[self.frame_idx]

                for marker in self.marker_names:
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if all(col in frame_data.index for col in [x_col, y_col, z_col]):
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if not any(pd.isna(val) for val in [x, y, z]):
                                marker_positions[marker] = [x, y, z]

                    except (KeyError, IndexError):
                        continue

            # Render skeleton lines (simplified fallback version)
            if marker_positions:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

                # Get customized settings
                line_width = self.marker_visual_settings.get_skeleton_line_width() if self.marker_visual_settings else 2.0
                normal_color = self.marker_visual_settings.get_skeleton_normal_color() if self.marker_visual_settings else (0.7, 0.7, 0.7)
                outlier_color = self.marker_visual_settings.get_skeleton_outlier_color() if self.marker_visual_settings else (1.0, 0.0, 0.0)
                opacity = self.marker_visual_settings.get_skeleton_opacity() if self.marker_visual_settings else 0.8

                # Pass 1: Draw normal skeleton lines
                GL.glLineWidth(line_width)
                GL.glColor4f(normal_color[0], normal_color[1], normal_color[2], opacity)
                GL.glBegin(GL.GL_LINES)

                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]

                        # Check outlier status
                        outlier_status1 = False
                        outlier_status2 = False
                        if hasattr(self, 'outliers'):
                            outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                            outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]

                        is_outlier = outlier_status1 or outlier_status2
                        if not is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)

                GL.glEnd()

                # Pass 2: Draw outlier skeleton lines
                outlier_line_width = (line_width + 1.5) if self.marker_visual_settings else 3.5
                GL.glLineWidth(outlier_line_width)
                GL.glColor4f(outlier_color[0], outlier_color[1], outlier_color[2], 1.0)  # Full opacity for outliers
                GL.glBegin(GL.GL_LINES)

                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]

                        # Check outlier status
                        outlier_status1 = False
                        outlier_status2 = False
                        if hasattr(self, 'outliers'):
                            outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                            outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]

                        is_outlier = outlier_status1 or outlier_status2
                        if is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)

                GL.glEnd()

                # Reset line width
                GL.glLineWidth(1.0)
                GL.glDisable(GL.GL_BLEND)

        except Exception as e:
            logger.error(f"Immediate skeleton rendering error: {e}")

    def _cache_skeleton_geometry(self):
        """Cache skeleton geometry for the current frame to optimize camera interactions."""
        if not (hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs')):
            self._skeleton_cache_valid = False
            return

        if self.data is None or not self.marker_names:
            self._skeleton_cache_valid = False
            return

        # Check if cache is already valid for current frame
        if (self._skeleton_cache_valid and
            self._cached_frame_idx == self.frame_idx and
            self._skeleton_display_list is not None):
            return

        try:
            # Clear existing display list
            if self._skeleton_display_list is not None:
                GL.glDeleteLists(self._skeleton_display_list, 1)
                self._skeleton_display_list = None

            # Collect marker positions for current frame
            marker_positions = {}
            if hasattr(self.data, 'iloc'):
                frame_data = self.data.iloc[self.frame_idx]

                for marker in self.marker_names:
                    try:
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if all(col in frame_data.index for col in [x_col, y_col, z_col]):
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            if not any(pd.isna(val) for val in [x, y, z]):
                                marker_positions[marker] = [x, y, z]

                    except (KeyError, IndexError):
                        continue

            # Create display list for skeleton geometry with proper type handling
            if marker_positions:
                skeleton_list_raw = GL.glGenLists(1)
                self._skeleton_display_list = int(skeleton_list_raw)  # Convert to standard int to avoid numpy type issues
                GL.glNewList(self._skeleton_display_list, GL.GL_COMPILE)

                # Enable blending and smoothing
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

                # Pass 1: Normal skeleton lines with customized settings
                line_width = self.marker_visual_settings.get_skeleton_line_width() if self.marker_visual_settings else 2.0
                normal_color = self.marker_visual_settings.get_skeleton_normal_color() if self.marker_visual_settings else (0.7, 0.7, 0.7)
                opacity = self.marker_visual_settings.get_skeleton_opacity() if self.marker_visual_settings else 0.8

                GL.glLineWidth(line_width)
                GL.glColor4f(normal_color[0], normal_color[1], normal_color[2], opacity)
                GL.glBegin(GL.GL_LINES)

                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]

                        # Check outlier status if available
                        outlier_status1 = False
                        outlier_status2 = False
                        if hasattr(self, 'outliers'):
                            outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                            outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]

                        is_outlier = outlier_status1 or outlier_status2
                        if not is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)

                GL.glEnd()

                # Pass 2: Outlier skeleton lines with customized settings
                outlier_line_width = (line_width + 1.5) if self.marker_visual_settings else 3.5
                outlier_color = self.marker_visual_settings.get_skeleton_outlier_color() if self.marker_visual_settings else (1.0, 0.0, 0.0)

                GL.glLineWidth(outlier_line_width)
                GL.glColor4f(outlier_color[0], outlier_color[1], outlier_color[2], 1.0)
                GL.glBegin(GL.GL_LINES)

                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]

                        # Check outlier status if available
                        outlier_status1 = False
                        outlier_status2 = False
                        if hasattr(self, 'outliers'):
                            outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                            outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]

                        is_outlier = outlier_status1 or outlier_status2
                        if is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)

                GL.glEnd()

                # Add explicit torso lines
                explicit_torso_pairs = [
                    ("RHip", "RShoulder"),
                    ("LHip", "LShoulder"),
                    ("RHip", "LHip"),
                    ("RShoulder", "LShoulder")
                ]

                GL.glLineWidth(line_width)
                GL.glColor4f(normal_color[0], normal_color[1], normal_color[2], opacity)
                GL.glBegin(GL.GL_LINES)

                for pair in explicit_torso_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        GL.glVertex3fv(p1)
                        GL.glVertex3fv(p2)

                GL.glEnd()

                # Reset OpenGL state
                GL.glLineWidth(1.0)
                GL.glDisable(GL.GL_BLEND)

                GL.glEndList()

                # Mark cache as valid
                self._skeleton_cache_valid = True
                self._cached_frame_idx = self.frame_idx

        except Exception as e:
            logger.error(f"Skeleton caching error: {e}")
            self._skeleton_cache_valid = False

    def _render_trajectories_immediate(self):
        """Render trajectories immediately for camera interactions - optimized version."""
        # Skip if trajectory is not enabled or no data available
        if not (hasattr(self, 'show_trajectory') and self.show_trajectory):
            return

        if self.data is None or not self.marker_names:
            return

        try:
            # Choose marker for trajectory: override current_marker in analysis mode
            marker_to_trace = self.current_marker
            if getattr(self, 'analysis_mode_active', False):
                sel = self.analysis_selection
                if len(sel) == 1:
                    marker_to_trace = sel[0]
                elif len(sel) == 2:
                    marker_to_trace = sel[1]
                elif len(sel) >= 3:
                    marker_to_trace = sel[1]

            if marker_to_trace is not None:
                trajectory_points = []

                for i in range(0, self.frame_idx + 1):
                    try:
                        x = self.data.loc[i, f'{marker_to_trace}_X']
                        y = self.data.loc[i, f'{marker_to_trace}_Y']
                        z = self.data.loc[i, f'{marker_to_trace}_Z']

                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            continue

                        trajectory_points.append([x, y, z])

                    except KeyError:
                        continue

                if trajectory_points:
                    GL.glLineWidth(0.8)
                    GL.glColor3f(1.0, 0.9, 0.4)  # Light yellow
                    GL.glBegin(GL.GL_LINE_STRIP)

                    for point in trajectory_points:
                        GL.glVertex3fv(point)

                    GL.glEnd()

        except Exception as e:
            logger.error(f"Immediate trajectory rendering error: {e}")

    def _render_marker_names_immediate(self):
        """Render marker names immediately for camera interactions - optimized version."""
        # Skip if marker names are not enabled or no data available
        if not (hasattr(self, 'show_marker_names') and self.show_marker_names):
            return

        if self.data is None or not self.marker_names:
            return

        try:
            # Quick marker position collection for name rendering
            marker_positions = {}
            valid_markers = []

            # Use optimized data access
            if hasattr(self.data, 'iloc'):
                frame_data = self.data.iloc[self.frame_idx]
                current_marker_str = str(self.current_marker) if self.current_marker is not None else ""

                for marker in self.marker_names:
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if x_col in frame_data.index and y_col in frame_data.index and z_col in frame_data.index:
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                                continue

                            marker_positions[marker] = [x, y, z]
                            valid_markers.append(marker)

                    except (KeyError, IndexError):
                        continue

            # Render marker names (simplified version for immediate rendering)
            if valid_markers and marker_positions:
                try:
                    # Save current OpenGL state
                    GL.glPushMatrix()
                    GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT)

                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""

                    # Render all normal marker names (white)
                    for marker in valid_markers:
                        marker_str = str(marker)
                        if marker_str == current_marker_str:
                            continue  # Render selected marker later

                        pos = marker_positions[marker]

                        # Render normal marker names in white
                        GL.glColor3f(1.0, 1.0, 1.0)  # White
                        GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])

                        # Render marker name
                        for c in marker_str:
                            try:
                                GLUT.glutBitmapCharacter(SMALL_FONT, ord(c))
                            except:
                                pass

                    # Render selected marker name in yellow (separate pass)
                    if self.current_marker is not None:
                        for marker in valid_markers:
                            marker_str = str(marker)
                            if marker_str == current_marker_str:
                                pos = marker_positions[marker]

                                # Render selected marker name in yellow
                                GL.glColor3f(1.0, 0.9, 0.4)  # Light yellow
                                GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])

                                # Render marker name
                                for c in marker_str:
                                    try:
                                        GLUT.glutBitmapCharacter(SMALL_FONT, ord(c))
                                    except:
                                        pass
                                break

                    # Restore OpenGL state
                    GL.glPopAttrib()
                    GL.glPopMatrix()

                except Exception as text_error:
                    logger.error(f"Immediate text rendering error: {text_error}")

        except Exception as e:
            logger.error(f"Immediate marker names rendering error: {e}")

    def _render_analysis_immediate(self):
        """Render analysis mode visualizations immediately for camera interactions - complete version."""
        # Skip if analysis mode is not active or no data available
        if not (hasattr(self, 'analysis_mode_active') and self.analysis_mode_active):
            return

        if not (hasattr(self, 'analysis_selection') and len(self.analysis_selection) >= 1):
            return

        if self.data is None or not self.marker_names:
            return

        try:
            # Quick marker position collection for analysis rendering
            marker_positions = {}

            # Use optimized data access
            if hasattr(self.data, 'iloc'):
                frame_data = self.data.iloc[self.frame_idx]

                for marker in self.marker_names:
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if x_col in frame_data.index and y_col in frame_data.index and z_col in frame_data.index:
                            x, y, z = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                                continue

                            marker_positions[marker] = [x, y, z]

                    except (KeyError, IndexError):
                        continue

            # Highlight selected analysis markers (Green, larger size based on customization)
            base_marker_size = self.marker_visual_settings.get_marker_size() if self.marker_visual_settings else 5.0
            analysis_marker_size = base_marker_size + 5.0  # Always 5 pixels larger than base size
            GL.glPointSize(analysis_marker_size)
            GL.glColor3f(0.0, 1.0, 0.0)  # Green color
            GL.glBegin(GL.GL_POINTS)

            analysis_positions_raw = {}
            valid_analysis_markers = []

            for marker_name in self.analysis_selection:
                if marker_name in marker_positions:
                    pos = marker_positions[marker_name]
                    analysis_positions_raw[marker_name] = np.array(pos)  # Store as numpy array
                    GL.glVertex3fv(pos)
                    valid_analysis_markers.append(marker_name)

            GL.glEnd()
            GL.glPointSize(base_marker_size)  # Reset to base marker size

            # Complete analysis visualization based on selection count
            num_valid_analysis = len(valid_analysis_markers)

            # -- Velocity and Acceleration (1 Marker Selected) --
            if num_valid_analysis == 1:
                marker_name = valid_analysis_markers[0]
                current_pos = analysis_positions_raw[marker_name]
                frame_idx = self.frame_idx
                frame_rate = float(self.parent.fps_var.get()) if hasattr(self.parent, 'fps_var') else 30.0

                # Get positions for velocity and acceleration calculation
                pos_data = {}
                valid_indices = True
                for i in range(frame_idx - 2, frame_idx + 3):  # Need i-2 to i+2 for accel calc
                    if 0 <= i < self.num_frames:
                        try:
                            pos_data[i] = self.data.loc[i, [f'{marker_name}_{c}' for c in 'XYZ']].values
                            if np.isnan(pos_data[i]).any():
                                valid_indices = False
                                break
                        except KeyError:
                            valid_indices = False
                            break
                    else:
                        valid_indices = False
                        break

                # Calculate Velocity and Acceleration (if data is valid)
                velocity = None
                acceleration = None
                if valid_indices:
                    # Calculate velocity at current frame
                    velocity = calculate_velocity(pos_data[frame_idx-1], current_pos, pos_data[frame_idx+1], frame_rate)

                    # Calculate velocities at previous and next frames for acceleration
                    vel_prev = calculate_velocity(pos_data[frame_idx-2], pos_data[frame_idx-1], current_pos, frame_rate)
                    vel_next = calculate_velocity(current_pos, pos_data[frame_idx+1], pos_data[frame_idx+2], frame_rate)

                    if vel_prev is not None and vel_next is not None:
                        acceleration = calculate_acceleration(vel_prev, vel_next, frame_rate)

                # Display Text
                analysis_text_lines = []

                if velocity is not None:
                    speed = np.linalg.norm(velocity)
                    analysis_text_lines.append(f"{speed:.2f} m/s")

                if acceleration is not None:
                    accel_mag = np.linalg.norm(acceleration)
                    analysis_text_lines.append(f"{accel_mag:.2f} m/s²")

                # Render text if available
                if analysis_text_lines:
                    text_color = (1.0, 1.0, 0.0)  # Yellow
                    text_base_pos = [current_pos[0], current_pos[1] + 0.04, current_pos[2]]
                    line_height_offset = 0.02

                    GL.glPushMatrix()
                    GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_DEPTH_BUFFER_BIT)
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glColor3fv(text_color)

                    for i, line in enumerate(analysis_text_lines):
                        current_text_pos = [text_base_pos[0], text_base_pos[1] - i * line_height_offset, text_base_pos[2]]
                        GL.glRasterPos3f(current_text_pos[0], current_text_pos[1], current_text_pos[2])
                        for char in line:
                            try:
                                GLUT.glutBitmapCharacter(SMALL_FONT, ord(char))
                            except Exception:
                                pass

                    GL.glPopAttrib()
                    GL.glPopMatrix()

            # -- Distance / Angle (2 or 3 Markers Selected) --
            elif num_valid_analysis >= 2:
                # Get positions in the selection order
                analysis_positions_ordered = [analysis_positions_raw[m] for m in valid_analysis_markers]

                # Draw thicker lines between selected markers based on customization
                base_line_width = self.marker_visual_settings.get_skeleton_line_width() if self.marker_visual_settings else 2.0
                analysis_line_width = base_line_width + 1.0  # Always 1 pixel thicker than base skeleton width
                GL.glLineWidth(analysis_line_width)
                GL.glColor3f(0.0, 1.0, 0.0)  # Green color for analysis lines

                if len(analysis_positions_ordered) == 2:
                    GL.glBegin(GL.GL_LINES)
                    GL.glVertex3fv(analysis_positions_ordered[0])
                    GL.glVertex3fv(analysis_positions_ordered[1])
                    GL.glEnd()

                    # Draw reference horizontal line and arc for segment angle
                    pA = analysis_positions_ordered[0]
                    pB = analysis_positions_ordered[1]
                    v = pA - pB
                    norm_v = np.linalg.norm(v)
                    if norm_v > 0:
                        # Get reference vector from state manager
                        if hasattr(self.parent, 'state_manager'):
                            u = self.parent.state_manager.get_reference_vector()
                        else:
                            u = np.array([1.0, 0.0, 0.0])  # Default to X-axis

                        # Store reference line endpoints for click detection (fixed length)
                        self.ref_line_start = pB.copy()
                        self.ref_line_end = pB + u * REF_LINE_FIXED_LENGTH

                        # reference line with hover effect based on customization
                        ref_line_width = base_line_width + (1.0 if self.ref_line_hover else 0.0)
                        line_color = [0.5, 0.9, 0.5] if self.ref_line_hover else [0.3, 0.7, 0.3]

                        GL.glLineWidth(ref_line_width)
                        GL.glColor3fv(line_color)
                        GL.glBegin(GL.GL_LINES)
                        GL.glVertex3fv(self.ref_line_start)
                        GL.glVertex3fv(self.ref_line_end)
                        GL.glEnd()

                        # Add axis label at the end of reference line
                        if hasattr(self.parent, 'state_manager'):
                            current_axis = self.parent.state_manager.editing_state.reference_axis
                            self._render_axis_label(self.ref_line_end, current_axis)

                        # arc
                        radius = norm_v * 0.2
                        p_ref = pB + u * radius
                        p_seg = pB + (v / norm_v) * radius
                        pts = calculate_arc_points(vertex=pB, p1=p_ref, p3=p_seg, radius=radius, num_segments=20)
                        if pts:
                            GL.glEnable(GL.GL_BLEND)
                            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                            GL.glDepthMask(GL.GL_FALSE)
                            GL.glDisable(GL.GL_CULL_FACE)
                            GL.glColor4f(1.0, 0.6, 0.0, 0.5)
                            GL.glBegin(GL.GL_TRIANGLE_FAN)
                            GL.glVertex3fv(pB)
                            for pt in pts:
                                GL.glVertex3fv(pt)
                            GL.glEnd()
                            GL.glEnable(GL.GL_CULL_FACE)
                            arc_outline_width = max(1.0, base_line_width * 0.5)  # Half of base line width, minimum 1.0
                            GL.glLineWidth(arc_outline_width)
                            GL.glColor4f(1.0, 0.6, 0.0, 0.8)
                            GL.glBegin(GL.GL_LINE_STRIP)
                            for pt in pts:
                                GL.glVertex3fv(pt)
                            GL.glEnd()
                            GL.glDepthMask(GL.GL_TRUE)
                        GL.glLineWidth(base_line_width)

                elif len(analysis_positions_ordered) == 3:
                    GL.glBegin(GL.GL_LINES)
                    # Draw lines based on selection order: 0->1 and 1->2
                    GL.glVertex3fv(analysis_positions_ordered[0])
                    GL.glVertex3fv(analysis_positions_ordered[1])
                    GL.glVertex3fv(analysis_positions_ordered[1])
                    GL.glVertex3fv(analysis_positions_ordered[2])
                    GL.glEnd()

                GL.glLineWidth(base_line_width)  # Reset to base line width

                # Calculate and prepare text for display
                dist_text = None
                dist_pos = None
                angle_text = None
                angle_pos = None

                if len(analysis_positions_ordered) == 2:
                    distance = calculate_distance(analysis_positions_ordered[0], analysis_positions_ordered[1])
                    if distance is not None:
                        pA = analysis_positions_ordered[0]
                        pB = analysis_positions_ordered[1]
                        # distance text at midpoint
                        mid_pt = (pA + pB) / 2
                        dist_text = f"{distance:.3f} m"
                        dist_pos = [mid_pt[0], mid_pt[1] + 0.02, mid_pt[2]]
                        # compute angle relative to reference axis
                        if hasattr(self.parent, 'state_manager'):
                            u = self.parent.state_manager.get_reference_vector()
                        else:
                            u = np.array([1.0, 0.0, 0.0])  # Default to X-axis
                        ref_point = pB + u
                        angle_val = calculate_angle(ref_point, pB, pA)
                        if angle_val is not None:
                            angle_text = f"{angle_val:.1f}\u00B0"
                            angle_pos = [pB[0], pB[1] + 0.03, pB[2]]

                elif len(analysis_positions_ordered) == 3:
                    # Angle at the vertex (second selected marker)
                    angle = calculate_angle(analysis_positions_ordered[0], analysis_positions_ordered[1], analysis_positions_ordered[2])
                    if angle is not None:
                        angle_text = f"{angle:.1f}\u00B0"
                        angle_pos = [analysis_positions_ordered[1][0], analysis_positions_ordered[1][1] + 0.03, analysis_positions_ordered[1][2]]

                        # Calculate and draw the arc
                        arc_radius = min(np.linalg.norm(analysis_positions_ordered[0]-analysis_positions_ordered[1]),
                                         np.linalg.norm(analysis_positions_ordered[2]-analysis_positions_ordered[1])) * 0.2
                        arc_points = calculate_arc_points(vertex=analysis_positions_ordered[1],
                                                          p1=analysis_positions_ordered[0],
                                                          p3=analysis_positions_ordered[2],
                                                          radius=max(0.01, arc_radius),
                                                          num_segments=20)

                        if arc_points:
                            # Draw the filled, semi-transparent arc using TRIANGLE_FAN
                            GL.glEnable(GL.GL_BLEND)
                            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                            GL.glDepthMask(GL.GL_FALSE)
                            GL.glDisable(GL.GL_CULL_FACE)
                            GL.glColor4f(1.0, 1.0, 0.0, 0.3)  # yellow with 30% alpha

                            GL.glBegin(GL.GL_TRIANGLE_FAN)
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            for point in arc_points:
                                GL.glVertex3fv(point)
                            GL.glEnd()

                            GL.glEnable(GL.GL_CULL_FACE)
                            arc_outline_width = max(1.0, base_line_width * 0.5)  # Half of base line width, minimum 1.0
                            GL.glLineWidth(arc_outline_width)
                            GL.glColor4f(1.0, 1.0, 0.0, 0.7)
                            GL.glBegin(GL.GL_LINE_STRIP)
                            for point in arc_points:
                                GL.glVertex3fv(point)
                            GL.glEnd()
                            GL.glDepthMask(GL.GL_TRUE)

                # Render the analysis text if available
                if (dist_text and dist_pos) or (angle_text and angle_pos):
                    GL.glPushMatrix()
                    GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_DEPTH_BUFFER_BIT)
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glColor3f(0.0, 1.0, 0.0)

                    if dist_text and dist_pos:
                        GL.glRasterPos3f(dist_pos[0], dist_pos[1], dist_pos[2])
                        for ch in dist_text:
                            try:
                                GLUT.glutBitmapCharacter(LARGE_FONT, ord(ch))
                            except:
                                pass

                    if angle_text and angle_pos:
                        GL.glRasterPos3f(angle_pos[0], angle_pos[1], angle_pos[2])
                        for ch in angle_text:
                            try:
                                GLUT.glutBitmapCharacter(LARGE_FONT, ord(ch))
                            except:
                                pass

                    GL.glPopAttrib()
                    GL.glPopMatrix()

        except Exception as e:
            logger.error(f"Immediate analysis rendering error: {e}")

    def pick_marker(self, x, y):
        """
        Select marker (picking)
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
        """
        if not self.gl_initialized or not hasattr(self, 'picking_texture'):
            return
        
        # Check picking texture initialization and initialize if necessary
        if not self.picking_texture.initialized:
            width, height = self.winfo_width(), self.winfo_height()
            if width <= 0 or height <= 0 or not self.picking_texture.init(width, height):
                return
        
        try:
            # Activate context
            self.tkMakeCurrent()
            
            # Render to picking texture
            if not self.picking_texture.enable_writing():
                return
            
            # Initialize buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            # Set perspective projection
            width, height = self.winfo_width(), self.winfo_height()
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height)
            # Use perspective projection (like in pick_marker)
            GLU.gluPerspective(45, aspect, 0.1, 100.0) # fov, aspect, near, far

            # 3. Switch back to the modelview matrix for camera/object transformations
            GL.glMatrixMode(GL.GL_MODELVIEW)
            # --- Viewport and Projection Setup --- END
            
            # Initialize frame (Clear after setting viewport/projection)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0) # Ensure clear color is set
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity() # Reset modelview matrix before camera setup
            
            # Set camera position (zoom, translation, rotation)
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            
            # Additional rotation if Z-up coordinate system
            if self.is_z_up:
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)
            
            # Set state for picking rendering
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_POINT_SMOOTH)
            GL.glDisable(GL.GL_LINE_SMOOTH)
            
            # Check marker information
            if self.data is None or len(self.marker_names) == 0:
                self.picking_texture.disable_writing()
                return
            
            # Set large point size for picking
            GL.glPointSize(12.0)
            
            # Render markers with unique ID colors
            GL.glBegin(GL.GL_POINTS)
            
            # Optimized batch data access for better performance
            frame_data = self.data.iloc[self.frame_idx] if hasattr(self.data, 'iloc') else None
            if frame_data is not None:
                for idx, marker in enumerate(self.marker_names):
                    try:
                        # Batch access to marker coordinates
                        x_col, y_col, z_col = f'{marker}_X', f'{marker}_Y', f'{marker}_Z'
                        if x_col in frame_data.index and y_col in frame_data.index and z_col in frame_data.index:
                            x_val, y_val, z_val = frame_data[x_col], frame_data[y_col], frame_data[z_col]

                            # Skip NaN values
                            if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val):
                                continue

                            # Set marker ID starting from 1 (0 is background)
                            marker_id = idx + 1

                            # Unique color encoding for each marker
                            # R channel: Normalized value of marker ID
                            r = float(marker_id) / float(len(self.marker_names) + 1)
                            g = float(marker_id % 256) / 255.0  # Additional info
                            b = 1.0  # Constant for marker identification

                            GL.glColor3f(r, g, b)
                            GL.glVertex3f(x_val, y_val, z_val)

                    except (KeyError, IndexError):
                        continue
            
            GL.glEnd()
            
            # Verify rendering completion
            GL.glFinish()
            GL.glFlush()
            
            # Read pixel information (OpenGL coordinate system conversion)
            y_inverted = height - y - 1
            pixel_info = self.read_pixel_at(x, y_inverted)
            
            # Disable picking texture
            self.picking_texture.disable_writing()
            
            # If pixel info exists, select marker
            if pixel_info is not None:
                r_value = pixel_info[0]
                
                # Restore ID value (encoding scheme: r = marker_id / (len(marker_names) + 1))
                actual_id = int(r_value * (len(self.marker_names) + 1) + 0.5)
                
                # Convert marker ID (starts from 1) to index
                marker_idx = actual_id - 1
                
                # Check if marker index is valid
                if 0 <= marker_idx < len(self.marker_names):
                    selected_marker = self.marker_names[marker_idx]
                    
                    # --- Mode-Dependent Handling ---
                    # Handle ANALYSIS mode selection (using left-click, hence in pick_marker)
                    if self.analysis_mode_active:
                        self.parent.handle_analysis_marker_selection(selected_marker)
                        # In analysis mode, we don't update self.current_marker or call _notify_marker_selected
                        # Highlighting is handled by analysis_selection list during redraw.

                    # Handle PATTERN selection mode (using right-click, logic remains here for now)
                    elif self.pattern_selection_mode:
                        # Notify the parent (TRCViewer) to handle the selection change
                        self.parent.handle_pattern_marker_selection(selected_marker)
                            
                    # Handle NORMAL marker selection mode
                    else:
                        # If the already selected marker is clicked again, deselect it
                        if self.current_marker == selected_marker:
                            self.current_marker = None
                            self._notify_marker_selected(None)  # Notify deselection
                        # If a new marker is selected
                        else:
                            # Update current marker
                            self.current_marker = selected_marker
                            
                            # Notify parent class
                            self._notify_marker_selected(selected_marker)
        
            # Restore normal rendering state
            GL.glEnable(GL.GL_BLEND)
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)

            # Update screen - Removed redraw, parent (app.py) will handle the final update
            # self.redraw()
        
        except Exception as e:
            logger.error(f"Marker selection error: {e}")
            import traceback
            traceback.print_exc()

    def read_pixel_at(self, x, y):
        """
        Read pixel information at the specified position
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate (already converted to OpenGL coordinate system)
            
        Returns:
            tuple: (R, G, B) color value or None
        """
        try:
            # Read pixel from framebuffer
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.picking_texture.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # Check if pixel coordinates are within texture bounds
            width, height = self.picking_texture.width, self.picking_texture.height
            if x < 0 or x >= width or y < 0 or y >= height:
                return None
            
            # Read pixel information
            data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGB, GL.GL_FLOAT)
            pixel_info = np.frombuffer(data, dtype=np.float32)
            
            # Restore default settings
            GL.glReadBuffer(GL.GL_NONE)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            
            # Check for background pixel (R=0 means background)
            if pixel_info[0] == 0.0:
                return None
            
            # Return pixel color value
            return (pixel_info[0], pixel_info[1], pixel_info[2])
            
        except Exception as e:
            logger.error(f"Pixel read error: {e}")
            return None

    def _notify_marker_selected(self, marker_name):
        """
        Notify the parent window about the marker selection event
        
        Args:
            marker_name: Selected marker name or None (for deselection)
        """
        # Call parent window method
        if hasattr(self.master, 'on_marker_selected'):
            try:
                self.master.on_marker_selected(marker_name)
            except Exception as e:
                logger.error(f"Error notifying master of marker selection: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Warning: Master {self.master} does not have 'on_marker_selected' method.")
            
            
    def on_configure(self, event):
        """Handle widget resize/move/visibility changes with anti-flickering optimization."""
        if not self.gl_initialized:
            return

        # Get current window dimensions
        current_width = self.winfo_width()
        current_height = self.winfo_height()

        # Skip if dimensions are invalid
        if current_width <= 0 or current_height <= 0:
            return

        current_size = (current_width, current_height)

        # Check if this is actually a size change (not just a move/expose event)
        if current_size == self._last_viewport_size:
            return

        self._last_viewport_size = current_size
        self._resize_in_progress = True

        # Cancel any pending resize timer
        if self._resize_timer:
            self.after_cancel(self._resize_timer)

        # Use throttled resize to prevent excessive redraws during continuous resizing
        import time
        current_time = time.time() * 1000  # Convert to milliseconds

        if current_time - self._last_resize_time >= self._resize_throttle_ms:
            # Immediate resize for responsive feel
            self._perform_optimized_resize(current_width, current_height)
            self._last_resize_time = current_time
        else:
            # Throttled resize - schedule for later
            self._resize_timer = self.after(self._resize_throttle_ms,
                                          lambda: self._perform_optimized_resize(current_width, current_height))

    def _perform_optimized_resize(self, width, height):
        """Perform optimized resize operation with minimal flickering."""
        try:
            # Ensure OpenGL context is active
            self.tkMakeCurrent()

            # Update viewport with new dimensions
            GL.glViewport(0, 0, width, height)

            # Update projection matrix for new aspect ratio
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height) if height > 0 else 1.0
            GLU.gluPerspective(45, aspect, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)

            # Update picking texture size if initialized
            if hasattr(self, 'picking_texture') and self.picking_texture.initialized:
                try:
                    self.picking_texture.cleanup()
                    self.picking_texture.init(width, height)
                except Exception as e:
                    logger.warning(f"Error updating picking texture during resize: {e}")
                    # Continue with resize operation even if picking texture update fails

            # Use optimized redraw that minimizes state changes
            self._optimized_resize_redraw()

            # Mark resize as complete
            self._resize_in_progress = False

        except Exception as e:
            logger.error(f"Error during optimized resize: {e}")
            # Fallback to standard redraw
            self.redraw()
            self._resize_in_progress = False

    def _optimized_resize_redraw(self):
        """Optimized redraw specifically for resize operations with complete visualization."""
        try:
            # Clear buffers efficiently
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity()

            # Apply current camera transformations
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)

            # Apply coordinate system rotation if needed
            if getattr(self, 'is_z_up', False):
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)

            # Render all elements to maintain visual consistency during resize
            # Grid and axes (using display lists for efficiency)
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)

            # Render complete scene during resize to maintain visual consistency
            if self.data is not None and self.marker_names:
                # Use immediate rendering methods for complete visualization
                self._render_markers_immediate()
                self._render_skeleton_immediate()
                self._render_trajectories_immediate()
                self._render_marker_names_immediate()
                self._render_analysis_immediate()

            # Force immediate buffer swap for smooth resize
            self.tkSwapBuffers()

        except Exception as e:
            logger.error(f"Error during optimized resize redraw: {e}")



    def _is_mouse_over_reference_line(self, mouse_x, mouse_y):
        """Check if mouse is hovering over the reference line"""
        if self.ref_line_start is None or self.ref_line_end is None:
            return False

        # Convert 3D line endpoints to screen coordinates
        try:
            start_screen = self._world_to_screen(self.ref_line_start)
            end_screen = self._world_to_screen(self.ref_line_end)

            if start_screen is None or end_screen is None:
                return False

            # Calculate distance from mouse to line segment
            distance = self._point_to_line_distance(
                [mouse_x, mouse_y], start_screen, end_screen
            )

            return distance <= self.ref_line_click_tolerance

        except Exception as e:
            logger.error(f"Error checking reference line hover: {e}")
            return False

    def _world_to_screen(self, world_pos):
        """Convert 3D world coordinates to 2D screen coordinates"""
        try:
            # Get current matrices
            modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
            projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)

            # Project to screen coordinates
            screen_coords = GLU.gluProject(
                world_pos[0], world_pos[1], world_pos[2],
                modelview, projection, viewport
            )

            return [screen_coords[0], viewport[3] - screen_coords[1]]  # Flip Y coordinate

        except Exception as e:
            logger.error(f"Error converting world to screen coordinates: {e}")
            return None

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        try:
            # Vector from line start to end
            line_vec = [line_end[0] - line_start[0], line_end[1] - line_start[1]]
            line_length_sq = line_vec[0]**2 + line_vec[1]**2

            if line_length_sq == 0:
                # Line is a point
                return ((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)**0.5

            # Vector from line start to point
            point_vec = [point[0] - line_start[0], point[1] - line_start[1]]

            # Project point onto line
            t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_length_sq))

            # Find closest point on line segment
            closest = [line_start[0] + t * line_vec[0], line_start[1] + t * line_vec[1]]

            # Return distance to closest point
            return ((point[0] - closest[0])**2 + (point[1] - closest[1])**2)**0.5

        except Exception as e:
            logger.error(f"Error calculating point to line distance: {e}")
            return float('inf')

    def _handle_reference_line_click(self):
        """Handle click on reference line to cycle through axes"""
        try:
            # Get state manager from parent
            if hasattr(self.parent, 'state_manager'):
                new_axis = self.parent.state_manager.cycle_reference_axis()
                logger.info(f"Reference axis cycled to: {new_axis}")

                # Trigger visual feedback and redraw
                self._show_axis_change_feedback(new_axis)
                self.redraw()
            else:
                logger.warning("State manager not found, cannot cycle reference axis")

        except Exception as e:
            logger.error(f"Error handling reference line click: {e}")

    def _show_axis_change_feedback(self, new_axis):
        """Show brief visual feedback when axis changes"""
        # This could be enhanced with visual effects like brief color change
        # For now, just log the change
        logger.info(f"Visual feedback: Reference axis changed to {new_axis}")
        # Future enhancement: Add brief animation or color change

    def _render_axis_label(self, position, axis_name):
        """Render axis label at the specified position"""
        try:
            # Set color for axis label (brighter when hovered)
            if self.ref_line_hover:
                GL.glColor3f(0.8, 1.0, 0.8)  # Bright green when hovered
            else:
                GL.glColor3f(0.6, 0.8, 0.6)  # Normal green

            # Position label slightly offset from line end
            label_pos = [position[0] + 0.02, position[1] + 0.02, position[2]]
            GL.glRasterPos3fv(label_pos)

            # Render axis name
            for char in axis_name:
                try:
                    GLUT.glutBitmapCharacter(SMALL_FONT, ord(char))
                except:
                    pass  # Skip if GLUT not available

        except Exception as e:
            logger.error(f"Error rendering axis label: {e}")

    def set_analysis_state(self, is_active: bool, selected_markers: list):
        """Sets the analysis mode state and the list of selected markers."""
        logger.debug(f"Setting analysis_mode_active to {is_active}, selection: {selected_markers}")
        self.analysis_mode_active = is_active
        # Make a copy to avoid direct modification issues if parent list changes elsewhere
        self.analysis_selection = list(selected_markers)
        logger.debug(f"Renderer analysis state updated: Active={self.analysis_mode_active}, Selection={self.analysis_selection}")
        # Use immediate redraw for instant visual feedback (consistent with marker selection)
        self._immediate_redraw()

    def cleanup_all_resources(self):
        """
        Comprehensive cleanup of all OpenGL resources.

        This method should be called during application shutdown or when the renderer
        is being destroyed to ensure all OpenGL resources are properly released.
        """
        if self._cleanup_performed:
            return  # Prevent multiple cleanup attempts

        try:
            logger.info("Starting comprehensive OpenGL resource cleanup...")

            # Check if widget and OpenGL context are still valid
            context_available = False
            if hasattr(self, 'tkMakeCurrent') and hasattr(self, 'winfo_exists'):
                try:
                    # Check if the widget still exists
                    if self.winfo_exists():
                        self.tkMakeCurrent()
                        context_available = True
                        logger.debug("OpenGL context activated for cleanup")
                    else:
                        logger.debug("Widget no longer exists, skipping OpenGL context activation")
                except Exception as context_error:
                    logger.debug(f"Could not activate OpenGL context for cleanup: {context_error}")
                    # Continue cleanup without OpenGL context

            # Clean up OpenGL resources only if context is available
            if context_available:
                # Clean up picking texture
                if hasattr(self, 'picking_texture') and self.picking_texture:
                    try:
                        self.picking_texture.cleanup()
                        logger.debug("Cleaned up picking texture")
                    except Exception as e:
                        logger.warning(f"Error cleaning up picking texture: {e}")

                # Clean up display lists with proper error handling
                display_lists = [
                    ('grid_list', 'Grid display list'),
                    ('axes_list', 'Axes display list'),
                    ('_skeleton_display_list', 'Skeleton display list')
                ]

                for attr_name, description in display_lists:
                    if hasattr(self, attr_name):
                        display_list = getattr(self, attr_name)
                        if display_list is not None and display_list != 0:
                            try:
                                list_id = int(display_list)  # Ensure it's a standard int
                                GL.glDeleteLists(list_id, 1)
                                setattr(self, attr_name, None)
                                logger.debug(f"Cleaned up {description}")
                            except Exception as e:
                                logger.warning(f"Error cleaning up {description}: {e}")
            else:
                logger.debug("OpenGL context not available, skipping OpenGL resource cleanup")
                # Just reset the references without OpenGL calls
                if hasattr(self, 'picking_texture'):
                    self.picking_texture = None
                for attr_name in ['grid_list', 'axes_list', '_skeleton_display_list']:
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, None)

            # Reset all OpenGL-related flags and caches
            self.gl_initialized = False
            self.initialized = False
            self._context_active = False
            self._skeleton_cache_valid = False
            self._skeleton_cache.clear()

            # Cancel any pending resize operations
            if hasattr(self, '_resize_timer') and self._resize_timer:
                try:
                    self.after_cancel(self._resize_timer)
                    self._resize_timer = None
                except Exception as e:
                    logger.warning(f"Error canceling resize timer: {e}")

            self._cleanup_performed = True
            logger.info("OpenGL resource cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")
            # Mark cleanup as performed even if there were errors to prevent infinite retry
            self._cleanup_performed = True

    def __del__(self):
        """Destructor to ensure cleanup is performed."""
        try:
            if not self._cleanup_performed:
                self.cleanup_all_resources()
        except Exception as e:
            # Don't raise exceptions in destructor
            logger.warning(f"Error in destructor cleanup: {e}")