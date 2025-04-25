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
            # Create FBO
            self.fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            
            # Create texture for ID information
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, width, height, 
                           0, GL.GL_RGB, GL.GL_FLOAT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, 
                                    GL.GL_TEXTURE_2D, self.texture, 0)
            
            # Create texture for depth information
            self.depth_texture = GL.glGenTextures(1)
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
        """Clean up resources"""
        try:
            if self.texture != 0:
                GL.glDeleteTextures(self.texture)
                self.texture = 0
                
            if self.depth_texture != 0:
                GL.glDeleteTextures(self.depth_texture)
                self.depth_texture = 0
                
            if self.fbo != 0:
                GL.glDeleteFramebuffers(1, [self.fbo])
                self.fbo = 0
                
            self.initialized = False
        except Exception as e:
            logger.error(f"Picking texture cleanup error: {e}")
    
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
        self.bind("<Configure>", self.on_configure) # Add binding for Configure event
        
        # Marker picking related variables
        self.picking_texture = PickingTexture()
        self.dragging = False
        
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
            
        self.axes_list = GL.glGenLists(1)
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
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('X'))
        except:
            pass  # Skip label rendering if GLUT is unavailable
        
        # Y Label
        GL.glColor3f(1.0, 1.0, 0.0)  # Yellow
        GL.glRasterPos3f(0, axis_length + text_offset + offset_y, 0)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Y'))
        except:
            pass
        
        # Z Label
        GL.glColor3f(0.0, 0.0, 1.0)  # Blue
        GL.glRasterPos3f(0, offset_y, axis_length + text_offset)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Z'))
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
            # Activate OpenGL context (for safety)
            try:
                self.tkMakeCurrent()
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
            
            # Collect valid marker data for the current frame
            for marker in self.marker_names:
                try:
                    x = self.data.loc[self.frame_idx, f'{marker}_X']
                    y = self.data.loc[self.frame_idx, f'{marker}_Y']
                    z = self.data.loc[self.frame_idx, f'{marker}_Z']
                    
                    # Skip NaN values
                    if pd.isna(x) or pd.isna(y) or pd.isna(z):
                        continue
                    
                    # Adjust position according to the coordinate system
                    pos = [x, y, z]
                        
                    marker_positions[marker] = pos
                    
                    # Set color
                    marker_str = str(marker)
                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""
                    
                    if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                        if marker in self.pattern_markers:
                            colors.append([1.0, 0.0, 0.0])  # Red
                        else:
                            colors.append([1.0, 1.0, 1.0])  # White
                    elif marker_str == current_marker_str:
                        colors.append([1.0, 0.9, 0.4])  # Light yellow
                    else:
                        colors.append([1.0, 1.0, 1.0])  # White
                        
                    positions.append(pos)
                    valid_markers.append(marker)
                    
                    if marker_str == current_marker_str:
                        selected_position = pos
                        
                except KeyError:
                    continue
            
            # Marker rendering - separated into 2 stages: normal markers -> pattern markers
            if positions:
                # Stage 1: Normal markers (unselected markers or when not in pattern mode)
                GL.glPointSize(5.0) # Normal size
                GL.glBegin(GL.GL_POINTS)
                for i, pos in enumerate(positions):
                    marker = valid_markers[i]
                    is_pattern_selected = self.pattern_selection_mode and marker in self.pattern_markers
                    if not is_pattern_selected:
                        GL.glColor3fv(colors[i])
                        GL.glVertex3fv(pos)
                GL.glEnd()
                
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
            
            # Skeleton line rendering
            if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs'):
                # --- Enable Blending and Smoothing (needed for normal lines) ---
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                # ---------------------------------------------------------------
                
                # Pass 1: Draw Normal Skeleton Lines (Gray, Semi-Transparent, Width 2.0)
                GL.glLineWidth(2.0)
                GL.glColor4f(0.7, 0.7, 0.7, 0.8) # Gray, Alpha 0.8
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
                
                # Pass 2: Draw Outlier Skeleton Lines (Red, Opaque, Width 4.0)
                # Blending is already enabled, just change width and color
                GL.glLineWidth(3.5)
                GL.glColor4f(1.0, 0.0, 0.0, 1.0) # Red, Alpha 1.0
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
                # Use the same style as normal skeleton lines (or adjust if needed)
                # Ensure Blend is enabled if needed
                # GL.glEnable(GL.GL_BLEND) # Already enabled from standard skeleton drawing
                GL.glLineWidth(2.0) # Match normal skeleton line width
                GL.glColor4f(0.7, 0.7, 0.7, 0.8) # Gray, Alpha 0.8
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
                    # Highlight selected analysis markers (Green, larger size)
                    GL.glPointSize(10.0) # Larger size for analysis selection
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
                    GL.glPointSize(5.0) # Reset point size

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
                                        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord(char)) # Smaller font
                                    except Exception: pass
                                    
                            GL.glPopAttrib()
                            GL.glPopMatrix()

                    # -- Distance / Angle (2 or 3 Markers Selected) --
                    elif num_valid_analysis >= 2:
                        # Get positions in the selection order
                        analysis_positions_ordered = [analysis_positions_raw[m] for m in valid_analysis_markers]

                        # Draw thicker lines between selected markers
                        GL.glLineWidth(3.0) # Thicker line for analysis
                        GL.glColor3f(0.0, 1.0, 0.0) # Green color for analysis lines
                        if len(analysis_positions_ordered) == 2:
                            GL.glBegin(GL.GL_LINES)
                            GL.glVertex3fv(analysis_positions_ordered[0])
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            GL.glEnd()
                        elif len(analysis_positions_ordered) == 3:
                            GL.glBegin(GL.GL_LINES)
                            # Draw lines based on selection order: 0->1 and 1->2
                            GL.glVertex3fv(analysis_positions_ordered[0])
                            GL.glVertex3fv(analysis_positions_ordered[1]) 
                            GL.glVertex3fv(analysis_positions_ordered[1])
                            GL.glVertex3fv(analysis_positions_ordered[2]) 
                            GL.glEnd()
                        GL.glLineWidth(1.0) # Reset line width

                        # Calculate and prepare text for display
                        analysis_text = ""
                        text_position = None
                        
                        if len(analysis_positions_ordered) == 2:
                            distance = calculate_distance(analysis_positions_ordered[0], analysis_positions_ordered[1])
                            if distance is not None:
                                analysis_text = f"{distance:.3f} m"
                                mid_point = (analysis_positions_ordered[0] + analysis_positions_ordered[1]) / 2
                                text_position = [mid_point[0], mid_point[1] + 0.02, mid_point[2]]
                                
                        elif len(analysis_positions_ordered) == 3:
                            # Angle at the vertex (second selected marker)
                            angle = calculate_angle(analysis_positions_ordered[0], analysis_positions_ordered[1], analysis_positions_ordered[2])
                            if angle is not None:
                                analysis_text = f"{angle:.1f}\xb0" # Degree symbol
                                text_position = [analysis_positions_ordered[1][0], analysis_positions_ordered[1][1] + 0.03, analysis_positions_ordered[1][2]]
                                
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
                                    
                                    # Set color to semi-transparent green (match highlight color)
                                    GL.glColor4f(1.0, 1.0, 0.0, 0.3) # yellow with 30% alpha
                                    
                                    GL.glBegin(GL.GL_TRIANGLE_FAN)
                                    # Center vertex of the fan is the angle vertex
                                    GL.glVertex3fv(analysis_positions_ordered[1]) 
                                    # Outer vertices are the points along the arc
                                    for point in arc_points:
                                        GL.glVertex3fv(point)
                                    GL.glEnd()
                                    
                                    # Optional: Draw outline of the arc (thinner line)
                                    GL.glLineWidth(1.0)
                                    GL.glColor4f(1.0, 1.0, 0.0, 0.7) # Slightly more opaque outline
                                    GL.glBegin(GL.GL_LINE_STRIP)
                                    for point in arc_points:
                                        GL.glVertex3fv(point)
                                    GL.glEnd()

                                    GL.glDepthMask(GL.GL_TRUE) # Re-enable depth writing
                                    # GL.glDisable(GL.GL_BLEND) # Optionally disable blend if not needed afterwards
                                    
                        # Render the analysis text if available
                        if analysis_text and text_position:
                            GL.glPushMatrix()
                            GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_DEPTH_BUFFER_BIT)
                            GL.glDisable(GL.GL_DEPTH_TEST) # Draw text on top
                            GL.glColor3f(0.0, 1.0, 0.0) # Green text
                            GL.glRasterPos3f(text_position[0], text_position[1], text_position[2])
                            for char in analysis_text:
                                try:
                                    GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord(char))
                                except Exception as glut_error:
                                    pass 
                            GL.glPopAttrib()
                            GL.glPopMatrix()
                            
                except Exception as analysis_error:
                     logger.error(f"Error during analysis visualization: {analysis_error}", exc_info=True)
            # --- Analysis Mode Visualization End ---

            # Trajectory rendering
            if self.current_marker is not None and hasattr(self, 'show_trajectory') and self.show_trajectory:
                trajectory_points = []
                
                for i in range(0, self.frame_idx + 1):
                    try:
                        x = self.data.loc[i, f'{self.current_marker}_X']
                        y = self.data.loc[i, f'{self.current_marker}_Y']
                        z = self.data.loc[i, f'{self.current_marker}_Z']
                        
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
                                GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord(c))
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
                                        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord(c))
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
            
            # Swap buffers (refresh screen)
            self.tkSwapBuffers()
        
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
        self.show_trajectory = show
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
        self.show_marker_names = show
        self.redraw()
        
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
        
        # Perform picking
        if self.data is not None and len(self.marker_names) > 0:
            self.pick_marker(event.x, event.y)

    def on_mouse_release(self, event):
        """Called when the left mouse button is released"""
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
            self.redraw()

    def on_right_mouse_press(self, event):
        """Handle right mouse button press event (start view translation or pattern selection mode)"""
        if not self.pattern_selection_mode: # Start view translation only when not in pattern selection mode
            self.dragging = True
            self.last_x = event.x
            self.last_y = event.y
            
    def on_right_mouse_release(self, event):
        """Handle right mouse button release event (end view translation or select pattern marker)"""
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
        
        self.redraw()

    def on_scroll(self, event):
        """Called when scrolling the mouse wheel (zoom)"""
        # On Windows: event.delta, other platforms may need different approaches
        self.zoom += event.delta * 0.001
        self.redraw()

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
            
            for idx, marker in enumerate(self.marker_names):
                try:
                    # Get marker coordinates for the current frame
                    x_val = self.data.loc[self.frame_idx, f'{marker}_X']
                    y_val = self.data.loc[self.frame_idx, f'{marker}_Y']
                    z_val = self.data.loc[self.frame_idx, f'{marker}_Z']
                    
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
                    
                except KeyError:
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
        """Handle widget resize/move/visibility changes."""
        # Check if GL is initialized before attempting to redraw
        if self.gl_initialized:
             self.redraw()

    def set_analysis_state(self, is_active: bool, selected_markers: list):
        """Sets the analysis mode state and the list of selected markers."""
        self.analysis_mode_active = is_active
        # Make a copy to avoid direct modification issues if parent list changes elsewhere
        self.analysis_selection = list(selected_markers) 
        logger.debug(f"Renderer analysis state updated: Active={self.analysis_mode_active}, Selection={self.analysis_selection}")
        # No redraw here, redraw will be triggered by the caller (TRCViewer) if needed
        # Or redraw might be called automatically if other state changes concurrently