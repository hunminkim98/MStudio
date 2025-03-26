import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3D

def create_plot(self):
    """
    Creates the main 3D plot for displaying marker data.
    This function was extracted from the main class to improve code organization.
    """
    self.fig = plt.Figure(figsize=(10, 10), facecolor='black')  # Changed to square figure
    self.ax = self.fig.add_subplot(111, projection='3d')
    self.ax.set_position([0.1, 0.1, 0.8, 0.8])  # Add proper spacing around plot
    
    _setup_plot_style(self)
    _draw_static_elements(self)
    _initialize_dynamic_elements(self)

    if hasattr(self, 'canvas') and self.canvas:
        self.canvas.get_tk_widget().destroy()
        self.canvas = None

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill='both', expand=True)

    self.canvas.mpl_connect('scroll_event', self.mouse_handler.on_scroll)
    self.canvas.mpl_connect('pick_event', self.mouse_handler.on_pick)
    self.canvas.mpl_connect('button_press_event', self.mouse_handler.on_mouse_press)
    self.canvas.mpl_connect('button_release_event', self.mouse_handler.on_mouse_release)
    self.canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_mouse_move)

    if self.data is None:
        # Set equal aspect ratio and limits
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_box_aspect([1,1,1])  # Force equal aspect ratio
    self.canvas.draw()

def _setup_plot_style(self):
    """
    Sets up the style of the plot (colors, margins, etc.).
    """
    self.ax.set_facecolor('black')
    self.fig.patch.set_facecolor('black')

    # 3D axis spacing removal
    # self.ax.dist = 11  # camera distance adjustment
    # self.fig.tight_layout(pad=10)  # minimum padding
    self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjusted margins for better aspect ratio
    
    for pane in [self.ax.xaxis.set_pane_color,
                 self.ax.yaxis.set_pane_color,
                 self.ax.zaxis.set_pane_color]:
        pane((0, 0, 0, 1))

    for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
        axis.label.set_color('white')
        axis.set_tick_params(colors='white')

    self.ax.set_xticks([])
    self.ax.set_yticks([])
    self.ax.set_zticks([])
    self.ax.set_xlabel('')
    self.ax.set_ylabel('')
    self.ax.set_zlabel('')

def _draw_static_elements(self):
    """
    Draw static elements like the ground grid based on the coordinate system.
    """
    grid_size = 2
    grid_divisions = 20
    x = np.linspace(-grid_size, grid_size, grid_divisions)
    y = np.linspace(-grid_size, grid_size, grid_divisions)

    # Clear existing grid lines (if any)
    if hasattr(self, 'grid_lines'):
        for line in self.grid_lines:
            line.remove()
    self.grid_lines = []

    # Draw grid based on coordinate system
    # Z-up: Grid on X-Y plane at Z=0
    for i in range(grid_divisions):
        line1, = self.ax.plot(x, [y[i]] * grid_divisions, [0] * grid_divisions, 'gray', alpha=0.2)
        line2, = self.ax.plot([x[i]] * grid_divisions, y, [0] * grid_divisions, 'gray', alpha=0.2)
        self.grid_lines.extend([line1, line2])

def _initialize_dynamic_elements(self):
    """
    Initializes dynamic elements of the plot (points, lines, etc.).
    """
    _update_coordinate_axes(self)

    if hasattr(self, 'markers_scatter'):
        self.markers_scatter.remove()
    if hasattr(self, 'selected_marker_scatter'):
        self.selected_marker_scatter.remove()

    self.markers_scatter = self.ax.scatter([], [], [], c='white', s=5, picker=5)
    self.selected_marker_scatter = self.ax.scatter([], [], [], c='yellow', s=15)

    if hasattr(self, 'skeleton_lines'):
        for line in self.skeleton_lines:
            line.remove()
    self.skeleton_lines = []

    if hasattr(self, 'skeleton_pairs') and self.skeleton_pairs:
        for _ in self.skeleton_pairs:
            line = Line3D([], [], [], color='gray', alpha=0.9)
            self.ax.add_line(line)
            self.skeleton_lines.append(line)

    if hasattr(self, 'marker_labels'):
        for label in self.marker_labels:
            label.remove()
    self.marker_labels = []

def _update_coordinate_axes(self):
    """
    Update coordinate axes and labels based on the coordinate system.
    """
    # axis initialization
    if hasattr(self, 'coordinate_axes'):
        for line in self.coordinate_axes:
            line.remove()
    self.coordinate_axes = []

    if hasattr(self, 'axis_labels'):
        for label in self.axis_labels:
            label.remove()
    self.axis_labels = []

    # axis settings
    origin = np.zeros(3)
    axis_length = 0.4
    
    # axis colors
    x_color = 'red'
    y_color = 'yellow'
    z_color = 'blue'
    
    if self.is_z_up:
        # draw main axes for Z-up coordinate system
        # X-axis (red)
        line_x = self.ax.plot([origin[0], origin[0] + axis_length], 
                    [origin[1], origin[1]], 
                    [origin[2], origin[2]], 
                    color=x_color, alpha=0.8, linewidth=2)[0]
        
        # Y-axis (yellow)
        line_y = self.ax.plot([origin[0], origin[0]], 
                    [origin[1], origin[1] + axis_length], 
                    [origin[2], origin[2]], 
                    color=y_color, alpha=0.8, linewidth=2)[0]
        
        # Z-axis (blue)
        line_z = self.ax.plot([origin[0], origin[0]], 
                    [origin[1], origin[1]], 
                    [origin[2], origin[2] + axis_length], 
                    color=z_color, alpha=0.8, linewidth=2)[0]

        # label position
        label_x = self.ax.text(axis_length + 0.1, 0, 0, 'X', color=x_color, fontsize=12)
        label_y = self.ax.text(0, axis_length + 0.1, 0, 'Y', color=y_color, fontsize=12)
        label_z = self.ax.text(0, 0, axis_length + 0.1, 'Z', color=z_color, fontsize=12)
    else:
        # draw main axes for Y-up coordinate system (right-hand rule)
        # X-axis (red)
        line_x = self.ax.plot([origin[0], origin[0] + axis_length], 
                    [origin[2], origin[2]], 
                    [origin[1], origin[1]], 
                    color=x_color, alpha=0.8, linewidth=2)[0]
        
        # Z-axis (blue) - change direction
        line_z = self.ax.plot([origin[0], origin[0]], 
                    [origin[2], origin[2] - axis_length], 
                    [origin[1], origin[1]], 
                    color=z_color, alpha=0.8, linewidth=2)[0]
        
        # Y-axis (yellow)
        line_y = self.ax.plot([origin[0], origin[0]], 
                    [origin[2], origin[2]], 
                    [origin[1], origin[1] + axis_length], 
                    color=y_color, alpha=0.8, linewidth=2)[0]

        # label position
        label_x = self.ax.text(axis_length + 0.1, 0, 0, 'X', color=x_color, fontsize=12)
        label_z = self.ax.text(0, -axis_length - 0.1, 0, 'Z', color=z_color, fontsize=12)
        label_y = self.ax.text(0, 0, axis_length + 0.1, 'Y', color=y_color, fontsize=12)

    # save axes and labels
    self.coordinate_axes = [line_x, line_y, line_z]
    self.axis_labels = [label_x, label_y, label_z]
