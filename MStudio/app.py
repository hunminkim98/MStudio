import numpy as np
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib
import os

from MStudio.gui.TRCviewerWidgets import create_widgets
from MStudio.gui.markerPlot import show_marker_plot
from MStudio.gui.plotCreator import create_plot
from MStudio.gui.filterUI import on_filter_type_change, build_filter_parameter_widgets
from MStudio.gui.markerPlotUI import build_marker_plot_buttons

from MStudio.utils.dataLoader import open_file
from MStudio.utils.dataSaver import save_as
from MStudio.utils.skeletons import *
from MStudio.utils.viewToggles import (
    toggle_marker_names,
    toggle_trajectory,
    toggle_animation,
)
from MStudio.utils.viewReset import reset_main_view, reset_graph_view
from MStudio.utils.dataProcessor import *
from MStudio.utils.mouseHandler import MouseHandler

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


# Interactive mode on
plt.ion()
# Conditionally set backend based on DISPLAY environment variable
if os.environ.get('DISPLAY'):
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg') # Use non-interactive backend for headless environments (CI)


# General TODO:
# 1. Current TRCViewer is too long and complex. It needs to be refactored.
# 2. The code is not documented well and should be english.
# 3. Add information about the author and the version of the software.
# 4. project.toml file

class TRCViewer(ctk.CTk): 
    def __init__(self):
        super().__init__()
        self.title("MStudio")
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")

        # coordinate system setting variable
        self.coordinate_system = "y-up"  # Default is y-up because Pose2Sim and OpenSim use y-up coordinate system

        # --- Data Related Attributes ---
        self.marker_names = []
        self.data = None
        self.original_data = None
        self.num_frames = 0
        self.frame_idx = 0
        self.outliers = {}

        # --- Main 3D Plot Attributes ---
        self.canvas = None
        self.gl_renderer = None # OpenGL renderer related attributes
        self.is_z_up = False   # coordinate system state (True = Z-up, False = Y-up)
        self.view_limits = None
        self.pan_enabled = False
        self.last_mouse_pos = None
        self.show_trajectory = False 
        self.trajectory_length = 10
        self.trajectory_line = None

        # --- Marker Graph Plot Attributes ---
        self.marker_last_pos = None
        self.marker_pan_enabled = False
        self.marker_canvas = None
        self.marker_axes = []
        self.marker_lines = []
        self.selection_in_progress = False

        # --- Filter Attributes ---
        self.filter_type_var = ctk.StringVar(value='butterworth')

        # --- Interpolation Attributes ---
        self.interp_methods = [
            'linear',
            'polynomial',
            'spline',
            'nearest',
            'zero',
            'slinear',
            'quadratic',
            'cubic',
            'pattern-based' # 11/05 added pattern-based interpolation method
        ]
        self.interp_method_var = ctk.StringVar(value='linear')
        self.order_var = ctk.StringVar(value='3')

        # --- Pattern-Based Interpolation Attributes ---
        self.pattern_markers = set()
        self._selected_markers_list = None

        # --- Skeleton Model Attributes ---
        self.available_models = {
            'No skeleton': None,
            'BODY_25B': BODY_25B,
            'BODY_25': BODY_25,
            'BODY_135': BODY_135,
            'BLAZEPOSE': BLAZEPOSE,
            'HALPE_26': HALPE_26,
            'HALPE_68': HALPE_68,
            'HALPE_136': HALPE_136,
            'COCO_133': COCO_133,
            'COCO': COCO,
            'MPII': MPII,
            'COCO_17': COCO_17
        }
        self.current_model = None
        self.skeleton_pairs = []

        # --- Animation Attributes ---
        self.is_playing = False
        self.playback_speed = 1.0
        self.animation_job = None
        self.fps_var = ctk.StringVar(value="60")

        # --- Timeline Attributes ---
        self.current_frame_line = None

        # --- Mouse Handling ---
        self.mouse_handler = MouseHandler(self)

        # --- Editing State ---
        self.edit_window = None
        self.is_editing = False # Add editing state flag
        self.edit_controls_frame = None # Placeholder for edit controls frame

        # --- Analysis Mode ---
        self.is_analysis_mode = False
        self.analysis_markers = [] # List to store selected markers for analysis

        # --- Key Bindings ---
        self.bind('<space>', lambda e: self.toggle_animation())
        self.bind('<Return>', lambda e: self.toggle_animation())
        self.bind('<Escape>', lambda e: self.stop_animation())
        self.bind('<Left>', lambda e: self.prev_frame())
        self.bind('<Right>', lambda e: self.next_frame())

        # --- Widget and Plot Creation ---
        self.create_widgets()
        self.create_plot()
        self.update_plot()


    #########################################
    ############ File managers ##############
    #########################################

    def open_file(self):
        open_file(self)


    def save_as(self):
        save_as(self)

    
    #########################################
    ############ View managers ##############
    #########################################

    def reset_main_view(self):
        reset_main_view(self)


    def reset_graph_view(self):
        reset_graph_view(self)


    def calculate_data_limits(self):
        try:
            x_coords = [col for col in self.data.columns if col.endswith('_X')]
            y_coords = [col for col in self.data.columns if col.endswith('_Y')]
            z_coords = [col for col in self.data.columns if col.endswith('_Z')]

            x_min = self.data[x_coords].min().min()
            x_max = self.data[x_coords].max().max()
            y_min = self.data[y_coords].min().min()
            y_max = self.data[y_coords].max().max()
            z_min = self.data[z_coords].min().min()
            z_max = self.data[z_coords].max().max()

            margin = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min

            self.data_limits = {
                'x': (x_min - x_range * margin, x_max + x_range * margin),
                'y': (y_min - y_range * margin, y_max + y_range * margin),
                'z': (z_min - z_range * margin, z_max + z_range * margin)
            }

            self.initial_limits = self.data_limits.copy()

        except Exception as e:
            logger.error("Error calculating data limits: %s", e, exc_info=True)
            self.data_limits = None
            self.initial_limits = None

    
    # ---------- Right panel resize ----------
    def start_resize(self, event):
        self.sizer_dragging = True
        self.initial_sizer_x = event.x_root
        self.initial_panel_width = self.right_panel.winfo_width()


    def do_resize(self, event):
        if self.sizer_dragging:
            dx = event.x_root - self.initial_sizer_x
            new_width = max(200, min(self.initial_panel_width - dx, self.winfo_width() - 200))
            self.right_panel.configure(width=new_width)


    def stop_resize(self, event):
        self.sizer_dragging = False


    #########################################
    ###### Show/Hide Name of markers ########
    #########################################

    def toggle_marker_names(self):
        toggle_marker_names(self)


    #########################################
    #### Show/Hide trajectory of markers ####
    #########################################

    # TODO for show/hide trajectory of markers:
    # 1. Users can choose the color of the trajectory
    # 2. Users can choose the width of the trajectory
    # 3. Users can choose the length of the trajectory

    def toggle_trajectory(self):
        toggle_trajectory(self)

        
    #########################################
    ####### Skeleton model manager ##########
    #########################################

    def on_model_change(self, choice):
        try:
            # Save the current frame
            current_frame = self.frame_idx

            # Update the model
            self.current_model = self.available_models[choice]

            # Update skeleton settings
            if self.current_model is None:
                self.skeleton_pairs = []
                self.show_skeleton = False
            else:
                self.show_skeleton = True
                self.update_skeleton_pairs()

            # Deliver skeleton pairs and show skeleton to OpenGL renderer
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_skeleton_pairs(self.skeleton_pairs)
                self.gl_renderer.set_show_skeleton(self.show_skeleton)

            # Re-detect outliers with new skeleton pairs
            self.detect_outliers()
            
            # Deliver outliers to OpenGL renderer
            if hasattr(self, 'gl_renderer') and hasattr(self, 'outliers'):
                self.gl_renderer.set_outliers(self.outliers)

            # Update the plot with the current frame data
            self.update_plot()
            self.update_frame(current_frame)

            # If a marker is currently selected, update its plot
            if hasattr(self, 'current_marker') and self.current_marker:
                self.show_marker_plot(self.current_marker)

        except Exception as e:
            logger.error("Error in on_model_change: %s", e, exc_info=True)


    def update_skeleton_pairs(self):
        """update skeleton pairs"""
        self.skeleton_pairs = []
        if self.current_model is not None:
            for node in self.current_model.descendants:
                if node.parent:
                    parent_name = node.parent.name
                    node_name = node.name
                    
                    # check if marker names are in the data
                    if (f"{parent_name}_X" in self.data.columns and 
                        f"{node_name}_X" in self.data.columns):
                        self.skeleton_pairs.append((parent_name, node_name))


    #########################################
    ########## Outlier detection ############
    #########################################

    # TODO for outlier detection:
    # 1. Find a better way to detect outliers
    # 2. Add a threshold for outlier detection

    def detect_outliers(self):
        if not self.skeleton_pairs:
            return

        self.outliers = {marker: np.zeros(len(self.data), dtype=bool) for marker in self.marker_names}

        for frame in range(len(self.data)):
            for pair in self.skeleton_pairs:
                try:
                    p1 = np.array([
                        self.data.loc[frame, f'{pair[0]}_X'],
                        self.data.loc[frame, f'{pair[0]}_Y'],
                        self.data.loc[frame, f'{pair[0]}_Z']
                    ])
                    p2 = np.array([
                        self.data.loc[frame, f'{pair[1]}_X'],
                        self.data.loc[frame, f'{pair[1]}_Y'],
                        self.data.loc[frame, f'{pair[1]}_Z']
                    ])

                    current_length = np.linalg.norm(p2 - p1)

                    if frame > 0:
                        p1_prev = np.array([
                            self.data.loc[frame-1, f'{pair[0]}_X'],
                            self.data.loc[frame-1, f'{pair[0]}_Y'],
                            self.data.loc[frame-1, f'{pair[0]}_Z']
                        ])
                        p2_prev = np.array([
                            self.data.loc[frame-1, f'{pair[1]}_X'],
                            self.data.loc[frame-1, f'{pair[1]}_Y'],
                            self.data.loc[frame-1, f'{pair[1]}_Z']
                        ])
                        prev_length = np.linalg.norm(p2_prev - p1_prev)

                        if abs(current_length - prev_length) / prev_length > 0.3:
                            self.outliers[pair[0]][frame] = True
                            self.outliers[pair[1]][frame] = True

                except KeyError:
                    continue
                    
        # Deliver outliers to OpenGL renderer
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_outliers(self.outliers)


    #########################################
    ############ Mouse handling #############
    #########################################

    def connect_mouse_events(self):
        # OpenGL renderer handles mouse events internally

        # Marker canvas (matplotlib) still needs to be connected
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
            self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
            self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
            self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)


    def disconnect_mouse_events(self):
        """disconnect mouse events"""
        # Marker canvas (matplotlib) still needs to be connected
        if hasattr(self, 'marker_canvas') and self.marker_canvas and hasattr(self.marker_canvas, 'callbacks') and self.marker_canvas.callbacks:
             # Iterate through all event types and their registered callback IDs
             all_cids = []
             for event_type in list(self.marker_canvas.callbacks.callbacks.keys()): # Use list() for safe iteration
                 all_cids.extend(list(self.marker_canvas.callbacks.callbacks[event_type].keys()))

             # Disconnect each callback ID
             for cid in all_cids:
                 try:
                     self.marker_canvas.mpl_disconnect(cid)
                 except Exception as e:
                     # Log potential issues if a cid is invalid
                     logger.error("Could not disconnect cid %d: %s", cid, e)


    #########################################
    ########## Marker selection #############
    #########################################

    def on_marker_selected(self, marker_name):
        """Handle marker selection event"""
        
        # If the clicked marker is already selected, deselect it
        if marker_name == self.current_marker:
            marker_name = None

        # Save current view state
        current_view_state = None
        if hasattr(self, 'gl_renderer'):
            current_view_state = {
                'rot_x': self.gl_renderer.rot_x,
                'rot_y': self.gl_renderer.rot_y,
                'zoom': self.gl_renderer.zoom,
                'trans_x': self.gl_renderer.trans_x,
                'trans_y': self.gl_renderer.trans_y
            }
        
        self.current_marker = marker_name
        
        # Update selection state in markers list
        if hasattr(self, 'markers_list') and self.markers_list:
            try:
                # Clear selection in markers list
                self.markers_list.selection_clear(0, "end")
                
                # Select marker in markers list if it is selected
                if marker_name is not None:
                    # Find index of selected marker
                    for i, item in enumerate(self.markers_list.get(0, "end")):
                        if item == marker_name:
                            self.markers_list.selection_set(i)  # Set selection
                            self.markers_list.see(i)  # Scroll to show
                            break
            except Exception as e:
                logger.error("Error updating markers list: %s", e, exc_info=True)
        
        # Display marker plot (if marker is selected) or hide if deselected
        if marker_name is not None and hasattr(self, 'show_marker_plot'):
            try:
                self.show_marker_plot(marker_name)
            except Exception as e:
                logger.error("Error displaying marker plot: %s", e, exc_info=True)
        elif marker_name is None:
            # Hide graph frame, sizer, and right panel if they exist and are visible
            if hasattr(self, 'graph_frame') and self.graph_frame.winfo_ismapped():
                self.graph_frame.pack_forget()
            if hasattr(self, 'sizer') and self.sizer.winfo_ismapped():
                self.sizer.pack_forget()
            if hasattr(self, 'right_panel') and self.right_panel.winfo_ismapped():
                self.right_panel.pack_forget()
            
            # Disconnect mouse events from the (now hidden) marker canvas
            self.disconnect_mouse_events()
            # Clear the reference to the marker canvas
            if hasattr(self, 'marker_canvas'):
                del self.marker_canvas
        
        # Deliver selected marker information to OpenGL renderer
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_current_marker(marker_name)

        # Restore view state *before* final plot update
        if hasattr(self, 'gl_renderer') and current_view_state:
            self.gl_renderer.rot_x = current_view_state['rot_x']
            self.gl_renderer.rot_y = current_view_state['rot_y']
            self.gl_renderer.zoom = current_view_state['zoom']
            self.gl_renderer.trans_x = current_view_state['trans_x']
            self.gl_renderer.trans_y = current_view_state['trans_y']
            # No need for extra redraw here, update_plot will handle it

        # Update screen (now rendering with restored view state)
        self.update_plot()


    def show_marker_plot(self, marker_name):
        show_marker_plot(self, marker_name)
        self.update_timeline()


    def update_selected_markers_list(self):
        """Update selected markers list"""
        try:
            # check if pattern selection window exists and is valid
            if (hasattr(self, 'pattern_window') and 
                self.pattern_window.winfo_exists() and 
                self._selected_markers_list and 
                self._selected_markers_list.winfo_exists()):
                
                self._selected_markers_list.configure(state='normal')
                self._selected_markers_list.delete('1.0', 'end')
                for marker in sorted(self.pattern_markers):
                    self._selected_markers_list.insert('end', f"• {marker}\n")
                self._selected_markers_list.configure(state='disabled')
        except Exception as e:
            logger.error("Error updating markers list: %s", e, exc_info=True)
            # initialize related variables if error occurs
            if hasattr(self, 'pattern_window'):
                delattr(self, 'pattern_window')
            self._selected_markers_list = None


    #########################################
    ############## Updaters #################
    #########################################

    def update_timeline(self):
        if self.data is None:
            return
            
        self.timeline_ax.clear()
        frames = np.arange(self.num_frames)
        fps = float(self.fps_var.get())
        times = frames / fps
        
        # add horizontal baseline (y=0)
        self.timeline_ax.axhline(y=0, color='white', alpha=0.3, linewidth=1)
        
        display_mode = self.timeline_display_var.get()
        light_yellow = '#FFEB3B'
        
        if display_mode == "time":
            # major ticks every 10 seconds
            major_time_ticks = np.arange(0, times[-1] + 10, 10)
            for time in major_time_ticks:
                if time <= times[-1]:
                    frame = int(time * fps)
                    self.timeline_ax.axvline(frame, color='white', alpha=0.3, linewidth=1)
                    self.timeline_ax.text(frame, -0.7, f"{time:.0f}s", 
                                        color='white', fontsize=8, 
                                        horizontalalignment='center',
                                        verticalalignment='top')
            
            # minor ticks every 1 second
            minor_time_ticks = np.arange(0, times[-1] + 1, 1)
            for time in minor_time_ticks:
                if time <= times[-1] and time % 10 != 0:  # not overlap with 10-second ticks
                    frame = int(time * fps)
                    self.timeline_ax.axvline(frame, color='white', alpha=0.15, linewidth=0.5)
                    self.timeline_ax.text(frame, -0.7, f"{time:.0f}s", 
                                        color='white', fontsize=6, alpha=0.5,
                                        horizontalalignment='center',
                                        verticalalignment='top')
            
            current_time = self.frame_idx / fps
            current_display = f"{current_time:.2f}s"
        else:  # frame mode
            # major ticks every 100 frames
            major_frame_ticks = np.arange(0, self.num_frames, 100)
            for frame in major_frame_ticks:
                self.timeline_ax.axvline(frame, color='white', alpha=0.3, linewidth=1)
                self.timeline_ax.text(frame, -0.7, f"{frame}", 
                                    color='white', fontsize=6, alpha=0.5,
                                    horizontalalignment='center',
                                    verticalalignment='top')
            
            current_display = f"{self.frame_idx}"
        
        # current frame display (light yellow line)
        self.timeline_ax.axvline(self.frame_idx, color=light_yellow, alpha=0.8, linewidth=1.5)
        
        # update label
        self.current_info_label.configure(text=current_display)
        
        # timeline settings
        self.timeline_ax.set_xlim(0, self.num_frames - 1)
        self.timeline_ax.set_ylim(-1, 1)
        
        # hide y-axis
        self.timeline_ax.set_yticks([])
        
        # border style
        self.timeline_ax.spines['top'].set_visible(False)
        self.timeline_ax.spines['right'].set_visible(False)
        self.timeline_ax.spines['left'].set_visible(False)
        self.timeline_ax.spines['bottom'].set_color('white')
        self.timeline_ax.spines['bottom'].set_alpha(0.3)
        self.timeline_ax.spines['bottom'].set_color('white')
        self.timeline_ax.spines['bottom'].set_alpha(0.3)
        
        # hide x-axis ticks (we draw them manually)
        self.timeline_ax.set_xticks([])
        # adjust figure margins (to avoid text clipping)
        self.timeline_fig.subplots_adjust(bottom=0.2)
        
        self.timeline_canvas.draw_idle()


    def update_frame_from_timeline(self, x_pos):
        if x_pos is not None and self.data is not None:
            frame = int(max(0, min(x_pos, self.num_frames - 1)))
            self.frame_idx = frame
            self._update_display_after_frame_change()

            # update vertical line if marker graph is displayed
            self._update_marker_plot_vertical_line_data()
            # Check if marker_canvas exists before drawing
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()


    def update_plot(self):
        """
        Update method for 3D marker visualization.
        Previously used the external plotUpdater.py module,
        but now directly calls the OpenGL renderer.
        """
        if hasattr(self, 'gl_renderer'):
            # Deliver data
            if self.data is not None:
                # Check coordinate system
                coordinate_system = "z-up" if self.is_z_up else "y-up"
                
                # Deliver outliers
                if hasattr(self, 'outliers') and self.outliers:
                    self.gl_renderer.set_outliers(self.outliers)
                
                # Deliver current frame data
                try:
                    self.gl_renderer.set_frame_data(
                        self.data, 
                        self.frame_idx, 
                        self.marker_names,
                        getattr(self, 'current_marker', None),
                        getattr(self, 'show_names', False),
                        getattr(self, 'show_trajectory', False),
                        coordinate_system,
                        self.skeleton_pairs if hasattr(self, 'skeleton_pairs') else None
                    )
                except Exception as e:
                    logger.error("Error setting OpenGL data: %s", e, exc_info=True)
            
            # Update OpenGL renderer screen
            self.gl_renderer.update_plot()


    def _update_marker_plot_vertical_line_data(self):
        """Helper function to update the x-data of the vertical lines on the marker plot."""
        if hasattr(self, 'marker_lines') and self.marker_lines:
            for line in self.marker_lines:
                line.set_xdata([self.frame_idx, self.frame_idx])


    def _update_display_after_frame_change(self):
        """Helper function to update the main plot and the timeline after a frame change."""
        self.update_plot()
        self.update_timeline()


    def update_frame(self, value):
        if self.data is not None:
            self.frame_idx = int(float(value))
            self._update_display_after_frame_change()

            # update vertical line if marker graph is displayed
            self._update_marker_plot_vertical_line_data()
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()
        
        # Update marker graph vertical line if it exists
        self._update_marker_plot_vertical_line_data()


    def update_fps_label(self):
        fps = self.fps_var.get()
        if hasattr(self, 'fps_label'):
            self.fps_label.configure(text=f"FPS: {fps}")


    def _update_marker_plot_vertical_line_data(self):
        """Updates the vertical line data in the marker plot."""
        if self.data is None or not hasattr(self, 'marker_canvas') or self.marker_canvas is None:
            return

        if hasattr(self, 'marker_lines') and self.marker_lines:
            for line in self.marker_lines:
                line.set_xdata([self.frame_idx, self.frame_idx])


    #########################################
    ############## Creators #################
    #########################################

    def create_widgets(self):
        create_widgets(self)


    def create_plot(self):
        create_plot(self)


    #########################################
    ############## Clearers #################
    #########################################

    def clear_current_state(self):
        try:
            if hasattr(self, 'graph_frame') and self.graph_frame.winfo_ismapped():
                self.graph_frame.pack_forget()
                for widget in self.graph_frame.winfo_children():
                    widget.destroy()

            if hasattr(self, 'fig'):
                plt.close(self.fig)
                del self.fig
            if hasattr(self, 'marker_plot_fig'):
                plt.close(self.marker_plot_fig)
                del self.marker_plot_fig

            # canvas related processing
            if hasattr(self, 'canvas') and self.canvas:
                try:
                    # OpenGL renderer case - always this case
                    if hasattr(self, 'gl_renderer'):
                        if self.canvas == self.gl_renderer:
                            if hasattr(self.gl_renderer, 'pack_forget'):
                                self.gl_renderer.pack_forget()
                            if hasattr(self, 'gl_renderer'):
                                del self.gl_renderer
                except Exception as e:
                    logger.error("Error clearing canvas: %s", e, exc_info=True)
                
                self.canvas = None

            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                try:
                    if hasattr(self.marker_canvas, 'get_tk_widget'):
                        self.marker_canvas.get_tk_widget().destroy()
                except Exception as e:
                    logger.error("Error clearing marker canvas: %s", e, exc_info=True)
                
                if hasattr(self, 'marker_canvas'):
                    del self.marker_canvas
                
                self.marker_canvas = None

            if hasattr(self, 'ax'):
                del self.ax
            if hasattr(self, 'marker_axes'):
                del self.marker_axes

            self.data = None
            self.original_data = None
            self.marker_names = []
            self.num_frames = 0
            self.frame_idx = 0
            self.outliers = {}
            self.current_marker = None
            self.marker_axes = []
            self.marker_lines = []

            self.view_limits = None
            self.data_limits = None
            self.initial_limits = None

            self.selection_data = {
                'start': None,
                'end': None,
                'rects': [],
                'current_ax': None,
                'rect': None
            }

            # frame_slider related code
            self.title_label.configure(text="")
            self.show_names = False
            self.show_skeleton = True
            self.current_file = None

            # timeline initialization
            if hasattr(self, 'timeline_ax'):
                self.timeline_ax.clear()
                self.timeline_canvas.draw_idle()

        except Exception as e:
            logger.error("Error clearing state: %s", e, exc_info=True)


    def clear_selection(self):
        if 'rects' in self.selection_data and self.selection_data['rects']:
            for rect in self.selection_data['rects']:
                rect.remove()
            self.selection_data['rects'] = []
        if hasattr(self, 'marker_canvas'):
            self.marker_canvas.draw_idle()
        self.selection_in_progress = False


    def clear_pattern_selection(self):
        """Initialize pattern markers"""
        self.pattern_markers.clear()
        self.update_selected_markers_list()
        self.update_plot()


    #########################################
    ########## Playing Controllers ##########
    #########################################

    def toggle_animation(self):
        toggle_animation(self)


    def prev_frame(self):
        """Move to the previous frame when left arrow key is pressed."""
        if self.data is not None and self.frame_idx > 0:
            self.frame_idx -= 1
            self._update_display_after_frame_change()
            
            # Update marker graph vertical line if it exists
            self._update_marker_plot_vertical_line_data()
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()
            # self.update_frame_counter()


    def next_frame(self):
        """Move to the next frame when right arrow key is pressed."""
        if self.data is not None and self.frame_idx < self.num_frames - 1:
            self.frame_idx += 1
            self._update_display_after_frame_change()
            
            # Update marker graph vertical line if it exists
            self._update_marker_plot_vertical_line_data()
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()
            # self.update_frame_counter()


    def change_timeline_mode(self, mode):
        """Change timeline mode and update button style"""
        self.timeline_display_var.set(mode)
        
        # highlight selected button
        if mode == "time":
            self.time_btn.configure(fg_color="#444444", text_color="white")
            self.frame_btn.configure(fg_color="transparent", text_color="#888888")
        else:
            self.frame_btn.configure(fg_color="#444444", text_color="white")
            self.time_btn.configure(fg_color="transparent", text_color="#888888")
        
        self.update_timeline()


    # ---------- animation ----------
    def animate(self):
        if self.is_playing:
            if self.frame_idx < self.num_frames - 1:
                self.frame_idx += 1
            else:
                if self.loop_var.get():
                    self.frame_idx = 0
                else:
                    self.stop_animation()
                    return

            self._update_display_after_frame_change()

            # Update marker graph vertical line if it exists (Added)
            self._update_marker_plot_vertical_line_data()

            # remove speed slider related code and use default FPS
            base_fps = float(self.fps_var.get())
            delay = int(1000 / base_fps)
            delay = max(1, delay)

            self.animation_job = self.after(delay, self.animate)


    def play_animation(self):
        self.is_playing = True
        self.play_pause_button.configure(text="⏸")
        self.stop_button.configure(state='normal')
        self.animate()


    def pause_animation(self):
        self.is_playing = False
        self.play_pause_button.configure(text="▶")
        if self.animation_job:
            self.after_cancel(self.animation_job)
            self.animation_job = None


    def stop_animation(self):
        # if playing, stop
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.configure(text="▶")
            if self.animation_job:
                self.after_cancel(self.animation_job)
                self.animation_job = None
        
        # go back to first frame
        self.frame_idx = 0
        self._update_display_after_frame_change()
        # Update marker graph vertical line if it exists (Added)
        self._update_marker_plot_vertical_line_data()
        if hasattr(self, 'marker_canvas'):
            self.marker_canvas.draw() # Use draw() here as it's a single event
        self.stop_button.configure(state='disabled')


    #########################################
    ############### Editors #################
    #########################################

    # TODO for edit mode:
    # 1. Create a new file for edit mode
    def toggle_edit_mode(self):
        """Toggles the editing mode for the marker plot."""
        if not self.current_marker: # Ensure a marker plot is shown
            return

        self.is_editing = not self.is_editing
        # Re-render plot area with different controls based on edit state
        if hasattr(self, 'graph_frame') and self.graph_frame and self.graph_frame.winfo_ismapped():
            # Get the button frame (bottom frame of graph area)
            button_frame = None
            for widget in self.graph_frame.winfo_children():
                if isinstance(widget, ctk.CTkFrame) and not widget.winfo_ismapped():
                    continue
                if widget != self.marker_canvas.get_tk_widget() and isinstance(widget, ctk.CTkFrame):
                    button_frame = widget
                    break
            
            if button_frame:
                # Call our helper to rebuild the buttons with the new mode
                build_marker_plot_buttons(self, button_frame)
                
                # Update pattern selection mode based on interpolation method
                if self.is_editing and self.interp_method_var.get() == 'pattern-based':
                    self.pattern_selection_mode = True
                else:
                    self.pattern_selection_mode = False
                    
                # Force update of the UI
                self.graph_frame.update_idletasks()
        
        # Update the plot to reflect any changes in selection mode
        self.update_plot()
    
    # NOTE: This function should be moved to the other file.
    # The original _build_marker_plot_buttons function (lines 1006-1129) is removed here.
    
    # ---------- Select data ----------
    def highlight_selection(self):
        if self.selection_data.get('start') is None or self.selection_data.get('end') is None:
            return

        start_frame = min(self.selection_data['start'], self.selection_data['end'])
        end_frame = max(self.selection_data['start'], self.selection_data['end'])

        if 'rects' in self.selection_data:
            for rect in self.selection_data['rects']:
                rect.remove()

        self.selection_data['rects'] = []
        for ax in self.marker_axes:
            ylim = ax.get_ylim()
            rect = plt.Rectangle((start_frame, ylim[0]),
                                 end_frame - start_frame,
                                 ylim[1] - ylim[0],
                                 facecolor='yellow',
                                 alpha=0.2)
            self.selection_data['rects'].append(ax.add_patch(rect))
        self.marker_canvas.draw()


    def start_new_selection(self, event):
        self.selection_data = {
            'start': event.xdata,
            'end': event.xdata,
            'rects': [],
            'current_ax': None,
            'rect': None
        }
        self.selection_in_progress = True

        for ax in self.marker_axes:
            ylim = ax.get_ylim()
            rect = plt.Rectangle((event.xdata, ylim[0]),
                                 0,
                                 ylim[1] - ylim[0],
                                 facecolor='yellow',
                                 alpha=0.2)
            self.selection_data['rects'].append(ax.add_patch(rect))
        self.marker_canvas.draw_idle()


    # ---------- Delete selected data ----------
    def delete_selected_data(self):
        if self.selection_data['start'] is None or self.selection_data['end'] is None:
            return

        view_states = []
        for ax in self.marker_axes:
            view_states.append({
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim()
            })

        current_selection = {
            'start': self.selection_data['start'],
            'end': self.selection_data['end']
        }

        start_frame = min(int(self.selection_data['start']), int(self.selection_data['end']))
        end_frame = max(int(self.selection_data['start']), int(self.selection_data['end']))

        for coord in ['X', 'Y', 'Z']:
            col_name = f'{self.current_marker}_{coord}'
            self.data.loc[start_frame:end_frame, col_name] = np.nan

        self.show_marker_plot(self.current_marker)

        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])

        self.update_plot()

        self.selection_data['start'] = current_selection['start']
        self.selection_data['end'] = current_selection['end']
        self.highlight_selection()

        # Update button state *only if* the edit button exists (i.e., not in edit mode)
        # and the widget itself hasn't been destroyed
        if not self.is_editing and hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
            self.edit_button.configure(fg_color="#555555")


    # ---------- Restore original data ----------
    def restore_original_data(self):
        if self.original_data is not None:
            self.data = self.original_data.copy(deep=True)
            self.detect_outliers()
            # Check if a marker plot is currently displayed before trying to update it
            if hasattr(self, 'current_marker') and self.current_marker:
                self.show_marker_plot(self.current_marker)
            self.update_plot()

            # Update button state *only if* the edit button exists (i.e., not in edit mode)
            if hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
                 self.edit_button.configure(fg_color="#3B3B3B") # Reset to default color, not gray

            # Consider exiting edit mode upon restoring?
            # if self.is_editing:
            #     self.toggle_edit_mode()

            # print("Data has been restored to the original state.")
        else:
            messagebox.showinfo("Restore Data", "No original data to restore.")


    # ---------- Filter selected data ----------
    def filter_selected_data(self):
        filter_selected_data(self)


    def on_filter_type_change(self, choice):
        on_filter_type_change(self, choice)


    def _on_filter_type_change_in_panel(self, choice):
        """Updates filter parameter widgets directly in the panel."""
        self._build_filter_param_widgets(choice) # Just call the builder


    def _build_filter_param_widgets(self, filter_type):
        """Builds the specific parameter entry widgets for the selected filter type."""
        # Clear previous widgets first
        widgets_to_destroy = list(self.filter_params_container.winfo_children())
        for widget in widgets_to_destroy:
             widget.destroy()

        # Force Tkinter to process the destruction events immediately
        self.filter_params_container.update_idletasks()

        # Save current parameter values before recreating StringVars
        current_values = {}
        if hasattr(self, 'filter_params') and filter_type in self.filter_params:
            for param, var in self.filter_params[filter_type].items():
                current_values[param] = var.get()
        
        # Recreate StringVar objects for the selected filter type
        if hasattr(self, 'filter_params') and filter_type in self.filter_params:
            for param in self.filter_params[filter_type]:
                # Get current value or use default
                value = current_values.get(param, self.filter_params[filter_type][param].get())
                # Create a new StringVar with the same value
                self.filter_params[filter_type][param] = ctk.StringVar(value=value)

        params_frame = self.filter_params_container # Use the container directly

        # Call the reusable function from filterUI
        if hasattr(self, 'filter_params'):
            build_filter_parameter_widgets(params_frame, filter_type, self.filter_params)
        else:
            logger.error("Error: filter_params attribute not found on TRCViewer.")

        
    # ---------- Interpolate selected data ----------
    def interpolate_selected_data(self):
        interpolate_selected_data(self)


    # NOTE: Currently, this function is not stable.
    def interpolate_with_pattern(self):
        """
        Pattern-based interpolation using reference markers to interpolate target marker
        """
        interpolate_with_pattern(self)


    def on_pattern_selection_confirm(self):
        """Process pattern selection confirmation"""
        on_pattern_selection_confirm(self)


    def _on_interp_method_change_in_panel(self, choice):
        """Updates interpolation UI elements based on selected method."""
        # Enable/disable Order field based on method type
        if choice in ['polynomial', 'spline']:
            self.interp_order_label.configure(state='normal')
            self.interp_order_entry.configure(state='normal')
        else:
            self.interp_order_label.configure(state='disabled')
            self.interp_order_entry.configure(state='disabled')
            
        # Special handling for pattern-based interpolation
        if choice == 'pattern-based':
            # Clear any existing pattern markers on the main app
            self.pattern_markers.clear()
            # Set pattern selection mode on the main app
            self.pattern_selection_mode = True
            # **Update the renderer's mode**
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_pattern_selection_mode(True, self.pattern_markers)
            messagebox.showinfo("Pattern Selection", 
                "Left-click markers in the 3D view to select/deselect them as reference patterns.\n"
                "Selected markers will be shown in red.")
        else:
            # Disable pattern selection mode on the main app
            self.pattern_selection_mode = False
            # **Update the renderer's mode**
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_pattern_selection_mode(False)
            
        # Update main 3D view if needed (redraws with correct marker colors)
        self.update_plot()
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            self.marker_canvas.draw_idle()


    def handle_pattern_marker_selection(self, marker_name):
        """Handles the selection/deselection of a marker for pattern-based interpolation."""
        if not self.pattern_selection_mode:
            return # Should not happen if called correctly, but as a safeguard

        if marker_name in self.pattern_markers:
            self.pattern_markers.remove(marker_name)
            logger.info(f"Removed {marker_name} from pattern markers.")
        else:
            self.pattern_markers.add(marker_name)
            logger.info(f"Added {marker_name} to pattern markers.")

        # Update the UI list showing selected markers
        self.update_selected_markers_list()
        
        # Update the renderer state (important for visual feedback)
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_pattern_selection_mode(True, self.pattern_markers)
            # Trigger redraw in the renderer to show color changes
            self.gl_renderer.redraw() 


    def toggle_analysis_mode(self):
        """Toggles the analysis mode on/off."""
        self.is_analysis_mode = not self.is_analysis_mode
        logger.info(f"Analysis mode {'enabled' if self.is_analysis_mode else 'disabled'}.")

        if not self.is_analysis_mode:
            # Clear analysis markers when exiting the mode
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

        # Visually update the button state (assuming button exists)
        if hasattr(self, 'analysis_button'):
            if self.is_analysis_mode:
                # Indicate active state (e.g., change color, text)
                self.analysis_button.configure(fg_color="#00529B") # Example: Blue color when active
            else:
                # Indicate inactive state (e.g., default color)
                self.analysis_button.configure(fg_color=["#3B3B3B", "#3B3B3B"]) # Example: Default button color


    def handle_analysis_marker_selection(self, marker_name):
        """Handles marker selection/deselection in analysis mode."""
        if not self.is_analysis_mode:
            logger.warning("handle_analysis_marker_selection called when not in analysis mode.")
            return

        if marker_name in self.analysis_markers:
            self.analysis_markers.remove(marker_name)
            logger.info(f"Removed {marker_name} from analysis markers.")
        else:
            if len(self.analysis_markers) < 3:
                # Append marker. Order might be important for angle calculation (e.g., vertex is middle).
                self.analysis_markers.append(marker_name) 
                logger.info(f"Added {marker_name} to analysis markers: {self.analysis_markers}")
            else:
                # Notify user that the limit is reached
                logger.warning(f"Cannot select more than 3 markers for analysis. Click existing marker to deselect.")
                messagebox.showwarning("Analysis Mode", "You can select a maximum of 3 markers for analysis. Click on an already selected marker to deselect it.")
                return # Do not proceed further if limit reached

        # Update the renderer state with the new list and trigger redraw
        if hasattr(self, 'gl_renderer'):
            # Ensure the renderer knows the current mode state and the updated list
            self.gl_renderer.set_analysis_state(self.is_analysis_mode, self.analysis_markers)
            self.gl_renderer.redraw() # Redraw to show selection changes


    def handle_pattern_marker_selection(self, marker_name):
        """Handles the selection/deselection of a marker for pattern-based interpolation."""
        if not self.pattern_selection_mode:
            return # Should not happen if called correctly, but as a safeguard

        if marker_name in self.pattern_markers:
            self.pattern_markers.remove(marker_name)
            logger.info(f"Removed {marker_name} from pattern markers.")
        else:
            self.pattern_markers.add(marker_name)
            logger.info(f"Added {marker_name} to pattern markers.")

        # Update the UI list showing selected markers
        self.update_selected_markers_list()
        
        # Update the renderer state (important for visual feedback)
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_pattern_selection_mode(True, self.pattern_markers)
            # Trigger redraw in the renderer to show color changes
            self.gl_renderer.redraw() 
