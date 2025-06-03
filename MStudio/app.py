import logging
import os
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib

from MStudio.gui.TRCviewerWidgets import create_widgets
from MStudio.gui.markerPlot import show_marker_plot
from MStudio.gui.plotCreator import create_plot
from MStudio.gui.filterUI import on_filter_type_change, build_filter_parameter_widgets
from MStudio.gui.markerPlotUI import build_marker_plot_buttons

from MStudio.utils.dataLoader import open_file
from MStudio.utils.dataSaver import save_as
from MStudio.utils.skeletons import (
    BODY_25B, BODY_25, BODY_135, BLAZEPOSE, HALPE_26, HALPE_68,
    HALPE_136, COCO_133, COCO, MPII, COCO_17
)
from MStudio.utils.viewToggles import (
    toggle_marker_names,
    toggle_trajectory,
    toggle_animation,
    toggle_analysis_mode,
)
from MStudio.utils.viewReset import reset_main_view, reset_graph_view
from MStudio.utils.dataProcessor import (
    filter_selected_data,
    interpolate_selected_data,
    interpolate_with_pattern,
    on_pattern_selection_confirm
)
from MStudio.utils.mouseHandler import MouseHandler
from MStudio.utils.performance_utils import PerformanceTimer, memoize, animation_optimized

# Core components
from MStudio.core.data_manager import DataManager
from MStudio.core.animation_controller import AnimationController
from MStudio.core.outlier_detector import OutlierDetector
from MStudio.core.state_manager import StateManager

# Configure logging
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

        # Initialize core components
        self.data_manager = DataManager()
        self.animation_controller = AnimationController(self)
        self.outlier_detector = OutlierDetector()
        self.state_manager = StateManager()

        # Initialize performance manager for animation optimization
        from MStudio.utils.performance_utils import AnimationPerformanceManager
        self.performance_manager = AnimationPerformanceManager()

        # Setup callbacks for core components
        self._setup_core_callbacks()

        # --- Frame tracking (now managed by animation_controller) ---
        self.frame_idx = 0  # Synchronized with animation_controller

        # --- Main 3D Plot Attributes ---
        self.canvas = None
        self.gl_renderer = None # OpenGL renderer related attributes
        self.view_limits = None
        self.pan_enabled = False
        self.last_mouse_pos = None
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

        # --- Legacy selection attributes (will be moved to state_manager) ---
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

        # --- Timeline Attributes ---
        self.current_frame_line = None
        self.fps_var = ctk.StringVar(value="60")

        # --- Mouse Handling ---
        self.mouse_handler = MouseHandler(self)

        # --- Editing State (legacy - will be moved to state_manager) ---
        self.edit_window = None
        self.edit_controls_frame = None # Placeholder for edit controls frame

        # --- Key Bindings ---
        self.bind('<space>', lambda _: self.toggle_animation())
        self.bind('<Return>', lambda _: self.toggle_animation())
        self.bind('<Escape>', lambda _: self.stop_animation())
        self.bind('<Left>', lambda _: self.prev_frame())
        self.bind('<Right>', lambda _: self.next_frame())

        # --- Widget and Plot Creation ---
        self.create_widgets()
        self.create_plot()
        self.update_plot()

    def _setup_core_callbacks(self) -> None:
        """Setup callbacks for core component communication."""
        # Animation controller callbacks
        self.animation_controller.set_frame_update_callback(self._on_frame_changed)
        self.animation_controller.set_animation_state_callback(self._on_animation_state_changed)

        # State manager callbacks
        self.state_manager.register_view_callback(self._on_view_state_changed)
        self.state_manager.register_selection_callback(self._on_selection_state_changed)
        self.state_manager.register_editing_callback(self._on_editing_state_changed)

    def _on_frame_changed(self, frame_idx: int) -> None:
        """Callback for when the current frame changes - optimized for animation performance."""
        self.frame_idx = frame_idx  # Update legacy attribute

        # During animation, use optimized update path
        if self.animation_controller.is_playing:
            self._update_display_during_animation()
        else:
            # Full update when not animating (manual frame changes)
            self._update_display_after_frame_change()

    def _on_animation_state_changed(self, is_playing: bool) -> None:
        """Callback for when animation state changes."""
        # OPTIMIZATION: Update performance manager state
        self.performance_manager.set_animation_active(is_playing)

        # Apply animation optimizations to renderer
        if hasattr(self, 'gl_renderer'):
            self.performance_manager.optimize_for_animation(self.gl_renderer)

        # Update UI elements based on animation state
        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.configure(text="⏸" if is_playing else "▶")
        if hasattr(self, 'stop_button'):
            self.stop_button.configure(state='normal' if is_playing else 'disabled')

        # When animation starts or stops, ensure timeline is properly updated
        if hasattr(self, 'timeline_ax'):
            if is_playing:
                # Starting animation - ensure current frame line exists for optimization
                if not hasattr(self, '_current_frame_line') or self._current_frame_line is None:
                    self.update_timeline()
            else:
                # Stopping animation - redraw the timeline to ensure proper yellow line display
                self.update_timeline()

    def _on_view_state_changed(self, _view_state) -> None:
        """Callback for when view state changes."""
        # Update UI elements and renderer based on view state
        self.update_plot()

    def _on_selection_state_changed(self, _selection_state) -> None:
        """Callback for when selection state changes."""
        # Update UI elements based on selection state
        self.update_plot()

    def _on_editing_state_changed(self, _editing_state) -> None:
        """Callback for when editing state changes."""
        # Update UI elements based on editing state
        pass

    # --- Direct access to core components (optimized) ---
    # Remove redundant property wrappers for better performance
    # Access data_manager, animation_controller, etc. directly

    # Legacy properties kept for critical backward compatibility only
    @property
    def current_file(self):
        """Get current file path."""
        return getattr(self, '_current_file', None)

    @current_file.setter
    def current_file(self, value):
        """Set current file path."""
        self._current_file = value


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
        """Calculate data limits using the DataManager."""
        self.data_manager.calculate_data_limits()

    
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


    def do_resize_optimized(self, event):
        """Optimized resize method with throttling to prevent flickering."""
        if not self.sizer_dragging:
            return

        # Cancel any pending resize timer
        if hasattr(self, '_sizer_resize_timer') and self._sizer_resize_timer:
            self.after_cancel(self._sizer_resize_timer)

        # Use throttled resize to prevent excessive updates during continuous resizing
        import time
        current_time = time.time() * 1000  # Convert to milliseconds

        if current_time - self._sizer_last_resize_time >= self._sizer_throttle_ms:
            # Immediate resize for responsive feel
            self._perform_sizer_resize(event)
            self._sizer_last_resize_time = current_time
        else:
            # Throttled resize - schedule for later
            self._sizer_resize_timer = self.after(self._sizer_throttle_ms,
                                                lambda: self._perform_sizer_resize(event))

    def _perform_sizer_resize(self, event):
        """Perform the actual sizer resize operation."""
        try:
            dx = event.x_root - self.initial_sizer_x
            new_width = max(200, min(self.initial_panel_width - dx, self.winfo_width() - 200))

            # Use configure with explicit width to minimize layout recalculations
            self.right_panel.configure(width=new_width)

            # Force immediate layout update to prevent visual lag
            self.right_panel.update_idletasks()

        except Exception as e:
            # Fallback to standard resize if optimization fails
            self.do_resize(event)

    def stop_resize(self, _event):
        self.sizer_dragging = False

        # Cancel any pending resize operations
        if hasattr(self, '_sizer_resize_timer') and self._sizer_resize_timer:
            self.after_cancel(self._sizer_resize_timer)
            self._sizer_resize_timer = None

        # Force final layout update
        if hasattr(self, 'right_panel'):
            self.right_panel.update_idletasks()


    #########################################
    ###### Show/Hide Name of markers ########
    #########################################

    def toggle_marker_names(self):
        """Toggle marker name visibility using StateManager."""
        self.state_manager.toggle_marker_names()
        # Update button text for backward compatibility
        if hasattr(self, 'names_button'):
            text = "Hide Names" if self.state_manager.view_state.show_names else "Show Names"
            self.names_button.configure(text=text)


    #########################################
    #### Show/Hide trajectory of markers ####
    #########################################

    # TODO for show/hide trajectory of markers:
    # 1. Users can choose the color of the trajectory
    # 2. Users can choose the width of the trajectory
    # 3. Users can choose the length of the trajectory

    def toggle_trajectory(self):
        """Toggle trajectory visibility using StateManager."""
        self.state_manager.toggle_trajectory()
        # Update button text for backward compatibility
        if hasattr(self, 'trajectory_button'):
            text = "Hide Trajectory" if self.state_manager.view_state.show_trajectory else "Show Trajectory"
            self.trajectory_button.configure(text=text)

        
    #########################################
    ####### Skeleton model manager ##########
    #########################################

    def update_keypoint_names(self):
        """
        Update keypoint names based on the selected skeleton model using DataManager.
        """
        if self.data_manager.update_keypoint_names(self.state_manager.current_skeleton_model):
            logger.info("Keypoint names updated successfully")

    def on_model_change(self, choice):
        try:
            # Save the current frame
            current_frame = self.frame_idx

            # Update the model
            self.state_manager.current_skeleton_model = self.available_models[choice]

            # Update skeleton settings
            if self.state_manager.current_skeleton_model is None:
                self.state_manager.skeleton_pairs = []
                self.state_manager.view_state.show_skeleton = False
            else:
                self.state_manager.view_state.show_skeleton = True
                # Update keypoint names based on the selected skeleton model
                self.update_keypoint_names()
                # Update skeleton pairs after keypoint names have been updated
                self.update_skeleton_pairs()

            # Deliver skeleton pairs and show skeleton to OpenGL renderer
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_skeleton_pairs(self.state_manager.skeleton_pairs)
                self.gl_renderer.set_show_skeleton(self.state_manager.view_state.show_skeleton)
                # Update marker names in the renderer
                self.gl_renderer.set_marker_names(self.data_manager.marker_names)

            # Re-detect outliers with new skeleton pairs
            self.detect_outliers()
            
            # Deliver outliers to OpenGL renderer
            if hasattr(self, 'gl_renderer') and hasattr(self, 'outliers'):
                self.gl_renderer.set_outliers(self.outliers)

            # Update the plot with the current frame data
            self.update_plot()
            self.update_frame(current_frame)

            # If a marker is currently selected, update its plot
            current_marker = self.state_manager.selection_state.current_marker
            if current_marker:
                self.show_marker_plot(current_marker)

        except Exception as e:
            logger.error("Error in on_model_change: %s", e, exc_info=True)


    def update_skeleton_pairs(self):
        """update skeleton pairs"""
        self.state_manager.skeleton_pairs = []
        if self.state_manager.current_skeleton_model is not None:
            for node in self.state_manager.current_skeleton_model.descendants:
                if node.parent:
                    parent_name = node.parent.name
                    node_name = node.name

                    # check if marker names are in the data
                    data = self.data_manager.data
                    if (f"{parent_name}_X" in data.columns and
                        f"{node_name}_X" in data.columns):
                        self.state_manager.skeleton_pairs.append((parent_name, node_name))


    #########################################
    ########## Outlier detection ############
    #########################################

    # TODO for outlier detection:
    # 1. Find a better way to detect outliers
    # 2. Add a threshold for outlier detection

    def detect_outliers(self):
        """Detect outliers using the optimized OutlierDetector."""
        if not self.state_manager.skeleton_pairs or not self.data_manager.has_data():
            self.outliers = {}
            return

        with PerformanceTimer("Outlier detection"):
            # Use the optimized outlier detector
            self.outliers = self.outlier_detector.detect_outliers(
                self.data_manager.data,
                self.data_manager.marker_names,
                self.state_manager.skeleton_pairs
            )

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
        if marker_name == self.state_manager.selection_state.current_marker:
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
        
        self.state_manager.set_current_marker(marker_name)
        
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
                for marker in sorted(self.state_manager.selection_state.pattern_markers):
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

    def update_timeline(self, current_frame_only=False):
        """Optimized timeline update with optional current frame only mode for animation."""
        if not self.data_manager.has_data():
            return

        light_yellow = '#FFEB3B'

        # Fast path for animation - only update current frame indicator
        if current_frame_only and hasattr(self, 'timeline_ax'):
            self._update_current_frame_indicator_only(light_yellow)
            return

        # Full timeline redraw
        self.timeline_ax.clear()
        frames = np.arange(self.data_manager.num_frames)
        fps = float(self.fps_var.get())
        times = frames / fps

        # add horizontal baseline (y=0)
        self.timeline_ax.axhline(y=0, color='white', alpha=0.3, linewidth=1)

        display_mode = self.timeline_display_var.get()

        if display_mode == "time":
            self._draw_time_ticks(times, fps)
            current_time = self.frame_idx / fps
            current_display = f"{current_time:.2f}s"
        else:  # frame mode
            self._draw_frame_ticks()
            current_display = f"{self.frame_idx}"

        # current frame display (light yellow line)
        # Always create the line during full timeline redraw
        self._current_frame_line = self.timeline_ax.axvline(self.frame_idx, color=light_yellow, alpha=0.8, linewidth=1.5)

        # update label
        self.current_info_label.configure(text=current_display)

        # timeline settings
        self.timeline_ax.set_xlim(0, self.data_manager.num_frames - 1)
        self.timeline_ax.set_ylim(-1, 1)

        # hide y-axis
        self.timeline_ax.set_yticks([])

        # border style
        self.timeline_ax.spines['top'].set_visible(False)
        self.timeline_ax.spines['right'].set_visible(False)
        self.timeline_ax.spines['left'].set_visible(False)
        self.timeline_ax.spines['bottom'].set_color('white')
        self.timeline_ax.spines['bottom'].set_alpha(0.3)

        # hide x-axis ticks (we draw them manually)
        self.timeline_ax.set_xticks([])
        # adjust figure margins (to avoid text clipping)
        self.timeline_fig.subplots_adjust(bottom=0.2)

        self.timeline_canvas.draw_idle()

    def _draw_time_ticks(self, times, fps):
        """Helper method to draw time-based ticks on timeline."""
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

    def _draw_frame_ticks(self):
        """Helper method to draw frame-based ticks on timeline."""
        # major ticks every 100 frames
        major_frame_ticks = np.arange(0, self.data_manager.num_frames, 100)
        for frame in major_frame_ticks:
            self.timeline_ax.axvline(frame, color='white', alpha=0.3, linewidth=1)
            self.timeline_ax.text(frame, -0.7, f"{frame}",
                                color='white', fontsize=6, alpha=0.5,
                                horizontalalignment='center',
                                verticalalignment='top')

    def _update_current_frame_indicator_only(self, light_yellow):
        """Optimized method to update only the current frame indicator during animation."""
        if not hasattr(self, 'timeline_ax'):
            return

        # OPTIMIZATION: Use efficient line position update instead of remove/add
        if hasattr(self, '_current_frame_line') and self._current_frame_line:
            try:
                # Check if the line is still valid and in the axes
                if self._current_frame_line in self.timeline_ax.lines:
                    # Simply update the x-position of existing line (much faster)
                    self._current_frame_line.set_xdata([self.frame_idx, self.frame_idx])

                    # Update frame display label
                    self._update_frame_display_label()

                    # Use draw_idle for better performance during animation
                    self.timeline_canvas.draw_idle()
                    return  # Skip expensive remove/add operations
                else:
                    # Line is no longer in axes, need to recreate
                    self._current_frame_line = None
            except (ValueError, AttributeError):
                # Line is invalid, fall back to recreation
                self._current_frame_line = None

        # Fallback: Create new line only if needed
        # Remove only the tracked current frame line (not all yellow lines)
        if hasattr(self, '_current_frame_line') and self._current_frame_line:
            try:
                self._current_frame_line.remove()
            except ValueError:
                pass  # Line already removed

        # Add new current frame line
        self._current_frame_line = self.timeline_ax.axvline(
            self.frame_idx, color=light_yellow, alpha=0.8, linewidth=1.5
        )

        # Update frame display label
        self._update_frame_display_label()

        # Use draw_idle for better performance during animation
        self.timeline_canvas.draw_idle()

    def _update_frame_display_label(self):
        """Helper method to update the frame display label."""
        if hasattr(self, 'current_info_label'):
            display_mode = self.timeline_display_var.get()
            if display_mode == "time":
                fps = float(self.fps_var.get())
                current_time = self.frame_idx / fps
                current_display = f"{current_time:.2f}s"
            else:
                current_display = f"{self.frame_idx}"
            self.current_info_label.configure(text=current_display)


    def update_frame_from_timeline(self, x_pos):
        """Update frame position from timeline interaction (dragging/clicking)."""
        if x_pos is not None and self.data_manager.has_data():
            frame = int(max(0, min(x_pos, self.data_manager.num_frames - 1)))

            # Sync both the main frame_idx and AnimationController
            self.frame_idx = frame
            self.animation_controller.set_frame(frame, from_external=True)

            # Update display
            self._update_display_after_frame_change()

            # update vertical line if marker graph is displayed
            self._update_marker_plot_vertical_line_data()
            # Check if marker_canvas exists before drawing
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()

            # Update timeline to reflect the new position
            self.update_timeline()


    def update_plot(self):
        """
        Update method for 3D marker visualization with performance optimizations.
        """
        if not hasattr(self, 'gl_renderer') or self.gl_renderer is None:
            return

        # Use DataManager to check if we have data
        if not self.data_manager.has_data():
            return

        try:
            # Get current state from StateManager
            coordinate_system = self.state_manager.view_state.coordinate_system

            # Deliver outliers if available
            if hasattr(self, 'outliers') and self.outliers:
                self.gl_renderer.set_outliers(self.outliers)

            # Deliver current frame data using optimized access
            self.gl_renderer.set_frame_data(
                self.data_manager.data,
                self.frame_idx,
                self.data_manager.marker_names,
                self.state_manager.selection_state.current_marker,
                self.state_manager.view_state.show_names,
                self.state_manager.view_state.show_trajectory,
                self.state_manager.view_state.show_skeleton,
                coordinate_system,
                self.state_manager.skeleton_pairs
            )

            # Update OpenGL renderer screen
            self.gl_renderer.update_plot()

        except Exception as e:
            logger.error("Error updating plot: %s", e, exc_info=True)


    def _update_marker_plot_vertical_line_data(self):
        """Helper function to update the x-data of the vertical lines on the marker plot."""
        if hasattr(self, 'marker_lines') and self.marker_lines:
            for line in self.marker_lines:
                line.set_xdata([self.frame_idx, self.frame_idx])


    def _update_display_after_frame_change(self):
        """Helper function to update the main plot and the timeline after a frame change."""
        self.update_plot()
        self.update_timeline()

    def _update_display_during_animation(self):
        """Optimized update method for smooth animation playback."""
        # Only update the 3D plot during animation - skip expensive timeline redraw
        self.update_plot()

        # Update only the current frame indicator on timeline (much faster)
        self.update_timeline(current_frame_only=True)

        # Update marker plot vertical line efficiently if needed
        if hasattr(self, 'marker_lines') and self.marker_lines:
            self._update_marker_plot_vertical_line_data()
            # Use draw_idle for better performance during animation
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw_idle()


    def update_frame(self, value):
        """Update frame from external input (e.g., slider, keyboard)."""
        if self.data_manager.has_data():
            frame = int(float(value))

            # Sync both the main frame_idx and AnimationController
            self.frame_idx = frame
            self.animation_controller.set_frame(frame, from_external=True)

            # Update display
            self._update_display_after_frame_change()

            # update vertical line if marker graph is displayed
            self._update_marker_plot_vertical_line_data()
            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                self.marker_canvas.draw()

            # Update timeline to reflect the new position
            self.update_timeline()


    def update_fps_label(self):
        fps = self.fps_var.get()
        if hasattr(self, 'fps_label'):
            self.fps_label.configure(text=f"FPS: {fps}")


    def _update_marker_plot_vertical_line_data(self):
        """Updates the vertical line data in the marker plot."""
        if not self.data_manager.has_data() or not hasattr(self, 'marker_canvas') or self.marker_canvas is None:
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

            # Clear data through managers
            self.data_manager.clear_data()
            self.state_manager.reset_all_states()

            self.frame_idx = 0
            self.outliers = {}
            self.marker_axes = []
            self.marker_lines = []

            self.view_limits = None

            self.selection_data = {
                'start': None,
                'end': None,
                'rects': [],
                'current_ax': None,
                'rect': None
            }

            # frame_slider related code
            self.title_label.configure(text="")
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
        self.state_manager.selection_state.pattern_markers.clear()
        self.update_selected_markers_list()
        self.update_plot()


    #########################################
    ########## Playing Controllers ##########
    #########################################

    def toggle_animation(self):
        """Toggle animation play/pause using the AnimationController."""
        # Before toggling, sync the AnimationController with current UI frame position
        if not self.animation_controller.is_playing:
            # Starting animation - sync with current UI frame
            self.animation_controller.set_frame(self.frame_idx)

        self.animation_controller.toggle_play_pause()

    def prev_frame(self):
        """Move to the previous frame using the AnimationController."""
        self.animation_controller.prev_frame()

    def next_frame(self):
        """Move to the next frame using the AnimationController."""
        self.animation_controller.next_frame()


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


    # ---------- animation (legacy methods for backward compatibility) ----------
    def animate(self):
        """Legacy method - animation is now handled by AnimationController."""
        pass  # This is now handled by the AnimationController

    def play_animation(self):
        """Start animation using the AnimationController."""
        # Sync with current UI frame position before starting
        self.animation_controller.set_frame(self.frame_idx)
        self.animation_controller.play()

    def pause_animation(self):
        """Pause animation using the AnimationController."""
        self.animation_controller.pause()

    def stop_animation(self):
        """Stop animation using the AnimationController."""
        self.animation_controller.stop()

    def _on_loop_checkbox_changed(self):
        """Callback for when the loop checkbox state changes."""
        # BUG FIX: Connect loop checkbox to animation controller
        loop_enabled = self.loop_var.get()
        self.animation_controller.set_loop(loop_enabled)
        logger.info(f"Loop checkbox changed: {'enabled' if loop_enabled else 'disabled'}")


    #########################################
    ############### Editors #################
    #########################################

    # TODO for edit mode:
    # 1. Create a new file for edit mode
    def toggle_edit_mode(self):
        """Toggles the editing mode for the marker plot."""
        current_marker = self.state_manager.selection_state.current_marker
        if not current_marker: # Ensure a marker plot is shown
            return

        self.state_manager.set_editing_mode(not self.state_manager.editing_state.is_editing)
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
                is_editing = self.state_manager.editing_state.is_editing
                if is_editing and self.interp_method_var.get() == 'pattern-based':
                    self.state_manager.editing_state.pattern_selection_mode = True
                else:
                    self.state_manager.editing_state.pattern_selection_mode = False
                    
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

        current_marker = self.state_manager.selection_state.current_marker
        for coord in ['X', 'Y', 'Z']:
            col_name = f'{current_marker}_{coord}'
            self.data_manager.data.loc[start_frame:end_frame, col_name] = np.nan

        self.show_marker_plot(current_marker)

        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])

        self.update_plot()

        self.selection_data['start'] = current_selection['start']
        self.selection_data['end'] = current_selection['end']
        self.highlight_selection()

        # Update button state *only if* the edit button exists (i.e., not in edit mode)
        # and the widget itself hasn't been destroyed
        is_editing = self.state_manager.editing_state.is_editing
        if not is_editing and hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
            self.edit_button.configure(fg_color="#555555")


    # ---------- Restore original data ----------
    def restore_original_data(self):
        """Restore data to original state using DataManager."""
        if self.data_manager.restore_original_data():
            self.detect_outliers()
            # Check if a marker plot is currently displayed before trying to update it
            current_marker = self.state_manager.selection_state.current_marker
            if current_marker:
                self.show_marker_plot(current_marker)
            self.update_plot()

            # Update button state *only if* the edit button exists (i.e., not in edit mode)
            if hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
                 self.edit_button.configure(fg_color="#3B3B3B") # Reset to default color, not gray

            logger.info("Data has been restored to the original state.")
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
            self.state_manager.selection_state.pattern_markers.clear()
            # Set pattern selection mode on the main app
            self.state_manager.editing_state.pattern_selection_mode = True
            # **Update the renderer's mode**
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_pattern_selection_mode(True, self.state_manager.selection_state.pattern_markers)
            messagebox.showinfo("Pattern Selection", 
                "Left-click markers in the 3D view to select/deselect them as reference patterns.\n"
                "Selected markers will be shown in red.")
        else:
            # Disable pattern selection mode on the main app
            self.state_manager.editing_state.pattern_selection_mode = False
            # **Update the renderer's mode**
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_pattern_selection_mode(False)
            
        # Update main 3D view if needed (redraws with correct marker colors)
        self.update_plot()
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            self.marker_canvas.draw_idle()


    def handle_pattern_marker_selection(self, marker_name):
        """Handles the selection/deselection of a marker for pattern-based interpolation."""
        pattern_selection_mode = self.state_manager.editing_state.pattern_selection_mode
        if not pattern_selection_mode:
            return # Should not happen if called correctly, but as a safeguard

        pattern_markers = self.state_manager.selection_state.pattern_markers
        if marker_name in pattern_markers:
            pattern_markers.remove(marker_name)
            logger.info(f"Removed {marker_name} from pattern markers.")
        else:
            pattern_markers.add(marker_name)
            logger.info(f"Added {marker_name} to pattern markers.")

        # Update the UI list showing selected markers
        self.update_selected_markers_list()
        
        # Update the renderer state (important for visual feedback)
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_pattern_selection_mode(True, self.state_manager.selection_state.pattern_markers)
            # Trigger redraw in the renderer to show color changes
            self.gl_renderer.redraw()


    def handle_analysis_marker_selection(self, marker_name):
        """Handles marker selection/deselection in analysis mode."""
        if not self.state_manager.editing_state.is_analysis_mode:
            logger.warning("handle_analysis_marker_selection called when not in analysis mode.")
            return

        analysis_markers = self.state_manager.selection_state.analysis_markers
        if marker_name in analysis_markers:
            analysis_markers.remove(marker_name)
            logger.info(f"Removed {marker_name} from analysis markers.")
        else:
            if len(analysis_markers) < 3:
                # Append marker. Order might be important for angle calculation (e.g., vertex is middle).
                analysis_markers.append(marker_name)
                logger.info(f"Added {marker_name} to analysis markers: {analysis_markers}")
            else:
                # Notify user that the limit is reached
                logger.warning(f"Cannot select more than 3 markers for analysis. Click existing marker to deselect.")
                messagebox.showwarning("Analysis Mode", "You can select a maximum of 3 markers for analysis. Click on an already selected marker to deselect it.")
                return # Do not proceed further if limit reached

        # Update the renderer state with the new list and trigger redraw
        if hasattr(self, 'gl_renderer'):
            # Ensure the renderer knows the current mode state and the updated list
            self.gl_renderer.set_analysis_state(self.state_manager.editing_state.is_analysis_mode, self.state_manager.selection_state.analysis_markers)
            self.gl_renderer.redraw() # Redraw to show selection changes
