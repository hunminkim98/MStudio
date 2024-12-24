from turtle import width
import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Pose2Sim.skeletons import *
from Pose2Sim.filtering import *
from matplotlib.figure import Figure
import matplotlib
import os
from utils.data_loader import read_data_from_c3d, read_data_from_trc
from utils.data_saver import save_to_trc, save_to_c3d
from gui.EditWindow import EditWindow
from utils.mouse_handler import MouseHandler
from utils.trajectory import MarkerTrajectory

# Interactive mode on
plt.ion()
matplotlib.use('TkAgg')

class TRCViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TRC Viewer")
        self.geometry("1920x1080")

        # 기 변수 초기화
        self.marker_names = []
        self.data = None
        self.original_data = None
        self.num_frames = 0
        self.frame_idx = 0
        self.canvas = None
        self.selection_in_progress = False
        self.outliers = {}

        self.marker_last_pos = None
        self.marker_pan_enabled = False
        self.marker_canvas = None
        self.marker_axes = []
        self.marker_lines = []

        self.show_trajectory = False 
        self.trajectory_length = 10
        self.trajectory_line = None 

        self.view_limits = None
        self.is_z_up = True

        # 필터 타입 변수 초기화
        self.filter_type_var = ctk.StringVar(value='butterworth')

        # 마우스 핸들러 초기화
        self.mouse_handler = MouseHandler(self)
        
        # 보간 메소드 리스트 추가
        self.interp_methods = [
            'linear',
            'polynomial',
            'spline',
            'nearest',
            'zero',
            'slinear',
            'quadratic',
            'cubic',
            'pattern-based' # 11/05
        ]
        
        # 보간 메소드 변수 초기화
        self.interp_method_var = ctk.StringVar(value='linear')
        self.order_var = ctk.StringVar(value='3')

        # 패턴 마커 관련 속성 초기화
        self.pattern_markers = set()
        self._selected_markers_list = None

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

        self.pan_enabled = False
        self.last_mouse_pos = None

        self.is_playing = False
        self.playback_speed = 1.0
        self.animation_job = None
        self.fps_var = ctk.StringVar(value="60")

        self.current_frame_line = None

        self.bind('<space>', lambda e: self.toggle_animation())
        self.bind('<Return>', lambda e: self.toggle_animation())
        self.bind('<Escape>', lambda e: self.stop_animation())
        self.bind('<Left>', lambda e: self.prev_frame())
        self.bind('<Right>', lambda e: self.next_frame())

        self.create_widgets()

        # initialize plot
        self.create_plot()
        self.update_plot()

        self.edit_window = None
        
        # Initialize trajectory handler
        self.trajectory_handler = MarkerTrajectory()
        
        # Keep these for compatibility with existing code
        self.show_trajectory = False
        self.trajectory_length = 10
        self.trajectory_line = None
        self.marker_lines = []

    def create_widgets(self):
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10, padx=10, fill='x')

        button_style = {
            "fg_color": "#333333",
            "hover_color": "#444444"
        }

        left_button_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
        left_button_frame.pack(side='left', fill='x')

        self.reset_view_button = ctk.CTkButton(
            left_button_frame,
            text="🎥",
            width=30,
            command=self.reset_main_view,
            **button_style
        )
        self.reset_view_button.pack(side='left', padx=5)

        self.open_button = ctk.CTkButton(
            left_button_frame,
            text="Open TRC File",
            command=self.open_file,
            **button_style
        )
        self.open_button.pack(side='left', padx=5)

        self.coord_button = ctk.CTkButton(
            button_frame,
            text="Switch to Y-up",
            command=self.toggle_coordinates,
            **button_style
        )
        self.coord_button.pack(side='left', padx=5)

        self.names_button = ctk.CTkButton(
            button_frame,
            text="Hide Names",
            command=self.toggle_marker_names,
            **button_style
        )
        self.names_button.pack(side='left', padx=5)

        self.trajectory_button = ctk.CTkButton(
            button_frame,
            text="Show Trajectory",
            command=self.toggle_trajectory,
            **button_style
        )
        self.trajectory_button.pack(side='left', padx=5)

        self.save_button = ctk.CTkButton(
            button_frame,
            text="Save As...",
            command=self.save_as,
            **button_style
        )
        self.save_button.pack(side='left', padx=5)

        self.model_var = ctk.StringVar(value='No skeleton')
        self.model_combo = ctk.CTkComboBox(
            button_frame,
            values=list(self.available_models.keys()),
            variable=self.model_var,
            command=self.on_model_change
        )
        self.model_combo.pack(side='left', padx=5)

        self.main_content = ctk.CTkFrame(self)
        self.main_content.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.view_frame = ctk.CTkFrame(self.main_content, fg_color="black")
        self.view_frame.pack(side='left', fill='both', expand=True)

        self.right_panel = ctk.CTkFrame(self.main_content, fg_color="black")
        self.right_panel.pack_forget()  # 처음에는 숨김
        self.right_panel.pack_propagate(False)  # 크기 고정

        self.graph_frame = ctk.CTkFrame(self.right_panel, fg_color="black")
        
        viewer_top_frame = ctk.CTkFrame(self.view_frame)
        viewer_top_frame.pack(fill='x', pady=(5, 0))

        self.title_label = ctk.CTkLabel(viewer_top_frame, text="", font=("Arial", 14))
        self.title_label.pack(side='left', expand=True)

        canvas_container = ctk.CTkFrame(self.view_frame)
        canvas_container.pack(fill='both', expand=True)

        self.canvas_frame = ctk.CTkFrame(canvas_container)
        self.canvas_frame.pack(expand=True, fill='both')
        self.canvas_frame.pack_propagate(False)

        self.control_frame = ctk.CTkFrame(
            self,
            border_width=1,  
            fg_color="#1A1A1A"  # background color
        )
        self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

        # control button style
        control_style = {
            "width": 30,
            "fg_color": "#333333",
            "hover_color": "#444444"
        }

        # control button frame
        button_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        button_frame.pack(side='left', padx=5)

        # play control buttons
        self.play_pause_button = ctk.CTkButton(
            button_frame,
            text="▶",
            command=self.toggle_animation,
            **control_style
        )
        self.play_pause_button.pack(side='left', padx=2)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="■",
            command=self.stop_animation,
            # state='disabled',
            **control_style
        )
        self.stop_button.pack(side='left', padx=2)

        # loop checkbox style
        checkbox_style = {
            "width": 60,
            "fg_color": "#1A1A1A",  # transparent instead of background color
            "border_color": "#666666",  # border color
            "hover_color": "#1A1A1A",  # hover color
            "checkmark_color": "#00A6FF",  # checkmark color
            "border_width": 2  # border width
        }

        # loop checkbox
        self.loop_var = ctk.BooleanVar(value=False)
        self.loop_checkbox = ctk.CTkCheckBox(
            button_frame,
            text="Loop",
            variable=self.loop_var,
            text_color="#FFFFFF",
            **checkbox_style
        )
        self.loop_checkbox.pack(side='left', padx=5)

        # timeline menu frame
        timeline_menu_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        timeline_menu_frame.pack(side='left', padx=(5, 10))

        # current frame/time display label
        self.current_info_label = ctk.CTkLabel(
            timeline_menu_frame,
            text="0.00s",
            font=("Arial", 14),
            text_color="#FFFFFF"
        )
        self.current_info_label.pack(side='left', padx=5)

        # mode selection button frame
        mode_frame = ctk.CTkFrame(timeline_menu_frame, fg_color="#222222", corner_radius=6)
        mode_frame.pack(side='left', padx=2)

        # time/frame mode button
        button_style = {
            "width": 60,
            "height": 24,
            "corner_radius": 4,
            "font": ("Arial", 11),
            "fg_color": "transparent",
            "text_color": "#888888",
            "hover_color": "#333333"
        }

        self.timeline_display_var = ctk.StringVar(value="time")
        
        self.time_btn = ctk.CTkButton(
            mode_frame,
            text="Time",
            command=lambda: self.change_timeline_mode("time"),
            **button_style
        )
        self.time_btn.pack(side='left', padx=2, pady=2)

        self.frame_btn = ctk.CTkButton(
            mode_frame,
            text="Frame",
            command=lambda: self.change_timeline_mode("frame"),
            **button_style
        )
        self.frame_btn.pack(side='left', padx=2, pady=2)

        # timeline figure
        self.timeline_fig = Figure(figsize=(5, 0.8), facecolor='black')
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        self.timeline_ax.set_facecolor('black')
        
        # timeline canvas
        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, master=self.control_frame)
        self.timeline_canvas.get_tk_widget().pack(fill='x', expand=True, padx=5, pady=5)
        
        # timeline event connection
        self.timeline_canvas.mpl_connect('button_press_event', self.mouse_handler.on_timeline_click)
        self.timeline_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_timeline_drag)
        self.timeline_canvas.mpl_connect('button_release_event', self.mouse_handler.on_timeline_release)
        
        self.timeline_dragging = False

        # initial timeline mode
        self.change_timeline_mode("time")

        self.marker_label = ctk.CTkLabel(self, text="")
        self.marker_label.pack(pady=5)

        if self.canvas:
            self.canvas.mpl_connect('button_press_event', self.mouse_handler.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.mouse_handler.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_mouse_move)
        
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

            # Remove existing skeleton lines
            if hasattr(self, 'skeleton_lines'):
                for line in self.skeleton_lines:
                    line.remove()
                self.skeleton_lines = []

            # Initialize new skeleton lines
            if self.show_skeleton:
                for _ in self.skeleton_pairs:
                    line = Line3D([], [], [], color='gray', alpha=0.9)
                    self.ax.add_line(line)
                    self.skeleton_lines.append(line)

            # Re-detect outliers with new skeleton pairs
            self.detect_outliers()

            # Update the plot with the current frame data
            self.update_plot()
            self.update_frame(current_frame)

            # If a marker is currently selected, update its plot
            if hasattr(self, 'current_marker') and self.current_marker:
                self.show_marker_plot(self.current_marker)

            # Refresh the canvas
            if hasattr(self, 'canvas'):
                self.canvas.draw()
                self.canvas.flush_events()

        except Exception as e:
            print(f"Error in on_model_change: {e}")
            import traceback
            traceback.print_exc()

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

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Motion files", "*.trc;*.c3d"), ("TRC files", "*.trc"), ("C3D files", "*.c3d"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.clear_current_state()

                self.current_file = file_path
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                self.title_label.configure(text=file_name)

                if file_extension == '.trc':
                    header_lines, self.data, self.marker_names, frame_rate = read_data_from_trc(file_path)
                elif file_extension == '.c3d':
                    header_lines, self.data, self.marker_names, frame_rate = read_data_from_c3d(file_path)
                else:
                    raise Exception("Unsupported file format")

                self.num_frames = self.data.shape[0]
                self.original_data = self.data.copy(deep=True)
                self.calculate_data_limits()

                self.fps_var.set(str(int(frame_rate)))
                self.update_fps_label()

                # frame_slider related code
                self.frame_idx = 0
                self.update_timeline()

                self.current_model = self.available_models[self.model_var.get()]
                self.update_skeleton_pairs()
                self.detect_outliers()

                self.create_plot()
                self.reset_main_view()
                self.update_plot()
                # self.update_frame_counter()

                if hasattr(self, 'canvas'):
                    self.canvas.draw()
                    self.canvas.flush_events()

                self.play_pause_button.configure(state='normal')
                # self.speed_slider.configure(state='normal')
                self.loop_checkbox.configure(state='normal')

                self.is_playing = False
                self.play_pause_button.configure(text="▶")
                self.stop_button.configure(state='disabled')

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

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

            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

            if hasattr(self, 'marker_canvas') and self.marker_canvas and hasattr(self.marker_canvas, 'get_tk_widget'):
                self.marker_canvas.get_tk_widget().destroy()
                del self.marker_canvas
                self.marker_canvas = None

            if hasattr(self, 'ax'):
                del self.ax
            if hasattr(self, 'marker_axes'):
                del self.marker_axes

            self.data = None
            self.original_data = None
            self.marker_names = None
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
            print(f"Error clearing state: {e}")

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
            print(f"Error calculating data limits: {e}")
            self.data_limits = None
            self.initial_limits = None

    def create_plot(self):
        self.fig = plt.Figure(figsize=(10, 10), facecolor='black')  # Changed to square figure
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])  # Add proper spacing around plot
        
        self._setup_plot_style()
        self._draw_static_elements()
        self._initialize_dynamic_elements()

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
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')

        # 3D 축의 여백 제거
        # self.ax.dist = 11  # 카메라 거리 조절
        # self.fig.tight_layout(pad=10)  # 여백 최소
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
        """Draw static elements like the ground grid based on the coordinate system."""
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
        self._update_coordinate_axes()

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
        """Update coordinate axes and labels based on the coordinate system."""
        # 축과 레이블 초기화
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

    def update_plot(self):
        if self.data is None:
            return

        # Update trajectories using the handler
        if hasattr(self, 'trajectory_handler'):
            self.trajectory_handler.update_trajectory(self.data, self.frame_idx, self.marker_names, self.ax)

        # handle empty 3D space when data is None
        if self.data is None:
            # initialize markers and skeleton
            if hasattr(self, 'markers_scatter'):
                self.markers_scatter._offsets3d = ([], [], [])
            if hasattr(self, 'selected_marker_scatter'):
                self.selected_marker_scatter._offsets3d = ([], [], [])
                self.selected_marker_scatter.set_visible(False)
            if hasattr(self, 'skeleton_lines'):
                for line in self.skeleton_lines:
                    line.set_data_3d([], [], [])

            # set axis ranges
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-1, 1])
            self.ax.set_zlim([-1, 1])

            self.canvas.draw()
            return

        # remove existing trajectory line
        if hasattr(self, 'trajectory_line') and self.trajectory_line is not None:
            self.trajectory_line.remove()
            self.trajectory_line = None

        if self.current_marker is not None and self.show_trajectory:
            x_vals = []
            y_vals = []
            z_vals = []
            for i in range(0, self.frame_idx + 1):
                try:
                    x = self.data.loc[i, f'{self.current_marker}_X']
                    y = self.data.loc[i, f'{self.current_marker}_Y']
                    z = self.data.loc[i, f'{self.current_marker}_Z']
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        continue
                    if self.is_z_up:
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)
                    else:
                        x_vals.append(x)
                        y_vals.append(-z)
                        z_vals.append(y)
                except KeyError:
                    continue
            if len(x_vals) > 0:
                self.trajectory_line, = self.ax.plot(x_vals, y_vals, z_vals, color='yellow', alpha=0.5, linewidth=1)
        else:
            self.trajectory_line = None

        prev_elev = self.ax.elev
        prev_azim = self.ax.azim
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()
        prev_zlim = self.ax.get_zlim()

        # collect marker position data
        positions = []
        colors = []
        alphas = []
        selected_position = []
        marker_positions = {}
        valid_markers = []

        # collect valid markers for the current frame
        for marker in self.marker_names:
            try:
                x = self.data.loc[self.frame_idx, f'{marker}_X']
                y = self.data.loc[self.frame_idx, f'{marker}_Y']
                z = self.data.loc[self.frame_idx, f'{marker}_Z']
                
                # skip NaN values or deleted data
                if pd.isna(x) or pd.isna(y) or pd.isna(z):
                    continue
                    
                # add valid data
                if self.is_z_up:
                    marker_positions[marker] = np.array([x, y, z])
                    positions.append([x, y, z])
                else:
                    marker_positions[marker] = np.array([x, -z, y])
                    positions.append([x, -z, y])

                # add colors and alphas for valid markers
                if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                    if marker in self.pattern_markers:
                        colors.append('red')
                        alphas.append(0.3)
                    else:
                        colors.append('white')
                        alphas.append(1.0)
                elif marker == self.current_marker:
                    colors.append('yellow')
                    alphas.append(1.0)
                else:
                    colors.append('white')
                    alphas.append(1.0)

                if marker == self.current_marker:
                    if self.is_z_up:
                        selected_position.append([x, y, z])
                    else:
                        selected_position.append([x, -z, y])
                valid_markers.append(marker)
            except KeyError:
                continue

        # array conversion
        positions = np.array(positions) if positions else np.zeros((0, 3))
        selected_position = np.array(selected_position) if selected_position else np.zeros((0, 3))

        # update scatter plot - display valid data
        if len(positions) > 0:
            try:
                # remove existing scatter
                if hasattr(self, 'markers_scatter'):
                    self.markers_scatter.remove()
                
                # create new scatter plot
                self.markers_scatter = self.ax.scatter(
                    positions[:, 0], 
                    positions[:, 1], 
                    positions[:, 2],
                    c=colors[:len(positions)],  # length match
                    alpha=alphas[:len(positions)],  # length match
                    s=30,
                    picker=5
                )
            except Exception as e:
                print(f"Error updating scatter plot: {e}")
                # create default scatter plot if error occurs
                if hasattr(self, 'markers_scatter'):
                    self.markers_scatter.remove()
                self.markers_scatter = self.ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c='white',
                    s=30,
                    picker=5
                )
        else:
            # create empty scatter plot if data is None
            if hasattr(self, 'markers_scatter'):
                self.markers_scatter.remove()
            self.markers_scatter = self.ax.scatter([], [], [], c='white', s=30, picker=5)

        # update selected marker
        if len(selected_position) > 0:
            self.selected_marker_scatter._offsets3d = (
                selected_position[:, 0],
                selected_position[:, 1],
                selected_position[:, 2]
            )
            self.selected_marker_scatter.set_visible(True)
        else:
            self.selected_marker_scatter._offsets3d = ([], [], [])
            self.selected_marker_scatter.set_visible(False)

        # update skeleton lines
        if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_lines'):
            for line, pair in zip(self.skeleton_lines, self.skeleton_pairs):
                if pair[0] in marker_positions and pair[1] in marker_positions:
                    p1 = marker_positions[pair[0]]
                    p2 = marker_positions[pair[1]]

                    outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                    outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                    is_outlier = outlier_status1 or outlier_status2

                    line.set_data_3d(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]]
                    )
                    line.set_visible(True)
                    line.set_color('red' if is_outlier else 'gray')
                    line.set_alpha(1 if is_outlier else 0.8)
                    line.set_linewidth(3 if is_outlier else 2)
                else:
                    line.set_visible(False)

        # update marker names
        # remove existing labels
        for label in self.marker_labels:
            label.remove()
        self.marker_labels.clear()

        for marker in valid_markers:
            pos = marker_positions[marker]
            color = 'white'
            alpha = 1.0
            
            # pattern-based selected markers are always displayed
            if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode and marker in self.pattern_markers:
                color = 'red'  # pattern-based selected marker
                alpha = 0.7
                label = self.ax.text(pos[0], pos[1], pos[2], marker, color=color, alpha=alpha, fontsize=8)
                self.marker_labels.append(label)
            # display other markers if show_names is True
            elif self.show_names:
                if marker == self.current_marker:
                    color = 'yellow'
                label = self.ax.text(pos[0], pos[1], pos[2], marker, color=color, alpha=alpha, fontsize=8)
                self.marker_labels.append(label)

        # update current frame line when marker graph is displayed
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            # remove existing current_frame_line code
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata([self.frame_idx, self.frame_idx])
                self.marker_canvas.draw_idle()

        self.ax.view_init(elev=prev_elev, azim=prev_azim)
        self.ax.set_xlim(prev_xlim)
        self.ax.set_ylim(prev_ylim)
        self.ax.set_zlim(prev_zlim)

        self.canvas.draw_idle()

    def connect_mouse_events(self):
        if self.canvas:
            self.canvas.mpl_connect('scroll_event', self.mouse_handler.on_scroll)
            self.canvas.mpl_connect('button_press_event', self.mouse_handler.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.mouse_handler.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_mouse_move)
            
            if self.marker_canvas:
                self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
                self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
                self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
                self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)
            
    def disconnect_mouse_events(self):
        """disconnect mouse events"""
        if hasattr(self, 'canvas'):
            for cid in self.canvas.callbacks.callbacks.copy():
                self.canvas.mpl_disconnect(cid)

    def update_frame(self, value):
        if self.data is not None:
            self.frame_idx = int(float(value))
            self.update_plot()
            self.update_timeline()

            # update vertical line if marker graph is displayed
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata([self.frame_idx, self.frame_idx])
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()

    def show_marker_plot(self, marker_name):
        # Save current states
        was_editing = getattr(self, 'editing', False)
        
        # Save previous filter parameters if they exist
        prev_filter_params = None
        if hasattr(self, 'filter_params'):
            prev_filter_params = {
                filter_type: {
                    param: var.get() for param, var in params.items()
                } for filter_type, params in self.filter_params.items()
            }
        prev_filter_type = getattr(self, 'filter_type_var', None)
        if prev_filter_type:
            prev_filter_type = prev_filter_type.get()

        if not self.graph_frame.winfo_ismapped():
            # Right panel 표시
            self.right_panel.pack(side='right', fill='both')
            
            # 초기 너비 설정 (전체 창의 1/3)
            initial_width = self.winfo_width() // 3
            self.right_panel.configure(width=initial_width)
            
            # Sizer 생성 및 설정
            if not hasattr(self, 'sizer') or self.sizer is None:
                self.sizer = ctk.CTkFrame(self.main_content, width=5, height=self.main_content.winfo_height(),
                                        fg_color="#666666", bg_color="black")
                self.sizer.pack(side='left', fill='y')
                self.sizer.pack_propagate(False)
                
                # Sizer bindings
                self.sizer.bind('<Enter>', lambda e: (
                    self.sizer.configure(fg_color="#888888"),
                    self.sizer.configure(cursor="sb_h_double_arrow")
                ))
                self.sizer.bind('<Leave>', lambda e: self.sizer.configure(fg_color="#666666"))
                self.sizer.bind('<Button-1>', self.start_resize)
                self.sizer.bind('<B1-Motion>', self.do_resize)
                self.sizer.bind('<ButtonRelease-1>', self.stop_resize)
        
            self.graph_frame.pack(fill='both', expand=True)

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        self.marker_plot_fig = Figure(figsize=(6, 8), facecolor='black')
        self.marker_plot_fig.patch.set_facecolor('black')

        self.current_marker = marker_name

        self.marker_axes = []
        self.marker_lines = []
        coords = ['X', 'Y', 'Z']

        if not hasattr(self, 'outliers') or marker_name not in self.outliers:
            self.outliers = {marker_name: np.zeros(len(self.data), dtype=bool)}

        outlier_frames = np.where(self.outliers[marker_name])[0]

        for i, coord in enumerate(coords):
            ax = self.marker_plot_fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('black')

            data = self.data[f'{marker_name}_{coord}']
            frames = np.arange(len(data))

            ax.plot(frames[~self.outliers[marker_name]],
                    data[~self.outliers[marker_name]],
                    color='white',
                    label='Normal')

            if len(outlier_frames) > 0:
                ax.plot(frames[self.outliers[marker_name]],
                        data[self.outliers[marker_name]],
                        'ro',
                        markersize=3,
                        label='Outlier')

            ax.set_title(f'{marker_name} - {coord}', color='white')
            ax.grid(True, color='gray', alpha=0.3)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

            self.marker_axes.append(ax)

            if len(outlier_frames) > 0:
                ax.legend(facecolor='black',
                        labelcolor='white',
                        loc='upper right',
                        bbox_to_anchor=(1.0, 1.0))

        # initialize current frame display line
        self.marker_lines = []  # initialize existing lines
        for ax in self.marker_axes:
            line = ax.axvline(x=self.frame_idx, color='red', linestyle='--', alpha=0.8)
            self.marker_lines.append(line)

        self.marker_plot_fig.tight_layout()

        self.marker_canvas = FigureCanvasTkAgg(self.marker_plot_fig, master=self.graph_frame)
        self.marker_canvas.draw()
        self.marker_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.initial_graph_limits = []
        for ax in self.marker_axes:
            self.initial_graph_limits.append({
                'x': ax.get_xlim(),
                'y': ax.get_ylim()
            })

        self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
        self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
        self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
        self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)

        button_frame = ctk.CTkFrame(self.graph_frame)
        button_frame.pack(fill='x', padx=5, pady=5)

        reset_button = ctk.CTkButton(button_frame,
                                    text="Reset View",
                                    command=self.reset_graph_view,
                                    width=80,
                                    fg_color="#333333",
                                    hover_color="#444444")
        reset_button.pack(side='right', padx=5)

        # Edit button to open the new window
        self.edit_button = ctk.CTkButton(button_frame,
                                        text="Edit",
                                        command=self.toggle_edit_window,  # window rather than menu
                                        width=80,
                                        fg_color="#333333",
                                        hover_color="#444444")
        self.edit_button.pack(side='right', padx=5)

        # Initialize filter parameters if not already present
        if not hasattr(self, 'filter_params'):
            self.filter_params = {
                'butterworth': {
                    'order': ctk.StringVar(value="4"),
                    'cut_off_frequency': ctk.StringVar(value="10")
                },
                'kalman': {
                    'trust_ratio': ctk.StringVar(value="20"),
                    'smooth': ctk.StringVar(value="1")
                },
                'gaussian': {
                    'sigma_kernel': ctk.StringVar(value="3")
                },
                'LOESS': {
                    'nb_values_used': ctk.StringVar(value="10")
                },
                'median': {
                    'kernel_size': ctk.StringVar(value="3")
                }
            }
        
        # Restore previous parameter values if they exist
        if prev_filter_params:
            for filter_type, params in prev_filter_params.items():
                for param, value in params.items():
                    self.filter_params[filter_type][param].set(value)

        # Backwards compatibility for filter parameters
        self.hz_var = self.filter_params['butterworth']['cut_off_frequency']
        self.filter_order_var = self.filter_params['butterworth']['order']

        self.selection_data = {
            'start': None,
            'end': None,
            'rects': [],
            'current_ax': None,
            'rect': None
        }

        self.connect_mouse_events()

        # Restore edit state if it was active
        if was_editing:
            self.start_edit()

        # connect marker canvas events
        self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
        self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
        self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)
        
        # initialize selection_data
        self.selection_data = {
            'start': None,
            'end': None,
            'rects': []
        }

        # initialize selection_in_progress
        self.selection_in_progress = False

        # update marker name display logic
        if self.show_names or (hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode):
            for marker in self.marker_names:
                x = self.data.loc[self.frame_idx, f'{marker}_X']
                y = self.data.loc[self.frame_idx, f'{marker}_Y']
                z = self.data.loc[self.frame_idx, f'{marker}_Z']
                
                # determine marker name color
                if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode and marker in self.pattern_markers:
                    name_color = 'red'  # pattern-based selected marker
                else:
                    name_color = 'black'  # normal marker
                    
                if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                    if self.is_z_up:
                        self.ax.text(x, y, z, marker, color=name_color)
                    else:
                        self.ax.text(x, z, y, marker, color=name_color)

    def on_interp_method_change(self, choice):
        """보간 방법 변경 시 처리"""
        if choice != 'pattern-based':
            # initialize pattern markers
            self.pattern_markers.clear()
            self.pattern_selection_mode = False
            
            # update screen
            self.update_plot()
            self.canvas.draw_idle()
        else:
            # activate pattern selection mode when pattern-based is selected
            self.pattern_selection_mode = True
            messagebox.showinfo("Pattern Selection", 
                "Right-click markers to select/deselect them as reference patterns.\n"
                "Selected markers will be shown in red.")
        
        # change Order input field state only if EditWindow is open
        if hasattr(self, 'edit_window') and self.edit_window:
            if choice in ['polynomial', 'spline']:
                self.edit_window.order_entry.configure(state='normal')
                self.edit_window.order_label.configure(state='normal')
            else:
                self.edit_window.order_entry.configure(state='disabled')
                self.edit_window.order_label.configure(state='disabled')

    def toggle_edit_window(self):
        try:
            # focus on existing edit_window if it exists
            if hasattr(self, 'edit_window') and self.edit_window:
                self.edit_window.focus()
            else:
                # create new EditWindow
                self.edit_window = EditWindow(self)
                self.edit_window.focus()
                
        except Exception as e:
            print(f"Error in toggle_edit_window: {e}")
            import traceback
            traceback.print_exc()

    def clear_selection(self):
        if 'rects' in self.selection_data and self.selection_data['rects']:
            for rect in self.selection_data['rects']:
                rect.remove()
            self.selection_data['rects'] = []
        if hasattr(self, 'marker_canvas'):
            self.marker_canvas.draw_idle()
        self.selection_in_progress = False

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

    def filter_selected_data(self):
        try:
            # save current selection area
            current_selection = None
            if hasattr(self, 'selection_data'):
                current_selection = {
                    'start': self.selection_data.get('start'),
                    'end': self.selection_data.get('end')
                }

            # If no selection, use entire range
            if self.selection_data.get('start') is None or self.selection_data.get('end') is None:
                start_frame = 0
                end_frame = len(self.data) - 1
            else:
                start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
                end_frame = int(max(self.selection_data['start'], self.selection_data['end']))

            # Store current view states
            view_states = []
            for ax in self.marker_axes:
                view_states.append({
                    'xlim': ax.get_xlim(),
                    'ylim': ax.get_ylim()
                })

            # Get filter parameters
            filter_type = self.filter_type_var.get()

            if filter_type == 'butterworth':
                try:
                    cutoff_freq = float(self.filter_params['butterworth']['cut_off_frequency'].get())
                    filter_order = int(self.filter_params['butterworth']['order'].get())
                    
                    if cutoff_freq <= 0:
                        messagebox.showerror("Input Error", "Hz must be greater than 0")
                        return
                    if filter_order < 1:
                        messagebox.showerror("Input Error", "Order must be at least 1")
                        return
                        
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numbers for Hz and Order")
                    return

                # Create config dict for Pose2Sim
                config_dict = {
                    'filtering': {
                        'butterworth': {
                            'order': filter_order,
                            'cut_off_frequency': cutoff_freq
                        }
                    }
                }
            else:
                config_dict = {
                    'filtering': {
                        filter_type: {k: float(v.get()) for k, v in self.filter_params[filter_type].items()}
                    }
                }

            # Get frame rate and apply filter
            frame_rate = float(self.fps_var.get())
            
            for coord in ['X', 'Y', 'Z']:
                col_name = f'{self.current_marker}_{coord}'
                series = self.data[col_name]
                
                # Apply Pose2Sim filter
                filtered_data = filter1d(series, config_dict, filter_type, frame_rate)
                
                # Update data
                self.data[col_name] = filtered_data

            # Update plots
            self.detect_outliers()
            self.show_marker_plot(self.current_marker)

            # Restore view states
            for ax, view_state in zip(self.marker_axes, view_states):
                ax.set_xlim(view_state['xlim'])
                ax.set_ylim(view_state['ylim'])

            # Restore selection if it existed
            if current_selection and current_selection['start'] is not None:
                self.selection_data['start'] = current_selection['start']
                self.selection_data['end'] = current_selection['end']
                self.highlight_selection()

            self.update_plot()

            if hasattr(self, 'edit_window') and self.edit_window:
                self.edit_window.focus()
                # update edit_button state
                if hasattr(self, 'edit_button'):
                    self.edit_button.configure(fg_color="#555555")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during filtering: {str(e)}")
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()

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

        # Update edit button state if it exists
        if hasattr(self, 'edit_button'):
            self.edit_button.configure(fg_color="#555555")

    def interpolate_selected_data(self):
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

        start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
        end_frame = int(max(self.selection_data['start'], self.selection_data['end']))

        method = self.interp_method_var.get()
        
        if method == 'pattern-based':
            self.interpolate_with_pattern()
        else:
            order = None
            if method in ['polynomial', 'spline']:
                try:
                    order = self.order_var.get()
                except:
                    messagebox.showerror("Error", "Please enter a valid order number")
                    return

            for coord in ['X', 'Y', 'Z']:
                col_name = f'{self.current_marker}_{coord}'
                series = self.data[col_name]

                self.data.loc[start_frame:end_frame, col_name] = np.nan

                interp_kwargs = {}
                if order is not None:
                    interp_kwargs['order'] = order

                try:
                    self.data[col_name] = series.interpolate(method=method, **interp_kwargs)
                except Exception as e:
                    messagebox.showerror("Interpolation Error", f"Error interpolating {coord} with method '{method}': {e}")
                    return

        self.detect_outliers()
        self.show_marker_plot(self.current_marker)

        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])

        self.update_plot()

        self.selection_data['start'] = current_selection['start']
        self.selection_data['end'] = current_selection['end']
        self.highlight_selection()

        # Update edit button state if it exists
        if hasattr(self, 'edit_button'):
            self.edit_button.configure(fg_color="#555555")

    def interpolate_with_pattern(self):
        """
        Pattern-based interpolation using reference markers to interpolate target marker
        """
        try:
            print(f"\nStarting pattern-based interpolation:")
            print(f"Target marker to interpolate: {self.current_marker}")
            print(f"Reference markers: {list(self.pattern_markers)}")
            
            reference_markers = list(self.pattern_markers)
            if not reference_markers:
                print("Error: No reference markers selected")
                messagebox.showerror("Error", "Please select reference markers")
                return

            start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
            end_frame = int(max(self.selection_data['start'], self.selection_data['end']))
            print(f"Frame range for interpolation: {start_frame} to {end_frame}")
            
            # search for valid frames in entire dataset
            print("\nSearching for valid target marker data...")
            all_valid_frames = []
            for frame in range(len(self.data)):
                if not any(pd.isna(self.data.loc[frame, f'{self.current_marker}_{coord}']) 
                          for coord in ['X', 'Y', 'Z']):
                    all_valid_frames.append(frame)
            
            if not all_valid_frames:
                print("Error: No valid data found for target marker in entire dataset")
                messagebox.showerror("Error", "No valid data found for target marker in entire dataset")
                return
                
            print(f"Found {len(all_valid_frames)} valid frames for target marker")
            print(f"Valid frames range: {min(all_valid_frames)} to {max(all_valid_frames)}")
            
            # find the closest valid frame
            closest_frame = min(all_valid_frames, 
                              key=lambda x: min(abs(x - start_frame), abs(x - end_frame)))
            print(f"\nUsing frame {closest_frame} as reference frame")
            
            # Get initial positions using closest valid frame
            target_pos_init = np.array([
                self.data.loc[closest_frame, f'{self.current_marker}_X'],
                self.data.loc[closest_frame, f'{self.current_marker}_Y'],
                self.data.loc[closest_frame, f'{self.current_marker}_Z']
            ])
            print(f"Initial target position: {target_pos_init}")
            
            # Calculate initial distances and positions
            marker_distances = {}
            marker_positions_init = {}
            
            print("\nCalculating initial distances:")
            for ref_marker in reference_markers:
                ref_pos = np.array([
                    self.data.loc[closest_frame, f'{ref_marker}_X'],
                    self.data.loc[closest_frame, f'{ref_marker}_Y'],
                    self.data.loc[closest_frame, f'{ref_marker}_Z']
                ])
                marker_positions_init[ref_marker] = ref_pos
                marker_distances[ref_marker] = np.linalg.norm(target_pos_init - ref_pos)
                print(f"{ref_marker}:")
                print(f"  Initial position: {ref_pos}")
                print(f"  Distance from target: {marker_distances[ref_marker]:.3f}")
            
            # Interpolate missing frames
            print("\nStarting frame interpolation:")
            interpolated_count = 0
            frames = range(start_frame, end_frame + 1)
            for frame in frames:
                # Check if target marker needs interpolation
                if any(pd.isna(self.data.loc[frame, f'{self.current_marker}_{coord}']) 
                      for coord in ['X', 'Y', 'Z']):
                    
                    weighted_pos = np.zeros(3)
                    total_weight = 0
                    
                    # Use each reference marker to estimate position
                    for ref_marker in reference_markers:
                        current_ref_pos = np.array([
                            self.data.loc[frame, f'{ref_marker}_X'],
                            self.data.loc[frame, f'{ref_marker}_Y'],
                            self.data.loc[frame, f'{ref_marker}_Z']
                        ])
                        
                        # Calculate expected position based on initial distance
                        init_distance = marker_distances[ref_marker]
                        init_direction = target_pos_init - marker_positions_init[ref_marker]
                        init_unit_vector = init_direction / np.linalg.norm(init_direction)
                        
                        # Weight based on initial distance
                        weight = 1.0 / (init_distance + 1e-6)
                        weighted_pos += weight * (current_ref_pos + init_unit_vector * init_distance)
                        total_weight += weight
                    
                    # Calculate final interpolated position
                    interpolated_pos = weighted_pos / total_weight
                    
                    # Update target marker position
                    self.data.loc[frame, f'{self.current_marker}_X'] = interpolated_pos[0]
                    self.data.loc[frame, f'{self.current_marker}_Y'] = interpolated_pos[1]
                    self.data.loc[frame, f'{self.current_marker}_Z'] = interpolated_pos[2]
                    
                    interpolated_count += 1
                    
                    if frame % 10 == 0:
                        print(f"  Interpolated position: {interpolated_pos}")
                
                elif frame % 10 == 0:
                    print(f"\nSkipping frame {frame} (valid data exists)")
            
            print(f"\nInterpolation completed successfully")
            print(f"Total frames interpolated: {interpolated_count}")
            
            # end pattern-based mode and initialize
            self.pattern_selection_mode = False
            self.pattern_markers.clear()
            
            # update UI
            self.update_plot()
            self.show_marker_plot(self.current_marker)
            
        except Exception as e:
            print(f"\nFATAL ERROR during interpolation: {e}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Interpolation Error", f"Error during pattern-based interpolation: {str(e)}")
        finally:
            # reset mouse events and UI state
            print("\nResetting mouse events and UI state")
            self.disconnect_mouse_events()
            self.connect_mouse_events()

    def on_pattern_selection_confirm(self):
        """Process pattern selection confirmation"""
        try:
            print("\nPattern selection confirmation:")
            print(f"Selected markers: {self.pattern_markers}")
            
            if not self.pattern_markers:
                print("Error: No markers selected")
                messagebox.showwarning("No Selection", "Please select at least one pattern marker")
                return
            
            print("Starting interpolation")
            self.interpolate_selected_data()
            
            # pattern selection window is closed in interpolate_with_pattern
            
        except Exception as e:
            print(f"Error in pattern selection confirmation: {e}")
            traceback.print_exc()
            
            # initialize related variables if error occurs
            if hasattr(self, 'pattern_window'):
                delattr(self, 'pattern_window')
            self._selected_markers_list = None

    def restore_original_data(self):
        if self.original_data is not None:
            self.data = self.original_data.copy(deep=True)
            self.detect_outliers()
            self.show_marker_plot(self.current_marker)
            self.update_plot()
            
            # Update edit button state if it exists
            if hasattr(self, 'edit_button'):
                self.edit_button.configure(fg_color="#555555")
                
            print("Data has been restored to the original state.")
        else:
            messagebox.showinfo("Restore Data", "No original data to restore.")

    def toggle_coordinates(self):
        """Toggle between Z-up and Y-up coordinate systems."""
        if self.data is None:
            return

        self.is_z_up = not self.is_z_up
        self.coord_button.configure(text="Switch to Y-up" if self.is_z_up else "Switch to Z-up")

        # Redraw static elements and coordinate axes
        self._draw_static_elements()
        self._update_coordinate_axes()

        # Update the plot with new data
        self.update_plot()
        self._draw_static_elements()
        self._update_coordinate_axes()

        # Update the plot with new data
        self.update_plot()

    def toggle_trajectory(self):
        """Toggle the visibility of marker trajectories"""
        self.show_trajectory = self.trajectory_handler.toggle_trajectory()
        self.trajectory_button.configure(text="Hide Trajectory" if self.show_trajectory else "Show Trajectory")
        self.update_plot()


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

                        if abs(current_length - prev_length) / prev_length > 0.2:
                            self.outliers[pair[0]][frame] = True
                            self.outliers[pair[1]][frame] = True

                except KeyError:
                    continue

    def prev_frame(self):
        """Move to the previous frame when left arrow key is pressed."""
        if self.data is not None and self.frame_idx > 0:
            self.frame_idx -= 1
            self.update_plot()
            self.update_timeline()
            
            # Update marker graph vertical line if it exists
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata([self.frame_idx, self.frame_idx])
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()
            # self.update_frame_counter()

    def next_frame(self):
        """Move to the next frame when right arrow key is pressed."""
        if self.data is not None and self.frame_idx < self.num_frames - 1:
            self.frame_idx += 1
            self.update_plot()
            self.update_timeline()
            
            # Update marker graph vertical line if it exists
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata([self.frame_idx, self.frame_idx])
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()
            # self.update_frame_counter()

    def toggle_marker_names(self):
        self.show_names = not self.show_names
        self.names_button.configure(text="Show Names" if not self.show_names else "Hide Names")
        self.update_plot()

    def reset_main_view(self):
        if self.data_limits:
            self.ax.set_xlim(self.data_limits['x'])
            self.ax.set_ylim(self.data_limits['y'])
            self.ax.set_zlim(self.data_limits['z'])
            self.ax.set_box_aspect([1,1,1])  # Force equal aspect ratio
            self.canvas.draw()

    def reset_graph_view(self):
        if hasattr(self, 'marker_axes') and hasattr(self, 'initial_graph_limits'):
            for ax, limits in zip(self.marker_axes, self.initial_graph_limits):
                ax.set_xlim(limits['x'])
                ax.set_ylim(limits['y'])
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

    def toggle_animation(self):
        if not self.data is None:
            if self.is_playing:
                self.pause_animation()
            else:
                self.play_animation()

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
        # 재생 중이었다면 멈춤
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.configure(text="▶")
            if self.animation_job:
                self.after_cancel(self.animation_job)
                self.animation_job = None
        
        # 첫 프레임으로 되돌아감
        self.frame_idx = 0
        self.update_plot()
        self.update_timeline()
        self.stop_button.configure(state='disabled')

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

            self.update_plot()
            self.update_timeline()

            # remove speed slider related code and use default FPS
            base_fps = float(self.fps_var.get())
            delay = int(1000 / base_fps)
            delay = max(1, delay)

            self.animation_job = self.after(delay, self.animate)

    def update_fps_label(self):
        fps = self.fps_var.get()
        if hasattr(self, 'fps_label'):
            self.fps_label.configure(text=f"FPS: {fps}")

    def save_as(self):
        if self.data is None:
            messagebox.showinfo("No Data", "There is no data to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".trc",
            filetypes=[("TRC files", "*.trc"), ("C3D files", "*.c3d"), ("All files", "*.*")]
        )

        if not file_path:
            return

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.trc':
                save_to_trc(file_path, self.data, self.fps_var.get(), self.marker_names, self.num_frames)
            elif file_extension == '.c3d':
                save_to_c3d(file_path, self.data, self.fps_var.get(), self.marker_names, self.num_frames)
            else:
                messagebox.showerror("Unsupported Format", "Unsupported file format.")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {e}")

    def update_frame_from_timeline(self, x_pos):
        if x_pos is not None and self.data is not None:
            frame = int(max(0, min(x_pos, self.num_frames - 1)))
            self.frame_idx = frame
            self.update_plot()
            # self.update_frame_counter()
            self.update_timeline()

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

    def on_filter_type_change(self, choice):
        if self.current_params_frame:
            self.current_params_frame.destroy()
        
        self.current_params_frame = ctk.CTkFrame(self.filter_params_frame)
        self.current_params_frame.pack(side='left', padx=5)
        
        if choice == 'butterworth':
            order_label = ctk.CTkLabel(self.current_params_frame, text="Order:")
            order_label.pack(side='left', padx=2)
            order_entry = ctk.CTkEntry(self.current_params_frame, 
                                     textvariable=self.parent.filter_params['butterworth']['order'],
                                     width=50)
            order_entry.pack(side='left', padx=2)

            cutoff_label = ctk.CTkLabel(self.current_params_frame, text="Cutoff (Hz):")
            cutoff_label.pack(side='left', padx=2)
            cutoff_entry = ctk.CTkEntry(self.current_params_frame,
                                      textvariable=self.parent.filter_params['butterworth']['cut_off_frequency'],
                                      width=50)
            cutoff_entry.pack(side='left', padx=2)

        elif choice == 'kalman':
            trust_label = ctk.CTkLabel(self.current_params_frame, text="Trust Ratio:")
            trust_label.pack(side='left', padx=2)
            trust_entry = ctk.CTkEntry(self.current_params_frame,
                                     textvariable=self.parent.filter_params['kalman']['trust_ratio'],
                                     width=50)
            trust_entry.pack(side='left', padx=2)

            smooth_label = ctk.CTkLabel(self.current_params_frame, text="Smooth:")
            smooth_label.pack(side='left', padx=2)
            smooth_entry = ctk.CTkEntry(self.current_params_frame,
                                      textvariable=self.parent.filter_params['kalman']['smooth'],
                                      width=50)
            smooth_entry.pack(side='left', padx=2)

        elif choice == 'gaussian':
            kernel_label = ctk.CTkLabel(self.current_params_frame, text="Sigma Kernel:")
            kernel_label.pack(side='left', padx=2)
            kernel_entry = ctk.CTkEntry(self.current_params_frame,
                                      textvariable=self.parent.filter_params['gaussian']['sigma_kernel'],
                                      width=50)
            kernel_entry.pack(side='left', padx=2)

        elif choice == 'LOESS':
            values_label = ctk.CTkLabel(self.current_params_frame, text="Values Used:")
            values_label.pack(side='left', padx=2)
            values_entry = ctk.CTkEntry(self.current_params_frame,
                                      textvariable=self.parent.filter_params['LOESS']['nb_values_used'],
                                      width=50)
            values_entry.pack(side='left', padx=2)

        elif choice == 'median':
            kernel_label = ctk.CTkLabel(self.current_params_frame, text="Kernel Size:")
            kernel_label.pack(side='left', padx=2)
            kernel_entry = ctk.CTkEntry(self.current_params_frame,
                                      textvariable=self.parent.filter_params['median']['kernel_size'],
                                      width=50)
            kernel_entry.pack(side='left', padx=2)

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
            print(f"Error updating markers list: {e}")
            # initialize related variables if error occurs
            if hasattr(self, 'pattern_window'):
                delattr(self, 'pattern_window')
            self._selected_markers_list = None

    def clear_pattern_selection(self):
        """Initialize pattern markers"""
        self.pattern_markers.clear()
        self.update_selected_markers_list()
        self.update_plot()

    def on_marker_selected(self, marker_name):
        """Handle marker selection event"""
        self.current_marker = marker_name
        if hasattr(self, 'trajectory_handler'):
            self.trajectory_handler.set_current_marker(marker_name)
        self.update_plot()

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

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = TRCViewer()
    app.mainloop()
