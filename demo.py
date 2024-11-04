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

# Interactive mode on
plt.ion()
matplotlib.use('TkAgg')

def read_data_from_c3d(c3d_file_path):
    """
    Read data from a C3D file and return header lines, data frame, marker names, and frame rate.
    """
    try:
        import c3d
        with open(c3d_file_path, 'rb') as f:
            reader = c3d.Reader(f)
            point_labels = reader.point_labels
            frame_rate = reader.header.frame_rate
            first_frame = reader.header.first_frame
            last_frame = reader.header.last_frame

            point_labels = [label.strip() for label in point_labels if label.strip()]
            point_labels = list(dict.fromkeys(point_labels))

            frames = []
            times = []
            marker_data = {label: {'X': [], 'Y': [], 'Z': []} for label in point_labels}

            for i, points, analog in reader.read_frames():
                frames.append(i)
                times.append(i / frame_rate)
                points_meters = points[:, :3] / 1000.0

                for j, label in enumerate(point_labels):
                    if j < len(points_meters):
                        marker_data[label]['X'].append(points_meters[j, 0])
                        marker_data[label]['Y'].append(points_meters[j, 1])
                        marker_data[label]['Z'].append(points_meters[j, 2])

            data_dict = {'Frame#': frames, 'Time': times}

            for label in point_labels:
                if label in marker_data:
                    data_dict[f'{label}_X'] = marker_data[label]['X']
                    data_dict[f'{label}_Y'] = marker_data[label]['Y']
                    data_dict[f'{label}_Z'] = marker_data[label]['Z']

            data = pd.DataFrame(data_dict)

            header_lines = [
                f"PathFileType\t4\t(X/Y/Z)\t{c3d_file_path}\n",
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
                f"{frame_rate}\t{frame_rate}\t{len(frames)}\t{len(point_labels)}\tm\t{frame_rate}\t{first_frame}\t{last_frame}\n",
                "\t".join(['Frame#', 'Time'] + point_labels) + "\n",
                "\t".join(['', ''] + ['X\tY\tZ' for _ in point_labels]) + "\n"
            ]

            return header_lines, data, point_labels, frame_rate

    except Exception as e:
        raise Exception(f"Error reading C3D file: {str(e)}")

def read_data_from_trc(trc_file_path):
    """
    Read data from a TRC file and return header lines, data frame, marker names, and frame rate.
    """
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()

    header_lines = lines[:5]

    try:
        frame_rate = float(header_lines[2].split('\t')[0])
    except (IndexError, ValueError):
        frame_rate = 30.0

    marker_names_line = lines[3].strip().split('\t')[2:]

    marker_names = []
    for name in marker_names_line:
        if name.strip() and name not in marker_names:
            marker_names.append(name.strip())

    column_names = ['Frame#', 'Time']
    for marker in marker_names:
        column_names.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])

    data = pd.read_csv(trc_file_path, sep='\t', skiprows=6, names=column_names)

    return header_lines, data, marker_names, frame_rate

class TRCViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TRC Viewer")
        self.geometry("1200x1000")

        self.data = None
        self.original_data = None
        self.marker_names = None
        self.num_frames = 0
        self.frame_idx = 0
        self.canvas = None
        self.selection_in_progress = False

        self.marker_last_pos = None
        self.marker_pan_enabled = False
        self.marker_canvas = None
        self.marker_axes = []
        self.marker_lines = []

        self.view_limits = None
        self.is_z_up = True
        self.outliers = {}

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
        self.fps_var = ctk.StringVar(value="30")

        self.bind('<space>', lambda e: self.toggle_animation())
        self.bind('<Return>', lambda e: self.toggle_animation())
        self.bind('<Escape>', lambda e: self.stop_animation())
        self.bind('<Left>', lambda e: self.prev_frame())
        self.bind('<Right>', lambda e: self.next_frame())

        self.create_widgets()

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
        self.main_content.pack(fill='both', expand=True, padx=10)

        self.viewer_frame = ctk.CTkFrame(self.main_content)
        self.viewer_frame.pack(side='left', fill='both', expand=True)

        self.graph_frame = ctk.CTkFrame(self.main_content)
        self.graph_frame.pack_forget()

        viewer_top_frame = ctk.CTkFrame(self.viewer_frame)
        viewer_top_frame.pack(fill='x', pady=(5, 0))

        self.title_label = ctk.CTkLabel(viewer_top_frame, text="", font=("Arial", 14))
        self.title_label.pack(side='left', expand=True)

        canvas_container = ctk.CTkFrame(self.viewer_frame)
        canvas_container.pack(fill='both', expand=True)

        self.canvas_frame = ctk.CTkFrame(canvas_container)
        self.canvas_frame.pack(side='left', fill='both', expand=True)

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.animation_frame = ctk.CTkFrame(self.control_frame)
        self.animation_frame.pack(fill='x', padx=5, pady=(5, 0))

        control_style = {
            "width": 30,
            "fg_color": "#333333",
            "hover_color": "#444444"
        }

        self.play_pause_button = ctk.CTkButton(
            self.animation_frame,
            text="▶",
            command=self.toggle_animation,
            **control_style
        )
        self.play_pause_button.pack(side='left', padx=5)

        self.stop_button = ctk.CTkButton(
            self.animation_frame,
            text="■",
            command=self.stop_animation,
            state='disabled',
            **control_style
        )
        self.stop_button.pack(side='left', padx=5)

        speed_frame = ctk.CTkFrame(self.animation_frame, fg_color="transparent")
        speed_frame.pack(side='left', fill='x', expand=True, padx=10)

        self.speed_label = ctk.CTkLabel(speed_frame, text="Speed: 1.0x")
        self.speed_label.pack(side='left', padx=5)

        self.speed_var = ctk.DoubleVar(value=1.0)
        self.speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.1,
            to=3.0,
            number_of_steps=99,
            variable=self.speed_var,
            command=self.update_playback_speed
        )
        self.speed_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.loop_var = ctk.BooleanVar(value=False)
        self.loop_checkbox = ctk.CTkCheckBox(
            self.animation_frame,
            text="Loop",
            variable=self.loop_var
        )
        self.loop_checkbox.pack(side='left', padx=5)

        self.frame_counter = ctk.CTkLabel(
            self.animation_frame,
            text="Frame: 0/0"
        )
        self.frame_counter.pack(side='right', padx=5)

        self.bottom_frame = ctk.CTkFrame(self.control_frame)
        self.bottom_frame.pack(fill='x', padx=5)

        self.prev_button = ctk.CTkButton(
            self.bottom_frame,
            text="◀",
            width=30,
            command=self.prev_frame
        )
        self.prev_button.pack(side='left', padx=5)

        self.frame_slider = ctk.CTkSlider(
            self.bottom_frame,
            from_=0,
            to=1,
            command=self.update_frame
        )
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.next_button = ctk.CTkButton(
            self.bottom_frame,
            text="▶",
            width=30,
            command=self.next_frame
        )
        self.next_button.pack(side='left', padx=5)

        self.marker_label = ctk.CTkLabel(self, text="")
        self.marker_label.pack(pady=5)

        if self.canvas:
            self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_model_change(self, choice):
        """모델 선택이 변경될 때 호출되는 메소드"""
        try:
            # 현재 프레임 저장
            current_frame = self.frame_idx
            
            # 모델 업데이트
            self.current_model = self.available_models[choice]
            
            # 스켈레톤 설정 업데이트
            if self.current_model is None:
                self.skeleton_pairs = []
                self.show_skeleton = False
            else:
                self.show_skeleton = True
                self.update_skeleton_pairs()

            # 기존 스켈레톤 라인 제거
            if hasattr(self, 'skeleton_lines'):
                for line in self.skeleton_lines:
                    if line in self.ax.lines:
                        self.ax.lines.remove(line)
                self.skeleton_lines = []

            # 새로운 스켈레톤 라인 초기화
            if self.show_skeleton:
                for _ in self.skeleton_pairs:
                    line = Line3D([], [], [], color='gray', alpha=0.5)
                    self.ax.add_line(line)
                    self.skeleton_lines.append(line)

            # 현재 프레임 데이터로 플롯 업데이트
            self.update_plot()
            self.update_frame(current_frame)

            # 캔버스 새로고침
            if hasattr(self, 'canvas'):
                self.canvas.draw()
                self.canvas.flush_events()

        except Exception as e:
            print(f"Error in on_model_change: {e}")
            import traceback
            traceback.print_exc()

    def update_skeleton_pairs(self):
        """스켈레톤 페어 업데이트"""
        self.skeleton_pairs = []
        if self.current_model is not None:
            for node in self.current_model.descendants:
                if node.parent:
                    parent_name = node.parent.name
                    node_name = node.name
                    
                    # 마커 이름이 데이터에 있는지 확인
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

                self.frame_slider.configure(to=self.num_frames - 1)
                self.frame_idx = 0
                self.frame_slider.set(0)

                self.current_model = self.available_models[self.model_var.get()]
                self.update_skeleton_pairs()
                self.detect_outliers()

                self.create_plot()

                self.reset_main_view()

                self.update_plot()
                self.update_frame_counter()

                if hasattr(self, 'canvas'):
                    self.canvas.draw()
                    self.canvas.flush_events()

                self.play_pause_button.configure(state='normal')
                self.speed_slider.configure(state='normal')
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

            if hasattr(self, 'canvas') and self.canvas and hasattr(self.canvas, 'get_tk_widget'):
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

            self.frame_slider.set(0)
            self.frame_slider.configure(to=1)

            self.title_label.configure(text="")
            self.show_names = False
            self.show_skeleton = True
            self.current_file = None

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
        self.fig = plt.Figure(facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')

        self._setup_plot_style()

        self._draw_static_elements()

        self._initialize_dynamic_elements()

        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def _setup_plot_style(self):
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')

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

        # Define grid based on the coordinate system
        if self.is_z_up:
            # Grid on the X-Y plane at Z=0
            for i in range(grid_divisions):
                line1, = self.ax.plot(x, [y[i]] * grid_divisions, [0] * grid_divisions, 'gray', alpha=0.2)
                line2, = self.ax.plot([x[i]] * grid_divisions, y, [0] * grid_divisions, 'gray', alpha=0.2)
                self.grid_lines.extend([line1, line2])
        else:
            # Grid on the X-Z plane at Y=0
            for i in range(grid_divisions):
                line1, = self.ax.plot(x, [0] * grid_divisions, [y[i]] * grid_divisions, 'gray', alpha=0.2)
                line2, = self.ax.plot([x[i]] * grid_divisions, [0] * grid_divisions, y, 'gray', alpha=0.2)
                self.grid_lines.extend([line1, line2])


    def _initialize_dynamic_elements(self):
        self._update_coordinate_axes()

        if hasattr(self, 'markers_scatter'):
            self.markers_scatter.remove()
        if hasattr(self, 'selected_marker_scatter'):
            self.selected_marker_scatter.remove()

        self.markers_scatter = self.ax.scatter([], [], [], c='white', s=50, picker=5)
        self.selected_marker_scatter = self.ax.scatter([], [], [], c='yellow', s=70)

        if hasattr(self, 'skeleton_lines'):
            for line in self.skeleton_lines:
                line.remove()
        self.skeleton_lines = []

        if hasattr(self, 'skeleton_pairs') and self.skeleton_pairs:
            for _ in self.skeleton_pairs:
                line = Line3D([], [], [], color='gray', alpha=0.5)
                self.ax.add_line(line)
                self.skeleton_lines.append(line)

        if hasattr(self, 'marker_labels'):
            for label in self.marker_labels:
                label.remove()
        self.marker_labels = []

    def _update_coordinate_axes(self):
        """Update coordinate axes and labels based on the coordinate system."""
        axis_length = 0.5
        colors = {'x': 'red', 'y': 'yellow', 'z': 'blue'}

        # Clear existing axes and labels (if any)
        if hasattr(self, 'coordinate_axes'):
            for line in self.coordinate_axes:
                line.remove()
        self.coordinate_axes = []

        if hasattr(self, 'axis_labels'):
            for label in self.axis_labels:
                label.remove()
        self.axis_labels = []

        # Define axes and label positions based on the coordinate system
        if self.is_z_up:
            # Z-up coordinate system
            axes = [
                ([0, axis_length], [0, 0], [0, 0], 'x'),  # X-axis
                ([0, 0], [0, axis_length], [0, 0], 'y'),  # Y-axis
                ([0, 0], [0, 0], [0, axis_length], 'z')   # Z-axis
            ]
            label_positions = {
                'x': (axis_length + 0.1, 0, 0),
                'y': (0, axis_length + 0.1, 0),
                'z': (0, 0, axis_length + 0.1)
            }
        else:
            # Y-up coordinate system
            axes = [
                ([0, axis_length], [0, 0], [0, 0], 'x'),  # X-axis
                ([0, 0], [0, 0], [0, axis_length], 'y'),  # Y-axis
                ([0, 0], [axis_length, 0], [0, 0], 'z')   # Z-axis
            ]
            label_positions = {
                'x': (axis_length + 0.1, 0, 0),
                'y': (0, 0, axis_length + 0.1),
                'z': (0, axis_length + 0.1, 0)
            }

        # Plot axes and add labels at the specified positions
        for x, y, z, axis in axes:
            line, = self.ax.plot(x, y, z, color=colors[axis], alpha=0.8, linewidth=2)
            self.coordinate_axes.append(line)

            label_x, label_y, label_z = label_positions[axis]
            label = self.ax.text(label_x, label_y, label_z, axis.upper(), color=colors[axis], fontsize=12)
            self.axis_labels.append(label)



    def update_plot(self):
        if self.canvas is None:
            return

        prev_elev = self.ax.elev
        prev_azim = self.ax.azim
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()
        prev_zlim = self.ax.get_zlim()

        positions = []
        selected_position = []
        marker_positions = {}
        valid_markers = []

        for marker in self.marker_names:
            try:
                x = self.data.loc[self.frame_idx, f'{marker}_X']
                y = self.data.loc[self.frame_idx, f'{marker}_Y']
                z = self.data.loc[self.frame_idx, f'{marker}_Z']

                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue

                if self.is_z_up:
                    pos = [x, y, z]
                else:
                    pos = [x, z, y]

                marker_positions[marker] = np.array(pos)

                if hasattr(self, 'current_marker') and marker == self.current_marker:
                    selected_position.append(pos)
                else:
                    positions.append(pos)

                valid_markers.append(marker)
            except KeyError:
                continue

        self.valid_markers = valid_markers
        positions = np.array(positions) if positions else np.empty((0, 3))
        selected_position = np.array(selected_position) if selected_position else np.empty((0, 3))

        if len(positions) > 0:
            self.markers_scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        else:
            self.markers_scatter._offsets3d = ([], [], [])

        if len(selected_position) > 0:
            self.selected_marker_scatter._offsets3d = (
                selected_position[:, 0],
                selected_position[:, 1],
                selected_position[:, 2]
            )
        else:
            self.selected_marker_scatter._offsets3d = ([], [], [])

        if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_lines'):
            for line, pair in zip(self.skeleton_lines, self.skeleton_pairs):
                if pair[0] in marker_positions and pair[1] in marker_positions:
                    p1 = marker_positions[pair[0]]
                    p2 = marker_positions[pair[1]]

                    is_outlier = (
                        self.outliers[pair[0]][self.frame_idx] or
                        self.outliers[pair[1]][self.frame_idx]
                    )

                    line.set_data_3d(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]]
                    )

                    line.set_color('red' if is_outlier else 'gray')
                    line.set_alpha(0.8 if is_outlier else 0.5)
                    line.set_linewidth(3 if is_outlier else 2)

        if self.show_names:
            for label in self.marker_labels:
                label.remove()
            self.marker_labels.clear()

            for marker in valid_markers:
                pos = marker_positions[marker]
                color = 'yellow' if (hasattr(self, 'current_marker') and marker == self.current_marker) else 'white'
                label = self.ax.text(pos[0], pos[1], pos[2], marker, color=color, fontsize=8)
                self.marker_labels.append(label)

        self.ax.view_init(elev=prev_elev, azim=prev_azim)
        self.ax.set_xlim(prev_xlim)
        self.ax.set_ylim(prev_ylim)
        self.ax.set_zlim(prev_zlim)

        self.canvas.draw()

    def connect_mouse_events(self):
        if self.canvas:
            self.canvas.mpl_disconnect('scroll_event')
            self.canvas.mpl_disconnect('pick_event')
            self.canvas.mpl_disconnect('button_press_event')
            self.canvas.mpl_disconnect('button_release_event')
            self.canvas.mpl_disconnect('motion_notify_event')

            self.canvas.mpl_connect('scroll_event', self.on_scroll)
            self.canvas.mpl_connect('pick_event', self.on_pick)
            self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def update_frame(self, value):
        if self.data is not None:
            self.frame_idx = int(float(value))
            self.update_plot()
            self.update_frame_counter()

            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata([self.frame_idx, self.frame_idx])
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()

    def on_pick(self, event):
        try:
            if event.mouseevent.button != 3:
                return

            current_view = {
                'elev': self.ax.elev,
                'azim': self.ax.azim,
                'xlim': self.ax.get_xlim(),
                'ylim': self.ax.get_ylim(),
                'zlim': self.ax.get_zlim()
            }

            if not hasattr(self, 'valid_markers') or not self.valid_markers:
                return

            ind = event.ind[0]
            if ind >= len(self.valid_markers):
                return

            self.current_marker = self.valid_markers[ind]

            if event.mouseevent.button == 3:
                if self.current_marker in self.marker_names:
                    self.show_marker_plot(self.current_marker)
                else:
                    return

            self.update_plot()

            self.ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
            self.ax.set_xlim(current_view['xlim'])
            self.ax.set_ylim(current_view['ylim'])
            self.ax.set_zlim(current_view['zlim'])
            self.canvas.draw()

        except Exception as e:
            print(f"Error in on_pick: {str(e)}")

        finally:
            self.connect_mouse_events()

    def show_marker_plot(self, marker_name):
        prev_interp_method = None
        prev_order = None
        if hasattr(self, 'interp_method_var'):
            prev_interp_method = self.interp_method_var.get()
        if hasattr(self, 'order_var'):
            prev_order = self.order_var.get()

        if not self.graph_frame.winfo_ismapped():
            self.graph_frame.pack(side='right', fill='both', expand=True)

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

            line = ax.axvline(x=self.frame_idx, color='red', linestyle='--')
            self.marker_lines.append(line)
            self.marker_axes.append(ax)

            if len(outlier_frames) > 0:
                ax.legend(facecolor='black',
                          labelcolor='white',
                          loc='upper right',
                          bbox_to_anchor=(1.0, 1.0))

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

        self.marker_canvas.mpl_connect('scroll_event', self.on_marker_scroll)
        self.marker_canvas.mpl_connect('button_press_event', self.on_marker_mouse_press)
        self.marker_canvas.mpl_connect('button_release_event', self.on_marker_mouse_release)
        self.marker_canvas.mpl_connect('motion_notify_event', self.on_marker_mouse_move)

        button_frame = ctk.CTkFrame(self.graph_frame)
        button_frame.pack(fill='x', padx=5, pady=5)

        reset_button = ctk.CTkButton(button_frame,
                                     text="Reset View",
                                     command=self.reset_graph_view,
                                     width=80,
                                     fg_color="#333333",
                                     hover_color="#444444")
        reset_button.pack(side='right', padx=5)

        self.edit_button = ctk.CTkButton(button_frame,
                                         text="Edit",
                                         command=self.toggle_edit_menu,
                                         width=80,
                                         fg_color="#333333",
                                         hover_color="#444444")
        self.edit_button.pack(side='right', padx=5)

        self.edit_menu = ctk.CTkFrame(self.graph_frame)

        edit_buttons = [
            ("Delete", self.delete_selected_data),
            ("Interpolate", self.interpolate_selected_data),
            ("Restore", self.restore_original_data),
            ("Cancel", lambda: self.edit_menu.pack_forget())
        ]

        for text, command in edit_buttons:
            btn = ctk.CTkButton(self.edit_menu,
                                text=text,
                                command=command,
                                width=80,
                                fg_color="#333333",
                                hover_color="#444444")
            btn.pack(side='left', padx=5, pady=5)

        self.interp_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
                               'polynomial', 'spline', 'barycentric', 'krogh', 'pchip', 'akima', 'from_derivatives']
        self.interp_method_var = ctk.StringVar(value='linear' if prev_interp_method is None else prev_interp_method)
        interp_label = ctk.CTkLabel(self.edit_menu, text="Interpolation Method:")
        interp_label.pack(side='left', padx=5)
        self.interp_combo = ctk.CTkComboBox(self.edit_menu,
                                            values=self.interp_methods,
                                            variable=self.interp_method_var,
                                            command=self.on_interp_method_change)
        self.interp_combo.pack(side='left', padx=5)

        self.order_var = ctk.IntVar(value=2 if prev_order is None else prev_order)
        self.order_entry = ctk.CTkEntry(self.edit_menu, textvariable=self.order_var, width=50)
        self.order_label = ctk.CTkLabel(self.edit_menu, text="Order:")
        self.order_label.pack(side='left', padx=5)
        self.order_entry.pack(side='left', padx=5)

        if prev_interp_method in ['polynomial', 'spline']:
            self.order_entry.configure(state='normal')
        else:
            self.order_entry.configure(state='disabled')

        self.selection_data = {
            'start': None,
            'end': None,
            'rects': []
        }

        self.connect_mouse_events()

    def on_interp_method_change(self, choice):
        if choice in ['polynomial', 'spline']:
            self.order_entry.configure(state='normal')
        else:
            self.order_entry.configure(state='disabled')

    def toggle_edit_menu(self):
        if self.edit_menu.winfo_ismapped():
            self.edit_menu.pack_forget()
            self.edit_button.configure(fg_color="#333333")
            self.clear_selection()
        else:
            self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
            self.edit_button.configure(fg_color="#555555")

    def clear_selection(self):
        if 'rects' in self.selection_data and self.selection_data['rects']:
            for rect in self.selection_data['rects']:
                rect.remove()
            self.selection_data['rects'] = []
        if hasattr(self, 'marker_canvas'):
            self.marker_canvas.draw_idle()
        self.selection_in_progress = False

    def on_marker_mouse_press(self, event):
        if event.inaxes is None:
            return

        if event.button == 2:
            self.marker_pan_enabled = True
            self.marker_last_pos = (event.xdata, event.ydata)
        elif event.button == 1 and hasattr(self, 'edit_menu') and self.edit_menu.winfo_ismapped():
            if event.xdata is not None:
                if self.selection_data.get('rects'):
                    start = min(self.selection_data['start'], self.selection_data['end'])
                    end = max(self.selection_data['start'], self.selection_data['end'])
                    if not (start <= event.xdata <= end):
                        self.clear_selection()
                        self.start_new_selection(event)
                else:
                    self.start_new_selection(event)

    def on_marker_mouse_release(self, event):
        if event.button == 2:
            self.marker_pan_enabled = False
            self.marker_last_pos = None
        elif event.button == 1 and hasattr(self, 'edit_menu') and self.edit_menu.winfo_ismapped():
            if self.selection_data.get('start') is not None and event.xdata is not None:
                self.selection_data['end'] = event.xdata
                self.selection_in_progress = False
                self.highlight_selection()

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

        self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
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

        self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
        self.edit_button.configure(fg_color="#555555")

    def restore_original_data(self):
        if self.original_data is not None:
            self.data = self.original_data.copy(deep=True)
            self.detect_outliers()
            self.show_marker_plot(self.current_marker)
            self.update_plot()
            self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
            self.edit_button.configure(fg_color="#555555")
            print("Data has been restored to the original state.")
        else:
            messagebox.showinfo("Restore Data", "No original data to restore.")

    def on_scroll(self, event):
        try:
            if event.inaxes != self.ax:
                return

            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            z_min, z_max = self.ax.get_zlim()

            scale_factor = 0.9 if event.button == 'up' else 1.1

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            x_range = (x_max - x_min) * scale_factor
            y_range = (y_max - y_min) * scale_factor
            z_range = (z_max - z_min) * scale_factor

            min_range = 1e-3
            max_range = 1e5

            x_range = max(min(x_range, max_range), min_range)
            y_range = max(min(y_range, max_range), min_range)
            z_range = max(min(z_range, max_range), min_range)

            self.ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            self.ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            self.ax.set_zlim(z_center - z_range / 2, z_center + z_range / 2)

            self.canvas.draw_idle()
        except Exception as e:
            print(f"Scroll event error: {e}")
            self.connect_mouse_events()

    def on_marker_scroll(self, event):
        if not event.inaxes:
            return

        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2

        scale_factor = 0.9 if event.button == 'up' else 1.1

        new_x_range = (x_max - x_min) * scale_factor
        new_y_range = (y_max - y_min) * scale_factor

        x_left = x_center - new_x_range * (x_center - x_min) / (x_max - x_min)
        x_right = x_center + new_x_range * (x_max - x_center) / (x_max - x_min)
        y_bottom = y_center - new_y_range * (y_center - y_min) / (y_max - y_min)
        y_top = y_center + new_y_range * (y_max - y_center) / (y_max - y_min)

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

        self.marker_canvas.draw_idle()

    def toggle_coordinates(self):
        """Toggle between Z-up and Y-up coordinate systems."""
        if self.data is None:
            return

        self.is_z_up = not self.is_z_up
        self.coord_button.configure(text="Switch to Y-up" if self.is_z_up else "Switch to Z-up")

        # Swap Y and Z data in self.data
        for marker in self.marker_names:
            y_col = f'{marker}_Y'
            z_col = f'{marker}_Z'
            self.data[y_col], self.data[z_col] = self.data[z_col].copy(), self.data[y_col].copy()

        # Redraw static elements and coordinate axes
        self._draw_static_elements()
        self._update_coordinate_axes()

        # Adjust camera view based on the coordinate system
        if self.is_z_up:
            self.ax.view_init(elev=30, azim=45)  # Z-up view
        else:
            self.ax.view_init(elev=0, azim=90)  # Y-up view for appropriate orientation

        # Update the plot with new data
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
        if self.data is not None and self.frame_idx > 0:
            self.frame_idx -= 1
            self.frame_slider.set(self.frame_idx)
            self.update_plot()
            self.update_frame_counter()

    def next_frame(self):
        if self.data is not None and self.frame_idx < self.num_frames - 1:
            self.frame_idx += 1
            self.frame_slider.set(self.frame_idx)
            self.update_plot()
            self.update_frame_counter()

    def toggle_marker_names(self):
        self.show_names = not self.show_names
        self.names_button.configure(text="Show Names" if not self.show_names else "Hide Names")
        self.update_plot()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.pan_enabled = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1:
            self.pan_enabled = False
            self.last_mouse_pos = None

    def on_mouse_move(self, event):
        if self.pan_enabled and event.xdata is not None and event.ydata is not None:
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]

            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            min_limit = -1e5
            max_limit = 1e5

            new_x_min = max(new_x_min, min_limit)
            new_x_max = min(new_x_max, max_limit)
            new_y_min = max(new_y_min, min_limit)
            new_y_max = min(new_y_max, max_limit)

            self.ax.set_xlim(new_x_min, new_x_max)
            self.ax.set_ylim(new_y_min, new_y_max)

            self.canvas.draw_idle()

            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_marker_mouse_move(self, event):
        if not hasattr(self, 'marker_pan_enabled'):
            self.marker_pan_enabled = False
        if not hasattr(self, 'selection_in_progress'):
            self.selection_in_progress = False

        if self.marker_pan_enabled and self.marker_last_pos:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self.marker_last_pos[0]
                dy = event.ydata - self.marker_last_pos[1]

                ax = event.inaxes
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                ax.set_xlim(x_min - dx, x_max - dx)
                ax.set_ylim(y_min - dy, y_max - dy)

                self.marker_last_pos = (event.xdata, event.ydata)

                self.marker_canvas.draw_idle()
        elif self.selection_in_progress and event.xdata is not None:
            self.selection_data['end'] = event.xdata

            start_x = min(self.selection_data['start'], self.selection_data['end'])
            width = abs(self.selection_data['end'] - self.selection_data['start'])

            for rect in self.selection_data['rects']:
                rect.set_x(start_x)
                rect.set_width(width)

            self.marker_canvas.draw_idle()

    def reset_main_view(self):
        if hasattr(self, 'data_limits'):
            try:
                self.ax.view_init(elev=20, azim=45)

                if self.is_z_up:
                    self.ax.set_xlim(self.data_limits['x'])
                    self.ax.set_ylim(self.data_limits['y'])
                    self.ax.set_zlim(self.data_limits['z'])
                else:
                    self.ax.set_xlim(self.data_limits['x'])
                    self.ax.set_ylim(self.data_limits['z'])
                    self.ax.set_zlim(self.data_limits['y'])

                self.ax.grid(True)

                self.ax.set_box_aspect([1.0, 1.0, 1.0])

                self.canvas.draw()

                self.view_limits = {
                    'x': self.ax.get_xlim(),
                    'y': self.ax.get_ylim(),
                    'z': self.ax.get_zlim()
                }

            except Exception as e:
                print(f"Error resetting camera view: {e}")

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
            'rects': []
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
        self.is_playing = False
        self.play_pause_button.configure(text="▶")
        self.stop_button.configure(state='disabled')

        if self.animation_job:
            self.after_cancel(self.animation_job)
            self.animation_job = None

        self.frame_idx = 0
        self.frame_slider.set(0)
        self.update_plot()
        self.update_frame_counter()

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

            self.frame_slider.set(self.frame_idx)
            self.update_plot()
            self.update_frame_counter()

            base_fps = float(self.fps_var.get())
            delay = int(1000 / (self.playback_speed * base_fps))
            delay = max(1, delay)

            self.animation_job = self.after(delay, self.animate)

    def update_playback_speed(self, value):
        self.playback_speed = float(value)
        self.speed_label.configure(text=f"Speed: {self.playback_speed:.1f}x")

    def update_frame_counter(self):
        if self.data is not None:
            self.frame_counter.configure(
                text=f"Frame: {self.frame_idx}/{self.num_frames-1}"
            )

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
                self.save_to_trc(file_path)
            elif file_extension == '.c3d':
                self.save_to_c3d(file_path)
            else:
                messagebox.showerror("Unsupported Format", "Unsupported file format.")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {e}")

    def save_to_trc(self, file_path):
        header_lines = [
            "PathFileType\t4\t(X/Y/Z)\t{}\n".format(os.path.basename(file_path)),
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
            "{}\t{}\t{}\t{}\tm\t{}\t{}\t{}\n".format(
                self.fps_var.get(),
                self.fps_var.get(),
                self.num_frames,
                len(self.marker_names),
                self.fps_var.get(),
                1,
                self.num_frames
            ),
            "\t".join(['Frame#', 'Time'] + self.marker_names) + "\n",
            "\t".join(['', ''] + ['X\tY\tZ' for _ in self.marker_names]) + "\n"
        ]

        with open(file_path, 'w') as f:
            f.writelines(header_lines)
            self.data.to_csv(f, sep='\t', index=False, header=False)

        messagebox.showinfo("Save Successful", f"Data saved to {file_path}")

    def save_to_c3d(self, file_path):
        try:
            import c3d
        except ImportError:
            messagebox.showerror("c3d Library Missing", "Please install the 'c3d' library to save in C3D format.")
            return

        try:
            # Frame rate 설정
            frame_rate = float(self.fps_var.get())
            
            # Writer 객체 초기화 시 frame rate 설정
            writer = c3d.Writer(point_rate=frame_rate, analog_rate=0)

            # 마커 레이블 설정
            marker_labels = self.marker_names
            writer.set_point_labels(marker_labels)

            # 모든 프레임의 데이터를 한 번에 준비
            all_frames = []
            
            # 데이터 채우기
            for frame_idx in range(self.num_frames):
                # 각 프레임의 데이터 준비
                points = np.zeros((len(marker_labels), 5))  # 열 개수를 5로 변경
                    
                for i, marker in enumerate(marker_labels):
                    try:
                        x = self.data.loc[frame_idx, f'{marker}_X'] * 1000.0  # mm 단위로 변환
                        y = self.data.loc[frame_idx, f'{marker}_Y'] * 1000.0
                        z = self.data.loc[frame_idx, f'{marker}_Z'] * 1000.0

                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            points[i, :3] = [0.0, 0.0, 0.0]
                            points[i, 3] = -1.0  # Residual
                            points[i, 4] = 0    # Camera_Mask
                        else:
                            points[i, :3] = [x, y, z]
                            points[i, 3] = 0.0   # Residual
                            points[i, 4] = 0     # Camera_Mask (필요에 따라 변경 가능)
                    except Exception as e:
                        print(f"Error processing marker {marker} at frame {frame_idx}: {e}")
                        points[i, :3] = [0.0, 0.0, 0.0]
                        points[i, 3] = -1.0  # Residual
                        points[i, 4] = 0     # Camera_Mask

                # 디버깅 정보 출력
                if frame_idx % 100 == 0:
                    print(f"Frame {frame_idx}:")
                    print(f"Points shape: {points.shape}")

                # 프레임 데이터를 리스트에 추가 (아날로그 데이터는 빈 넘파이 배열로)
                all_frames.append((points, np.empty((0, 0))))

            # 모든 프레임을 한 번에 추가
            writer.add_frames(all_frames)

            # 파일 저장
            with open(file_path, 'wb') as h:
                writer.write(h)

            messagebox.showinfo("Save Successful", f"Data saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {str(e)}\n\nPlease check the console for more details.")
            print(f"Detailed error: {e}")



if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = TRCViewer()
    app.mainloop()
