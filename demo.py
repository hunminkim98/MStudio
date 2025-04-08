import os
import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from Pose2Sim.skeletons import *
from Pose2Sim.filtering import *
import matplotlib

from gui.TRCviewerWidgets import create_widgets
from gui.markerPlot import show_marker_plot
from gui.plotCreator import create_plot
from gui.filterUI import on_filter_type_change, build_filter_parameter_widgets

from utils.dataLoader import open_file
from utils.dataSaver import save_as
from utils.viewToggles import toggle_marker_names, toggle_trajectory, toggle_animation
from utils.viewReset import reset_main_view, reset_graph_view
from utils.dataProcessor import *
from utils.mouseHandler import MouseHandler


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
from importlib.metadata import version
# __version__ = version('')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"


# Interactive mode on
plt.ion()
matplotlib.use('TkAgg')


# General TODO:
# 1. Current TRCViewer is too long and complex. It needs to be refactored.
# 2. The code is not documented well and should be english.
# 3. Add information about the author and the version of the software.
# 4. project.toml file

class TRCViewer(ctk.CTk): 
    def __init__(self):
        super().__init__()
        self.title("MarkerStudio")
        self.geometry("1920x1080") # May be changed based on the user's screen size

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
            print(f"Error calculating data limits: {e}")
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
    ###### Coordinate system manager  #######
    #########################################

    # TODO for coordinate system manager:
    # 1. Add a X-up coordinate system
    # 2. Add a left-handed coordinate system

    def toggle_coordinates(self):
        """Toggle between Z-up and Y-up coordinate systems"""
        # 이전 상태 저장
        previous_state = self.is_z_up
        
        # 좌표계 상태 전환
        self.is_z_up = not self.is_z_up
        
        # 버튼 텍스트 업데이트
        button_text = "Switch to Y-up" if self.is_z_up else "Switch to Z-up"
        if hasattr(self, 'coord_button'):
            self.coord_button.configure(text=button_text)
            self.update_idletasks()  # UI 즉시 갱신
        
        # 좌표계 설정 변경
        self.coordinate_system = "z-up" if self.is_z_up else "y-up"
        
        # OpenGL 렌더러에 좌표계 변경 전달
        if hasattr(self, 'gl_renderer'):
            if hasattr(self.gl_renderer, 'set_coordinate_system'):
                self.gl_renderer.set_coordinate_system(self.is_z_up)
            
            # 화면 강제 갱신 요청 - 약간의 지연 후 실행
            self.after(50, self._force_update_opengl)


    def _force_update_opengl(self):
        """OpenGL 렌더러의 화면을 강제로 갱신합니다."""
        if not hasattr(self, 'gl_renderer'):
            return
            
        try:
            # 현재 프레임을 다시 설정하여 강제 갱신
            if self.data is not None:
                self.gl_renderer.set_frame_data(
                    data=self.data,
                    frame_idx=self.frame_idx,
                    marker_names=self.marker_names,
                    current_marker=self.current_marker,
                    show_marker_names=self.show_names,
                    show_trajectory=self.show_trajectory,
                    coordinate_system="z-up" if self.is_z_up else "y-up",
                    skeleton_pairs=self.skeleton_pairs if hasattr(self, 'skeleton_pairs') else None
                )
                
                # 화면 갱신 명령
                self.gl_renderer._force_redraw()
                
                # 한 번 더 갱신 요청 (보험)
                self.after(100, lambda: self.gl_renderer.redraw())
                
        except Exception:
            pass


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
    ############ Analysis Mode ##############
    #########################################

    # TODO for analysis mode:
    # 1. Distance (and dotted line?) visualization between two selected markers
    # 2. Joint angle (and arc)visualization for three selected markers

    def toggle_analysis_mode(self):
        """Toggles the analysis mode on and off."""
        self.is_analysis_mode = not self.is_analysis_mode
        if self.is_analysis_mode:
            print("Analysis mode activated.")
            # Potentially change button appearance or disable other interactions
            self.analysis_button.configure(fg_color="#00A6FF") # Example: Highlight button
        else:
            print("Analysis mode deactivated.")
            # Restore button appearance and re-enable other interactions
            button_style = {
                "fg_color": "#333333",
                "hover_color": "#444444"
            }
            self.analysis_button.configure(**button_style) # Example: Restore default style
        
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

            # OpenGL 렌더러에 스켈레톤 쌍 정보 전달
            if hasattr(self, 'gl_renderer'):
                self.gl_renderer.set_skeleton_pairs(self.skeleton_pairs)
                self.gl_renderer.set_show_skeleton(self.show_skeleton)

            # Re-detect outliers with new skeleton pairs
            self.detect_outliers()
            
            # OpenGL 렌더러에 이상치 정보 전달
            if hasattr(self, 'gl_renderer') and hasattr(self, 'outliers'):
                self.gl_renderer.set_outliers(self.outliers)

            # Update the plot with the current frame data
            self.update_plot()
            self.update_frame(current_frame)

            # If a marker is currently selected, update its plot
            if hasattr(self, 'current_marker') and self.current_marker:
                self.show_marker_plot(self.current_marker)

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

                        if abs(current_length - prev_length) / prev_length > 0.25:
                            self.outliers[pair[0]][frame] = True
                            self.outliers[pair[1]][frame] = True

                except KeyError:
                    continue
                    
        # OpenGL 렌더러에 이상치 정보 전달
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_outliers(self.outliers)


    #########################################
    ############ Mouse handling #############
    #########################################

    def connect_mouse_events(self):
        # OpenGL 렌더러에서는 마우스 이벤트가 이미 내부적으로 처리됨

        # 마커 캔버스(matplotlib)는 여전히 연결 필요
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
            self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
            self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
            self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)


    def disconnect_mouse_events(self):
        """disconnect mouse events"""
        # 마커 캔버스(matplotlib) 이벤트 연결 해제
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
                     print(f"Could not disconnect cid {cid}: {e}")


    #########################################
    ########## Marker selection #############
    #########################################

    def on_marker_selected(self, marker_name):
        """Handle marker selection event"""
        self.current_marker = marker_name
        
        # 마커 목록에서 선택 상태 업데이트
        if hasattr(self, 'markers_list') and self.markers_list:
            try:
                # 마커 리스트에서 선택 항목 모두 해제
                self.markers_list.selection_clear(0, "end")
                
                # 마커가 선택된 경우에만 목록에서 선택
                if marker_name is not None:
                    # 선택된 마커의 인덱스 찾기
                    for i, item in enumerate(self.markers_list.get(0, "end")):
                        if item == marker_name:
                            self.markers_list.selection_set(i)  # 선택 항목 설정
                            self.markers_list.see(i)  # 스크롤하여 보이게
                            break
            except Exception as e:
                print(f"마커 목록 업데이트 오류: {e}")
        
        # 마커 그래프 표시 (마커가 선택된 경우에만)
        if marker_name is not None and hasattr(self, 'show_marker_plot'):
            try:
                self.show_marker_plot(marker_name)
            except Exception as e:
                import traceback
                print(f"Error displaying marker plot: {e}")
                traceback.print_exc()
        
        # OpenGL 렌더러에게 선택된 마커 정보 전달
        if hasattr(self, 'gl_renderer'):
            self.gl_renderer.set_current_marker(marker_name)
        
        # 화면 업데이트
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
            print(f"Error updating markers list: {e}")
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
        3D 마커 시각화 업데이트 메서드
        이전에는 외부 plotUpdater.py 모듈을 사용했지만 
        이제는 OpenGL 렌더러를 직접 호출합니다.
        """
        if hasattr(self, 'gl_renderer'):
            # 데이터 전달
            if self.data is not None:
                # 좌표계 설정 확인
                coordinate_system = "z-up" if self.is_z_up else "y-up"
                
                # 이상치(outliers) 데이터 전달
                if hasattr(self, 'outliers') and self.outliers:
                    self.gl_renderer.set_outliers(self.outliers)
                
                # 현재 프레임 데이터 전달
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
                    print(f"OpenGL 데이터 설정 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
            
            # OpenGL 렌더러 화면 갱신
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
        # if hasattr(self, 'marker_canvas'):
        #     self.marker_canvas.draw()


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

            # canvas 관련 처리를 더 안전하게 수정
            if hasattr(self, 'canvas') and self.canvas:
                try:
                    # OpenGL 렌더러인 경우 - 이제 항상 이 경우
                    if hasattr(self, 'gl_renderer'):
                        if self.canvas == self.gl_renderer:
                            if hasattr(self.gl_renderer, 'pack_forget'):
                                self.gl_renderer.pack_forget()
                            if hasattr(self, 'gl_renderer'):
                                del self.gl_renderer
                except Exception as e:
                    print(f"Canvas 정리 중 오류: {e}")
                
                self.canvas = None

            if hasattr(self, 'marker_canvas') and self.marker_canvas:
                try:
                    if hasattr(self.marker_canvas, 'get_tk_widget'):
                        self.marker_canvas.get_tk_widget().destroy()
                except Exception as e:
                    print(f"Marker canvas 정리 중 오류: {e}")
                
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
            print(f"Error clearing state: {e}")
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
            # if hasattr(self, 'marker_canvas'):
            #     self.marker_canvas.draw_idle() # Use draw_idle for potentially better performance in animation loop

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
                self._build_marker_plot_buttons(button_frame)
                
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
    def _build_marker_plot_buttons(self, parent_frame):
        """Helper method to build buttons for the marker plot, adjusting height for edit mode."""
        # Destroy existing button frame contents if they exist
        for widget in parent_frame.winfo_children():
            widget.destroy()

        button_style = {
            "width": 80, "height": 28, "fg_color": "#3B3B3B", "hover_color": "#4B4B4B",
            "text_color": "#FFFFFF", "corner_radius": 6, "border_width": 1, "border_color": "#555555"
        }

        if self.is_editing:
            # --- Build Edit Controls (Multi-row) ---
            parent_frame.configure(height=90) # Increase height for edit mode with interpolation

            # Main container for edit controls
            self.edit_controls_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
            self.edit_controls_frame.pack(fill='both', expand=True, padx=5, pady=2)

            # Top Row Frame
            top_row_frame = ctk.CTkFrame(self.edit_controls_frame, fg_color="transparent")
            top_row_frame.pack(side='top', fill='x', pady=(0, 5)) # Add padding below

            # 1. Filter Type Frame (in top row)
            filter_type_frame = ctk.CTkFrame(top_row_frame, fg_color="transparent")
            filter_type_frame.pack(side='left', padx=(0, 10)) # Add padding to the right
            
            # Filter label with increased padding
            filter_label = ctk.CTkLabel(filter_type_frame, text="Filter:", width=50, anchor="e")
            filter_label.pack(side='left', padx=(3, 8))
            
            self.filter_type_combo = ctk.CTkComboBox(
                filter_type_frame,
                width=150, # Adjust width if needed
                values=['kalman', 'butterworth', 'butterworth_on_speed', 'gaussian', 'LOESS', 'median'],
                variable=self.filter_type_var,
                command=self._on_filter_type_change_in_panel
            )
            self.filter_type_combo.pack(side='left', padx=(42, 0))

            # 2. Dynamic Filter Parameters Container (in top row)
            self.filter_params_container = ctk.CTkFrame(top_row_frame, fg_color="transparent")
            self.filter_params_container.pack(side='left', fill='x', expand=True)
            # Initial population of parameters based on current filter type
            self._build_filter_param_widgets(self.filter_type_var.get())

            # Middle Row Frame for Interpolation
            middle_row_frame = ctk.CTkFrame(self.edit_controls_frame, fg_color="transparent")
            middle_row_frame.pack(side='top', fill='x', pady=(0, 5))
            
            # Interpolation Method Frame
            interp_frame = ctk.CTkFrame(middle_row_frame, fg_color="transparent")
            interp_frame.pack(side='left', padx=(0, 10))
            
            # Interpolation label with consistent styling
            interp_label = ctk.CTkLabel(interp_frame, text="Interpolation:", width=90, anchor="e")
            interp_label.pack(side='left', padx=(5, 8))
            
            # Interpolation ComboBox
            self.interp_method_combo = ctk.CTkComboBox(
                interp_frame,
                width=150,
                values=self.interp_methods,
                variable=self.interp_method_var,
                command=self._on_interp_method_change_in_panel
            )
            self.interp_method_combo.pack(side='left')
            
            # Interpolation Order Frame
            interp_order_frame = ctk.CTkFrame(middle_row_frame, fg_color="transparent")
            interp_order_frame.pack(side='left')
            
            # Order Label and Entry - consistent styling with other labels
            self.interp_order_label = ctk.CTkLabel(interp_order_frame, text="Order:", width=40, anchor='e')
            self.interp_order_label.pack(side='left', padx=(0, 4))
            
            self.interp_order_entry = ctk.CTkEntry(interp_order_frame, textvariable=self.order_var, width=60)
            self.interp_order_entry.pack(side='left', padx=(0, 0))
            
            # Set initial state based on current method
            current_method = self.interp_method_var.get()
            if current_method not in ['polynomial', 'spline']:
                self.interp_order_label.configure(state='disabled')
                self.interp_order_entry.configure(state='disabled')

            # Bottom Row Frame
            bottom_row_frame = ctk.CTkFrame(self.edit_controls_frame, fg_color="transparent")
            bottom_row_frame.pack(side='top', fill='x')

            # 3. Action Buttons Frame (in bottom row)
            action_buttons_frame = ctk.CTkFrame(bottom_row_frame, fg_color="transparent")
            action_buttons_frame.pack(side='left', padx=(15, 0)) # 왼쪽에 20px 패딩 추가
            action_buttons = [
                ("Filter", self.filter_selected_data),
                ("Delete", self.delete_selected_data),
                ("Interpolate", self.interpolate_selected_data),
                ("Restore", self.restore_original_data)
            ]
            # Use smaller width for action buttons if needed
            action_button_style = {**button_style, "width": 80, "height": 28}
            for text, command in action_buttons:
                btn = ctk.CTkButton(action_buttons_frame, text=text, command=command, **action_button_style)
                btn.pack(side='left', padx=3)

            # 4. Done Button (in bottom row, packed to the right)
            done_button = ctk.CTkButton(
                bottom_row_frame, text="Done", command=self.toggle_edit_mode, **button_style
            )
            # Pack Done button to the far right of the bottom row
            done_button.pack(side='right', padx=(10, 0)) # Add padding to the left

        else:
            # --- Build View Controls (Single Row) ---
            parent_frame.configure(height=40) # Set default height for view mode

            reset_button = ctk.CTkButton(
                parent_frame, text="Reset View", command=self.reset_graph_view, **button_style
            )
            reset_button.pack(side='right', padx=5, pady=5)

            self.edit_button = ctk.CTkButton(
                parent_frame, text="Edit", command=self.toggle_edit_mode, **button_style
            )
            self.edit_button.pack(side='right', padx=5, pady=5)

    
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

            print("Data has been restored to the original state.")
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
            print("Error: filter_params attribute not found on TRCViewer.")

        
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
                "Right-click markers in the 3D view to select/deselect them as reference patterns.\n"
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
