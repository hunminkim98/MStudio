import os
import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from Pose2Sim.skeletons import *
from Pose2Sim.filtering import *
import matplotlib
import argparse
import sys

from gui.TRCviewerWidgets import create_widgets
from gui.markerPlot import show_marker_plot
from gui.plotCreator import create_plot, _setup_plot_style, _draw_static_elements, _initialize_dynamic_elements, _update_coordinate_axes
from gui.plotUpdater import update_plot
from gui.filterUI import on_filter_type_change

from utils.dataLoader import read_data_from_c3d, read_data_from_trc, open_file
from utils.dataSaver import save_as, save_to_trc, save_to_c3d
from utils.viewToggles import toggle_marker_names, toggle_coordinates, toggle_trajectory, toggle_edit_window, toggle_animation
from utils.viewReset import reset_main_view, reset_graph_view
from utils.dataProcessor import *
from utils.mouseHandler import MouseHandler
from utils.trajectory import MarkerTrajectory

# OpenGL 렌더러 사용 가능 여부 확인 함수
def check_opengl_available():
    """
    OpenGL 렌더링에 필요한 라이브러리가 설치되어 있는지 확인합니다.
    
    Returns:
        bool: 필요한 라이브러리가 모두 설치되어 있으면 True, 아니면 False
    """
    try:
        import pyopengltk
        from OpenGL import GL
        from OpenGL import GLU
        return True
    except ImportError as e:
        print(f"OpenGL 렌더링에 필요한 라이브러리 임포트 오류: {e}")
        print("이 기능을 사용하려면 다음을 설치하세요: pip install pyopengltk pyopengl pyopengl-accelerate")
        return False

# OpenGL 렌더러 모듈 조건부 임포트
opengl_available = check_opengl_available()
if opengl_available:
    try:
        # OpenGL 기반 렌더러 모듈 임포트 시도
        from gui.opengl.GLMarkerRenderer import MarkerGLRenderer
        print("OpenGL 렌더러 모듈 임포트 성공")
    except ImportError as e:
        print(f"OpenGL 렌더러 모듈 임포트 오류: {e}")
        opengl_available = False

# Interactive mode on
plt.ion()
matplotlib.use('TkAgg')

class TRCViewer(ctk.CTk):
    def __init__(self, use_opengl=False):
        super().__init__()
        self.title("TRC Viewer")
        self.geometry("1920x1080")

        # 렌더링 모드 설정
        self.use_opengl = use_opengl
        # OpenGL 좌표계 설정 변수
        self.coordinate_system = "y-up"  # 기본값은 y-up

        # initialize variables
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
        self.is_z_up = False   # 좌표계 상태 (True = Z-up, False = Y-up)
        
        # OpenGL 렌더러 관련 속성
        self.gl_renderer = None

        # filter type variable initialization
        self.filter_type_var = ctk.StringVar(value='butterworth')

        # mouse handler initialization
        self.mouse_handler = MouseHandler(self)
        
        # interpolation method list
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
        
        # interpolation method variable initialization
        self.interp_method_var = ctk.StringVar(value='linear')
        self.order_var = ctk.StringVar(value='3')

        # pattern marker related attributes initialization
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
        create_widgets(self)

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
        open_file(self)

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
                    # OpenGL 렌더러인 경우
                    if self.use_opengl and hasattr(self, 'gl_renderer'):
                        if self.canvas == self.gl_renderer:
                            if hasattr(self.gl_renderer, 'pack_forget'):
                                self.gl_renderer.pack_forget()
                            if hasattr(self, 'gl_renderer'):
                                del self.gl_renderer
                    # Matplotlib 캔버스인 경우
                    elif hasattr(self.canvas, 'get_tk_widget'):
                        self.canvas.get_tk_widget().destroy()
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
        create_plot(self)

    def _setup_plot_style(self):
        _setup_plot_style(self)

    def _draw_static_elements(self):
        """Draw static elements like the ground grid based on the coordinate system."""
        _draw_static_elements(self)

    def _initialize_dynamic_elements(self):
        _initialize_dynamic_elements(self)

    def _update_coordinate_axes(self):
        """Update coordinate axes and labels based on the coordinate system."""
        _update_coordinate_axes(self)

    def update_plot(self):
        update_plot(self)

    def connect_mouse_events(self):
        if self.canvas:
            self.canvas.mpl_connect('scroll_event', self.mouse_handler.on_scroll)
            self.canvas.mpl_connect('pick_event', self.mouse_handler.on_pick)
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
        show_marker_plot(self, marker_name)

    def on_interp_method_change(self, choice):
        """Interpolation method change processing"""
        on_interp_method_change(self, choice)

    def on_pattern_selection_confirm(self):
        """Process pattern selection confirmation"""
        on_pattern_selection_confirm(self)

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
        interpolate_selected_data(self)

    def interpolate_with_pattern(self):
        """
        Pattern-based interpolation using reference markers to interpolate target marker
        """
        interpolate_with_pattern(self)

    def toggle_edit_window(self):
        toggle_edit_window(self)

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
        filter_selected_data(self)

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
        
        # OpenGL 렌더러를 사용 중인 경우
        if hasattr(self, 'use_opengl') and self.use_opengl and hasattr(self, 'gl_renderer'):
            # 좌표계 설정 변경만 먼저 수행 (나머지는 set_coordinate_system 내부에서 처리)
            self.coordinate_system = "z-up" if self.is_z_up else "y-up"
            if hasattr(self.gl_renderer, 'set_coordinate_system'):
                self.gl_renderer.set_coordinate_system(self.is_z_up)
            
            # 화면 강제 갱신 요청 - 약간의 지연 후 실행
            self.after(50, self._force_update_opengl)
            
        # Matplotlib 렌더러를 사용 중인 경우
        else:
            if self.data is not None:
                # 좌표계 변경
                self.coordinate_system = "z-up" if self.is_z_up else "y-up"
                
                # 좌표축 및 그리드 업데이트
                _draw_static_elements(self)
                _update_coordinate_axes(self)
                
                # 전체 시각화 업데이트 - 마커 좌표는 변환하지 않음
                self.update_plot()

    def _force_update_opengl(self):
        """OpenGL 렌더러의 화면을 강제로 갱신합니다."""
        if not (hasattr(self, 'use_opengl') and self.use_opengl and hasattr(self, 'gl_renderer')):
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

    def toggle_trajectory(self):
        """Toggle the visibility of marker trajectories"""
        toggle_trajectory(self)

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

    def toggle_marker_names(self):
        toggle_marker_names(self)

    def toggle_animation(self):
        toggle_animation(self)

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

    def reset_main_view(self):
        reset_main_view(self)

    def reset_graph_view(self):
        reset_graph_view(self)

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
        save_as(self)

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
        on_filter_type_change(self, choice)

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

