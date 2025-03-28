import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3D
import traceback
import pandas as pd

def create_plot(self):
    """
    Creates the main 3D plot for displaying marker data.
    This function was extracted from the main class to improve code organization.
    
    Now supports both matplotlib and OpenGL rendering modes based on self.use_opengl flag.
    """
    if not self.use_opengl:
        # 기존 matplotlib 렌더러 초기화
        self.fig = plt.Figure(figsize=(10, 10), facecolor='black')  # Changed to square figure
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])  # Add proper spacing around plot
        
        _setup_plot_style(self)
        _draw_static_elements(self)
        _initialize_dynamic_elements(self)

        if hasattr(self, 'canvas') and self.canvas:
            if hasattr(self.canvas, 'get_tk_widget'):
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
    else:
        # OpenGL 렌더러 초기화
        try:
            from gui.opengl.GLMarkerRenderer import MarkerGLRenderer
            
            # 기존 matplotlib 캔버스가 있다면 제거
            if hasattr(self, 'canvas') and self.canvas:
                if hasattr(self.canvas, 'get_tk_widget'):
                    self.canvas.get_tk_widget().destroy()
                self.canvas = None
            
            # OpenGL 렌더러 초기화
            self.gl_renderer = MarkerGLRenderer(self.canvas_frame, bg='black')
            self.gl_renderer.pack(fill='both', expand=True)
            
            # 마커 데이터 설정
            if self.data is not None:
                # 데이터 범위 설정 (이제 올바른 방식으로 접근)
                if hasattr(self, 'data_limits') and self.data_limits is not None:
                    # data_limits가 딕셔너리 형태로 되어있으므로 키로 접근
                    if 'x' in self.data_limits and 'y' in self.data_limits and 'z' in self.data_limits:
                        x_range = self.data_limits['x']
                        y_range = self.data_limits['y']
                        z_range = self.data_limits['z']
                        
                        # set_data_limits 메서드 호출 전 존재 여부 확인
                        if hasattr(self.gl_renderer, 'set_data_limits'):
                            self.gl_renderer.set_data_limits(x_range, y_range, z_range)
            
            # 렌더러 초기화 및 그리기
            self.gl_renderer.initialize()
            
            # 캔버스 참조 저장 (기존 코드와의 호환성 유지)
            self.canvas = self.gl_renderer
            
            print("OpenGL 렌더러가 초기화되었습니다.")
        except Exception as e:
            print(f"OpenGL 렌더러 초기화 중 오류 발생: {type(e).__name__} - {e}")
            print("--- Traceback ---")
            traceback.print_exc()
            print("-----------------")
            # OpenGL 초기화 실패 시 matplotlib로 폴백
            self.use_opengl = False
            print("matplotlib 렌더링 모드로 폴백합니다.")
            create_plot(self)  # matplotlib 모드로 재귀적 호출

def _setup_plot_style(self):
    """
    Sets up the style of the plot (colors, margins, etc.).
    """
    if not self.use_opengl:
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
    마커 좌표는 변환하지 않고 그리드와 축만 좌표계에 맞게 조정합니다.
    MatPlotLib 그리드는 삭제되었습니다.
    """
    # MatPlotLib 그리드 생성이 제거되었습니다
    pass

def _initialize_dynamic_elements(self):
    """
    Initializes dynamic elements of the plot (points, lines, etc.).
    """
    if not self.use_opengl:
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
    마커 좌표는 변환하지 않고 좌표계 표시만 적절히 조정합니다.
    """
    if not self.use_opengl:
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
            # Z-up coordinate system
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

            # Label position
            label_x = self.ax.text(axis_length + 0.1, 0, 0, 'X', color=x_color, fontsize=12)
            label_y = self.ax.text(0, axis_length + 0.1, 0, 'Y', color=y_color, fontsize=12)
            label_z = self.ax.text(0, 0, axis_length + 0.1, 'Z', color=z_color, fontsize=12)
        else:
            # Y-up coordinate system
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

            # Label position
            label_x = self.ax.text(axis_length + 0.1, 0, 0, 'X', color=x_color, fontsize=12)
            label_y = self.ax.text(0, axis_length + 0.1, 0, 'Y', color=y_color, fontsize=12)
            label_z = self.ax.text(0, 0, axis_length + 0.1, 'Z', color=z_color, fontsize=12)

        # 축 및 레이블 저장
        self.coordinate_axes = [line_x, line_y, line_z]
        self.axis_labels = [label_x, label_y, label_z]

        # 좌표계 설정에 맞게 카메라 각도 조정
        if self.is_z_up:
            self.ax.view_init(elev=30, azim=45)  # Z-up을 위한 기본 각도
        else:
            self.ax.view_init(elev=10, azim=45)  # Y-up을 위한 기본 각도
