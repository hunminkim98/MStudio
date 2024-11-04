import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Pose2Sim.skeletons import *
from Pose2Sim.filtering import *
from matplotlib.figure import Figure
import matplotlib
import os
import matplotlib.pyplot as plt

# 대화형 모드 활성화
plt.ion()  # Interactive mode on

matplotlib.use('TkAgg')

# C3D 파일 읽기 함수
def read_data_from_c3d(c3d_file_path):
    try:
        import c3d
        # C3D 파일을 바이너리 모드로 열기
        with open(c3d_file_path, 'rb') as f:
            reader = c3d.Reader(f)
            
            # 기본 정보 추출
            point_labels = reader.point_labels
            frame_rate = reader.header.frame_rate
            first_frame = reader.header.first_frame
            last_frame = reader.header.last_frame
            
            # 마커 이름에서 공백 제거 및 정리
            point_labels = [label.strip() for label in point_labels if label.strip()]
            # 중복 제거
            point_labels = list(dict.fromkeys(point_labels))
            
            # 데이터 프레임을 위한 빈 리스트 생성
            frames = []
            times = []
            marker_data = {label: {'X': [], 'Y': [], 'Z': []} for label in point_labels}
            
            # 프레임별 데이터 읽기
            for i, points, analog in reader.read_frames():
                frames.append(i)
                times.append(i / frame_rate)
                
                # points 데이터는 mm 단위로 저장되어 있으므로 m 단위로 변환 (1000으로 나누기)
                points_meters = points[:, :3] / 1000.0
                
                # 각 마커의 위치 데이터 저장
                for j, label in enumerate(point_labels):
                    if j < len(points_meters):  # 인덱스 범위 체크
                        marker_data[label]['X'].append(points_meters[j, 0])
                        marker_data[label]['Y'].append(points_meters[j, 1])
                        marker_data[label]['Z'].append(points_meters[j, 2])
            
            # DataFrame 생성을 위한 데이터 딕셔너리
            data_dict = {'Frame#': frames, 'Time': times}
            
            # 마커 데이터 추가
            for label in point_labels:
                if label in marker_data:  # 키 존재 여부 확인
                    data_dict[f'{label}_X'] = marker_data[label]['X']
                    data_dict[f'{label}_Y'] = marker_data[label]['Y']
                    data_dict[f'{label}_Z'] = marker_data[label]['Z']
            
            # DataFrame 생성
            data = pd.DataFrame(data_dict)
            
            # 헤더 라인 생성 (TRC 형식과 유사하게)
            header_lines = [
                f"PathFileType\t4\t(X/Y/Z)\t{c3d_file_path}\n",
                f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
                f"{frame_rate}\t{frame_rate}\t{len(frames)}\t{len(point_labels)}\tm\t{frame_rate}\t{first_frame}\t{last_frame}\n",
                "\t".join(['Frame#', 'Time'] + point_labels) + "\n",
                "\t".join(['', ''] + ['X\tY\tZ' for _ in point_labels]) + "\n"
            ]
            
            return header_lines, data, point_labels
            
    except Exception as e:
        raise Exception(f"C3D 파일 읽기 오류: {str(e)}")

# TRC 파일 읽기 함수
def read_data_from_trc(trc_file_path):
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()

    # 헤더 라인 추출
    header_lines = lines[:5]
    
    # 마커 이름 추출 (3번째 행)
    marker_names_line = lines[3].strip().split('\t')[2:]  # 'Frame#', 'Time' 제외
    
    # 고유한 마커 이름만 추출 (빈 문자열 제외)
    marker_names = []
    for name in marker_names_line:
        if name.strip() and name not in marker_names:  # 빈 문자열이 아니고 중복되지 않은 경우
            marker_names.append(name.strip())
    
    # 컬럼 이름 생성
    column_names = ['Frame#', 'Time']
    for marker in marker_names:
        column_names.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])
    
    # 데이터 읽기
    data = pd.read_csv(trc_file_path, sep='\t', skiprows=6, names=column_names)
    
    return header_lines, data, marker_names
    
# 메인 애플리케이션 클래스
class TRCViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TRC Viewer")
        self.geometry("1200x1000")

        # 데이터 관련 변수 초기화
        self.data = None
        self.original_data = None  # 원본 데이터 저장 변수 추가
        self.marker_names = None
        self.num_frames = 0
        self.frame_idx = 0
        self.canvas = None
        self.selection_in_progress = False

        # 마커 그래프 관련 변수 초기화
        self.marker_last_pos = None
        self.marker_pan_enabled = False
        self.marker_canvas = None
        self.marker_axes = []
        self.marker_lines = []

        # 뷰 범위 저장 변수 추가
        self.view_limits = None
        self.is_z_up = True
        self.outliers = {}

        # 사용 가능한 스켈레톤 모델들
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

        # 3D 뷰어의 이동 제한을 위한 변수
        self.pan_enabled = False
        self.last_mouse_pos = None

        # UI 구성
        self.create_widgets()

    def on_separator_drag(self, event):
        """구분선 드래그로 창 크기 조절"""
        if not self.graph_frame.winfo_ismapped():
            return

        try:
            # 전체 너비 계산
            total_width = self.main_content.winfo_width()
            if total_width <= 0:
                return

            # 새로운 뷰어 너비 계산
            new_viewer_width = max(0, event.x_root - self.viewer_frame.winfo_rootx())
            
            # 최소/최대 크기 제한
            min_width = total_width * 0.2  # 최소 20%
            max_width = total_width * 0.8  # 최대 80%
            new_viewer_width = max(min_width, min(new_viewer_width, max_width))

            # 그래프 프레임의 최소 너비 보장
            remaining_width = total_width - new_viewer_width
            if remaining_width < min_width:
                new_viewer_width = total_width - min_width

            # 크기 비율 계산
            viewer_ratio = new_viewer_width / total_width
            graph_ratio = 1 - viewer_ratio

            # 프레임 크기 설정
            self.viewer_frame.pack_configure(side='left', fill='both', expand=True)
            self.graph_frame.pack_configure(side='right', fill='both', expand=True)
            
            # 실제 너비 설정
            self.viewer_frame.configure(width=int(new_viewer_width))
            self.graph_frame.configure(width=int(total_width - new_viewer_width))

            # 즉시 업데이트
            self.update_idletasks()

        except Exception as e:
            print(f"Error in separator drag: {e}")

    def create_widgets(self):
        # 상단 버튼 프레임
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10, padx=10, fill='x')

        # 버튼 스타일 통일
        button_style = {
            "fg_color": "#333333",  # 짙은 회색
            "hover_color": "#444444"  # 호버 시 약간 밝은 회색
        }

        # 왼쪽 버튼들을 담을 프레임
        left_button_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
        left_button_frame.pack(side='left', fill='x')

        # 카메라 초기화 버튼
        self.reset_view_button = ctk.CTkButton(left_button_frame,
                                             text="🎥",
                                             width=30,
                                             command=self.reset_main_view,
                                             **button_style)
        self.reset_view_button.pack(side='left', padx=5)

        # 파일 열기 버튼
        self.open_button = ctk.CTkButton(left_button_frame, 
                                       text="Open TRC File", 
                                       command=self.open_file,
                                       **button_style)
        self.open_button.pack(side='left', padx=5)

        # 좌표계 전환 버튼
        self.coord_button = ctk.CTkButton(button_frame, 
                                        text="Switch to Y-up", 
                                        command=self.toggle_coordinates,
                                        **button_style)
        self.coord_button.pack(side='left', padx=5)

        # 마커 이름 표시/숨김 버튼
        self.show_names = False
        self.names_button = ctk.CTkButton(button_frame, 
                                        text="Hide Names", 
                                        command=self.toggle_marker_names,
                                        **button_style)
        self.names_button.pack(side='left', padx=5)

        # # 스켈레톤 라인 표시/숨김 버튼
        # self.show_skeleton = True
        # self.skeleton_button = ctk.CTkButton(button_frame, 
        #                                    text="Hide Skeleton", 
        #                                    command=self.toggle_skeleton,
        #                                    **button_style)
        # self.skeleton_button.pack(side='left', padx=5)

        # 모델 선택 콤보박스
        self.model_var = ctk.StringVar(value='No skeleton')
        self.model_combo = ctk.CTkComboBox(button_frame, 
                                         values=list(self.available_models.keys()),
                                         variable=self.model_var,
                                         command=self.on_model_change)
        self.model_combo.pack(side='left', padx=5)

        # 중앙 컨텐츠 프레임
        self.main_content = ctk.CTkFrame(self)
        self.main_content.pack(fill='both', expand=True, padx=10)

        # 3D 뷰어 프레임
        self.viewer_frame = ctk.CTkFrame(self.main_content)
        self.viewer_frame.pack(side='left', fill='both', expand=True)

        # Separator 추가
        self.separator = ctk.CTkFrame(self.main_content, width=5, bg_color='gray50')
        self.separator.pack(side='left', fill='y', padx=2)
        self.separator.bind('<Enter>', lambda e: self.separator.configure(bg_color='gray30'))
        self.separator.bind('<Leave>', lambda e: self.separator.configure(bg_color='gray50'))
        self.separator.bind('<B1-Motion>', self.on_separator_drag)

        # 그래프 프레임 (오른쪽, 초기에는 숨김)
        self.graph_frame = ctk.CTkFrame(self.main_content)
        self.graph_frame.pack_forget()

        # 뷰어 상단 프레임 (컨트롤 버튼들과 파일 이름을 위한 프레임)
        viewer_top_frame = ctk.CTkFrame(self.viewer_frame)
        viewer_top_frame.pack(fill='x', pady=(5, 0))

        # 파일 이름 표시 레이블 (중앙)
        self.title_label = ctk.CTkLabel(viewer_top_frame, text="", font=("Arial", 14))
        self.title_label.pack(side='left', expand=True)

        canvas_container = ctk.CTkFrame(self.viewer_frame)
        canvas_container.pack(fill='both', expand=True)

        # 캔버스 프레임
        self.canvas_frame = ctk.CTkFrame(canvas_container)
        self.canvas_frame.pack(side='left', fill='both', expand=True)

        # 하단 컨트롤 영역
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

        # 프레임 번호 표시 레이블 (슬라이더 바로 위)
        self.frame_label = ctk.CTkLabel(self.control_frame, text="Frame 0", font=("Arial", 12))
        self.frame_label.pack(pady=(0, 5))

        # 하단 컨트롤 프레임 (슬라이더와 버튼)
        self.bottom_frame = ctk.CTkFrame(self.control_frame)
        self.bottom_frame.pack(fill='x', padx=5)

        # 이전 프레임 버튼
        self.prev_button = ctk.CTkButton(self.bottom_frame, 
                                       text="◀", 
                                       width=30,
                                       command=self.prev_frame)
        self.prev_button.pack(side='left', padx=5)

        # 프레임 슬라이더
        self.frame_slider = ctk.CTkSlider(self.bottom_frame, 
                                        from_=0, 
                                        to=1, 
                                        command=self.update_frame)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=5)

        # 다음 프레임 버튼
        self.next_button = ctk.CTkButton(self.bottom_frame, 
                                       text="▶", 
                                       width=30,
                                       command=self.next_frame)
        self.next_button.pack(side='left', padx=5)

        # 마커 정보 레이블
        self.marker_label = ctk.CTkLabel(self, text="")
        self.marker_label.pack(pady=5)

        # 마우스 이벤트 바인딩
        if self.canvas:
            self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_model_change(self, choice):
        """스켈레톤 모델 변경 시 호출"""
        self.current_model = self.available_models[choice]
        if self.current_model is None: 
            self.skeleton_pairs = []
            self.show_skeleton = False
        else:
            self.show_skeleton = True
            self.update_skeleton_pairs()
        
        if self.data is not None:
            self.detect_outliers()
            self.update_plot()

    def update_skeleton_pairs(self):
        """현재 모델의 스켈레톤 연결 쌍 업데이트"""
        self.skeleton_pairs = []
        if self.current_model is not None:
            for node in self.current_model.descendants:
                if node.parent:
                    self.skeleton_pairs.append((node.parent.name, node.name))

    def open_file(self):
        """TRC/C3D 파일 열기"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Motion files", "*.trc;*.c3d"), ("TRC files", "*.trc"), ("C3D files", "*.c3d"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 기존 데이터 및 상태 초기화
                self.clear_current_state()
                
                # 새 파일 로드
                self.current_file = file_path
                file_name = os.path.basename(file_path)
                self.title_label.configure(text=file_name)
                
                # 파일 확장자에 따라 적절한 로더 사용
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.trc':
                    header_lines, self.data, self.marker_names = read_data_from_trc(file_path)
                elif file_ext == '.c3d':
                    header_lines, self.data, self.marker_names = read_data_from_c3d(file_path)
                else:
                    raise Exception("Unsupported file format")
                
                self.num_frames = self.data.shape[0]
                
                # 원본 데이터 저장
                self.original_data = self.data.copy(deep=True)
                
                # 데이터 범위 계산
                self.calculate_data_limits()
                
                # 프레임 슬라이더 초기화
                self.frame_slider.configure(to=self.num_frames - 1)
                self.frame_idx = 0
                self.frame_slider.set(0)
                
                # 현재 선택된 모델로 스켈레톤 구성
                self.current_model = self.available_models[self.model_var.get()]
                self.update_skeleton_pairs()
                self.detect_outliers()
                
                # 그래프 생성
                self.create_plot()
                
                # 초기 뷰 설정
                self.reset_main_view()
                
                print(f"Successfully loaded: {file_name}")
                
            except Exception as e:
                print(f"Error loading file: {e}")
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def clear_current_state(self):
        """현재 상태 초기화"""
        try:
            # 구분선 숨기기
            if hasattr(self, 'separator'):
                self.separator.pack_forget()
            
            # 그래프 프레임 초기화
            if hasattr(self, 'graph_frame') and self.graph_frame.winfo_ismapped():
                self.graph_frame.pack_forget()
                for widget in self.graph_frame.winfo_children():
                    widget.destroy()
            
            # matplotlib 객체 정리
            if hasattr(self, 'fig'):
                plt.close(self.fig)
                del self.fig
            if hasattr(self, 'marker_plot_fig'):
                plt.close(self.marker_plot_fig)
                del self.marker_plot_fig
            
            # 캔버스 정리 - 추가 검사 추가
            if hasattr(self, 'canvas') and self.canvas and hasattr(self.canvas, 'get_tk_widget'):
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            
            if hasattr(self, 'marker_canvas') and self.marker_canvas and hasattr(self.marker_canvas, 'get_tk_widget'):
                self.marker_canvas.get_tk_widget().destroy()
                del self.marker_canvas
                self.marker_canvas = None
            
            # Axes 객체 삭제
            if hasattr(self, 'ax'):
                del self.ax
            if hasattr(self, 'marker_axes'):
                del self.marker_axes
            if hasattr(self, 'ax2'):
                del self.ax2
            
            # 데이터 변수 초기화
            self.data = None
            self.original_data = None  # 원본 데이터 초기화
            self.marker_names = None
            self.num_frames = 0
            self.frame_idx = 0
            self.outliers = {}
            self.current_marker = None
            self.marker_axes = []
            self.marker_lines = []
            
            # 뷰 관련 변수 초기화
            self.view_limits = None
            self.data_limits = None
            self.initial_limits = None
            
            # 선택 데이터 초기화
            self.selection_data = {
                'start': None,
                'end': None,
                'rects': [],
                'current_ax': None,
                'rect': None
            }
            
            # 슬라이더 및 레이블 초기화
            self.frame_slider.set(0)
            self.frame_slider.configure(to=1)
            self.frame_label.configure(text="Frame 0")
            
            # 기타 상태 초기화
            self.title_label.configure(text="")
            self.show_names = False
            self.show_skeleton = True
            self.current_file = None

            print("Current state cleared successfully")
        except Exception as e:
            print(f"Error clearing state: {e}")


    def calculate_data_limits(self):
        """이터의 전체 범위 산"""
        try:
            x_coords = [col for col in self.data.columns if col.endswith('_X')]
            y_coords = [col for col in self.data.columns if col.endswith('_Y')]
            z_coords = [col for col in self.data.columns if col.endswith('_Z')]
            
            # 각 축의 최소/최대값 계산
            x_min = self.data[x_coords].min().min()
            x_max = self.data[x_coords].max().max()
            y_min = self.data[y_coords].min().min()
            y_max = self.data[y_coords].max().max()
            z_min = self.data[z_coords].min().min()
            z_max = self.data[z_coords].max().max()
            
            # 여유 공간 추가 (10%)
            margin = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            self.data_limits = {
                'x': (x_min - x_range * margin, x_max + x_range * margin),
                'y': (y_min - y_range * margin, y_max + y_range * margin),
                'z': (z_min - z_range * margin, z_max + z_range * margin)
            }
            
            # 초기 뷰 범위도 저장
            self.initial_limits = self.data_limits.copy()
            
        except Exception as e:
            print(f"Error calculating data limits: {e}")
            self.data_limits = None
            self.initial_limits = None



    def create_plot(self):
        """새로운 Figure와 Canvas를 생성하여 플롯을 초기화"""
        # 새로운 matplotlib Figure 생성
        self.fig = plt.Figure(facecolor='black')  # 검은색 배경
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 축과 배경 스타일 설정
        self.ax.set_facecolor('black')  # 플롯 배경색
        self.fig.patch.set_facecolor('black')  # Figure 배경색

        # 축 색상 설정
        self.ax.xaxis.set_pane_color((0, 0, 0, 1))  # 축 평면 색상
        self.ax.yaxis.set_pane_color((0, 0, 0, 1))
        self.ax.zaxis.set_pane_color((0, 0, 0, 1))

        # 축 라벨 색상 설정
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')

        # 축 눈금 색상 설정
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')

        # 기존 캔버스 제거 후 새로운 캔버스를 생성하여 추가
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        # 새로운 캔버스를 생성하고 설정
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # 이벤트 연결
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 초기 데이터 플롯
        self.update_plot()



    def connect_mouse_events(self):
        """마우스 이벤트 연결을 위한 새로운 메서드"""
        if self.canvas:
            self.canvas.mpl_disconnect('scroll_event')  # 기존 연결 해제
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
        """슬라이더 값이 변경될 때 호출"""
        self.frame_idx = int(float(value))
        self.update_plot()
        
        # 마커 그래프가 표시되어 있다면 수직선 업데이
        if hasattr(self, 'marker_lines') and self.marker_lines:
            for line in self.marker_lines:
                line.set_xdata([self.frame_idx, self.frame_idx])
            if hasattr(self, 'marker_canvas'):
                self.marker_canvas.draw()

    def update_plot(self):
        if self.canvas is None:
            return

        # 프레임 레이블 업데이트
        self.frame_label.configure(text=f"Frame {self.frame_idx}")

        # 현재 뷰 상태 저장
        try:
            prev_elev = self.ax.elev
            prev_azim = self.ax.azim
            prev_xlim = self.ax.get_xlim()
            prev_ylim = self.ax.get_ylim()
            prev_zlim = self.ax.get_zlim()
        except AttributeError:
            # 초기 상태에서는 기본값 사용
            prev_elev, prev_azim = 20, -60  # 원하는 초기 각도
            prev_xlim, prev_ylim, prev_zlim = None, None, None

        self.ax.clear()
        
        # 파일 이름 표시 (3D 뷰어 상단)
        if hasattr(self, 'current_file'):
            file_name = os.path.basename(self.current_file)
            # self.ax.set_title(file_name, color='white', pad=10)
        
        # 축과 경 스타일 설정
        self.ax.set_facecolor('black')
        self.ax.xaxis.set_pane_color((0, 0, 0, 1))
        self.ax.yaxis.set_pane_color((0, 0, 0, 1))
        self.ax.zaxis.set_pane_color((0, 0, 0, 1))
        
        # 눈금과 눈금 레이블 제거
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # 축 레이블 제거 (XYZ 텍스트 제거)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')

        # 바닥 그리드 추가
        grid_size = 2  # 그리드 크기
        grid_divisions = 20  # 그리드 분할 수
        x = np.linspace(-grid_size, grid_size, grid_divisions)
        y = np.linspace(-grid_size, grid_size, grid_divisions)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # 바닥면은 z=0에 위치

        # 그리드 라인 그리기
        for i in range(grid_divisions):
            self.ax.plot(x, [y[i]] * grid_divisions, [0] * grid_divisions, 'gray', alpha=0.2)
            self.ax.plot([x[i]] * grid_divisions, y, [0] * grid_divisions, 'gray', alpha=0.2)

        
        # 메인 XYZ 축 선 추가
        origin = np.zeros(3)
        axis_length = 0.5
        
        # 축 색상 정의
        x_color = 'red'
        y_color = 'yellow'
        z_color = 'blue'
        
        if self.is_z_up:
            # Z-up 좌표계 메인 축 그리기
            # X축 (빨간색)
            self.ax.plot([origin[0], origin[0] + axis_length], 
                        [origin[1], origin[1]], 
                        [origin[2], origin[2]], 
                        color=x_color, alpha=0.8, linewidth=2)
            
            # Y축 (노란색)
            self.ax.plot([origin[0], origin[0]], 
                        [origin[1], origin[1] + axis_length], 
                        [origin[2], origin[2]], 
                        color=y_color, alpha=0.8, linewidth=2)
            
            # Z축 (파란색)
            self.ax.plot([origin[0], origin[0]], 
                        [origin[1], origin[1]], 
                        [origin[2], origin[2] + axis_length], 
                        color=z_color, alpha=0.8, linewidth=2)
        else:
            # Y-up 좌표계 메인 축 그리기
            # X축 (빨간색)
            self.ax.plot([origin[0], origin[0] + axis_length], 
                        [origin[2], origin[2]], 
                        [origin[1], origin[1]], 
                        color=x_color, alpha=0.8, linewidth=2)
            
            # Z축 (파란색)
            self.ax.plot([origin[0], origin[0]], 
                        [origin[2], origin[2] + axis_length], 
                        [origin[1], origin[1]], 
                        color=z_color, alpha=0.8, linewidth=2)
            
            # Y축 (노란색)
            self.ax.plot([origin[0], origin[0]], 
                        [origin[2], origin[2]], 
                        [origin[1], origin[1] + axis_length], 
                        color=y_color, alpha=0.8, linewidth=2)
        
        # 오른쪽 하단에 새로운 좌표축 추가
        ax2 = self.fig.add_axes([0.85, 0.1, 0.14, 0.14], projection='3d')
        ax2.set_facecolor('none')
        ax2.set_navigate(False)  # ax2가 마우스 이벤트를 받지 않도록 설정
        self.ax2 = ax2  # 필요하다면 ax2를 인스턴스 변수로 저장
            
        # 작은 좌표축의 스타일 설정
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.grid(False)
        ax2.axis('off')
        
        # 작은 좌표축에 XYZ 축 그리기
        small_length = 0.7
        
        if self.is_z_up:
            # Z-up 좌표계 보조 축 그리기
            ax2.plot([0, small_length], [0, 0], [0, 0], color='red', linewidth=1.5)
            ax2.text(small_length * 1.2, 0, 0, 'X', color='red', fontsize=6)
            
            ax2.plot([0, 0], [0, small_length], [0, 0], color='yellow', linewidth=1.5)
            ax2.text(0, small_length * 1.2, 0, 'Y', color='yellow', fontsize=6)
            
            ax2.plot([0, 0], [0, 0], [0, small_length], color='blue', linewidth=1.5)
            ax2.text(0, 0, small_length * 1.2, 'Z', color='blue', fontsize=6)
        else:
            # Y-up 좌표계 보조 축 그리기
            ax2.plot([0, small_length], [0, 0], [0, 0], color='red', linewidth=1.5)
            ax2.text(small_length * 1.2, 0, 0, 'X', color='red', fontsize=6)
            
            ax2.plot([0, 0], [0, small_length], [0, 0], color='blue', linewidth=1.5)
            ax2.text(0, small_length * 1.2, 0, 'Z', color='blue', fontsize=6)
            
            ax2.plot([0, 0], [0, 0], [0, small_length], color='yellow', linewidth=1.5)
            ax2.text(0, 0, small_length * 1.2, 'Y', color='yellow', fontsize=6)
        
        # 작은 좌표축의 시점 설정
        ax2.view_init(elev=20, azim=45)
        ax2.set_box_aspect([1, 1, 1])
        
        positions = []
        valid_markers = []
        marker_positions = {}
        
        # 마커 위치 데이터 수집 - 좌표계 처리 수정
        for marker in self.marker_names:
            try:
                x = self.data.loc[self.frame_idx, f'{marker}_X']
                y = self.data.loc[self.frame_idx, f'{marker}_Y']
                z = self.data.loc[self.frame_idx, f'{marker}_Z']
                
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                    
                # Z-up 좌표계와 Y-up 좌표계 전환 수정
                if self.is_z_up:
                    marker_positions[marker] = np.array([x, y, z])
                    positions.append([x, y, z])
                else:
                    # Y-up 좌표계로 변환 (X, Z, Y 순서로 변경)
                    marker_positions[marker] = np.array([x, z, y])
                    positions.append([x, z, y])
                valid_markers.append(marker)
            except KeyError:
                continue
        
        positions = np.array(positions)
        
        # 데이터의 전체 범위 계산 (처음 로드할 때 한 번만)
        if not hasattr(self, 'data_limits'):
            x_data = self.data[[f'{marker}_X' for marker in self.marker_names]].values
            y_data = self.data[[f'{marker}_Y' for marker in self.marker_names]].values
            z_data = self.data[[f'{marker}_Z' for marker in self.marker_names]].values
            
            x_min, x_max = np.nanmin(x_data), np.nanmax(x_data)
            y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)
            z_min, z_max = np.nanmin(z_data), np.nanmax(z_data)
            
            margin = 0.2
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            self.data_limits = {
                'x': (x_min - x_range * margin, x_max + x_range * margin),
                'y': (y_min - y_range * margin, y_max + y_range * margin),
                'z': (z_min - z_range * margin, z_max + z_range * margin)
            }
            
            # 초기 뷰포트 설정 (처음 한 번만)
            if self.is_z_up:
                self.ax.set_xlim(self.data_limits['x'])
                self.ax.set_ylim(self.data_limits['y'])
                self.ax.set_zlim(self.data_limits['z'])
            else:
                self.ax.set_xlim(self.data_limits['x'])
                self.ax.set_ylim(self.data_limits['z'])  # Y-up에서는 Z와 Y를 교체
                self.ax.set_zlim(self.data_limits['y'])
        
        # 뷰 상태 복원
        if prev_xlim and prev_ylim and prev_zlim:
            self.ax.view_init(elev=prev_elev, azim=prev_azim)
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
            self.ax.set_zlim(prev_zlim)
        else:
            # 데이터 범위로 축 설정
            if self.is_z_up:
                self.ax.set_xlim(self.data_limits['x'])
                self.ax.set_ylim(self.data_limits['y'])
                self.ax.set_zlim(self.data_limits['z'])
            else:
                self.ax.set_xlim(self.data_limits['x'])
                self.ax.set_ylim(self.data_limits['z'])  # Y-up에서는 Z와 Y를 교체
                self.ax.set_zlim(self.data_limits['y'])
        
        # 마커 점 그리기
        if hasattr(self, 'current_marker'):
            # 선택된 마커와 나머지 마커 분리
            selected_indices = [i for i, marker in enumerate(valid_markers) if marker == self.current_marker]
            other_indices = [i for i, marker in enumerate(valid_markers) if marker != self.current_marker]
            
            # 선택되지 않은 마커 그리기 (흰색, 기본 크기)
            if other_indices:
                other_positions = positions[other_indices]
                self.ax.scatter(other_positions[:, 0], other_positions[:, 1], other_positions[:, 2], 
                            picker=5, color='white', s=30)
            
            # 선택된 마커 그리기 (연한 노란색, 중간 크기)
            if selected_indices:
                selected_positions = positions[selected_indices]
                self.ax.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2], 
                            picker=5, color='#FFFF99', s=50)  # 연한 노란색으로 변경, 크기 50으로 조정
        else:
            # 모든 마커를 흰색으로 그리기 (기본 상태)
            self.sc = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                                    picker=5, color='white', s=30)
        self.valid_markers = valid_markers
        
        # 마커 이름 표시 (토글 상태 따라)
        if self.show_names:
            for i, marker in enumerate(valid_markers):
                pos = positions[i]
                self.ax.text(pos[0], pos[1], pos[2], marker, 
                            color='white', fontsize=8)
        
        # 스켈레톤 라인 그리기 부분 수정
        if self.show_skeleton:
            for pair in self.skeleton_pairs:
                if pair[0] in marker_positions and pair[1] in marker_positions:
                    p1 = marker_positions[pair[0]]
                    p2 = marker_positions[pair[1]]
                    
                    # outlier 여부에 따라 색상과 두께 결정
                    is_outlier = (self.outliers[pair[0]][self.frame_idx] or 
                                self.outliers[pair[1]][self.frame_idx])
                    
                    line_color = 'red' if is_outlier else 'gray'
                    line_width = 2 if is_outlier else 1
                    line_alpha = 0.8 if is_outlier else 0.5
                    
                    # 좌표계에 따른 라인 그리기
                    self.ax.plot([p1[0], p2[0]], 
                            [p1[1], p2[1]], 
                            [p1[2], p2[2]], 
                            color=line_color, 
                            alpha=line_alpha, 
                            linewidth=line_width)

        self.canvas.draw()
        plt.pause(0.01)  # 즉시 반영하도록 대기 시간을 짧게 설정

    def on_pick(self, event):
        """마커 선택 시 이벤트 핸들러"""
        try:
            # 왼쪽 또는 오른쪽 클릭만 처리
            if event.mouseevent.button != 3:  # 3: 오른쪽
                return

            # 현재 뷰 상태 저장
            current_view = {
                'elev': self.ax.elev,
                'azim': self.ax.azim,
                'xlim': self.ax.get_xlim(),
                'ylim': self.ax.get_ylim(),
                'zlim': self.ax.get_zlim()
            }

            # 선택된 마커의 인덱스 확인
            if not hasattr(self, 'valid_markers') or not self.valid_markers:
                print("No valid markers available")
                return

            ind = event.ind[0]
            if ind >= len(self.valid_markers):
                print(f"Invalid marker index: {ind}")
                return

            # 선택된 마커 저장
            self.current_marker = self.valid_markers[ind]
            print(f"Selected Marker: {self.current_marker}")

            # 오른쪽 클릭일 때만 그래프 표시
            if event.mouseevent.button == 3:
                if self.current_marker in self.marker_names:  # 마커가 유효한지 확인
                    self.show_marker_plot(self.current_marker)
                else:
                    print(f"Invalid marker name: {self.current_marker}")
                    return

            # 뷰 상태를 복원하고 업데이트
            self.update_plot()

            # 저장된 뷰 상태 복원
            self.ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
            self.ax.set_xlim(current_view['xlim'])
            self.ax.set_ylim(current_view['ylim'])
            self.ax.set_zlim(current_view['zlim'])
            self.canvas.draw()

        except Exception as e:
            print(f"Error in on_pick: {str(e)}")
            import traceback
            traceback.print_exc()  # 상세한 에러 정보 출력

        finally:
            # 확대/축소 이벤트가 계속 작동하도록 이벤트 재연결
            self.connect_mouse_events()

    def show_marker_plot(self, marker_name):
        """마커 선택 시 오른쪽에 그래프 표시"""
        # 이전 interpolation 설정 저장
        prev_interp_method = None
        prev_order = None
        if hasattr(self, 'interp_method_var'):
            prev_interp_method = self.interp_method_var.get()
        if hasattr(self, 'order_var'):
            prev_order = self.order_var.get()

        # 그래프 프레임이 숨겨져 있으면 표시
        if not self.graph_frame.winfo_ismapped():
            # 구분선 표시
            self.separator.pack(side='left', fill='y', padx=2)
            # 그래프 프레임 표시
            self.graph_frame.pack(side='right', fill='both', expand=True)
            
        # 기존 그래프 제거
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # 새 그래프 생성
        self.marker_plot_fig = Figure(figsize=(6, 8), facecolor='black')
        self.marker_plot_fig.patch.set_facecolor('black')
        
        # 현재 선택된 마커 저장
        self.current_marker = marker_name
        
        # X, Y, Z 좌표 그래프
        self.marker_axes = []
        self.marker_lines = []  # 수직선 리스트 초기화
        coords = ['X', 'Y', 'Z']

        # outlier 데이터 확인 (No skeleton일 때 오류 발생)
        if not hasattr(self, 'outliers') or marker_name not in self.outliers:
            self.outliers = {marker_name: np.zeros(len(self.data), dtype=bool)}  # set to False
        
        # outlier 프레임 인덱스 가져오기
        outlier_frames = np.where(self.outliers[marker_name])[0]
        
        for i, coord in enumerate(coords):
            ax = self.marker_plot_fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('black')
            
            # 데이터 준비
            data = self.data[f'{marker_name}_{coord}']
            frames = np.arange(len(data))
            
            # 정상 데이터 플롯 (흰색)
            ax.plot(frames[~self.outliers[marker_name]], 
                   data[~self.outliers[marker_name]], 
                   color='white', 
                   label='Normal')
            
            # outlier 데이터 플롯 (빨간색)
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
            
            # 현재 프레임 표시
            line = ax.axvline(x=self.frame_idx, color='red', linestyle='--')
            self.marker_lines.append(line)
            self.marker_axes.append(ax)
            
            # 범례 표시 - 위치 고정
            if len(outlier_frames) > 0:
                ax.legend(facecolor='black', 
                         labelcolor='white',
                         loc='upper right',  # 우측 상단에 고정
                         bbox_to_anchor=(1.0, 1.0))
        
        self.marker_plot_fig.tight_layout()
        
        # 캔버스에 그래프 추가
        self.marker_canvas = FigureCanvasTkAgg(self.marker_plot_fig, master=self.graph_frame)
        self.marker_canvas.draw()
        self.marker_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 초기 그래프 범위 저
        self.initial_graph_limits = []
        for ax in self.marker_axes:
            self.initial_graph_limits.append({
                'x': ax.get_xlim(),
                'y': ax.get_ylim()
            })
        
        # 마우스 이벤트 바인딩 추가
        self.marker_canvas.mpl_connect('scroll_event', self.on_marker_scroll)
        self.marker_canvas.mpl_connect('button_press_event', self.on_marker_mouse_press)
        self.marker_canvas.mpl_connect('button_release_event', self.on_marker_mouse_release)
        self.marker_canvas.mpl_connect('motion_notify_event', self.on_marker_mouse_move)
        
        # 그래프 상단에 초기화 버튼 프레임 추가
        button_frame = ctk.CTkFrame(self.graph_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        # 그래프 초기화 버튼
        reset_button = ctk.CTkButton(button_frame,
                                    text="Reset View",
                                    command=self.reset_graph_view,
                                    width=80,
                                    fg_color="#333333",
                                    hover_color="#444444")
        reset_button.pack(side='right', padx=5)
        
        # Edit 버튼 추가
        self.edit_button = ctk.CTkButton(button_frame,
                                        text="Edit",
                                        command=self.toggle_edit_menu,
                                        width=80,
                                        fg_color="#333333",
                                        hover_color="#444444")
        self.edit_button.pack(side='right', padx=5)
        
        # 편집 메뉴 프레임 (초기에는 숨김)
        self.edit_menu = ctk.CTkFrame(self.graph_frame)
        
        # 편집 메뉴에 버튼들 추가
        edit_buttons = [
            ("Delete", self.delete_selected_data),
            ("Interpolate", self.interpolate_selected_data),
            ("Restore", self.restore_original_data),  # 원본 데이터 복원 버튼 추가
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
        
        # 편집 메뉴에 인터폴레이션 방법 선택을 위한 콤보박스 추가
        self.interp_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial', 'spline', 'barycentric', 'krogh', 'pchip', 'akima', 'from_derivatives']
        self.interp_method_var = ctk.StringVar(value='linear' if prev_interp_method is None else prev_interp_method)
        interp_label = ctk.CTkLabel(self.edit_menu, text="Interpolation Method:")
        interp_label.pack(side='left', padx=5)
        self.interp_combo = ctk.CTkComboBox(self.edit_menu, 
                                           values=self.interp_methods, 
                                           variable=self.interp_method_var, 
                                           command=self.on_interp_method_change)
        self.interp_combo.pack(side='left', padx=5)

        # 차수 입력 필드
        self.order_var = ctk.IntVar(value=2 if prev_order is None else prev_order)
        self.order_entry = ctk.CTkEntry(self.edit_menu, textvariable=self.order_var, width=50)
        self.order_label = ctk.CTkLabel(self.edit_menu, text="Order:")
        self.order_label.pack(side='left', padx=5)
        self.order_entry.pack(side='left', padx=5)

        # 이전 interpolation 방법에 따라 차수 입력 필드 상태 설정
        if prev_interp_method in ['polynomial', 'spline']:
            self.order_entry.configure(state='normal')
        else:
            self.order_entry.configure(state='disabled')
        
        # 선택 영역 변수 초기화 (각 축별로 독립적인 선택 영역)
        self.selection_data = {
            'start': None,
            'end': None,
            'current_ax': None,
            'rect': None
        }
        
        # 확대/축소 및 패닝 이벤트 재연결
        self.connect_mouse_events()
    
    def on_interp_method_change(self, choice):
        """인터폴레이션 방법이 변경될 때 호출"""
        if choice in ['polynomial', 'spline']:
            self.order_entry.configure(state='normal')
        else:
            self.order_entry.configure(state='disabled')

    def toggle_edit_menu(self):
        """편집 메뉴 표시/숨김 토글"""
        if self.edit_menu.winfo_ismapped():
            self.edit_menu.pack_forget()
            self.edit_button.configure(fg_color="#333333")
            self.clear_selection()
        else:
            self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
            self.edit_button.configure(fg_color="#555555")

    def clear_selection(self):
        """선택 영역 초기화"""
        if 'rects' in self.selection_data and self.selection_data['rects']:
            for rect in self.selection_data['rects']:
                rect.remove()
            self.selection_data['rects'] = []
        if hasattr(self, 'marker_canvas'):
            self.marker_canvas.draw_idle()
        self.selection_in_progress = False  # 드래그 상태 초기화


    def on_marker_mouse_press(self, event):
        """마커 그래프에서 마우스 버튼 눌렀을 때"""
        if event.inaxes is None:
            return

        if event.button == 2:  # 휠 클릭
            self.marker_pan_enabled = True
            self.marker_last_pos = (event.xdata, event.ydata)
        elif event.button == 1 and hasattr(self, 'edit_menu') and self.edit_menu.winfo_ismapped():
            if event.xdata is not None:
                # 현재 선택 영역이 있는지 확인
                if self.selection_data.get('rects'):
                    # 클릭 위치가 현재 선택 영역 밖인지 확인
                    start = min(self.selection_data['start'], self.selection_data['end'])
                    end = max(self.selection_data['start'], self.selection_data['end'])
                    if not (start <= event.xdata <= end):
                        # 영역 밖 클릭시 선택 영역 제거
                        self.clear_selection()
                        # 새로운 선택 시작
                        self.start_new_selection(event)
                else:
                    # 선택 영역이 없는 경우 새로운 선택 시작
                    self.start_new_selection(event)


    def on_marker_mouse_release(self, event):
        """마커 그래프에서 마우스 버튼을 놓았을 때"""
        if event.button == 2:  # 휠 클릭 해제
            self.marker_pan_enabled = False
            self.marker_last_pos = None
        elif event.button == 1 and hasattr(self, 'edit_menu') and self.edit_menu.winfo_ismapped():
            if self.selection_data.get('start') is not None and event.xdata is not None:
                self.selection_data['end'] = event.xdata
                self.selection_in_progress = False  # 드래그 종료
                self.highlight_selection()


    def highlight_selection(self):
        """선택 영역 강조 표시"""
        if self.selection_data.get('start') is None or self.selection_data.get('end') is None:
            return

        start_frame = min(self.selection_data['start'], self.selection_data['end'])
        end_frame = max(self.selection_data['start'], self.selection_data['end'])

        # 기존 선택 영역 제거
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

        # 현재 그래프의 뷰 상태 저장
        view_states = []
        for ax in self.marker_axes:
            view_states.append({
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim()
            })

        # 현재 선택 영역 저장
        current_selection = {
            'start': self.selection_data['start'],
            'end': self.selection_data['end']
        }

        start_frame = min(int(self.selection_data['start']), int(self.selection_data['end']))
        end_frame = max(int(self.selection_data['start']), int(self.selection_data['end']))

        for coord in ['X', 'Y', 'Z']:
            col_name = f'{self.current_marker}_{coord}'
            self.data.loc[start_frame:end_frame, col_name] = np.nan

        # 그래프 업데이트 (edit 모드 유지)
        self.show_marker_plot(self.current_marker)
        
        # 저장된 뷰 상태 복원
        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])
        
        # 3D 뷰어 업데이트
        self.update_plot()
        
        # 선택 영역 복원
        self.selection_data['start'] = current_selection['start']
        self.selection_data['end'] = current_selection['end']
        self.highlight_selection()
        
        # edit 모드 상태 복원
        self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
        self.edit_button.configure(fg_color="#555555")


    def interpolate_selected_data(self):
        """선택된 영역의 데이터를 보간"""
        if self.selection_data['start'] is None or self.selection_data['end'] is None:
            return

        # 현재 그래프의 뷰 상태 저장
        view_states = []
        for ax in self.marker_axes:
            view_states.append({
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim()
            })

        # 현재 선택 영역 저장
        current_selection = {
            'start': self.selection_data['start'],
            'end': self.selection_data['end']
        }

        start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
        end_frame = int(max(self.selection_data['start'], self.selection_data['end']))

        # 선택된 보간 방법과 차수 가져오기
        method = self.interp_method_var.get()
        order = None
        if method in ['polynomial', 'spline']:
            try:
                order = self.order_var.get()
            except:
                messagebox.showerror("Error", "Please enter a valid order number")
                return

        # 각 좌표에 대해 보간 수행
        for coord in ['X', 'Y', 'Z']:
            col_name = f'{self.current_marker}_{coord}'
            series = self.data[col_name]

            # 선택된 범위의 데이터를 NaN으로 설정
            self.data.loc[start_frame:end_frame, col_name] = np.nan

            # 보간 수행
            interp_kwargs = {}
            if order is not None:
                interp_kwargs['order'] = order

            try:
                self.data[col_name] = series.interpolate(method=method, **interp_kwargs)
            except Exception as e:
                print(f"Interpolation error for {coord} with method '{method}': {e}")
                messagebox.showerror("Interpolation Error", f"Error interpolating {coord} with method '{method}': {e}")
                return

        # 보간 후 outlier 재탐지
        self.detect_outliers()

        # 그래프 업데이트 (edit 모드 유지)
        self.show_marker_plot(self.current_marker)
        
        # 저장된 뷰 상태 복원
        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])
        
        # 3D 뷰어 업데이트
        self.update_plot()
        
        # 선택 영역 복원
        self.selection_data['start'] = current_selection['start']
        self.selection_data['end'] = current_selection['end']
        self.highlight_selection()
        
        # edit 모드 상태 복원
        self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
        self.edit_button.configure(fg_color="#555555")

    def restore_original_data(self):
        """원본 데이터로 복원"""
        if self.original_data is not None:
            self.data = self.original_data.copy(deep=True)
            self.detect_outliers()
            self.show_marker_plot(self.current_marker)
            self.update_plot()
            # edit 모드 상태 복원
            self.edit_menu.pack(after=self.edit_button.winfo_parent(), pady=5)
            self.edit_button.configure(fg_color="#555555")
            print("Data has been restored to the original state.")
        else:
            messagebox.showinfo("Restore Data", "No original data to restore.")

    def on_scroll(self, event):
        """마우스 휠 이벤트 처리"""
        try:
            # 이벤트가 유효한지 확인
            if event.inaxes != self.ax:
                return

            # 현재 축 범위 가져오기
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            z_min, z_max = self.ax.get_zlim()

            # 확대/축소 비율
            scale_factor = 0.9 if event.button == 'up' else 1.1

            # 축 범위의 중심 계산
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            # 새로운 범위 계산
            x_range = (x_max - x_min) * scale_factor
            y_range = (y_max - y_min) * scale_factor
            z_range = (z_max - z_min) * scale_factor

            # 최소/최대 축 범위 제한 설정
            min_range = 1e-3  # 너무 작은 값으로 축소되지 않도록 최소 범위 설정
            max_range = 1e5   # 너무 크게 확대되지 않도록 최대 범위 설정

            x_range = max(min(x_range, max_range), min_range)
            y_range = max(min(y_range, max_range), min_range)
            z_range = max(min(z_range, max_range), min_range)

            # 새로운 축 범위 설정
            self.ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            self.ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            self.ax.set_zlim(z_center - z_range / 2, z_center + z_range / 2)

            # 캔버스 업데이트
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Scroll event error: {e}")
            # 오류 발생 시 이벤트 재연결
            self.connect_mouse_events()

    def on_marker_scroll(self, event):
        """마커 그래프의 확대/축소 처리"""
        if not event.inaxes:
            return
        
        # 현재 축의 범위
        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # 마우스 위치를 기준으로 확대/축소
        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2
        
        # 확대/축소 비율
        scale_factor = 0.9 if event.button == 'up' else 1.1
        
        # 새로운 범위 계산
        new_x_range = (x_max - x_min) * scale_factor
        new_y_range = (y_max - y_min) * scale_factor
        
        # 마우스 위치를 중심으로 새로운 범위 설정
        x_left = x_center - new_x_range * (x_center - x_min) / (x_max - x_min)
        x_right = x_center + new_x_range * (x_max - x_center) / (x_max - x_min)
        y_bottom = y_center - new_y_range * (y_center - y_min) / (y_max - y_min)
        y_top = y_center + new_y_range * (y_max - y_center) / (y_max - y_min)
        
        # 범위 적용
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        
        # 그래프 업데이트
        self.marker_canvas.draw_idle()

    def toggle_coordinates(self):
        if self.data is None:
            return
        
        self.is_z_up = not self.is_z_up
        self.coord_button.configure(text="Switch to Y-up" if self.is_z_up else "Switch to Z-up")
        
        # view_limits 초기화 (새로운 좌표계에 맞게 재계산하기 위해)
        self.view_limits = None
        self.update_plot()

    def detect_outliers(self):
        """각 마커의 outlier 프레임 탐지"""
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
        """이전 프레임으로 이동"""
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.frame_slider.set(self.frame_idx)
            self.update_plot()
            
            # 마커 그래프가 표시되어 있다면 수직선 업데이트
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata(self.frame_idx)
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()

    def next_frame(self):
        """음 프레임으로 이동"""
        if self.frame_idx < self.num_frames - 1:
            self.frame_idx += 1
            self.frame_slider.set(self.frame_idx)
            self.update_plot()
            
            # 마커 그래프가 표시되어 있다면 수직선 업데이트
            if hasattr(self, 'marker_lines') and self.marker_lines:
                for line in self.marker_lines:
                    line.set_xdata(self.frame_idx)
                if hasattr(self, 'marker_canvas'):
                    self.marker_canvas.draw()

    def toggle_marker_names(self):
        """마커 이름 표시/숨김 토글"""
        self.show_names = not self.show_names
        self.names_button.configure(text="Show Names" if not self.show_names else "Hide Names")
        self.update_plot()

    def toggle_skeleton(self):
        """스켈레톤 라인 표시/숨김 토글"""
        self.show_skeleton = not self.show_skeleton
        self.skeleton_button.configure(text="Show Skeleton" if not self.show_skeleton else "Hide Skeleton")
        self.update_plot()

    def on_mouse_press(self, event):
        if event.button == 1:  # 좌클릭
            self.pan_enabled = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1:  # 좌클릭 해제
            self.pan_enabled = False
            self.last_mouse_pos = None

    def on_mouse_move(self, event):
        if self.pan_enabled and event.xdata is not None and event.ydata is not None:
            # 현재 축 범위 가져오기
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            z_min, z_max = self.ax.get_zlim()

            # 마우스 이동량 계산
            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]

            # 새로운 축 범위 계산
            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            # 최소/최대 축 범위 제한 설정
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
        """마커 그래프에서 마우스를 움직일 때"""
        if not hasattr(self, 'marker_pan_enabled'):
            self.marker_pan_enabled = False
        if not hasattr(self, 'selection_in_progress'):
            self.selection_in_progress = False

        if self.marker_pan_enabled and self.marker_last_pos:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                # 이동량 계산
                dx = event.xdata - self.marker_last_pos[0]
                dy = event.ydata - self.marker_last_pos[1]

                # 현재 축의 범위 가져오기
                ax = event.inaxes
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                # 새로운 범위 설정
                ax.set_xlim(x_min - dx, x_max - dx)
                ax.set_ylim(y_min - dy, y_max - dy)

                # 마지막 위치 업데이트
                self.marker_last_pos = (event.xdata, event.ydata)

                # 그래프 업데이트
                self.marker_canvas.draw_idle()
        elif self.selection_in_progress and event.xdata is not None:
            # 선택 영역 업데이트
            self.selection_data['end'] = event.xdata
            
            # 선택 영역 사각형 업데이트
            start_x = min(self.selection_data['start'], self.selection_data['end'])
            width = abs(self.selection_data['end'] - self.selection_data['start'])
            
            for rect in self.selection_data['rects']:
                rect.set_x(start_x)
                rect.set_width(width)
            
            self.marker_canvas.draw_idle()

    def reset_main_view(self):
        """메인 3D 뷰어의 시점을 초기화"""
        if hasattr(self, 'data_limits'):
            try:
                # 뷰 초기화
                self.ax.view_init(elev=20, azim=45)
                
                # 데이터 범위 초기화
                if self.is_z_up:
                    self.ax.set_xlim(self.data_limits['x'])
                    self.ax.set_ylim(self.data_limits['y'])
                    self.ax.set_zlim(self.data_limits['z'])
                else:
                    self.ax.set_xlim(self.data_limits['x'])
                    self.ax.set_ylim(self.data_limits['z'])
                    self.ax.set_zlim(self.data_limits['y'])
                
                # 그리드 설정 초기화
                self.ax.grid(True)
                
                # 종횡비 초기화
                self.ax.set_box_aspect([1.0, 1.0, 1.0])
                
                # 캔버스 업데이트
                self.canvas.draw()
                
                # 뷰 상태 저장
                self.view_limits = {
                    'x': self.ax.get_xlim(),
                    'y': self.ax.get_ylim(),
                    'z': self.ax.get_zlim()
                }
                
                print("Camera view reset successfully")  # 디버깅용 메시지
                
            except Exception as e:
                print(f"Error resetting camera view: {e}")  # 오류 발생 시 출력

    def reset_graph_view(self):
        """마커 그래프의 시점을 초기화"""
        if hasattr(self, 'marker_axes') and hasattr(self, 'initial_graph_limits'):
            for ax, limits in zip(self.marker_axes, self.initial_graph_limits):
                ax.set_xlim(limits['x'])
                ax.set_ylim(limits['y'])
            self.marker_canvas.draw()

    def start_new_selection(self, event):
        """새로운 선택 영역 시작"""
        self.selection_data = {
            'start': event.xdata,
            'end': event.xdata,  # 초기에는 시작점과 같게 설정
            'rects': []
        }
        self.selection_in_progress = True
        
        # 선택 영역 사각형 초기화
        for ax in self.marker_axes:
            ylim = ax.get_ylim()
            rect = plt.Rectangle((event.xdata, ylim[0]),
                                0,  # 초기 너비는 0
                                ylim[1] - ylim[0],
                                facecolor='yellow',
                                alpha=0.2)
            self.selection_data['rects'].append(ax.add_patch(rect))
        self.marker_canvas.draw_idle()

# 애플리케이션 실행
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = TRCViewer()
    app.mainloop()
