from gui.opengl.GLPlotCreator import MarkerGLFrame
from gui.opengl.GLPlotUpdater import update_plot
from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import numpy as np
import pandas as pd
from .GridUtils import create_opengl_grid

# 좌표계 회전 상수
COORDINATE_X_ROTATION_Y_UP = -270.0  # Y-up 좌표계에서 X축 회전 각도 (-270도)
COORDINATE_X_ROTATION_Z_UP = -90.0   # Z-up 좌표계에서 X축 회전 각도 (-90도)

# 좌표계 문자열 상수
COORDINATE_SYSTEM_Y_UP = "y-up"
COORDINATE_SYSTEM_Z_UP = "z-up"

class MarkerGLRenderer(MarkerGLFrame):
    """완전한 마커 시각화 OpenGL 렌더러"""
    
    def __init__(self, parent, **kwargs):
        """
        마커 데이터를 OpenGL로 렌더링하는 프레임 초기화
        
        좌표계:
        - Y-up: 기본 좌표계, Y축이 상단을 향함
        - Z-up: Z축이 상단을 향하고, X-Y가 바닥 평면
        """
        super().__init__(parent, **kwargs)
        
        # 기본 좌표계 설정 (Y-up)
        self.is_z_up = False
        self.coordinate_system = COORDINATE_SYSTEM_Y_UP
        
        # 내부 상태 및 데이터 저장용 변수
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
        self.rot_x = 90  # X축 기준으로 -270도 회전
        self.rot_y = 45.0
        self.zoom = -4.0
        
        # 초기화 완료 플래그
        self.initialized = False
        self.gl_initialized = False
        
    def initialize(self):
        """
        OpenGL 렌더러 초기화 - gui/plotCreator.py에서 호출됨
        pyopengltk는 initgl 메서드를 통해 OpenGL을 초기화하므로,
        여기서는 초기화 플래그만 설정합니다.
        """
        print("OpenGL 렌더러 초기화 시작")
        
        # 초기화 완료 표시 - 실제 OpenGL 초기화는 initgl에서 수행됨
        self.initialized = True
        
        print("OpenGL 렌더러 초기화 완료")
        
        # 화면 갱신
        self.update()  # 자동으로 initgl과 redraw를 호출함
        
    def initgl(self):
        """OpenGL 초기화 (pyopengltk에서 자동 호출)"""
        try:
            # 부모 클래스의 initgl 호출
            super().initgl()
            
            # 배경색 설정 (검정)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            
            # 깊이 테스트 활성화
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            # 포인트 크기와 선 폭 설정
            GL.glPointSize(5.0)
            GL.glLineWidth(2.0)
            
            # 조명 비활성화 - 모든 각도에서 일정한 색상을 위해
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_LIGHT0)
            
            # 기존 디스플레이 리스트 제거 (혹시 있다면)
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glDeleteLists(self.grid_list, 1)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glDeleteLists(self.axes_list, 1)
            
            # 이제 OpenGL 컨텍스트가 완전히 초기화된 상태에서 디스플레이 리스트 생성
            self._create_grid_display_list()
            self._create_axes_display_list()
            
            # OpenGL 초기화 완료 플래그 설정
            self.gl_initialized = True
            print("OpenGL 컨텍스트 초기화 완료")
        except Exception as e:
            print(f"OpenGL 초기화 오류: {e}")
            self.gl_initialized = False
        
    def _create_grid_display_list(self):
        """그리드 표시를 위한 디스플레이 리스트 생성"""
        if hasattr(self, 'grid_list') and self.grid_list is not None:
            GL.glDeleteLists(self.grid_list, 1)
            
        # 중앙화된 유틸리티 함수 사용
        is_z_up = getattr(self, 'is_z_up', True)
        self.grid_list = create_opengl_grid(
            grid_size=2.0, 
            grid_divisions=20, 
            color=(0.3, 0.3, 0.3),
            is_z_up=is_z_up
        )
        
    def _create_axes_display_list(self):
        """좌표축 표시를 위한 디스플레이 리스트 생성"""
        if hasattr(self, 'axes_list') and self.axes_list is not None:
            GL.glDeleteLists(self.axes_list, 1)
            
        self.axes_list = GL.glGenLists(1)
        GL.glNewList(self.axes_list, GL.GL_COMPILE)
        
        # 후면 컬링 비활성화
        GL.glDisable(GL.GL_CULL_FACE)
        
        # 축 길이 (원래 스타일 유지)
        axis_length = 0.2
        
        # 그리드와 확실히 구분되도록 원점을 이동 - 그리드 위에 띄우기
        offset_y = 0.001
        
        # 축 굵기 설정 (원래 스타일 유지)
        original_line_width = GL.glGetFloatv(GL.GL_LINE_WIDTH)
        GL.glLineWidth(3.0)
        
        # Z-up 좌표계에 맞게 축 그리기 (회전 매트릭스가 적용됨)
        # X축 (빨강)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(1.0, 0.0, 0.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(axis_length, offset_y, 0)
        
        # Y축 (노랑)
        GL.glColor3f(1.0, 1.0, 0.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(0, axis_length + offset_y, 0)
        
        # Z축 (파랑)
        GL.glColor3f(0.0, 0.0, 1.0)
        GL.glVertex3f(0, offset_y, 0)
        GL.glVertex3f(0, offset_y, axis_length)
        GL.glEnd()
        
        # 축 라벨 텍스트 그리기 (GLUT 사용 - 원래 스타일 유지)
        text_offset = 0.06  # 축 끝에서 텍스트를 띄울 거리
        
        # 조명 비활성화 (텍스트 색상이 제대로 나오도록)
        lighting_enabled = GL.glIsEnabled(GL.GL_LIGHTING)
        if lighting_enabled:
            GL.glDisable(GL.GL_LIGHTING)
        
        # X 라벨
        GL.glColor3f(1.0, 0.0, 0.0)  # 빨강
        GL.glRasterPos3f(axis_length + text_offset, offset_y, 0)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('X'))
        except:
            pass  # GLUT을 사용할 수 없는 경우 라벨 표시 생략
        
        # Y 라벨
        GL.glColor3f(1.0, 1.0, 0.0)  # 노랑
        GL.glRasterPos3f(0, axis_length + text_offset + offset_y, 0)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Y'))
        except:
            pass
        
        # Z 라벨
        GL.glColor3f(0.0, 0.0, 1.0)  # 파랑
        GL.glRasterPos3f(0, offset_y, axis_length + text_offset)
        try:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Z'))
        except:
            pass
        
        # 원래 상태 복원
        GL.glLineWidth(original_line_width)
        if lighting_enabled:
            GL.glEnable(GL.GL_LIGHTING)  # 조명 다시 활성화
        
        GL.glEnable(GL.GL_CULL_FACE)
        
        GL.glEndList()
    
    def redraw(self, *args):
        """프레임 업데이트 (pyopengltk에서 자동 호출)"""
        if not self.gl_initialized:
            return
            
        # 디버깅 코드 제거
        update_plot(self)
        
    def update_data(self, data, frame_idx):
        """외부에서 호출하여 데이터 업데이트 (하위 호환성 유지)"""
        self.data = data
        self.frame_idx = frame_idx
        if data is not None:
            self.num_frames = len(data)
        self.initialized = True
        self.redraw()
    
    def set_frame_data(self, data, frame_idx, marker_names, current_marker=None, 
                       show_marker_names=False, show_trajectory=False, 
                       coordinate_system="z-up", skeleton_pairs=None):
        """
        TRCViewer에서 호출되는 통합 데이터 업데이트 메서드
        
        Args:
            data: 전체 마커 데이터
            frame_idx: 현재 프레임 인덱스
            marker_names: 마커 이름 목록
            current_marker: 현재 선택된 마커 이름
            show_marker_names: 마커 이름 표시 여부
            show_trajectory: 궤적 표시 여부
            coordinate_system: 좌표계 시스템 ("z-up" 또는 "y-up")
            skeleton_pairs: 스켈레톤 페어 목록
        """
        self.data = data
        self.frame_idx = frame_idx
        self.marker_names = marker_names
        self.current_marker = current_marker
        self.show_marker_names = show_marker_names
        self.show_trajectory = show_trajectory
        self.coordinate_system = coordinate_system
        self.skeleton_pairs = skeleton_pairs
        
        # 데이터가 있는 경우 프레임 수 업데이트
        if data is not None:
            self.num_frames = len(data)
            
        # OpenGL 초기화 확인
        self.initialized = True
        
        # 즉시 다시 그리기
        self.redraw()
        
    def set_current_marker(self, marker_name):
        """현재 선택된 마커 설정"""
        self.current_marker = marker_name
        self.redraw()
    
    def set_show_skeleton(self, show):
        """
        스켈레톤 표시 여부 설정
        
        Args:
            show: 스켈레톤을 표시하려면 True, 아니면 False
        """
        self.show_skeleton = show
        self.redraw()
    
    def set_show_trajectory(self, show):
        """궤적 표시 설정"""
        self.show_trajectory = show
        self.redraw()
        
    def update_plot(self):
        """
        외부(plotUpdater.py)에서 호출하여 화면 업데이트
        redraw 메서드는 내부적으로 GLPlotUpdater의 update_plot을 호출함
        """
        if self.gl_initialized:
            self.redraw()
        
    def set_pattern_selection_mode(self, mode, pattern_markers=None):
        """패턴 선택 모드 설정"""
        self.pattern_selection_mode = mode
        if pattern_markers is not None:
            self.pattern_markers = pattern_markers
        self.redraw()
    
    def set_coordinate_system(self, is_z_up):
        """
        좌표계 설정 변경
        
        Args:
            is_z_up: Z-up 좌표계를 사용하려면 True, Y-up 좌표계를 사용하려면 False
        
        주의:
        좌표계 변경은 마커의 실제 좌표를 변경하지 않고 표시 방법만 변경합니다.
        데이터는 항상 원래 좌표계를 유지합니다.
        """
        # 변화가 없으면 불필요한 처리를 하지 않음
        if self.is_z_up == is_z_up:
            return
        
        # 좌표계 상태 업데이트
        self.is_z_up = is_z_up
        
        # 좌표계 문자열 업데이트
        self.coordinate_system = COORDINATE_SYSTEM_Z_UP if is_z_up else COORDINATE_SYSTEM_Y_UP
        
        # 좌표계에 따라 축 디스플레이 리스트 재생성
        if self.gl_initialized:
            try:
                # OpenGL 컨텍스트 활성화 - 필수
                self.tkMakeCurrent()
                
                # 기존 축과 그리드 디스플레이 리스트 삭제
                if hasattr(self, 'axes_list') and self.axes_list is not None:
                    GL.glDeleteLists(self.axes_list, 1)
                if hasattr(self, 'grid_list') and self.grid_list is not None:
                    GL.glDeleteLists(self.grid_list, 1)
                
                # 새 좌표계에 맞는 축과 그리드 생성
                self._create_axes_display_list()
                self._create_grid_display_list()
                
                # 화면 강제 갱신
                self.redraw()
                # 메인 이벤트 루프에서 좀 더 여유있게 갱신 요청
                self.after(20, self._force_redraw)
            except Exception as e:
                print(f"좌표계 변경 중 오류 발생: {e}")
    
    def _force_redraw(self):
        """강제로 화면을 다시 그립니다"""
        try:
            # OpenGL 상태 확인
            if not self.gl_initialized:
                return
                
            # 컨텍스트 활성화
            self.tkMakeCurrent()
            
            # 전체 화면을 지우고 다시 그림
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity()
            
            # 3D 장면 설정
            GL.glTranslatef(0.0, 0.0, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            
            # 디스플레이 리스트 호출
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)
            
            # 완전한 장면 갱신
            self.redraw()
            
            # 강제 버퍼 교체
            self.tkSwapBuffers()
            
            # TK 업데이트
            self.update()
            self.update_idletasks()
            
        except Exception as e:
            pass
    
    def reset_view(self):
        """
        뷰 초기화 - 기본 카메라 위치와 각도로 재설정
        """
        # 현재 좌표계에 맞는 X축 회전 각도 사용
        self.rot_x = COORDINATE_X_ROTATION_Y_UP  # Y-up이 기본 설정
        self.rot_y = 45.0
        self.zoom = -4.0
        self.redraw()
        
    def set_marker_names(self, marker_names):
        """마커 이름 목록 설정"""
        self.marker_names = marker_names
        self.redraw()
        
    def set_skeleton_pairs(self, skeleton_pairs):
        """스켈레톤 구성 쌍 설정"""
        self.skeleton_pairs = skeleton_pairs
        self.redraw()
        
    def set_outliers(self, outliers):
        """이상치 데이터 설정"""
        self.outliers = outliers
        self.redraw()
        
    # matplotlib 호환 메서드
    def get_figure(self):
        """matplotlib 호환 메서드"""
        # 호환성을 위한 더미 반환
        return self
    
    def get_axes(self):
        """matplotlib 호환 메서드"""
        # 호환성을 위한 더미 반환
        return self

    def set_data_limits(self, x_range, y_range, z_range):
        """
        데이터의 범위를 설정합니다.
        
        Args:
            x_range: X축 범위 (min, max)
            y_range: Y축 범위 (min, max)
            z_range: Z축 범위 (min, max)
        """
        self.data_limits = {
            'x': x_range,
            'y': y_range,
            'z': z_range
        }
        # 추가적으로 필요한 작업이 있다면 여기에 구현
