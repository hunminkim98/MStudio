from gui.opengl.GLPlotCreator import MarkerGLFrame
from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import numpy as np
import pandas as pd
from .GridUtils import create_opengl_grid
import ctypes

# 좌표계 회전 상수
COORDINATE_X_ROTATION_Y_UP = -270.0  # Y-up 좌표계에서 X축 회전 각도 (-270도)
COORDINATE_X_ROTATION_Z_UP = -90.0   # Z-up 좌표계에서 X축 회전 각도 (-90도)

# 좌표계 문자열 상수
COORDINATE_SYSTEM_Y_UP = "y-up"
COORDINATE_SYSTEM_Z_UP = "z-up"

# 픽킹 텍스처 클래스
class PickingTexture:
    """마커 선택을 위한 픽킹 텍스처 클래스"""
    
    def __init__(self):
        """픽킹 텍스처 초기화"""
        self.fbo = 0
        self.texture = 0
        self.depth_texture = 0
        self.width = 0
        self.height = 0
        self.initialized = False
        
    def init(self, width, height):
        """
        픽킹 텍스처 초기화
        
        Args:
            width: 텍스처 너비
            height: 텍스처 높이
        
        Returns:
            bool: 초기화 성공 여부
        """
        self.width = width
        self.height = height
        
        try:
            # FBO 생성
            self.fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            
            # ID 정보를 위한 텍스처 생성
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, width, height, 
                           0, GL.GL_RGB, GL.GL_FLOAT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, 
                                    GL.GL_TEXTURE_2D, self.texture, 0)
            
            # 깊이 정보를 위한 텍스처 생성
            self.depth_texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, width, height,
                           0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                    GL.GL_TEXTURE_2D, self.depth_texture, 0)
            
            # 읽기 버퍼 비활성화 (이전 GPU 호환성을 위함)
            GL.glReadBuffer(GL.GL_NONE)
            
            # 그리기 버퍼 설정
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # FBO 상태 확인
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                print(f"FBO 생성 오류, 상태: {status:x}")
                self.cleanup()
                return False
            
            # 기본 프레임버퍼로 복원
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"픽킹 텍스처 초기화 오류: {e}")
            self.cleanup()
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.texture != 0:
                GL.glDeleteTextures(self.texture)
                self.texture = 0
                
            if self.depth_texture != 0:
                GL.glDeleteTextures(self.depth_texture)
                self.depth_texture = 0
                
            if self.fbo != 0:
                GL.glDeleteFramebuffers(1, [self.fbo])
                self.fbo = 0
                
            self.initialized = False
        except Exception as e:
            print(f"픽킹 텍스처 정리 오류: {e}")
    
    def enable_writing(self):
        """픽킹 텍스처에 쓰기 활성화"""
        if not self.initialized:
            return False
            
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        return True
    
    def disable_writing(self):
        """픽킹 텍스처에 쓰기 비활성화"""
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
    
    def read_pixel(self, x, y):
        """
        위치에 해당하는 픽셀 정보 읽기
        
        Args:
            x: 화면의 X 좌표
            y: 화면의 Y 좌표
            
        Returns:
            tuple: (ObjectID, PrimID) 또는 None (선택된 객체가 없는 경우)
        """
        if not self.initialized:
            return None
        
        try:
            # 읽기 프레임버퍼로 FBO 바인딩
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # 픽셀 좌표가 텍스처 범위 내에 있는지 확인
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return None
            
            # 픽셀 정보 읽기
            data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGB, GL.GL_FLOAT)
            pixel_info = np.frombuffer(data, dtype=np.float32)
            
            # 기본 설정 복원
            GL.glReadBuffer(GL.GL_NONE)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            
            # 배경 픽셀 확인 (ID가 0이면 배경)
            if pixel_info[0] == 0.0:
                return None
                
            return (pixel_info[0], pixel_info[1], pixel_info[2])
            
        except Exception as e:
            print(f"픽셀 읽기 오류: {e}")
            return None

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
        self.parent = parent # Store the parent reference
        
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
        
        # 화면 이동(translation)을 위한 변수 추가
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.last_x = 0
        self.last_y = 0
        
        # 마우스 이벤트 바인딩 추가
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<B1-Motion>", self.on_mouse_move)
        self.bind("<ButtonPress-3>", self.on_right_mouse_press)
        self.bind("<ButtonRelease-3>", self.on_right_mouse_release)
        self.bind("<B3-Motion>", self.on_right_mouse_move)
        self.bind("<MouseWheel>", self.on_scroll)
        self.bind("<Configure>", self.on_configure) # Add binding for Configure event
        
        # 마커 픽킹 관련 변수
        self.picking_texture = PickingTexture()
        self.dragging = False
        
    def initialize(self):
        """
        OpenGL 렌더러 초기화 - gui/plotCreator.py에서 호출됨
        pyopengltk는 initgl 메서드를 통해 OpenGL을 초기화하므로,
        여기서는 초기화 플래그만 설정합니다.
        """
        
        # 초기화 완료 표시 - 실제 OpenGL 초기화는 initgl에서 수행됨
        self.initialized = True
        
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
            
            # 픽킹 텍스처 초기화
            width, height = self.winfo_width(), self.winfo_height()
            if width > 0 and height > 0:
                self.picking_texture.init(width, height)
            
            # OpenGL 초기화 완료 플래그 설정
            self.gl_initialized = True
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
    
    def redraw(self):
        """
        OpenGL 화면을 다시 그립니다.
        This is the main drawing method.
        """
        if not self.gl_initialized:
            return
            
        # 내부 update_plot 메서드 호출
        self._update_plot()
        
    def _update_plot(self):
        """
        OpenGL로 3D 마커 시각화 업데이트
        이전에 별도 파일 gui/opengl/GLPlotUpdater.py에 있던 기능을 클래스 내부로 통합
        
        좌표계:
        - Y-up: 기본 좌표계, Y축이 상단을 향함
        - Z-up: Z축이 상단을 향하고, X-Y가 바닥 평면
        """
        # OpenGL 초기화 확인
        if not self.gl_initialized:
            return
        
        # 현재 좌표계 상태 확인 (기본값: Y-up)
        is_z_up_local = getattr(self, 'is_z_up', False)
        
        try:
            # OpenGL 컨텍스트 활성화 (안전을 위해)
            try:
                self.tkMakeCurrent()
            except Exception as context_error:
                print(f"Error setting OpenGL context: {context_error}")
                return # Cannot proceed without context
            
            # --- Viewport and Projection Setup --- START
            width = self.winfo_width()
            height = self.winfo_height()
            if width <= 0 or height <= 0:
                 # Avoid division by zero or invalid viewport
                 return 

            # 1. Set the viewport to the entire widget area
            GL.glViewport(0, 0, width, height)

            # 2. Set up the projection matrix
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height)
            # Use perspective projection (like in pick_marker)
            GLU.gluPerspective(45, aspect, 0.1, 100.0) # fov, aspect, near, far

            # 3. Switch back to the modelview matrix for camera/object transformations
            GL.glMatrixMode(GL.GL_MODELVIEW)
            # --- Viewport and Projection Setup --- END
            
            # 프레임 초기화 (Clear after setting viewport/projection)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0) # Ensure clear color is set
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity() # Reset modelview matrix before camera setup
            
            # 카메라 위치 설정 (줌, 회전, 이동)
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            
            # 좌표계에 따른 추가 회전 적용
            # - Y-up: 기본 카메라 설정이 이미 Y-up (-270도)에 맞춰져 있으므로 추가 회전 불필요
            # - Z-up: Y-up 평면의 반대 방향을 보기 위해 X축 기준 -90도 추가 회전
            if is_z_up_local:
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)
            
            # 그리드와 축 표시 (디스플레이 리스트가 있는 경우에만)
            if hasattr(self, 'grid_list') and self.grid_list is not None:
                GL.glCallList(self.grid_list)
            if hasattr(self, 'axes_list') and self.axes_list is not None:
                GL.glCallList(self.axes_list)
            
            # 데이터가 없는 경우 기본 뷰만 표시하고 종료
            if self.data is None:
                self.tkSwapBuffers()
                return
            
            # 마커 위치 데이터 수집
            positions = []
            colors = []
            selected_position = None
            marker_positions = {}
            valid_markers = []
            
            # 현재 프레임의 유효한 마커 데이터 수집
            for marker in self.marker_names:
                try:
                    x = self.data.loc[self.frame_idx, f'{marker}_X']
                    y = self.data.loc[self.frame_idx, f'{marker}_Y']
                    z = self.data.loc[self.frame_idx, f'{marker}_Z']
                    
                    # NaN 값 건너뛰기
                    if pd.isna(x) or pd.isna(y) or pd.isna(z):
                        continue
                    
                    # 좌표계에 맞게 위치 조정
                    pos = [x, y, z]
                        
                    marker_positions[marker] = pos
                    
                    # 색상 설정
                    marker_str = str(marker)
                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""
                    
                    if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                        if marker in self.pattern_markers:
                            colors.append([1.0, 0.0, 0.0])  # 빨간색
                        else:
                            colors.append([1.0, 1.0, 1.0])  # 흰색
                    elif marker_str == current_marker_str:
                        colors.append([1.0, 0.9, 0.4])  # 연한 노란색
                    else:
                        colors.append([1.0, 1.0, 1.0])  # 흰색
                        
                    positions.append(pos)
                    valid_markers.append(marker)
                    
                    if marker_str == current_marker_str:
                        selected_position = pos
                        
                except KeyError:
                    continue
            
            # 마커 렌더링 - 2단계로 분리: 일반 마커 -> 패턴 마커
            if positions:
                # 1단계: 일반 마커 (선택되지 않은 마커 또는 패턴 모드가 아닐 때)
                GL.glPointSize(5.0) # 일반 크기
                GL.glBegin(GL.GL_POINTS)
                for i, pos in enumerate(positions):
                    marker = valid_markers[i]
                    is_pattern_selected = self.pattern_selection_mode and marker in self.pattern_markers
                    if not is_pattern_selected:
                        GL.glColor3fv(colors[i])
                        GL.glVertex3fv(pos)
                GL.glEnd()
                
                # 2단계: 선택된 패턴 마커 (패턴 모드일 때)
                if self.pattern_selection_mode and any(m in self.pattern_markers for m in valid_markers):
                    GL.glPointSize(8.0) # 큰 크기
                    GL.glBegin(GL.GL_POINTS)
                    for i, pos in enumerate(positions):
                        marker = valid_markers[i]
                        if marker in self.pattern_markers:
                            # 색상은 colors 리스트에서 이미 레드로 설정되어 있음
                            GL.glColor3fv(colors[i]) 
                            GL.glVertex3fv(pos)
                    GL.glEnd()
            
            # 선택된 마커 강조 표시
            if selected_position:
                GL.glPointSize(8.0)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor3f(1.0, 0.9, 0.4)  # 연한 노란색
                GL.glVertex3fv(selected_position)
                GL.glEnd()
            
            # 스켈레톤 라인 렌더링
            if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs'):
                # --- Enable Blending and Smoothing (needed for normal lines) ---
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                # ---------------------------------------------------------------
                
                # Pass 1: Draw Normal Skeleton Lines (Gray, Semi-Transparent, Width 2.0)
                GL.glLineWidth(2.0)
                GL.glColor4f(0.7, 0.7, 0.7, 0.8) # Gray, Alpha 0.8
                GL.glBegin(GL.GL_LINES)
                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        is_outlier = outlier_status1 or outlier_status2
                        if not is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)
                GL.glEnd()
                
                # Pass 2: Draw Outlier Skeleton Lines (Red, Opaque, Width 4.0)
                # Blending is already enabled, just change width and color
                GL.glLineWidth(3.5)
                GL.glColor4f(1.0, 0.0, 0.0, 1.0) # Red, Alpha 1.0
                GL.glBegin(GL.GL_LINES)
                for pair in self.skeleton_pairs:
                    if pair[0] in marker_positions and pair[1] in marker_positions:
                        p1 = marker_positions[pair[0]]
                        p2 = marker_positions[pair[1]]
                        outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                        is_outlier = outlier_status1 or outlier_status2
                        if is_outlier:
                            GL.glVertex3fv(p1)
                            GL.glVertex3fv(p2)
                GL.glEnd()
                
                # --- Reset LineWidth and Disable Blending --- 
                GL.glLineWidth(1.0) # Reset to OpenGL default
                GL.glDisable(GL.GL_BLEND)
                # GL.glDisable(GL.GL_LINE_SMOOTH) # Optional
                # ------------------------------------------
            
            # 궤적 렌더링
            if self.current_marker is not None and hasattr(self, 'show_trajectory') and self.show_trajectory:
                trajectory_points = []
                
                for i in range(0, self.frame_idx + 1):
                    try:
                        x = self.data.loc[i, f'{self.current_marker}_X']
                        y = self.data.loc[i, f'{self.current_marker}_Y']
                        z = self.data.loc[i, f'{self.current_marker}_Z']
                        
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            continue
                        
                        # 원본 데이터 그대로 사용 (Y-up/Z-up에 관계없이)
                        trajectory_points.append([x, y, z])
                            
                    except KeyError:
                        continue
                
                if trajectory_points:
                    GL.glLineWidth(0.8)
                    GL.glColor3f(1.0, 0.9, 0.4)  # 연한 노란색
                    GL.glBegin(GL.GL_LINE_STRIP)
                    
                    for point in trajectory_points:
                        GL.glVertex3fv(point)
                    
                    GL.glEnd()
            
            # 마커 이름 렌더링
            if self.show_marker_names and valid_markers:
                # 텍스트 렌더링을 위해 GLUT 필요
                try:
                    # 현재 투영 및 모델뷰 매트릭스 저장
                    GL.glPushMatrix()
                    
                    # OpenGL 렌더링 상태 초기화 및 저장
                    GL.glPushAttrib(GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT)
                    
                    # 현재 마커 문자열화
                    current_marker_str = str(self.current_marker) if self.current_marker is not None else ""
                    
                    # 먼저 일반 마커 이름을 모두 렌더링 (흰색)
                    for marker in valid_markers:
                        marker_str = str(marker)
                        if marker_str == current_marker_str:
                            continue  # 선택된 마커는 나중에 렌더링
                            
                        pos = marker_positions[marker]
                        
                        # 일반 마커 이름은 흰색으로 렌더링
                        GL.glColor3f(1.0, 1.0, 1.0)  # 흰색
                        GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])
                        
                        # 마커 이름 렌더링
                        for c in marker_str:
                            try:
                                GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord(c))
                            except:
                                pass
                    
                    # 선택된 마커 이름만 노란색으로 렌더링 (별도 패스)
                    GL.glFlush()  # 이전 렌더링 명령 실행 보장
                    
                    if self.current_marker is not None:
                        # 선택된 마커만 찾아서 렌더링
                        for marker in valid_markers:
                            marker_str = str(marker)
                            if marker_str == current_marker_str:
                                pos = marker_positions[marker]
                                
                                # 선택된 마커 이름은 노란색으로 렌더링
                                GL.glColor3f(1.0, 0.9, 0.4)  # 연한 노란색
                                GL.glRasterPos3f(pos[0], pos[1] + 0.03, pos[2])
                                
                                # 마커 이름 렌더링
                                for c in marker_str:
                                    try:
                                        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_12, ord(c))
                                    except:
                                        pass
                                
                                GL.glFlush()  # 렌더링 명령 즉시 실행
                                break
                    
                    # OpenGL 렌더링 상태 복원
                    GL.glPopAttrib()
                    
                    # 매트릭스 복원
                    GL.glPopMatrix()
                    
                except Exception as e:
                    print(f"텍스트 렌더링 오류: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 버퍼 교체 (화면 갱신)
            self.tkSwapBuffers()
        
        except Exception as e:
            # 디버깅을 위해 오류 로깅
            print(f"OpenGL 렌더링 오류: {e}")
        
    def update_data(self, data, frame_idx):
        """외부에서 호출하여 데이터 업데이트 (하위 호환성 유지)"""
        self.data = data
        self.frame_idx = frame_idx
        if data is not None:
            self.num_frames = len(data)
        
        # current_marker 속성을 변경하지 않고 유지
        
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
        
        # 선택된 마커 정보 유지 - current_marker가 None이 아닌 경우에만 업데이트
        # 또는 현재 마커가 없는 경우(self.current_marker가 None인 경우)에도 업데이트
        if current_marker is not None:
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
        # 마커 이름이 문자열이 아닌 경우 문자열로 변환
        if marker_name is not None and not isinstance(marker_name, str):
            marker_name = str(marker_name)
        
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
        외부에서 호출되는 화면 업데이트 메서드
        이전에는 외부 모듈에 있던 update_plot 호출이었으나 이제 내부 메서드 호출로 변경
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
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
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
        self.trans_x = 0.0  # 추가: translation 값도 초기화
        self.trans_y = 0.0  # 추가: translation 값도 초기화
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
        
    def set_show_marker_names(self, show):
        """
        마커 이름 표시 여부 설정
        
        Args:
            show: 마커 이름을 표시하려면 True, 아니면 False
        """
        self.show_marker_names = show
        self.redraw()
        
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

    # 마우스 이벤트 핸들러 메서드 추가
    def on_mouse_press(self, event):
        """왼쪽 마우스 버튼 누를 때 호출"""
        self.last_x, self.last_y = event.x, event.y
        self.dragging = False
        
        # 픽킹 수행
        if self.data is not None and len(self.marker_names) > 0:
            self.pick_marker(event.x, event.y)

    def on_mouse_release(self, event):
        """왼쪽 마우스 버튼 뗄 때 호출"""
        # 드래그 상태가 아닌 경우 클릭으로 간주
        if not self.dragging and self.data is not None:
            pass  # 픽킹은 press에서 처리

    def on_mouse_move(self, event):
        """왼쪽 마우스 버튼 드래그할 때 호출 (회전)"""
        dx, dy = event.x - self.last_x, event.y - self.last_y
        
        # 약간의 드래그가 발생했을 때만 드래그 상태로 전환
        if abs(dx) > 3 or abs(dy) > 3:
            self.dragging = True
        
        # 드래그 중에는 회전만 수행
        if self.dragging:
            self.last_x, self.last_y = event.x, event.y
            self.rot_y += dx * 0.5
            self.rot_x += dy * 0.5
            self.redraw()

    def on_right_mouse_press(self, event):
        """오른쪽 마우스 버튼 누름 이벤트 처리 (뷰 이동 시작 또는 패턴 선택 모드)"""
        if not self.pattern_selection_mode: # 패턴 선택 모드가 아닐 때만 뷰 이동 시작
            self.dragging = True
            self.last_x = event.x
            self.last_y = event.y
            
    def on_right_mouse_release(self, event):
        """오른쪽 마우스 버튼 뗌 이벤트 처리 (뷰 이동 종료 또는 패턴 마커 선택)"""
        if self.pattern_selection_mode:
             # 패턴 선택 모드: 마커 픽킹 시도
            self.pick_marker(event.x, event.y) 
        elif self.dragging:
            # 뷰 이동 모드 종료
            self.dragging = False

    def on_right_mouse_move(self, event):
        """오른쪽 마우스 버튼 드래그할 때 호출 (이동)"""
        dx, dy = event.x - self.last_x, event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        
        # 화면 이동 계산 (화면 크기에 대한 비율로 이동)
        self.trans_x += dx * 0.005
        self.trans_y -= dy * 0.005  # 좌표계 방향 반전 (화면 y는 아래로 증가)
        
        self.redraw()

    def on_scroll(self, event):
        """마우스 휠 스크롤할 때 호출 (줌)"""
        # Windows에서 event.delta, 다른 플랫폼에서는 다른 접근법 필요
        self.zoom += event.delta * 0.001
        self.redraw()

    def pick_marker(self, x, y):
        """
        마커 선택 (픽킹)
        
        Args:
            x: 화면 X 좌표
            y: 화면 Y 좌표
        """
        if not self.gl_initialized or not hasattr(self, 'picking_texture'):
            return
        
        # 픽킹 텍스처 초기화 확인 및 필요시 초기화
        if not self.picking_texture.initialized:
            width, height = self.winfo_width(), self.winfo_height()
            if width <= 0 or height <= 0 or not self.picking_texture.init(width, height):
                return
        
        try:
            # 컨텍스트 활성화
            self.tkMakeCurrent()
            
            # 픽킹 텍스처에 렌더링
            if not self.picking_texture.enable_writing():
                return
            
            # 버퍼 초기화
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            # 원근 투영 설정
            width, height = self.winfo_width(), self.winfo_height()
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            aspect = float(width) / float(height) if height > 0 else 1.0
            GLU.gluPerspective(45, aspect, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            
            # 카메라 위치 설정 (줌, 이동, 회전)
            GL.glTranslatef(self.trans_x, self.trans_y, self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            
            # Z-up 좌표계인 경우 추가 회전
            if self.is_z_up:
                GL.glRotatef(COORDINATE_X_ROTATION_Z_UP, 1.0, 0.0, 0.0)
            
            # 픽킹 렌더링을 위한 상태 설정
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_POINT_SMOOTH)
            GL.glDisable(GL.GL_LINE_SMOOTH)
            
            # 마커 정보 확인
            if self.data is None or len(self.marker_names) == 0:
                self.picking_texture.disable_writing()
                return
            
            # 픽킹을 위한 큰 점 크기 설정
            GL.glPointSize(12.0)
            
            # 마커를 고유 ID 색상으로 렌더링
            GL.glBegin(GL.GL_POINTS)
            
            for idx, marker in enumerate(self.marker_names):
                try:
                    # 현재 프레임의 마커 좌표 가져오기
                    x_val = self.data.loc[self.frame_idx, f'{marker}_X']
                    y_val = self.data.loc[self.frame_idx, f'{marker}_Y']
                    z_val = self.data.loc[self.frame_idx, f'{marker}_Z']
                    
                    # NaN 값 건너뛰기
                    if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val):
                        continue
                    
                    # 마커 ID를 1부터 시작하도록 설정 (0은 배경)
                    marker_id = idx + 1
                    
                    # 각 마커마다 고유한 색상 인코딩
                    # R 채널: 마커 ID를 정규화한 값
                    r = float(marker_id) / float(len(self.marker_names) + 1)
                    g = float(marker_id % 256) / 255.0  # 추가 정보
                    b = 1.0  # 마커 식별용 상수
                    
                    GL.glColor3f(r, g, b)
                    GL.glVertex3f(x_val, y_val, z_val)
                    
                except KeyError:
                    continue
            
            GL.glEnd()
            
            # 렌더링 완료 확인
            GL.glFinish()
            GL.glFlush()
            
            # 픽셀 정보 읽기 (OpenGL 좌표계 변환)
            y_inverted = height - y - 1
            pixel_info = self.read_pixel_at(x, y_inverted)
            
            # 픽킹 텍스처 비활성화
            self.picking_texture.disable_writing()
            
            # 픽셀 정보가 있으면 마커 선택
            if pixel_info is not None:
                r_value = pixel_info[0]
                
                # ID 값 복원 (인코딩 방식: r = marker_id / (len(marker_names) + 1))
                actual_id = int(r_value * (len(self.marker_names) + 1) + 0.5)
                
                # 마커 ID는 1부터 시작하므로 인덱스로 변환
                marker_idx = actual_id - 1
                
                # 유효한 마커 인덱스인지 확인
                if 0 <= marker_idx < len(self.marker_names):
                    selected_marker = self.marker_names[marker_idx]
                    
                    # 패턴 선택 모드 처리
                    if self.pattern_selection_mode:
                        if selected_marker in self.parent.pattern_markers:
                            self.parent.pattern_markers.remove(selected_marker)
                        else:
                            self.parent.pattern_markers.add(selected_marker)
                        
                        # Update the UI list in the parent (TRCViewer)
                        if hasattr(self.parent, 'update_selected_markers_list'):
                            self.parent.update_selected_markers_list()
                            
                    # 일반 마커 선택 모드 처리
                    else:
                        # 이미 선택된 마커를 다시 클릭한 경우 선택 취소
                        if self.current_marker == selected_marker:
                            self.current_marker = None
                            self._notify_marker_selected(None)  # 선택 취소 알림
                        # 새로운 마커를 선택한 경우
                        else:
                            # 현재 마커 업데이트
                            self.current_marker = selected_marker
                            
                            # 부모 클래스에 알림
                            self._notify_marker_selected(selected_marker)
        
            # 일반 렌더링 상태 복원
            GL.glEnable(GL.GL_BLEND)
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            
            # 화면 업데이트
            self.redraw()
        
        except Exception as e:
            print(f"마커 선택 오류: {e}")
            import traceback
            traceback.print_exc()

    def read_pixel_at(self, x, y):
        """
        지정된 위치의 픽셀 정보를 읽음
        
        Args:
            x: 화면의 X 좌표
            y: 화면의 Y 좌표 (이미 OpenGL 좌표계로 변환된 값)
            
        Returns:
            tuple: (R, G, B) 색상 값 또는 None
        """
        try:
            # 프레임버퍼에서 픽셀 읽기
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.picking_texture.fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            
            # 픽셀 좌표가 텍스처 범위 내에 있는지 확인
            width, height = self.picking_texture.width, self.picking_texture.height
            if x < 0 or x >= width or y < 0 or y >= height:
                return None
            
            # 픽셀 정보 읽기
            data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGB, GL.GL_FLOAT)
            pixel_info = np.frombuffer(data, dtype=np.float32)
            
            # 기본 설정 복원
            GL.glReadBuffer(GL.GL_NONE)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            
            # 배경 픽셀 확인 (R=0이면 배경)
            if pixel_info[0] == 0.0:
                return None
            
            # 픽셀 색상 값 반환
            return (pixel_info[0], pixel_info[1], pixel_info[2])
            
        except Exception as e:
            print(f"픽셀 읽기 오류: {e}")
            return None

    def _notify_marker_selected(self, marker_name):
        """
        마커 선택 이벤트를 부모 창에 알림
        
        Args:
            marker_name: 선택된 마커 이름 또는 None(선택 취소 시)
        """
        # 부모 창 메서드 호출
        if hasattr(self.master, 'on_marker_selected'):
            try:
                self.master.on_marker_selected(marker_name)
            except Exception as e:
                print(f"Error notifying master of marker selection: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Master {self.master} does not have 'on_marker_selected' method.")
            
        # 이벤트 기반 알림 시도 (Fallback)
        # ... (rest of the function)

    def on_configure(self, event):
        """Handle widget resize/move/visibility changes."""
        # Check if GL is initialized before attempting to redraw
        if self.gl_initialized:
             self.redraw()
        # Optionally, you could call self.update() if initgl needs to run on configure
        # self.update()
