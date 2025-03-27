import customtkinter as ctk
from pyopengltk import OpenGLFrame
from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT
import numpy as np
import sys

class MarkerGLFrame(OpenGLFrame):
    """OpenGL 기반 3D 마커 시각화 프레임"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # 기존 plotCreator.py에서 필요한 속성들 가져오기
        self.data = None
        self.marker_names = []
        self.is_z_up = True
        self.show_skeleton = True
        self.skeleton_pairs = []
        self.skeleton_lines = []
        self.current_marker = None
        self.marker_labels = []
        self.coordinate_axes = []
        self.axis_labels = []
        self.grid_lines = []
        
        # 마우스 제어 관련
        self.rot_x = 30.0  # 기본 회전 각도 (X축)
        self.rot_y = 45.0  # 기본 회전 각도 (Y축)
        self.zoom = -4.0   # 기본 줌 레벨
        self.last_x = 0
        self.last_y = 0
        
        # 초기화 플래그
        self.gl_initialized = False
        self.grid_list = None
        self.axes_list = None
        
        # 마우스 이벤트 연결
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<B1-Motion>", self.on_mouse_move)
        self.bind("<MouseWheel>", self.on_scroll)
    
    def initgl(self):
        """OpenGL 초기화 (pyopengltk에서 자동 호출)"""
        try:
            # GLUT 초기화 추가
            GLUT.glutInit(sys.argv)

            # 배경색 설정 (검정)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            
            # 깊이 테스트 활성화
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            # 포인트 크기와 선 폭 설정
            GL.glPointSize(5.0)
            GL.glLineWidth(2.0)
            
            # 조명 설정 (기본 조명)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
            
            # 물체 재질 설정
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            
            # 안티앨리어싱 설정
            GL.glEnable(GL.GL_POINT_SMOOTH)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
            
            # 뷰포트 초기화
            width, height = self.winfo_width(), self.winfo_height()
            if width > 1 and height > 1:  # 유효한 크기인지 확인
                GL.glViewport(0, 0, width, height)
                
            # 초기 투영 설정
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(45, float(width)/float(height) if width > 0 and height > 0 else 1.0, 0.1, 100.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            
            # 초기화 플래그 설정
            self.gl_initialized = True
            
            # 디스플레이 리스트 생성 - 초기화 이후에 실행
            self.after(100, self.create_display_lists)
            
        except Exception as e:
            print(f"OpenGL 초기화 오류: {str(e)}")
            self.gl_initialized = False
    
    def create_display_lists(self):
        """OpenGL 디스플레이 리스트 생성"""
        try:
            self.create_grid()
            self.create_axes()
        except Exception as e:
            print(f"디스플레이 리스트 생성 오류: {str(e)}")
    
    def reshape(self, width, height):
        """창 크기 변경 처리"""
        if not self.gl_initialized:
            return
            
        try:
            if width > 1 and height > 1:  # 유효한 크기인지 확인
                GL.glViewport(0, 0, width, height)
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                GLU.gluPerspective(45, float(width)/float(height), 0.1, 100.0)
                GL.glMatrixMode(GL.GL_MODELVIEW)
        except Exception as e:
            print(f"창 크기 변경 오류: {str(e)}")
    
    def create_grid(self):
        """바닥 그리드 생성"""
        grid_size = 2
        grid_divisions = 20
        
        # 그리드 리스트 생성
        self.grid_list = GL.glGenLists(1)
        GL.glNewList(self.grid_list, GL.GL_COMPILE)
        
        GL.glColor3f(0.3, 0.3, 0.3)  # 회색
        GL.glBegin(GL.GL_LINES)
        
        for i in range(-grid_divisions, grid_divisions + 1):
            x = i * (grid_size / grid_divisions)
            GL.glVertex3f(x, 0, -grid_size)
            GL.glVertex3f(x, 0, grid_size)
            
            z = i * (grid_size / grid_divisions)
            GL.glVertex3f(-grid_size, 0, z)
            GL.glVertex3f(grid_size, 0, z)
        
        GL.glEnd()
        GL.glEndList()
    
    def create_axes(self):
        """좌표축 생성"""
        # 기존 리스트가 있으면 삭제
        if hasattr(self, 'axes_list') and self.axes_list is not None:
            GL.glDeleteLists(self.axes_list, 1)

        self.axes_list = GL.glGenLists(1)
        GL.glNewList(self.axes_list, GL.GL_COMPILE)

        # 축 길이 줄이기
        axis_length = 0.2

        # 축을 그리드 위로 띄울 오프셋 값
        offset_y = 0.001

        # 축 굵기 설정
        original_line_width = GL.glGetFloatv(GL.GL_LINE_WIDTH)
        GL.glLineWidth(3.0)

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

        # 축 라벨 텍스트 그리기 (GLUT 사용)
        text_offset = 0.06 # 축 끝에서 텍스트를 띄울 거리

        # 조명 비활성화 (텍스트 색상이 제대로 나오도록)
        lighting_enabled = GL.glIsEnabled(GL.GL_LIGHTING)
        if lighting_enabled:
            GL.glDisable(GL.GL_LIGHTING)

        # X 라벨
        GL.glColor3f(1.0, 0.0, 0.0) # 빨강
        GL.glRasterPos3f(axis_length + text_offset, offset_y, 0)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('X'))

        # Y 라벨
        GL.glColor3f(1.0, 1.0, 0.0) # 노랑
        GL.glRasterPos3f(0, axis_length + text_offset + offset_y, 0)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Y'))

        # Z 라벨
        GL.glColor3f(0.0, 0.0, 1.0) # 파랑
        GL.glRasterPos3f(0, offset_y, axis_length + text_offset)
        GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord('Z'))

        # 원래 상태 복원
        GL.glLineWidth(original_line_width)
        if lighting_enabled:
            GL.glEnable(GL.GL_LIGHTING) # 조명 다시 활성화

        GL.glEndList()
    
    def setup_view(self):
        """기본 뷰 설정 (plotCreator._setup_plot_style 대체)"""
        # 초기화 시 자동으로 호출되므로 여기선 추가 작업 없음
        pass
    
    # 마우스 이벤트 핸들러 (기존 mouse_handler 클래스 기능 대체)
    def on_mouse_press(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def on_mouse_release(self, event):
        pass
    
    def on_mouse_move(self, event):
        dx, dy = event.x - self.last_x, event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        
        self.rot_y += dx * 0.5
        self.rot_x += dy * 0.5
        self.redraw()
    
    def on_scroll(self, event):
        # Windows에서 event.delta, 다른 플랫폼에서는 다른 접근법 필요
        self.zoom += event.delta * 0.001
        self.redraw()
        
    def pick_marker(self, x, y):
        """화면 좌표에서 마커 선택 (미구현 부분, 향후 개발 필요)"""
        # 픽킹 구현은 복잡하므로 현재는 기본 기능만 유지
        # 향후 픽킹 기능 추가 시 구현 필요
        pass
