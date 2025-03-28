from OpenGL import GL
import numpy as np

# 좌표계 기본 설정
DEFAULT_COORDINATE_SYSTEM = False  # False: Y-up (기본값), True: Z-up

def create_opengl_grid(grid_size=2.0, grid_divisions=20, color=(0.3, 0.3, 0.3), is_z_up=DEFAULT_COORDINATE_SYSTEM):
    """
    OpenGL 그리드 생성을 위한 중앙화된 유틸리티 함수
    
    이 함수는 현재 좌표계에 맞는 바닥 그리드를 생성합니다.
    - Y-up 좌표계: X-Z 평면(Y=0)에 그리드 생성
    - Z-up 좌표계: X-Y 평면(Z=0)에 그리드 생성
    
    Args:
        grid_size: 그리드 크기 (기본값: 2.0)
        grid_divisions: 그리드 분할 수 (기본값: 20)
        color: 그리드 색상 (R, G, B) (기본값: 어두운 회색)
        is_z_up: Z-up 좌표계 사용 여부 (True: Z-up, False: Y-up)
    
    Returns:
        grid_list: 생성된 OpenGL 디스플레이 리스트 ID
    
    주의:
    좌표계에 관계없이 데이터의 실제 좌표는 변경되지 않습니다.
    이 함수는 시각화 목적으로만 사용됩니다.
    """
    # 그리드 리스트 생성
    grid_list = GL.glGenLists(1)
    GL.glNewList(grid_list, GL.GL_COMPILE)
    
    # 양면 렌더링 및 후면 컬링 비활성화
    GL.glDisable(GL.GL_CULL_FACE)
    
    # 그리드 굵기 설정
    GL.glLineWidth(1.0)
    
    # 그리드 색상 설정
    GL.glColor3f(*color)
    GL.glBegin(GL.GL_LINES)
    
    # 그리드 간격 계산
    step = (grid_size * 2) / grid_divisions
    
    if is_z_up:
        # Z-up 기준 그리드 그리기 (X-Y 평면, Z=0)
        for i in range(grid_divisions + 1):
            x = -grid_size + i * step
            # X방향 선 (Y축 변화)
            GL.glVertex3f(x, -grid_size, 0)
            GL.glVertex3f(x, grid_size, 0)
            # Y방향 선 (X축 변화)
            GL.glVertex3f(-grid_size, x, 0)
            GL.glVertex3f(grid_size, x, 0)
    else:
        # Y-up 기준 그리드 그리기 (X-Z 평면, Y=0)
        for i in range(grid_divisions + 1):
            x = -grid_size + i * step
            # X방향 선 (Z축 변화)
            GL.glVertex3f(x, 0, -grid_size)
            GL.glVertex3f(x, 0, grid_size)
            # Z방향 선 (X축 변화)
            GL.glVertex3f(-grid_size, 0, x)
            GL.glVertex3f(grid_size, 0, x)
    
    GL.glEnd()
    
    # 원래 설정 복원
    GL.glEnable(GL.GL_CULL_FACE)
    
    GL.glEndList()
    
    return grid_list 