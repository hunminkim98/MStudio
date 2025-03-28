from OpenGL import GL
from OpenGL import GLU
import numpy as np
import pandas as pd

# 좌표계 회전 상수
COORDINATE_X_ROTATION_Y_UP = -270.0  # Y-up 좌표계에서 X축 회전 각도 (-270도)
COORDINATE_X_ROTATION_Z_UP = -90.0   # Z-up 좌표계에서 X축 회전 각도 (-90도)

def update_plot(self):
    """
    OpenGL로 3D 마커 시각화 업데이트
    plotUpdater.py의 update_plot 기능을 대체
    
    좌표계:
    - Y-up: 기본 좌표계, Y축이 상단을 향함
    - Z-up: Z축이 상단을 향하고, X-Y가 바닥 평면
    """
    # OpenGL 초기화 확인
    if not hasattr(self, 'gl_initialized') or not self.gl_initialized:
        return
    
    # 현재 좌표계 상태 확인 (기본값: Y-up)
    is_z_up_local = getattr(self, 'is_z_up', False)
    
    try:
        # OpenGL 컨텍스트 활성화 (안전을 위해)
        try:
            self.tkMakeCurrent()
        except Exception:
            pass
        
        # 프레임 초기화
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        
        # 카메라 위치 설정 (줌, 회전)
        GL.glTranslatef(0.0, 0.0, self.zoom)
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
                if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                    if marker in self.pattern_markers:
                        colors.append([1.0, 0.0, 0.0])  # 빨간색
                    else:
                        colors.append([1.0, 1.0, 1.0])  # 흰색
                elif marker == self.current_marker:
                    colors.append([1.0, 1.0, 0.0])  # 노란색
                else:
                    colors.append([1.0, 1.0, 1.0])  # 흰색
                    
                positions.append(pos)
                valid_markers.append(marker)
                
                if marker == self.current_marker:
                    selected_position = pos
                    
            except KeyError:
                continue
        
        # 마커 렌더링
        if positions:
            GL.glPointSize(5.0)
            GL.glBegin(GL.GL_POINTS)
            for i, pos in enumerate(positions):
                GL.glColor3fv(colors[i])
                GL.glVertex3fv(pos)
            GL.glEnd()
        
        # 선택된 마커 강조 표시
        if selected_position:
            GL.glPointSize(10.0)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(1.0, 1.0, 0.0)  # 노란색
            GL.glVertex3fv(selected_position)
            GL.glEnd()
        
        # 스켈레톤 라인 렌더링
        if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_pairs'):
            GL.glLineWidth(2.0)
            GL.glBegin(GL.GL_LINES)
            
            for pair in self.skeleton_pairs:
                if pair[0] in marker_positions and pair[1] in marker_positions:
                    p1 = marker_positions[pair[0]]
                    p2 = marker_positions[pair[1]]
                    
                    # 이상치 여부 확인
                    outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                    outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx] if hasattr(self, 'outliers') else False
                    is_outlier = outlier_status1 or outlier_status2
                    
                    if is_outlier:
                        GL.glColor3f(1.0, 0.0, 0.0)  # 빨간색
                    else:
                        GL.glColor3f(0.7, 0.7, 0.7)  # 회색
                    
                    GL.glVertex3fv(p1)
                    GL.glVertex3fv(p2)
            
            GL.glEnd()
        
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
                GL.glLineWidth(1.0)
                GL.glColor3f(1.0, 1.0, 0.0)  # 노란색
                GL.glBegin(GL.GL_LINE_STRIP)
                
                for point in trajectory_points:
                    GL.glVertex3fv(point)
                
                GL.glEnd()
        
        # 버퍼 교체 (화면 갱신)
        self.tkSwapBuffers()
    
    except Exception:
        pass
