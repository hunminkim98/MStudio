from OpenGL import GL
from OpenGL import GLU
import numpy as np
import pandas as pd

def update_plot(self):
    """
    OpenGL로 3D 마커 시각화 업데이트
    plotUpdater.py의 update_plot 기능을 대체
    """
    # OpenGL 초기화 확인
    if not hasattr(self, 'gl_initialized') or not self.gl_initialized:
        return
    
    # 현재 좌표계 상태
    coordinate_status = getattr(self, 'is_z_up', True)
    
    try:
        # OpenGL 컨텍스트 활성화 (안전을 위해)
        try:
            self.tkMakeCurrent()
        except Exception:
            pass
        
        # 프레임 초기화
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        
        # 카메라 위치 설정
        GL.glTranslatef(0.0, 0.0, self.zoom)
        GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        
        # 그리드와 축 표시는 디스플레이 리스트가 있는 경우에만
        if hasattr(self, 'grid_list') and self.grid_list is not None:
            GL.glCallList(self.grid_list)
        if hasattr(self, 'axes_list') and self.axes_list is not None:
            GL.glCallList(self.axes_list)
        
        # 데이터가 없는 경우 기본 뷰만 표시
        if self.data is None:
            # 버퍼 교체
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
                is_z_up_local = coordinate_status
                
                # 좌표계가 Z-up인 경우와 Y-up인 경우를 구분
                if is_z_up_local:
                    # Z-up: 원본 데이터 그대로 사용
                    pos = [x, y, z]
                else:
                    # Y-up: Z와 Y 축을 교환하고 Z 방향 반전
                    pos = [x, -z, y]
                    
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
                    
                    # 좌표계에 맞게 위치 조정
                    is_z_up_local = coordinate_status
                    
                    if is_z_up_local:
                        # Z-up: 원본 데이터 그대로 사용
                        trajectory_points.append([x, y, z])
                    else:
                        # Y-up: Z와 Y 축을 교환하고 Z 방향 반전
                        trajectory_points.append([x, -z, y])
                        
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
