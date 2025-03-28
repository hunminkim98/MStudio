import numpy as np
import traceback
import pandas as pd

def create_plot(self):
    """
    Creates the main 3D plot for displaying marker data.
    This function was extracted from the main class to improve code organization.
    
    Now supports only OpenGL rendering mode.
    """
    try:
        from gui.opengl.GLMarkerRenderer import MarkerGLRenderer
        
        # 기존 캔버스가 있다면 제거
        if hasattr(self, 'canvas') and self.canvas:
            if hasattr(self.canvas, 'get_tk_widget'):
                self.canvas.get_tk_widget().destroy()
            self.canvas = None
        
        # OpenGL 렌더러 초기화
        self.gl_renderer = MarkerGLRenderer(self.canvas_frame, bg='black')
        self.gl_renderer.pack(fill='both', expand=True)
        
        # 마커 데이터 설정
        if self.data is not None:
            # 데이터 범위 설정
            if hasattr(self, 'data_limits') and self.data_limits is not None:
                if 'x' in self.data_limits and 'y' in self.data_limits and 'z' in self.data_limits:
                    x_range = self.data_limits['x']
                    y_range = self.data_limits['y']
                    z_range = self.data_limits['z']
                    
                    if hasattr(self.gl_renderer, 'set_data_limits'):
                        self.gl_renderer.set_data_limits(x_range, y_range, z_range)
        
        # 렌더러 초기화 및 그리기
        self.gl_renderer.initialize()
        
        # 스켈레톤 관련 정보 설정
        if hasattr(self, 'skeleton_pairs'):
            self.gl_renderer.set_skeleton_pairs(self.skeleton_pairs)
        if hasattr(self, 'show_skeleton'):
            self.gl_renderer.set_show_skeleton(self.show_skeleton)
        
        # 좌표계 설정 (Y-up 또는 Z-up)
        if hasattr(self, 'is_z_up'):
            self.gl_renderer.set_coordinate_system(self.is_z_up)
        
        # 이상치 정보 설정
        if hasattr(self, 'outliers') and self.outliers:
            self.gl_renderer.set_outliers(self.outliers)
            
        # 캔버스 참조 저장
        self.canvas = self.gl_renderer
        
        print("OpenGL 렌더러가 초기화되었습니다.")
    except Exception as e:
        print(f"OpenGL 렌더러 초기화 중 오류 발생: {type(e).__name__} - {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        raise

# 아래 함수들은 OpenGL 모드에서 사용되지 않지만, 
# 기존 코드와의 호환성을 위해 빈 함수로 유지
def _setup_plot_style(self):
    """OpenGL 모드에서는 사용되지 않음"""
    pass

def _draw_static_elements(self):
    """OpenGL 모드에서는 사용되지 않음"""
    pass

def _initialize_dynamic_elements(self):
    """OpenGL 모드에서는 사용되지 않음"""
    pass

def _update_coordinate_axes(self):
    """OpenGL 모드에서는 사용되지 않음"""
    pass
