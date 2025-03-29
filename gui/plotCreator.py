import numpy as np
import traceback

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
                try:
                    # Attempt to destroy the widget if it exists
                    widget = self.canvas.get_tk_widget()
                    if widget.winfo_exists():
                        widget.destroy()
                except Exception as destroy_error:
                    print(f"Error destroying previous canvas widget: {destroy_error}")
            elif hasattr(self.canvas, 'destroy') and callable(self.canvas.destroy):
                 # Handle cases where canvas might be a direct Tk widget or similar
                 try:
                    if self.canvas.winfo_exists():
                        self.canvas.destroy()
                 except Exception as destroy_error:
                     print(f"Error destroying previous canvas object: {destroy_error}")
            self.canvas = None
        
        # OpenGL 렌더러 초기화
        self.gl_renderer = MarkerGLRenderer(self, bg='black')
        self.gl_renderer.pack(in_=self.canvas_frame, fill='both', expand=True)
        
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
        else:
            self.gl_renderer.set_show_skeleton(False)
        
        # 좌표계 설정 (Y-up 또는 Z-up)
        if hasattr(self, 'is_z_up'):
            self.gl_renderer.set_coordinate_system(self.is_z_up)
        else:
            self.gl_renderer.set_coordinate_system(False)
        
        # 이상치 정보 설정
        if hasattr(self, 'outliers') and self.outliers:
            self.gl_renderer.set_outliers(self.outliers)
            
        # 캔버스 참조 저장
        self.canvas = self.gl_renderer
        
    except ImportError as ie:
        print(f"OpenGL 렌더러 임포트 오류: {ie}")
        print("OpenGL 기능을 사용하려면 필요한 라이브러리를 설치하세요.")
        # Optionally, you could disable OpenGL features or show a message to the user here
        raise # Re-raise to indicate failure
    except Exception as e:
        print(f"OpenGL 렌더러 초기화 중 오류 발생: {type(e).__name__} - {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        raise
