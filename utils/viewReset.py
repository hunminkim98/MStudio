"""
This module contains functions for resetting view states in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

def reset_main_view(self):
    """
    Resets the main 3D view to show all data within the calculated data limits.
    Supports both matplotlib and OpenGL rendering modes.
    """
    # OpenGL 렌더링 모드인 경우
    if hasattr(self, 'use_opengl') and self.use_opengl:
        if hasattr(self, 'gl_renderer') and self.gl_renderer:
            # OpenGL 렌더러가 reset_view 메서드를 가지고 있으면 해당 메서드 호출
            if hasattr(self.gl_renderer, 'reset_view'):
                self.gl_renderer.reset_view()
            # 아니면 필요한 설정만 진행
            elif hasattr(self, 'data_limits') and self.data_limits:
                # gl_renderer에 필요한 설정 적용
                pass
            return
    
    # matplotlib 렌더링 모드인 경우
    if hasattr(self, 'ax') and self.ax and hasattr(self, 'data_limits') and self.data_limits:
        try:
            self.ax.set_xlim(self.data_limits['x'])
            self.ax.set_ylim(self.data_limits['y'])
            self.ax.set_zlim(self.data_limits['z'])
            self.ax.set_box_aspect([1,1,1])  # Force equal aspect ratio
            if hasattr(self, 'canvas') and hasattr(self.canvas, 'draw'):
                self.canvas.draw()
        except Exception as e:
            print(f"Error resetting main view: {e}")
            import traceback
            traceback.print_exc()

def reset_graph_view(self):
    """
    Resets the marker graph view to its initial limits.
    """
    if hasattr(self, 'marker_axes') and hasattr(self, 'initial_graph_limits'):
        for ax, limits in zip(self.marker_axes, self.initial_graph_limits):
            ax.set_xlim(limits['x'])
            ax.set_ylim(limits['y'])
        if hasattr(self, 'marker_canvas') and hasattr(self.marker_canvas, 'draw'):
            self.marker_canvas.draw()
