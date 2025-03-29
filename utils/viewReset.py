"""
This module contains functions for resetting view states in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

def reset_main_view(self):
    """
    Resets the main 3D OpenGL view to its default state based on data limits.
    """
    # Check if the OpenGL renderer exists
    if hasattr(self, 'gl_renderer') and self.gl_renderer:
        # Check if the renderer has the reset_view method and call it
        if hasattr(self.gl_renderer, 'reset_view'):
            try:
                self.gl_renderer.reset_view()
            except Exception as e:
                print(f"Error calling gl_renderer.reset_view: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("OpenGL renderer does not have a 'reset_view' method.")
    else:
        print("OpenGL renderer not found, cannot reset view.")

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
