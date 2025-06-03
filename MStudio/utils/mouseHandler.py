## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MStudio')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"

class MouseHandler:
    def __init__(self, parent):
        self.parent = parent
        self.marker_pan_enabled = False
        self.marker_last_pos = None
        self.selection_in_progress = False
        self.timeline_dragging = False

    # Marker View Mouse Events
    def on_marker_scroll(self, event):
        if not event.inaxes:
            return

        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2

        scale_factor = 0.9 if event.button == 'up' else 1.1

        new_x_range = (x_max - x_min) * scale_factor
        new_y_range = (y_max - y_min) * scale_factor

        x_left = x_center - new_x_range * (x_center - x_min) / (x_max - x_min)
        x_right = x_center + new_x_range * (x_max - x_center) / (x_max - x_min)
        y_bottom = y_center - new_y_range * (y_center - y_min) / (y_max - y_min)
        y_top = y_center + new_y_range * (y_max - y_center) / (y_max - y_min)

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

        self.parent.marker_canvas.draw_idle()

    def on_marker_mouse_move(self, event):
        if self.marker_pan_enabled and self.marker_last_pos:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self.marker_last_pos[0]
                dy = event.ydata - self.marker_last_pos[1]

                ax = event.inaxes
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                ax.set_xlim(x_min - dx, x_max - dx)
                ax.set_ylim(y_min - dy, y_max - dy)

                self.marker_last_pos = (event.xdata, event.ydata)

                self.parent.marker_canvas.draw_idle()
        elif self.selection_in_progress and event.xdata is not None:
            self.parent.selection_data['end'] = event.xdata

            start_x = min(self.parent.selection_data['start'], self.parent.selection_data['end'])
            width = abs(self.parent.selection_data['end'] - self.parent.selection_data['start'])

            for rect in self.parent.selection_data['rects']:
                rect.set_x(start_x)
                rect.set_width(width)

            self.parent.marker_canvas.draw_idle()

    def on_marker_mouse_press(self, event):
        if event.button == 1:
            # edit mode only
            if event.inaxes in self.parent.marker_axes and self.parent.state_manager.editing_state.is_editing:
                # remove existing selection
                self.parent.clear_selection()
                self.selection_in_progress = True
                self.parent.start_new_selection(event)
        elif event.button == 3:
            self.marker_pan_enabled = True
            self.marker_last_pos = (event.xdata, event.ydata)

    def on_marker_mouse_release(self, event):
        if event.button == 1:
            if self.selection_in_progress:
                self.selection_in_progress = False
                self.parent.highlight_selection()
        elif event.button == 3:
            self.marker_pan_enabled = False
            self.marker_last_pos = None

    # Timeline Mouse Events
    def on_timeline_click(self, event):
        if event.inaxes == self.parent.timeline_ax:
            self.timeline_dragging = True
            # Store the animation state when dragging starts
            self._was_playing_before_drag = self.parent.animation_controller.is_playing
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_drag(self, event):
        if self.timeline_dragging and event.inaxes == self.parent.timeline_ax:
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_release(self, event):
        if self.timeline_dragging:
            self.timeline_dragging = False
            # If animation was playing before drag, ensure it continues from new position
            if hasattr(self, '_was_playing_before_drag') and self._was_playing_before_drag:
                # Animation should continue from the new position
                # The AnimationController will handle this automatically due to our from_external=True logic
                pass
