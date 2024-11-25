class MouseHandler:
    def __init__(self, parent):
        self.parent = parent
        self.pan_enabled = False
        self.last_mouse_pos = None
        self.marker_pan_enabled = False
        self.marker_last_pos = None
        self.selection_in_progress = False
        self.timeline_dragging = False

    # 3D View Mouse Events
    def on_scroll(self, event):
        try:
            if event.inaxes != self.parent.ax:
                return

            x_min, x_max = self.parent.ax.get_xlim()
            y_min, y_max = self.parent.ax.get_ylim()
            z_min, z_max = self.parent.ax.get_zlim()

            scale_factor = 0.9 if event.button == 'up' else 1.1

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            x_range = (x_max - x_min) * scale_factor
            y_range = (y_max - y_min) * scale_factor
            z_range = (z_max - z_min) * scale_factor

            min_range = 1e-3
            max_range = 1e5

            x_range = max(min(x_range, max_range), min_range)
            y_range = max(min(y_range, max_range), min_range)
            z_range = max(min(z_range, max_range), min_range)

            self.parent.ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            self.parent.ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            self.parent.ax.set_zlim(z_center - z_range / 2, z_center + z_range / 2)

            self.parent.canvas.draw_idle()
        except Exception as e:
            print(f"Scroll event error: {e}")
            self.parent.connect_mouse_events()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.pan_enabled = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1:
            self.pan_enabled = False
            self.last_mouse_pos = None

    def on_mouse_move(self, event):
        if self.pan_enabled and event.xdata is not None and event.ydata is not None:
            x_min, x_max = self.parent.ax.get_xlim()
            y_min, y_max = self.parent.ax.get_ylim()

            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]

            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            min_limit = -1e5
            max_limit = 1e5

            new_x_min = max(new_x_min, min_limit)
            new_x_max = min(new_x_max, max_limit)
            new_y_min = max(new_y_min, min_limit)
            new_y_max = min(new_y_max, max_limit)

            self.parent.ax.set_xlim(new_x_min, new_x_max)
            self.parent.ax.set_ylim(new_y_min, new_y_max)

            self.parent.canvas.draw_idle()

            self.last_mouse_pos = (event.xdata, event.ydata)

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

    # Timeline Mouse Events
    def on_timeline_click(self, event):
        if event.inaxes == self.parent.timeline_ax:
            self.timeline_dragging = True
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_drag(self, event):
        if self.timeline_dragging and event.inaxes == self.parent.timeline_ax:
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_release(self, event):
        self.timeline_dragging = False
