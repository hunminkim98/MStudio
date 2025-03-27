from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

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

    def on_marker_mouse_press(self, event):
        if event.button == 1:
            # edit 모드일 때만 선택 가능하도록
            if event.inaxes in self.parent.marker_axes and hasattr(self.parent, 'edit_window') and self.parent.edit_window:
                # 기존 선택 영역 제거
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
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_drag(self, event):
        if self.timeline_dragging and event.inaxes == self.parent.timeline_ax:
            self.parent.update_frame_from_timeline(event.xdata)

    def on_timeline_release(self, event):
        self.timeline_dragging = False

    # Event Connection/Disconnection
    def connect_mouse_events(self):
        if self.parent.canvas:
            self.parent.canvas.mpl_disconnect('scroll_event')
            self.parent.canvas.mpl_disconnect('pick_event')
            self.parent.canvas.mpl_disconnect('button_press_event')
            self.parent.canvas.mpl_disconnect('button_release_event')
            self.parent.canvas.mpl_disconnect('motion_notify_event')

            self.parent.canvas.mpl_connect('scroll_event', self.on_scroll)
            self.parent.canvas.mpl_connect('pick_event', self.on_pick)
            self.parent.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.parent.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.parent.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def disconnect_mouse_events(self):
        if hasattr(self.parent, 'canvas'):
            for cid in self.parent.canvas.callbacks.callbacks.copy():
                self.parent.canvas.mpl_disconnect(cid)

    def on_pick(self, event):
        try:
            # 오른쪽 클릭이 아닌 경우 무시
            if event.mouseevent.button != 3:
                return
            
            # OpenGL 모드일 경우는 별도의 pick 로직을 가지므로 스킵
            if hasattr(self.parent, 'use_opengl') and self.parent.use_opengl:
                print("OpenGL mode: pick event is handled differently")
                return
            
            # 현재 뷰 상태 저장
            if hasattr(self.parent, 'ax'):
                current_view = {
                    'elev': self.parent.ax.elev,
                    'azim': self.parent.ax.azim,
                    'xlim': self.parent.ax.get_xlim(),
                    'ylim': self.parent.ax.get_ylim(),
                    'zlim': self.parent.ax.get_zlim()
                }

            # 마우스 이벤트 연결 해제
            self.disconnect_mouse_events()

            # 유효한 마커 찾기
            valid_markers = []
            for marker in self.parent.marker_names:
                try:
                    x = self.parent.data.loc[self.parent.frame_idx, f'{marker}_X']
                    y = self.parent.data.loc[self.parent.frame_idx, f'{marker}_Y']
                    z = self.parent.data.loc[self.parent.frame_idx, f'{marker}_Z']
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        valid_markers.append(marker)
                except (KeyError, TypeError, ValueError):
                    continue

            if not valid_markers:
                print("No valid markers in current frame")
                return

            # Get selected marker index
            ind = event.ind[0]
            if ind >= len(valid_markers):
                return

            # Store valid markers and selected marker
            self.parent.valid_markers = valid_markers
            selected_marker = valid_markers[ind]

            # pattern-based interpolation이 선택된 경우
            if hasattr(self.parent, 'pattern_selection_mode') and self.parent.pattern_selection_mode:
                # 현재 마커는 패턴 마커로 선택할 수 없음
                if selected_marker == self.parent.current_marker:
                    messagebox.showwarning("Invalid Selection", "Cannot select the current marker as a pattern marker")
                    return
                    
                # 패턴 마커 토글
                if selected_marker in self.parent.pattern_markers:
                    self.parent.pattern_markers.remove(selected_marker)
                else:
                    self.parent.pattern_markers.add(selected_marker)
                
                # 패턴 마커 선택 UI 업데이트
                self.parent.update_selected_markers_list()
                
                # 화면 업데이트
                self.parent.update_plot()
                
                # 마우스 이벤트 다시 연결
                self.connect_mouse_events()
                return
            
            # 일반 케이스: 마커 선택 및 상세 그래프 표시
            print(f"Selected marker: {selected_marker}")
            
            # 현재 마커 설정
            self.parent.current_marker = selected_marker
            
            # Trajectory handler에도 알림
            if hasattr(self.parent, 'trajectory_handler'):
                self.parent.trajectory_handler.set_current_marker(selected_marker)
            
            # 마커 그래프 표시
            try:
                self.parent.show_marker_plot(selected_marker)
            except Exception as e:
                print(f"Error showing marker plot: {e}")
                import traceback
                traceback.print_exc()
            
            # 메인 뷰 업데이트
            self.parent.update_plot()
            
            # 이벤트 다시 연결
            self.connect_mouse_events()
            
            # 뷰 상태 복원
            if hasattr(self.parent, 'ax') and 'elev' in current_view:
                self.parent.ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
                self.parent.ax.set_xlim(current_view['xlim'])
                self.parent.ax.set_ylim(current_view['ylim'])
                self.parent.ax.set_zlim(current_view['zlim'])
                self.parent.canvas.draw()
        
        except Exception as e:
            print(f"Error in on_pick: {e}")
            import traceback
            traceback.print_exc()
            # Try to reconnect events for safety
            self.connect_mouse_events()
