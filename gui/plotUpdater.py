import numpy as np
import pandas as pd

def update_plot(self):
    """
    Updates the 3D plot with the current frame's marker positions and other visual elements.
    This function was extracted from the main class to improve code organization.
    """
    if self.data is None:
        return

    # Update trajectories using the handler
    if hasattr(self, 'trajectory_handler'):
        self.trajectory_handler.update_trajectory(self.data, self.frame_idx, self.marker_names, self.ax)

    # handle empty 3D space when data is None
    if self.data is None:
        # initialize markers and skeleton
        if hasattr(self, 'markers_scatter'):
            self.markers_scatter._offsets3d = ([], [], [])
        if hasattr(self, 'selected_marker_scatter'):
            self.selected_marker_scatter._offsets3d = ([], [], [])
            self.selected_marker_scatter.set_visible(False)
        if hasattr(self, 'skeleton_lines'):
            for line in self.skeleton_lines:
                line.set_data_3d([], [], [])

        # set axis ranges
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])

        self.canvas.draw()
        return

    # remove existing trajectory line
    if hasattr(self, 'trajectory_line') and self.trajectory_line is not None:
        self.trajectory_line.remove()
        self.trajectory_line = None

    if self.current_marker is not None and self.show_trajectory:
        x_vals = []
        y_vals = []
        z_vals = []
        for i in range(0, self.frame_idx + 1):
            try:
                x = self.data.loc[i, f'{self.current_marker}_X']
                y = self.data.loc[i, f'{self.current_marker}_Y']
                z = self.data.loc[i, f'{self.current_marker}_Z']
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                if self.is_z_up:
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                else:
                    x_vals.append(x)
                    y_vals.append(-z)
                    z_vals.append(y)
            except KeyError:
                continue
        if len(x_vals) > 0:
            self.trajectory_line, = self.ax.plot(x_vals, y_vals, z_vals, color='yellow', alpha=0.5, linewidth=1)
    else:
        self.trajectory_line = None

    prev_elev = self.ax.elev
    prev_azim = self.ax.azim
    prev_xlim = self.ax.get_xlim()
    prev_ylim = self.ax.get_ylim()
    prev_zlim = self.ax.get_zlim()

    # collect marker position data
    positions = []
    colors = []
    alphas = []
    selected_position = []
    marker_positions = {}
    valid_markers = []

    # collect valid markers for the current frame
    for marker in self.marker_names:
        try:
            x = self.data.loc[self.frame_idx, f'{marker}_X']
            y = self.data.loc[self.frame_idx, f'{marker}_Y']
            z = self.data.loc[self.frame_idx, f'{marker}_Z']
            
            # skip NaN values or deleted data
            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                continue
                
            # add valid data
            if self.is_z_up:
                marker_positions[marker] = np.array([x, y, z])
                positions.append([x, y, z])
            else:
                marker_positions[marker] = np.array([x, -z, y])
                positions.append([x, -z, y])

            # add colors and alphas for valid markers
            if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode:
                if marker in self.pattern_markers:
                    colors.append('red')
                    alphas.append(0.3)
                else:
                    colors.append('white')
                    alphas.append(1.0)
            elif marker == self.current_marker:
                colors.append('yellow')
                alphas.append(1.0)
            else:
                colors.append('white')
                alphas.append(1.0)

            if marker == self.current_marker:
                if self.is_z_up:
                    selected_position.append([x, y, z])
                else:
                    selected_position.append([x, -z, y])
            valid_markers.append(marker)
        except KeyError:
            continue

    # array conversion
    positions = np.array(positions) if positions else np.zeros((0, 3))
    selected_position = np.array(selected_position) if selected_position else np.zeros((0, 3))

    # update scatter plot - display valid data
    if len(positions) > 0:
        try:
            # remove existing scatter
            if hasattr(self, 'markers_scatter'):
                self.markers_scatter.remove()
            
            # create new scatter plot
            self.markers_scatter = self.ax.scatter(
                positions[:, 0], 
                positions[:, 1], 
                positions[:, 2],
                c=colors[:len(positions)],  # length match
                alpha=alphas[:len(positions)],  # length match
                s=30,
                picker=5
            )
        except Exception as e:
            print(f"Error updating scatter plot: {e}")
    else:
        # create empty scatter plot if data is None
        if hasattr(self, 'markers_scatter'):
            self.markers_scatter.remove()
        self.markers_scatter = self.ax.scatter([], [], [], c='white', s=30, picker=5)

    # update selected marker
    if len(selected_position) > 0:
        self.selected_marker_scatter._offsets3d = (
            selected_position[:, 0],
            selected_position[:, 1],
            selected_position[:, 2]
        )
        self.selected_marker_scatter.set_visible(True)
    else:
        self.selected_marker_scatter._offsets3d = ([], [], [])
        self.selected_marker_scatter.set_visible(False)

    # update skeleton lines
    if hasattr(self, 'show_skeleton') and self.show_skeleton and hasattr(self, 'skeleton_lines'):
        for line, pair in zip(self.skeleton_lines, self.skeleton_pairs):
            if pair[0] in marker_positions and pair[1] in marker_positions:
                p1 = marker_positions[pair[0]]
                p2 = marker_positions[pair[1]]

                outlier_status1 = self.outliers.get(pair[0], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                outlier_status2 = self.outliers.get(pair[1], np.zeros(self.num_frames, dtype=bool))[self.frame_idx]
                is_outlier = outlier_status1 or outlier_status2

                line.set_data_3d(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]]
                )
                line.set_visible(True)
                line.set_color('red' if is_outlier else 'gray')
                line.set_alpha(1 if is_outlier else 0.8)
                line.set_linewidth(3 if is_outlier else 2)
            else:
                line.set_visible(False)

    # update marker names
    # remove existing labels
    for label in self.marker_labels:
        label.remove()
    self.marker_labels.clear()

    for marker in valid_markers:
        pos = marker_positions[marker]
        color = 'white'
        alpha = 1.0
        
        # pattern-based selected markers are always displayed
        if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode and marker in self.pattern_markers:
            color = 'red'  # pattern-based selected marker
            alpha = 0.7
            label = self.ax.text(pos[0], pos[1], pos[2], marker, color=color, alpha=alpha, fontsize=8)
            self.marker_labels.append(label)
        # display other markers if show_names is True
        elif self.show_names:
            if marker == self.current_marker:
                color = 'yellow'
            label = self.ax.text(pos[0], pos[1], pos[2], marker, color=color, alpha=alpha, fontsize=8)
            self.marker_labels.append(label)

    # update current frame line when marker graph is displayed
    if hasattr(self, 'marker_canvas') and self.marker_canvas:
        # remove existing current_frame_line code
        if hasattr(self, 'marker_lines') and self.marker_lines:
            for line in self.marker_lines:
                line.set_xdata([self.frame_idx, self.frame_idx])
            self.marker_canvas.draw_idle()

    self.ax.view_init(elev=prev_elev, azim=prev_azim)
    self.ax.set_xlim(prev_xlim)
    self.ax.set_ylim(prev_ylim)
    self.ax.set_zlim(prev_zlim)

    self.canvas.draw_idle()
