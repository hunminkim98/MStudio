"""
Vispy Canvas for MarkerStudio
Implements a 3D visualization canvas using Vispy to replace Matplotlib
"""

import numpy as np
from vispy import app, scene, visuals, color
from vispy.scene import SceneCanvas, ViewBox, Grid, Node
from vispy.scene.visuals import Markers, Line, Text


class MarkerStudioCanvas:
    """
    Vispy-based 3D visualization canvas for marker trajectories
    """
    def __init__(self, parent=None):
        # Create a new canvas
        self.canvas = SceneCanvas(keys='interactive', size=(800, 600), show=True)
        
        # Create a view with a 3D viewport
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'  # 3D camera type
        self.view.camera.fov = 45
        self.view.camera.distance = 10
        
        # Add a 3D axis for reference
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        
        # Initialize empty data
        self.data = None
        self.marker_names = []
        self.current_frame = 0
        self.num_frames = 0
        
        # Visualization elements
        self.markers_visual = None
        self.labels_visual = []
        self.trajectory_lines = []
        self.show_names = True
        self.show_trajectory = False
        self.trajectory_length = 10
        
        # Selection
        self.selected_markers = []
        self.selection_visual = None
        
        # Coordinate system
        self.is_z_up = True
        
        # Initialize an empty line collection for skeleton connections
        self.skeleton_lines = None
        self.skeleton_pairs = []
        
        # Default marker colors
        self.marker_base_color = np.array([1.0, 1.0, 1.0, 1.0])  # White
        self.marker_selected_color = np.array([1.0, 0.5, 0.0, 1.0])  # Orange
        self.marker_missing_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red
        self.marker_size = 10
        
        # Add a grid plane for better reference
        self.grid = scene.visuals.GridLines(parent=self.view.scene, scale=(1, 1))
        
        # Connect events
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.key_press.connect(self._on_key_press)
        
        # Mouse state for interaction
        self.mouse_state = {
            'button': None,
            'pos': None,
            'press_pos': None,
            'last_pos': None,
            'selected_marker': None,
            'is_dragging': False,
            'initial_marker_pos': None
        }
        
    def set_data(self, data, marker_names):
        """
        Set the marker trajectory data
        
        Args:
            data: numpy array of shape (num_frames, num_markers, 3)
            marker_names: list of marker names
        """
        self.data = data
        self.marker_names = marker_names
        self.num_frames = data.shape[0] if data is not None else 0
        self.current_frame = 0
        
        # Initialize markers visual if needed
        if self.markers_visual is None:
            self.markers_visual = Markers(parent=self.view.scene)
            
        # Clear existing labels
        for label in self.labels_visual:
            self.view.scene.remove(label)
        self.labels_visual = []
        
        # Create new labels for each marker
        if self.show_names and data is not None:
            for i, name in enumerate(marker_names):
                text = Text(name, parent=self.view.scene, color='white')
                self.labels_visual.append(text)
        
        # Initialize selection visual
        if self.selection_visual is None:
            self.selection_visual = Markers(parent=self.view.scene)
        
        # Reset selected markers
        self.selected_markers = []
        
        # Set the initial view
        self.update_view(0)
        
        # Reset camera position
        self.reset_view()
    
    def update_view(self, frame_idx):
        """Update the visualization for the given frame index"""
        if self.data is None or frame_idx >= self.num_frames:
            return
            
        self.current_frame = frame_idx
        
        # Get the current frame's marker positions
        positions = self.data[frame_idx]
        
        # Check for NaN values (missing markers)
        missing_mask = np.isnan(positions).any(axis=1)
        
        # Prepare colors array (one color per marker)
        colors = np.ones((positions.shape[0], 4)) * self.marker_base_color
        
        # Mark selected markers
        for i in self.selected_markers:
            if i < positions.shape[0]:
                colors[i] = self.marker_selected_color
        
        # Mark missing markers
        for i in range(positions.shape[0]):
            if missing_mask[i]:
                colors[i] = self.marker_missing_color
        
        # Use colors only for non-NaN positions
        valid_mask = ~missing_mask
        
        # Update markers - only display non-missing markers
        if np.any(valid_mask):
            self.markers_visual.set_data(
                pos=positions[valid_mask],
                face_color=colors[valid_mask],
                size=self.marker_size,
                edge_width=1,
                edge_color='black'
            )
        else:
            # If all markers are missing, set empty data
            self.markers_visual.set_data(
                pos=np.zeros((0, 3)),
                face_color=np.zeros((0, 4)),
                size=self.marker_size
            )
        
        # Update marker labels
        if self.show_names:
            for i, text in enumerate(self.labels_visual):
                if i < positions.shape[0] and not missing_mask[i]:
                    text.pos = positions[i] + np.array([0, 0, 0.1])  # Offset text slightly above marker
                    text.color = colors[i, :3]  # Use marker color for label
                    text.visible = True
                else:
                    text.visible = False
        
        # Update trajectory lines if enabled
        if self.show_trajectory:
            self.update_trajectories(frame_idx)
            
        # Update skeleton lines if available
        if self.skeleton_lines is not None and len(self.skeleton_pairs) > 0:
            self.update_skeleton(frame_idx)
            
        # Update selection visual
        self.update_selection()
    
    def update_selection(self):
        """Update the visualization of selected markers"""
        if self.data is None or not self.selected_markers:
            # If no selection, hide the selection visual
            self.selection_visual.set_data(
                pos=np.zeros((0, 3)),
                face_color=np.zeros((0, 4)),
                size=self.marker_size
            )
            return
        
        # Get positions of selected markers
        positions = self.data[self.current_frame]
        selected_positions = []
        selected_colors = []
        
        for idx in self.selected_markers:
            if idx < positions.shape[0]:
                pos = positions[idx]
                if not np.isnan(pos).any():  # Skip NaN positions
                    selected_positions.append(pos)
                    selected_colors.append(self.marker_selected_color)
        
        if selected_positions:
            # Show highlight around selected markers
            self.selection_visual.set_data(
                pos=np.array(selected_positions),
                face_color=np.array(selected_colors),
                size=self.marker_size + 4,  # Slightly larger than regular markers
                edge_width=2,
                edge_color='yellow'
            )
        else:
            # Hide selection if no valid selected markers
            self.selection_visual.set_data(
                pos=np.zeros((0, 3)),
                face_color=np.zeros((0, 4)),
                size=self.marker_size
            )
    
    def update_trajectories(self, frame_idx):
        """Update the trajectory lines for the given frame"""
        # Clear existing trajectory lines
        for line in self.trajectory_lines:
            self.view.scene.remove(line)
        self.trajectory_lines = []
        
        # Calculate the start frame for trajectory
        start_frame = max(0, frame_idx - self.trajectory_length)
        
        # Draw trajectory for each marker
        for i in range(self.data.shape[1]):  # For each marker
            # Check if this marker is selected
            is_selected = i in self.selected_markers
            line_color = 'orange' if is_selected else 'yellow'
            line_width = 3 if is_selected else 1
            
            # Get trajectory points
            traj_points = self.data[start_frame:frame_idx+1, i, :]
            
            # Filter out NaN values
            valid_mask = ~np.isnan(traj_points).any(axis=1)
            valid_points = traj_points[valid_mask]
            
            # Only draw if we have at least 2 valid points
            if len(valid_points) > 1:
                line = Line(pos=valid_points, color=line_color, width=line_width, parent=self.view.scene)
                self.trajectory_lines.append(line)
    
    def update_skeleton(self, frame_idx):
        """Update the skeleton connections for the given frame"""
        # Get current positions
        positions = self.data[frame_idx]
        
        # Create lines connecting the pairs
        connect = []
        for pair in self.skeleton_pairs:
            # Skip if either index is out of bounds
            if pair[0] >= len(self.marker_names) or pair[1] >= len(self.marker_names):
                continue
                
            # Skip if either marker is missing (NaN)
            if np.isnan(positions[pair[0]]).any() or np.isnan(positions[pair[1]]).any():
                continue
                
            connect.append([pair[0], pair[1]])
        
        # Update the connections
        if connect and len(connect) > 0:
            if self.skeleton_lines is None:
                self.skeleton_lines = Line(pos=positions, connect=np.array(connect),
                                          color='cyan', width=2, parent=self.view.scene)
            else:
                self.skeleton_lines.set_data(pos=positions, connect=np.array(connect))
        elif self.skeleton_lines is not None:
            # If no valid connections, hide the skeleton lines
            self.skeleton_lines.set_data(pos=np.zeros((0, 3)), connect=np.array([]))
    
    def set_skeleton_pairs(self, pairs):
        """Set the pairs of markers to connect as a skeleton"""
        self.skeleton_pairs = pairs
        if self.data is not None:
            self.update_view(self.current_frame)
    
    def toggle_marker_names(self):
        """Toggle visibility of marker names"""
        self.show_names = not self.show_names
        
        # Remove all existing labels
        for label in self.labels_visual:
            self.view.scene.remove(label)
        self.labels_visual = []
        
        # Re-add labels if needed
        if self.show_names and self.data is not None:
            positions = self.data[self.current_frame]
            missing_mask = np.isnan(positions).any(axis=1)
            
            for i, name in enumerate(self.marker_names):
                if i < positions.shape[0] and not missing_mask[i]:
                    text = Text(name, pos=positions[i] + np.array([0, 0, 0.1]), 
                               parent=self.view.scene, color='white')
                    self.labels_visual.append(text)
        
        # Update view to apply changes
        if self.data is not None:
            self.update_view(self.current_frame)
    
    def toggle_coordinates(self):
        """Toggle between Z-up and Y-up coordinate systems"""
        self.is_z_up = not self.is_z_up
        
        # Update camera orientation based on coordinate system
        if self.is_z_up:
            self.view.camera.elevation = 30  # Look from above
            self.view.camera.azimuth = 45    # Rotate around z-axis
        else:
            self.view.camera.elevation = 0   # Look from side
            self.view.camera.azimuth = 0     # Look along x-axis
            
        # Update the grid orientation
        if self.is_z_up:
            self.grid.set_gl_state('translucent')
            self.grid.transform = visuals.transforms.MatrixTransform()
        else:
            self.grid.set_gl_state('translucent')
            transform = visuals.transforms.MatrixTransform()
            transform.rotate(90, (1, 0, 0))  # Rotate grid to XZ plane
            self.grid.transform = transform
    
    def toggle_trajectory(self):
        """Toggle visibility of marker trajectories"""
        self.show_trajectory = not self.show_trajectory
        
        # Clear existing trajectories if turning off
        if not self.show_trajectory:
            for line in self.trajectory_lines:
                self.view.scene.remove(line)
            self.trajectory_lines = []
        else:
            # Update trajectories if turning on
            if self.data is not None:
                self.update_trajectories(self.current_frame)
    
    def reset_view(self):
        """Reset the camera view to default position"""
        if self.data is None:
            return
            
        # Calculate center of markers
        positions = self.data[self.current_frame]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            return  # No valid markers to center on
            
        center = np.mean(valid_positions, axis=0)
        
        # Set the camera center
        self.view.camera.center = center
        
        # Set the camera distance based on the spread of markers
        max_dist = np.max(np.ptp(valid_positions, axis=0))
        self.view.camera.distance = max_dist * 2.5
        
        # Reset orientation based on coordinate system
        self.toggle_coordinates()
        self.toggle_coordinates()  # Toggle twice to keep same system but reset view
    
    def set_selected_markers(self, marker_indices):
        """Set the selection to the specified marker indices"""
        self.selected_markers = marker_indices
        self.update_view(self.current_frame)
    
    def get_widget(self):
        """Return the native widget for integration with PySide6"""
        return self.canvas.native
        
    def _on_mouse_press(self, event):
        """Handle mouse press events for selection and interaction"""
        if event.button == 1:  # Left click
            # Store the mouse position and button
            self.mouse_state['button'] = event.button
            self.mouse_state['press_pos'] = event.pos
            self.mouse_state['last_pos'] = event.pos
            self.mouse_state['is_dragging'] = False
            
            # Check if a marker was clicked
            if self.data is not None:
                # Get current marker positions
                positions = self.data[self.current_frame]
                
                # Convert mouse position to scene coordinates
                pos = event.pos
                tr = self.view.scene.transform
                
                # Try to select a marker
                selected_idx = self._pick_marker(pos)
                
                if selected_idx is not None:
                    # Store the selected marker
                    self.mouse_state['selected_marker'] = selected_idx
                    
                    # Check if Ctrl key is pressed for multi-selection
                    modifiers = event.modifiers
                    
                    if not (modifiers & 2):  # Ctrl key not pressed
                        # Clear previous selection if Ctrl not pressed
                        self.selected_markers = [selected_idx]
                    else:
                        # Toggle selection if Ctrl pressed
                        if selected_idx in self.selected_markers:
                            self.selected_markers.remove(selected_idx)
                        else:
                            self.selected_markers.append(selected_idx)
                    
                    # Update view to show selection
                    self.update_view(self.current_frame)
                elif not (event.modifiers & 2):  # If Ctrl not pressed and no marker clicked
                    # Clear selection
                    self.selected_markers = []
                    self.update_view(self.current_frame)
    
    def _on_mouse_release(self, event):
        """Handle mouse release events"""
        if event.button == self.mouse_state['button']:
            self.mouse_state['button'] = None
            self.mouse_state['selected_marker'] = None
            self.mouse_state['is_dragging'] = False
    
    def _on_mouse_move(self, event):
        """Handle mouse move events for drag operations"""
        if self.mouse_state['button'] is not None:
            # Store current position
            self.mouse_state['pos'] = event.pos
            
            # Calculate the distance moved
            if self.mouse_state['last_pos'] is not None:
                dx = event.pos[0] - self.mouse_state['last_pos'][0]
                dy = event.pos[1] - self.mouse_state['last_pos'][1]
                
                # Check if we're dragging
                if not self.mouse_state['is_dragging']:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 5:  # Threshold to start dragging
                        self.mouse_state['is_dragging'] = True
                
                # TODO: Implement marker dragging for editing
                # This will require integrating with the main application
                # to update the underlying data
            
            # Update last position
            self.mouse_state['last_pos'] = event.pos
    
    def _on_key_press(self, event):
        """Handle key press events"""
        # TODO: Implement keyboard shortcuts for view manipulation
        pass
    
    def _pick_marker(self, mouse_pos):
        """
        Pick a marker at the given mouse position
        
        Args:
            mouse_pos: The mouse position in screen coordinates
            
        Returns:
            The index of the selected marker, or None if no marker was selected
        """
        if self.data is None:
            return None
            
        # Get current marker positions
        positions = self.data[self.current_frame]
        
        # Filter out NaN positions
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_positions = positions[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_positions) == 0:
            return None
            
        # Convert data to screen coordinates
        tr = self.view.scene.transform
        data_to_screen = self.view.camera.transform * tr
        
        screen_pos = []
        for pos in valid_positions:
            screen_pos.append(data_to_screen.map(pos)[:2])
        screen_pos = np.array(screen_pos)
        
        # Calculate distances to mouse position
        dists = np.sum((screen_pos - mouse_pos)**2, axis=1)
        
        # Find closest marker within a threshold
        threshold = 100  # pixels squared
        if np.min(dists) < threshold:
            closest_idx = np.argmin(dists)
            return valid_indices[closest_idx]
        
        return None
