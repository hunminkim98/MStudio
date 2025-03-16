"""
MarkerStudio v2 - Main Application
A PySide6-based version of the marker trajectory editing tool
"""

import sys
import os
import numpy as np

# Add parent directory to path so we can import from original utils and gui
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QPushButton, QWidget, QComboBox, QLabel, QFrame,
    QFileDialog, QSlider, QSpinBox, QMessageBox, QMenu, QMenuBar, QStatusBar, QSplitter, QCheckBox, QListWidget, QListWidgetItem,
    QGroupBox, QToolBar, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence

# Import Vispy canvas
from v2.vispy_canvas import MarkerStudioCanvas
from v2.data_processor import process_marker_data, detect_missing_markers, interpolate_gaps, process_skeleton_definition

# Import original utility modules
from utils.data_loader import read_data_from_c3d, read_data_from_trc
from utils.data_saver import save_to_trc, save_to_c3d
from utils.trajectory import MarkerTrajectory

# Import Pose2Sim skeletons
try:
    from Pose2Sim.skeletons import (
        BODY_25B, BODY_25, BODY_135, BLAZEPOSE, HALPE_26,
        HALPE_68, HALPE_136, COCO_133, COCO, MPII, COCO_17
    )
except ImportError:
    print("Warning: Pose2Sim not found. Skeleton functionality will be limited.")
    BODY_25B = BODY_25 = BODY_135 = BLAZEPOSE = HALPE_26 = None
    HALPE_68 = HALPE_136 = COCO_133 = COCO = MPII = COCO_17 = None

# We'll handle mouse_handler integration later since it's tightly coupled with matplotlib


class MarkerStudioApp(QMainWindow):
    """Main application window for MarkerStudio v2"""
    
    def __init__(self):
        super().__init__()
        
        # Setup window properties
        self.setWindowTitle("MarkerStudio v2")
        self.resize(1920, 1080)
        
        # Initialize variables
        self.marker_names = []
        self.data = None
        self.original_data = None
        self.processed_data = None
        self.time_data = None
        self.frame_rate = 60.0
        self.num_frames = 0
        self.frame_idx = 0
        self.current_file_path = None
        
        # Skeleton models
        self.available_models = {
            'No skeleton': None,
            'BODY_25B': BODY_25B,
            'BODY_25': BODY_25,
            'BODY_135': BODY_135,
            'BLAZEPOSE': BLAZEPOSE,
            'HALPE_26': HALPE_26,
            'HALPE_68': HALPE_68,
            'HALPE_136': HALPE_136,
            'COCO_133': COCO_133,
            'COCO': COCO,
            'MPII': MPII,
            'COCO_17': COCO_17
        }
        self.current_model = None
        self.skeleton_pairs = []
        
        # Interpolation options
        self.interp_methods = [
            'linear',
            'cubic',
            'nearest',
        ]
        self.interp_method = 'linear'
        
        # Selection options
        self.selected_markers = []
        self.selected_frames = []
        
        # Playback variables
        self.is_playing = False
        self.playback_speed = 1.0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
        
        # Set up the UI
        self.init_ui()
        
        # Initialize trajectory handler
        self.trajectory_handler = MarkerTrajectory()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top button bar
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create buttons with similar functionality to original app
        self.reset_view_button = QPushButton("🎥")
        self.reset_view_button.setFixedWidth(30)
        self.reset_view_button.setToolTip("Reset View")
        self.reset_view_button.clicked.connect(self.reset_main_view)
        
        self.open_button = QPushButton("Open File")
        self.open_button.setToolTip("Open TRC or C3D file")
        self.open_button.clicked.connect(self.open_file)
        
        self.coord_button = QPushButton("Switch to Y-up")
        self.coord_button.setToolTip("Toggle coordinate system orientation")
        self.coord_button.clicked.connect(self.toggle_coordinates)
        
        self.names_button = QPushButton("Hide Names")
        self.names_button.setToolTip("Toggle marker names visibility")
        self.names_button.clicked.connect(self.toggle_marker_names)
        
        self.trajectory_button = QPushButton("Show Trajectory")
        self.trajectory_button.setToolTip("Toggle marker trajectory visibility")
        self.trajectory_button.clicked.connect(self.toggle_trajectory)
        
        self.save_button = QPushButton("Save As...")
        self.save_button.setToolTip("Save data to TRC or C3D file")
        self.save_button.clicked.connect(self.save_as)
        
        # Model selection combo box
        self.model_label = QLabel("Skeleton:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.available_models.keys()))
        self.model_combo.setToolTip("Select skeleton model for visualization")
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        
        # Add buttons to layout
        button_layout.addWidget(self.reset_view_button)
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.coord_button)
        button_layout.addWidget(self.names_button)
        button_layout.addWidget(self.trajectory_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.model_label)
        button_layout.addWidget(self.model_combo)
        button_layout.addStretch()
        
        # Add button frame to main layout
        main_layout.addWidget(button_frame)
        
        # Create Vispy canvas for 3D visualization
        self.vispy_widget = MarkerStudioCanvas()
        main_layout.addWidget(self.vispy_widget.get_widget(), 1)  # Give this widget more space
        
        # Create frame control bar
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        # Add playback controls
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Play/pause animation")
        self.play_button.clicked.connect(self.toggle_animation)
        control_layout.addWidget(self.play_button)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Will be updated when data is loaded
        self.frame_slider.setToolTip("Navigate through frames")
        self.frame_slider.valueChanged.connect(self.on_slider_change)
        control_layout.addWidget(self.frame_slider, 1)  # Give slider more space
        
        # Frame number display and input
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)  # Will be updated when data is loaded
        self.frame_spinbox.setToolTip("Current frame number")
        self.frame_spinbox.valueChanged.connect(self.on_spinbox_change)
        control_layout.addWidget(self.frame_spinbox)
        
        # Add FPS control
        control_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(120)
        self.fps_spinbox.setValue(60)
        self.fps_spinbox.setToolTip("Playback speed in frames per second")
        self.fps_spinbox.valueChanged.connect(self.on_fps_change)
        control_layout.addWidget(self.fps_spinbox)
        
        # Add control frame to main layout
        main_layout.addWidget(control_frame)
        
        # Create marker list panel
        self.marker_list_widget = self.create_marker_list_panel()
        main_layout.addWidget(self.marker_list_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Status bar for frame information
        self.statusBar().showMessage("Ready")
        
        # Set keyboard shortcuts
        self.shortcuts()
        
    def create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_as)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        
        select_all_action = QAction("Select &All Markers", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.triggered.connect(self.select_all_markers)
        edit_menu.addAction(select_all_action)
        
        clear_selection_action = QAction("&Clear Selection", self)
        clear_selection_action.setShortcut("Escape")
        clear_selection_action.triggered.connect(self.clear_marker_selection)
        edit_menu.addAction(clear_selection_action)
        
        edit_menu.addSeparator()
        
        toggle_marker_visibility_action = QAction("Toggle Selected Markers &Visibility", self)
        toggle_marker_visibility_action.triggered.connect(self.toggle_selected_markers_visibility)
        edit_menu.addAction(toggle_marker_visibility_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        toggle_names_action = QAction("Toggle Marker &Names", self)
        toggle_names_action.setShortcut("Ctrl+N")
        toggle_names_action.triggered.connect(self.toggle_marker_names)
        view_menu.addAction(toggle_names_action)
        
        toggle_trajectory_action = QAction("Toggle &Trajectories", self)
        toggle_trajectory_action.setShortcut("Ctrl+T")
        toggle_trajectory_action.triggered.connect(self.toggle_trajectory)
        view_menu.addAction(toggle_trajectory_action)
        
        toggle_coordinate_system_action = QAction("Toggle &Coordinate System", self)
        toggle_coordinate_system_action.setShortcut("Ctrl+C")
        toggle_coordinate_system_action.triggered.connect(self.toggle_coordinates)
        view_menu.addAction(toggle_coordinate_system_action)
        
        view_menu.addSeparator()
        
        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self.reset_main_view)
        view_menu.addAction(reset_view_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Open file action
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Play control actions
        play_action = QAction("Play", self)
        play_action.triggered.connect(self.toggle_animation)
        toolbar.addAction(play_action)
        
        toolbar.addSeparator()
        
        # Visualization controls
        toggle_names_action = QAction("Names", self)
        toggle_names_action.setCheckable(True)
        toggle_names_action.setChecked(True)
        toggle_names_action.triggered.connect(self.toggle_marker_names)
        toolbar.addAction(toggle_names_action)
        
        toggle_trajectory_action = QAction("Trajectories", self)
        toggle_trajectory_action.setCheckable(True)
        toggle_trajectory_action.triggered.connect(self.toggle_trajectory)
        toolbar.addAction(toggle_trajectory_action)
        
        # Add spacer to push right-aligned controls
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Playback speed control
        toolbar.addWidget(QLabel("Speed:"))
        speed_combo = QComboBox()
        speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        speed_combo.setCurrentText("1x")
        speed_combo.currentTextChanged.connect(self.set_playback_speed)
        toolbar.addWidget(speed_combo)
    
    def create_marker_list_panel(self):
        """Create the marker list panel for selection and visibility control"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title_label = QLabel("Markers")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Search box
        self.search_box = QComboBox()
        self.search_box.setEditable(True)
        self.search_box.setPlaceholderText("Search markers...")
        self.search_box.editTextChanged.connect(self.filter_marker_list)
        layout.addWidget(self.search_box)
        
        # Marker list
        self.marker_list = QListWidget()
        self.marker_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.marker_list.itemSelectionChanged.connect(self.on_marker_selection_changed)
        layout.addWidget(self.marker_list)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_markers)
        button_layout.addWidget(select_all_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_marker_selection)
        button_layout.addWidget(clear_btn)
        
        layout.addLayout(button_layout)
        
        return panel
    
    def shortcuts(self):
        """Set up keyboard shortcuts"""
        # Space and Enter for play/pause
        self.play_shortcut = Qt.Key_Space
        # Left and Right arrows for frame navigation
        self.prev_shortcut = Qt.Key_Left
        self.next_shortcut = Qt.Key_Right
        # Escape to stop
        self.stop_shortcut = Qt.Key_Escape
        
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        key = event.key()
        
        if key == self.play_shortcut:
            self.toggle_animation()
        elif key == self.prev_shortcut:
            self.prev_frame()
        elif key == self.next_shortcut:
            self.next_frame()
        elif key == self.stop_shortcut:
            self.stop_animation()
        else:
            super().keyPressEvent(event)
    
    def reset_main_view(self):
        """Reset the 3D view to default position"""
        self.vispy_widget.reset_view()
    
    def open_file(self):
        """Open a file dialog to select data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Motion Files (*.trc *.c3d);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load data based on file extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.trc':
                header_lines, data, marker_names, frame_rate = read_data_from_trc(file_path)
            elif extension == '.c3d':
                header_lines, data, marker_names, frame_rate = read_data_from_c3d(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
            
            # Store raw data
            self.data = data
            self.original_data = data.copy()
            self.marker_names = marker_names
            self.frame_rate = frame_rate
            self.time_data = data['Time'].values if 'Time' in data else np.arange(len(data)) / frame_rate
            self.num_frames = len(data)
            self.frame_idx = 0
            self.current_file_path = file_path
            
            # Process data for visualization
            self.processed_data = process_marker_data(data, marker_names)
            
            # Update UI elements
            self.frame_slider.setMaximum(self.num_frames - 1)
            self.frame_spinbox.setMaximum(self.num_frames - 1)
            self.fps_spinbox.setValue(int(frame_rate))
            
            # Update 3D visualization
            self.vispy_widget.set_data(self.processed_data, marker_names)
            
            # Update status bar
            self.statusBar().showMessage(f"Loaded {file_path} with {len(marker_names)} markers and {self.num_frames} frames")
            
            # Reset view to fit all markers
            self.reset_main_view()
            
            # Apply current model if one is selected
            current_model = self.model_combo.currentText()
            if current_model != "No skeleton":
                self.on_model_change(current_model)
            
            # Update marker list
            self.update_marker_list()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def toggle_coordinates(self):
        """Toggle between Z-up and Y-up coordinate systems"""
        self.vispy_widget.toggle_coordinates()
        
        # Update button text
        if self.vispy_widget.is_z_up:
            self.coord_button.setText("Switch to Y-up")
        else:
            self.coord_button.setText("Switch to Z-up")
    
    def toggle_marker_names(self):
        """Toggle visibility of marker names"""
        self.vispy_widget.toggle_marker_names()
        
        # Update button text
        if self.vispy_widget.show_names:
            self.names_button.setText("Hide Names")
        else:
            self.names_button.setText("Show Names")
    
    def toggle_trajectory(self):
        """Toggle visibility of marker trajectories"""
        self.vispy_widget.toggle_trajectory()
        
        # Update button text
        if self.vispy_widget.show_trajectory:
            self.trajectory_button.setText("Hide Trajectory")
        else:
            self.trajectory_button.setText("Show Trajectory")
    
    def save_as(self):
        """Save data to a file"""
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        
        # Get default filename from current file path, if any
        default_dir = os.path.dirname(self.current_file_path) if self.current_file_path else ""
        default_name = os.path.basename(self.current_file_path) if self.current_file_path else "untitled.trc"
        default_path = os.path.join(default_dir, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save As", default_path, "Motion Files (*.trc *.c3d);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save data based on file extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.trc':
                save_to_trc(file_path, self.data, self.marker_names)
            elif extension == '.c3d':
                save_to_c3d(file_path, self.data, self.marker_names)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
            
            self.statusBar().showMessage(f"Saved to {file_path}")
            self.current_file_path = file_path
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def on_model_change(self, model_name):
        """Handle skeleton model change"""
        self.current_model = self.available_models[model_name]
        
        # Clear existing skeleton pairs
        self.skeleton_pairs = []
        
        # Set up new skeleton pairs if a model is selected and we have data
        if self.current_model is not None and model_name != 'No skeleton' and self.marker_names:
            # Process skeleton definition to get marker pairs
            self.skeleton_pairs = process_skeleton_definition(self.current_model, self.marker_names)
        
        # Update the visualization
        self.vispy_widget.set_skeleton_pairs(self.skeleton_pairs)
    
    def toggle_animation(self):
        """Toggle playback of animation"""
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()
    
    def start_animation(self):
        """Start playback animation"""
        if self.data is None:
            return
        
        self.is_playing = True
        self.play_button.setText("Pause")
        
        # Set timer based on FPS
        interval = int(1000 / self.fps_spinbox.value())
        self.playback_timer.start(interval)
    
    def stop_animation(self):
        """Stop playback animation"""
        self.is_playing = False
        self.play_button.setText("Play")
        self.playback_timer.stop()
    
    def next_frame(self):
        """Move to next frame"""
        if self.data is None:
            return
        
        # Increment frame index, loop back to start if at end
        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        self.update_current_frame()
    
    def prev_frame(self):
        """Move to previous frame"""
        if self.data is None:
            return
        
        # Decrement frame index, loop to end if at start
        self.frame_idx = (self.frame_idx - 1) % self.num_frames
        self.update_current_frame()
    
    def on_slider_change(self, value):
        """Handle frame slider value change"""
        if self.frame_idx != value:
            self.frame_idx = value
            self.update_current_frame(update_slider=False, update_spinbox=True)
    
    def on_spinbox_change(self, value):
        """Handle frame spinbox value change"""
        if self.frame_idx != value:
            self.frame_idx = value
            self.update_current_frame(update_slider=True, update_spinbox=False)
    
    def on_fps_change(self, value):
        """Handle FPS spinbox value change"""
        if self.is_playing:
            # Update timer interval if playing
            interval = int(1000 / value)
            self.playback_timer.setInterval(interval)
    
    def update_current_frame(self, update_slider=True, update_spinbox=True):
        """Update UI and visualization for the current frame"""
        # Update slider and spinbox if requested
        if update_slider:
            self.frame_slider.setValue(self.frame_idx)
        if update_spinbox:
            self.frame_spinbox.setValue(self.frame_idx)
        
        # Update 3D visualization
        self.vispy_widget.update_view(self.frame_idx)
        
        # Update status bar
        if self.time_data is not None and self.frame_idx < len(self.time_data):
            time_value = self.time_data[self.frame_idx]
            self.statusBar().showMessage(f"Frame: {self.frame_idx+1}/{self.num_frames} | Time: {time_value:.3f}s")
        else:
            self.statusBar().showMessage(f"Frame: {self.frame_idx+1}/{self.num_frames}")
    
    def select_markers(self, marker_indices):
        """Select markers by their indices"""
        self.selected_markers = marker_indices
        
        # Highlight selected markers in the visualization
        self.vispy_widget.set_selected_markers(marker_indices)
    
    def select_frames(self, start_frame, end_frame):
        """Select a range of frames"""
        self.selected_frames = list(range(start_frame, end_frame + 1))
        
        # Update visualization to highlight selected frame range
        # TODO: Update the vispy canvas to show selected frame range
    
    def update_marker_list(self):
        """Update the marker list panel with current marker names"""
        self.marker_list.clear()
        
        for name in self.marker_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.marker_list.addItem(item)
    
    def filter_marker_list(self, filter_text):
        """Filter the marker list based on search text"""
        for i in range(self.marker_list.count()):
            item = self.marker_list.item(i)
            item.setHidden(filter_text.lower() not in item.text().lower())
    
    def on_marker_selection_changed(self):
        """Handler for marker selection changes in the list widget"""
        selected_indices = [self.marker_list.row(item) for item in self.marker_list.selectedItems()]
        self.vispy_widget.set_selected_markers(selected_indices)
    
    def select_all_markers(self):
        """Select all markers in the list"""
        self.marker_list.selectAll()
    
    def clear_marker_selection(self):
        """Clear all marker selections"""
        self.marker_list.clearSelection()
    
    def toggle_selected_markers_visibility(self):
        """Toggle visibility of selected markers"""
        # Currently not implemented - will need to modify the canvas
        # to support marker visibility toggling
        pass
    
    def set_playback_speed(self, speed_text):
        """Set the playback speed based on the combo box selection"""
        if not self.playback_timer.isActive():
            return
            
        # Parse the speed factor from the text
        speed_factor = float(speed_text.rstrip('x'))
        
        # Calculate new interval
        base_interval = int(1000 / self.fps_spinbox.value())
        new_interval = int(base_interval / speed_factor)
        
        # Restart timer with new interval
        self.playback_timer.stop()
        self.playback_timer.start(new_interval)
    
    def show_about(self):
        """Show the about dialog"""
        # Implement about dialog
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply stylesheet for a dark theme similar to the original app
    dark_style = """
    QMainWindow, QWidget {
        background-color: #2d2d2d;
        color: #f0f0f0;
    }
    QPushButton {
        background-color: #444444;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 5px;
        color: #f0f0f0;
    }
    QPushButton:hover {
        background-color: #555555;
    }
    QPushButton:pressed {
        background-color: #666666;
    }
    QComboBox, QSpinBox {
        background-color: #444444;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 3px;
        color: #f0f0f0;
    }
    QSlider::groove:horizontal {
        height: 8px;
        background: #444444;
        margin: 2px 0;
    }
    QSlider::handle:horizontal {
        background: #999999;
        border: 1px solid #999999;
        width: 12px;
        border-radius: 3px;
        margin: -4px 0;
    }
    """
    
    app.setStyleSheet(dark_style)
    
    # Create and show the main window
    main_window = MarkerStudioApp()
    main_window.show()
    
    sys.exit(app.exec())
